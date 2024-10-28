import os
import sys
import dotenv
import pandas as pd
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
# from promptflow.tracing import start_trace
import random

dotenv.load_dotenv()

# enable langchain instrumentation
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from opentelemetry import trace, trace as trace_api
from token_counter import TokenCounterCallback

st.set_page_config(
    page_title="AI bot that can use a database as tools"
)

@st.cache_resource
def setup_tracing():
    exporter = AzureMonitorTraceExporter.from_connection_string(
        os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"]
    )
    tracer_provider = TracerProvider()
    trace.set_tracer_provider(tracer_provider)
    tracer = trace.get_tracer(__name__)
    span_processor = BatchSpanProcessor(exporter, schedule_delay_millis=60000)
    trace.get_tracer_provider().add_span_processor(span_processor)
    LangchainInstrumentor().instrument()
    return tracer

tracer = setup_tracing()

st.title("💬 AI bot that talk to a database")
st.caption("🚀 A Bot that can use tools to answer questions about relational data")

def get_session_id() -> str:
    id = random.randint(0, 1000000)
    return "00000000-0000-0000-0000-" + str(id).zfill(12)

@st.cache_resource
def create_session(st: st) -> None:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = get_session_id()
        print("started new session: " + st.session_state["session_id"])
        st.write("You are running in session: " + st.session_state["session_id"])

create_session(st)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, ToolMessage):
        with st.chat_message("Tool"):
            st.markdown(message.content)
    else:
        with st.chat_message("Agent"):
            st.markdown(message.content)

callback = TokenCounterCallback()

llm: AzureChatOpenAI = None
if "AZURE_OPENAI_API_KEY" in os.environ:
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_VERSION"),
        temperature=0,
        streaming=True,
        model_kwargs={"stream_options":{"include_usage": True}},
        callbacks=[callback]
    )
else:
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    llm = AzureChatOpenAI(
        azure_ad_token_provider=token_provider,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_VERSION"),
        temperature=0,
        openai_api_type="azure_ad",
        streaming=True,
        model_kwargs={"stream_options":{"include_usage": True}},
        callbacks=[callback]
    )

driver = '{ODBC Driver 18 for SQL Server}'
odbc_str = 'mssql+pyodbc:///?odbc_connect=' \
                'Driver='+driver+ \
                ';' + os.getenv("AZURE_SQL_CONNECTIONSTRING")

from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri(odbc_str)
print(db.dialect)
print(db.get_usable_table_names())
# db.run("select * from [dbo].[Categories]")

from langchain.chains import create_sql_query_chain

from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.runnables import chain

generatedSQLquery = ''

@chain
def query_inspector(text):
    text = str.replace(text, "```sql", "")
    text = str.replace(text, "```", "")
    print("Query inspector: ", text)
    generatedSQLquery = text
    return text

execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)
chain = write_query | execute_query

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question. Please provide a clear and concise answer, output the SQL query and explain shortly the essential part of the SQL query that was used to generate the result.

DO NOT make any changing statements to the database by using CREATE, INSERT, UPDATE, DELETE, DROP etc.
    
Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | query_inspector | execute_query
    )
    | answer_prompt
    | llm
    | StrOutputParser()
)

human_query = st.chat_input()

if human_query is not None and human_query != "":

    st.session_state.chat_history.append(HumanMessage(human_query))

    with tracer.start_as_current_span("agent-chain") as span:

        with st.chat_message("Human"):
            st.markdown(human_query)

        with st.chat_message("Agent"):
            response = chain.invoke({"question": human_query})
            print(response)
            print(chain.get_prompts()[0].pretty_print())
            st.write(response)

        span.set_attribute("gen_ai.response.completion_token",callback.completion_tokens) 
        span.set_attribute("gen_ai.response.prompt_tokens", callback.prompt_tokens) 
        span.set_attribute("gen_ai.response.total_tokens", callback.total_tokens)
            
        st.write("The total number of tokens used in this conversation was: ", callback.total_tokens)
