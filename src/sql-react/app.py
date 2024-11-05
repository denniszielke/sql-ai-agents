import os
import sys
import struct
from sqlalchemy import create_engine, event
from sqlalchemy.engine.url import URL
import dotenv
import pandas as pd
import tiktoken
from typing import Any, Dict, List, Literal, Annotated, TypedDict
from azure import identity
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
import random
from token_counter import TokenCounterCallback
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from opentelemetry import trace, trace as trace_api
dotenv.load_dotenv()

st.set_page_config(
    page_title="AI reactive bot that can interact with a database"
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

st.title("💬 AI reactive database query bot")
st.caption("🚀 A Bot that can use iterative tools to answer questions about relational data")

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

connection_string = os.environ["AZURE_SQL_CONNECTIONSTRING"]

driver = '{ODBC Driver 18 for SQL Server}'
odbc_str = 'mssql+pyodbc:///?odbc_connect=' \
                'Driver='+driver+ \
                ';' + os.getenv("AZURE_SQL_CONNECTIONSTRING")

from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri(odbc_str)
print(db.dialect)
print(db.get_usable_table_names())
# print(db.get_table_info())

from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
print(tools)

# list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
# get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

@tool
def db_query_tool(query: str) -> str:
    """
    Execute a SQL Server query against the database and get back the result.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    result = db.run_no_throw(query)
    if not result:
        return "Error: Query failed. Please rewrite your query and try again."
    return result

tools.append(db_query_tool)

from langchain_core.messages import SystemMessage

sql_react_tool_prompt = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQL Server query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

DO NOT USE the QuerySQLCheckerTool.

You have access to the following tables: {table_names}

To start you should ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant tables.

Assistant is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. 
Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 

TOOLS:

------

Assistant has access to the following tools:

{tools}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Provide only ONE action per $JSON_BLOB, as shown:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

To use a tool, please use the following format:

```

Thought: Do I need to use a tool? Yes

Action: the action to take, should be one of [{tool_names}]
Valid "action" values: "Final Answer" or {tool_names}
```
$JSON_BLOB
```

Action Input: the input to the action

Observation: the result of the action

```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```

Thought: Do I need to use a tool? No
Thought: I know what to respond
Action:
```
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}

```

Begin!

Previous conversation history:

{chat_history}

New input: {input}

{agent_scratchpad}

"""

from langchain.agents import create_structured_chat_agent
from langchain import agents
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(sql_react_tool_prompt)
agent = create_structured_chat_agent(llm, tools, prompt)

agent_executor = agents.AgentExecutor(
        name="Tools Agent",
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=10, return_intermediate_steps=False, 
        # handle errors
        error_message="I'm sorry, I couldn't understand that. Please try to describe your question in a different way.",
    )

human_query = st.chat_input()

if human_query is not None and human_query != "":

    st.session_state.chat_history.append(HumanMessage(human_query))

    with st.chat_message("Human"):
        st.markdown(human_query)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        with tracer.start_as_current_span("agent-chain") as span:
            response = agent_executor.invoke(
                {"input": human_query, "chat_history": st.session_state.chat_history, "table_names": db.get_usable_table_names()}, {"callbacks": [st_callback]}, 
            )

            span.set_attribute("gen_ai.response.completion_token",callback.completion_tokens) 
            span.set_attribute("gen_ai.response.prompt_tokens", callback.prompt_tokens) 
            span.set_attribute("gen_ai.response.total_tokens", callback.total_tokens)
            
            ai_response = st.write(response["output"])
            st.write("The total number of tokens used in this conversation was: ", callback.total_tokens)

            print(response)

