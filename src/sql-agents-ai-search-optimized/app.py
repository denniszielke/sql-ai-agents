import ast
import os
import sys
import random
from typing import Any, Dict, List, Literal, Annotated, TypedDict, cast
from uuid import UUID
import dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import streamlit as st
import tiktoken
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from IPython.display import Image
from langchain.agents.agent import AgentAction
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from langchain_core.tools import tool, BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace, trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import hashlib
from langchain_core.documents import Document
from langchain_community.vectorstores.azuresearch import AzureSearch
from token_counter import TokenCounterCallback

dotenv.load_dotenv()

st.set_page_config(
    page_title="AI agentic bot that can interact with a database"
)

st.title("💬 AI agentic RAG")
st.caption("🚀 A Bot that can use an agent to retrieve, augment, generate, validate and iterate")

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

def get_session_id() -> str:
    id = random.randint(0, 1000000)
    return "00000000-0000-0000-0000-" + str(id).zfill(12)

@st.cache_resource
def create_session(st: st) -> None:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = get_session_id()
        print("started new session: " + st.session_state["session_id"])
        st.write("You are running in session: " + st.session_state["session_id"])

tracer = setup_tracing()
create_session(st)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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
    embeddings_model = AzureOpenAIEmbeddings(    
        azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version = os.getenv("AZURE_OPENAI_VERSION"),
        model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
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
    embeddings_model = AzureOpenAIEmbeddings(    
        azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version = os.getenv("AZURE_OPENAI_VERSION"),
        model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        azure_ad_token_provider = token_provider
    )

driver = '{ODBC Driver 18 for SQL Server}'
odbc_str = 'mssql+pyodbc:///?odbc_connect=' \
                'Driver='+driver+ \
                ';' + os.getenv("AZURE_SQL_CONNECTIONSTRING")

db = SQLDatabase.from_uri(odbc_str)

@st.cache_resource
def create_search_index() -> AzureSearch:
    index_name: str = "sql-server-index"

    return AzureSearch(
        azure_search_endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
        azure_search_key=os.getenv("AZURE_AI_SEARCH_KEY"),
        index_name=index_name,
        embedding_function=embeddings_model.embed_query,
    )

search_index = create_search_index()

# Define the state for the agent
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Define a new graph
workflow = StateGraph(State)

#-----------------------------------------------------------------------------------------------

@tool
def db_query_tool(query: str) -> str:
    """
    Execute a SQL Server query against the database and get back the result.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    result = db.run_no_throw(query)
    if not result:
        return "Error: The query is not correct. Please rewrite the query and try again."
    return result

@tool
def index_search_tool(query: str) -> List[str]:
    """
    Search for relevant schema information in the vector index based on the user's query.

    Args:
        query (str): The input query. This is used to search the vector index.

    Returns:
        List[str]: The resulting list of schema information in the form [TableName:Name;ColumnName:Name;DataType:Type;RelevanceScore:Float].

    """

    results = search_index.similarity_search_with_relevance_scores(
        query=query,
        k=20,
        score_threshold=0.75,
    )
    return [f"{result[0].page_content}RelevanceScore:{result[1]}" for result in results]

def db_schema_tool(fields: List[str]) -> List[Document]:
    """
    Returns the schema information of the database for the given fields. Returns only relevant information,
    such as the table name, column name, and data type. It does not return the data as sql.
    """

    query = """
            SELECT table_name, column_name, data_type
            FROM INFORMATION_SCHEMA.columns
            WHERE table_schema='dbo'
        """

    result = db_query_tool(query)
    list_of_tuples = ast.literal_eval(result)
    results = []
    hash_algo = hashlib.sha256()

    for table, column, data_type in list_of_tuples:
        hash_algo.update(f"{table};{column};{data_type}".encode('utf-8'))
        id = hash_algo.hexdigest()
        results.append(Document(
            id = id,
            page_content=f"TableName:{table};ColumnName:{column};DataType:{data_type};",
        ))
    return results

@st.cache_resource
def index_database() -> None:
    docs = db_schema_tool([])
    search_index.add_texts(
        keys=[doc.id for doc in docs],
        texts=[doc.page_content for doc in docs],
    )

index_database()

#-----------------------------------------------------------------------------------------------

sql_schema_tools = [index_search_tool]

def query_compiler(state: State) -> dict[str, list[AIMessage]]:
    prompt = """You are a SQL expert with a strong attention to detail.

    Your task is to output a syntactically correct SQL Server query to answer the users question. 
    You answer is based on data you retrieve from your provide tools and your input.

    Warning: Make sure, that the user does not want you to make any DML statements (CREATE, INSERT, UPDATE, DELETE, DROP etc.) to the database. 
    If the users question requires a DML statement, return a final message stating the problem. Prefix the message with "Error:".

    Given an input, retrieve all required information from your tools to answer the question. 
    Filter the information to only include the necessary columns. Be as short as possible.
    Use ONLY the tools provided, to extract the information. Make sure, that you create your queries based on the information you have exacly. 
    Use only the table names provided.

    ---
    Input Data: {input}
    ---

    When generating the query:

    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.

    DO NOT make any DML statements (CREATE, INSERT, UPDATE, DELETE, DROP etc.) to the database. 
    """

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompt), ("placeholder", "{input}")]
    )
    call = prompt_template | llm.bind_tools(sql_schema_tools, tool_choice="auto")
    return {"messages": [call.invoke({"input": state["messages"]})]}


workflow.add_node("query_compiler", query_compiler)
workflow.add_node("sql_tools", ToolNode(sql_schema_tools))

#-----------------------------------------------------------------------------------------------

def query_check(state: State) -> dict[str, list[AIMessage]]:
    prompt = """You are a SQL expert with a strong attention to detail.
    Double check the SQL Server query given as input for common mistakes, including:
    - Using NOT IN with NULL values
    - Using UNION when UNION ALL should have been used
    - Using BETWEEN for exclusive ranges
    - Data type mismatch in predicates
    - Properly quoting identifiers
    - Using the correct number of arguments for functions
    - Casting to the correct data type
    - Using the proper columns for joins
    - Not using any create or drop statements
    - Use cast or convert methods to bring date string into the correct data type

    ---
    Query: {input}
    ---

    If there are mistakes, rewrite the query and output the changes. If there are no mistakes, 
    run the query via the tool. In that case, do not output the query again."""

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompt), ("placeholder", "{input}")]
    )
    call = prompt_template | llm.bind_tools([db_query_tool], tool_choice="auto")
    return {"messages": [call.invoke({"input": [state["messages"][-1]]})]}

workflow.add_node("correct_and_execute_query", query_check)
workflow.add_node("db_query_tool", ToolNode([db_query_tool]))

#-----------------------------------------------------------------------------------------------

def format_gen_node(state: State):
    prompt = """
    You receive an unformatted input message and need to format it into a human readable, meaningful response.

    If the input message contains tabular data, you should format it into a table. 
    If the input message contains a list of items, you should format it into a list.
    If the input message contains a single item, you should format it into a sentence.
    ---
    {input}
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompt), ("placeholder", "{input}")]
    )
    call = prompt_template | llm
    message = [state["messages"][-1]][-1]
    return {"messages": [call.invoke({"input": [message.content]})]}

workflow.add_node("format_gen", format_gen_node)

#-----------------------------------------------------------------------------------------------

def should_continue(state: State) -> Literal["sql_tools", "correct_and_execute_query"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "sql_tools"
    return "correct_and_execute_query"
    
def should_continue2(state: State) -> Literal["format_gen", "query_compiler"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.content.startswith("Error:"):
        return "query_compiler"
    else:
        return "format_gen"

# Specify the edges between the nodes
workflow.add_edge(START, "query_compiler")
workflow.add_edge("sql_tools", "query_compiler")
workflow.add_conditional_edges(
    "query_compiler",
    should_continue, # -> correct_and_execute_query
)
workflow.add_edge("correct_and_execute_query", "db_query_tool")
workflow.add_conditional_edges(
    "db_query_tool",
    should_continue2, # -> format_gen
)
workflow.add_edge("format_gen", END)

app = workflow.compile()

human_query = st.chat_input()

if human_query is not None and human_query != "":

    st.session_state.chat_history.append(HumanMessage(human_query))

    inputs = {
        "messages": [
            ("user", human_query),
        ]
    }

    with st.chat_message("Human"):
        st.markdown(human_query)

    with tracer.start_as_current_span("agent-chain") as span:
        for event in app.stream(inputs):  
            for value in event.values():
                print(value)
                message = value["messages"][-1]
                if ( isinstance(message, AIMessage) ):
                    print("AI:", message.content)
                    with st.chat_message("Agent"):
                        if (message.content == ''):
                            toolusage = ''
                            for tool in message.tool_calls:
                                print(tool)
                                toolusage += "name: " + tool["name"] + "  \n\n"
                            st.write("Using the following tools: \n", toolusage)
                        else:
                            st.write(message.content)
                
                if ( isinstance(message, ToolMessage) ):
                    print("Tool:", message.content)
                    with st.chat_message("Tool"):
                        st.write(message.content.replace('\n\n', ''))

        st.write("The conversation has ended. Those were the steps taken to answer your query.")
        st.write("The total number of tokens used in this conversation was: ", callback.total_tokens)
        st.image(
            app.get_graph(xray=True).draw_mermaid_png(
                draw_method=MermaidDrawMethod.API,
            )
        )
        span.set_attribute("gen_ai.response.completion_token",callback.completion_tokens) 
        span.set_attribute("gen_ai.response.prompt_tokens", callback.prompt_tokens) 
        span.set_attribute("gen_ai.response.total_tokens", callback.total_tokens)