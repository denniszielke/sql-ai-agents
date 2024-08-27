import os
import random
from typing import Any, Dict, List, Literal, Annotated, TypedDict

import dotenv
from langchain_openai import AzureChatOpenAI
import streamlit as st
import tiktoken
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from IPython.display import Image
from langchain.agents.agent import AgentAction
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.outputs.llm_result import LLMResult
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

def num_tokens_from_messages(messages: List[str]) -> int:
    '''
    Calculate the number of tokens in a list of messages. This is a somewhat naive implementation that simply concatenates 
    the messages and counts the tokens in the resulting string. A more accurate implementation would take into account the 
    fact that the messages are separate and should be counted as separate sequences.
    If available, the token count should be taken directly from the model response.
    '''
    encoding = tiktoken.encoding_for_model("gpt-4o")
    num_tokens = 0
    content = ' '.join(messages)
    num_tokens += len(encoding.encode(content))

    return num_tokens

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

class TokenCounterCallback(BaseCallbackHandler):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        self.completion_tokens += 1

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any) -> Any:
        self.prompt_tokens += num_tokens_from_messages( [message.content for message in messages[0]])
         

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
        callbacks=[callback]
    )

driver = '{ODBC Driver 18 for SQL Server}'
odbc_str = 'mssql+pyodbc:///?odbc_connect=' \
                'Driver='+driver+ \
                ';' + os.getenv("AZURE_SQL_CONNECTIONSTRING")

db = SQLDatabase.from_uri(odbc_str)

def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

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

# Define the state for the agent
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Define a new graph
workflow = StateGraph(State)

#-----------------------------------------------------------------------------------------------

def call_model(prompt: str, tools: dict[str, BaseTool], input: Any) -> dict[str, list[AIMessage]]:
    query_gen_prompt = ChatPromptTemplate.from_messages(
        [("system", prompt), ("placeholder", "{messages}")]
    )

    if not tools:
        model_get_schema = query_gen_prompt | llm
    else:
        model_get_schema = query_gen_prompt | llm.bind_tools(
            list(tools.values()), tool_choice="required"
        )

    message = model_get_schema.invoke(input)
    tool_messages = []
    for tool_call in message.tool_calls:
        selected_tool = tools[tool_call["name"]]
        result = selected_tool.invoke(tool_call["args"])    
        tool_messages.append(
            ToolMessage(
                content=result,
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": [message] + tool_messages}

#-----------------------------------------------------------------------------------------------

def first(state: State) -> dict[str, list[AIMessage]]:
    prompt = """You are a SQL expert with a strong attention to detail.
        Use the provided tool to extract the list of tables in the database.
    """
    return call_model(prompt, {"sql_db_list_tables": list_tables_tool}, {"messages": state["messages"]})

workflow.add_node("first_tool_call", first)

#-----------------------------------------------------------------------------------------------

def second(state: State) -> dict[str, list[AIMessage]]:
    prompt = """You are a SQL expert with a strong attention to detail.
        Take the tools to extract the schema and table information from the database.
    """
    return call_model(prompt, {"sql_db_schema": get_schema_tool}, {"messages": state["messages"]})

workflow.add_node("second_tool_call", second)

#-----------------------------------------------------------------------------------------------

def query_gen_node(state: State):
    query_gen_system = """You are a SQL expert with a strong attention to detail.

    Given an input question, output a syntactically correct SQL Server query.

    When generating the query:

    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.

    If you have enough information to answer the input question, simply invoke the appropriate tool to submit the final answer to the user.

    DO NOT make any DML statements (CREATE, INSERT, UPDATE, DELETE, DROP etc.) to the database."""

    return call_model(query_gen_system, {}, state)


workflow.add_node("query_gen", query_gen_node)

#-----------------------------------------------------------------------------------------------

def model_check_query(state: State) -> dict[str, list[AIMessage]]:
    """
    Use this tool to double-check if your query is correct before executing it.
    """

    query_check_system = """You are a SQL expert with a strong attention to detail.
    Double check the SQL Server query for common mistakes, including:
    - Using NOT IN with NULL values
    - Using UNION when UNION ALL should have been used
    - Using BETWEEN for exclusive ranges
    - Data type mismatch in predicates
    - Properly quoting identifiers
    - Using the correct number of arguments for functions
    - Casting to the correct data type
    - Using the proper columns for joins
    - Not using any create or drop statements

    If there are any of the above mistakes, rewrite the query. If there are no mistakes, 
    return the correct query and execute it against the database by using the tool provided.
    Return the results to the user.

    Result formatting guidlines:
    If the resulting message contains tabular data, you should format it into a table. 
    If the resulting message contains a list of items, you should format it into a list.
    If the resulting message contains a single item, you should format it into a sentence."""

    query_check_prompt = ChatPromptTemplate.from_messages(
        [("system", query_check_system), ("placeholder", "{messages}")]
    )
    query_check = query_check_prompt | llm.bind_tools(
        [db_query_tool], tool_choice="required"
    )
    test = call_model(query_check_system, {"db_query_tool": db_query_tool}, {"messages": [state["messages"][-1]]})
    message = {"messages": [query_check.invoke({"messages": [state["messages"][-1]]})]}
    return test

# Add a node for the model to check the query before executing it
workflow.add_node("correct_and_execute_query", model_check_query)

#-----------------------------------------------------------------------------------------------

def format_gen_node(state: State):
    format_system = """
    You receive an unformatted input message and need to format it into a human readable, meaningful response.

    If the input message contains tabular data, you should format it into a table. 
    If the input message contains a list of items, you should format it into a list.
    If the input message contains a single item, you should format it into a sentence.
    ---
    {input}
    """
    #return {"messages": [format_gen.invoke({"input": [state["messages"][-1].content]})]}
    return call_model(format_system, {}, {"input": [state["messages"][-1].content]})

#workflow.add_node("format_gen", format_gen_node)

#-----------------------------------------------------------------------------------------------

# Define a conditional edge to decide whether to continue or end the workflow
def should_continue(state: State) -> Literal["__end__", "query_gen"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.content.startswith("Error:"):
        return "query_gen"
    else:
        return "__end__"


# Specify the edges between the nodes
workflow.add_edge(START, "first_tool_call")
workflow.add_edge("first_tool_call", "second_tool_call")
workflow.add_edge("second_tool_call", "query_gen")
workflow.add_edge("query_gen", "correct_and_execute_query")
workflow.add_conditional_edges(
    "correct_and_execute_query",
    should_continue,
)

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
        st.write("The total number of tokens used in this conversation was: ", callback.completion_tokens + callback.prompt_tokens)
        st.image(
            app.get_graph(xray=True).draw_mermaid_png(
                draw_method=MermaidDrawMethod.API,
            )
        )
        span.set_attribute("gen_ai.response.completion_token",callback.completion_tokens) 
        span.set_attribute("gen_ai.response.prompt_tokens", callback.prompt_tokens) 
        span.set_attribute("gen_ai.response.total_tokens", callback.completion_tokens + callback.prompt_tokens)
