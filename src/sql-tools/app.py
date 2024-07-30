import os
import dotenv
import pandas as pd
from io import StringIO
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import streamlit as st
from langchain import agents
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent import RunnableAgent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from langchain.prompts.chat import ChatPromptTemplate
from langchain.agents import LLMSingleActionAgent, AgentOutputParser
from langchain.chains import LLMChain
from promptflow.tracing import start_trace
import random
import json
import pytz
from datetime import datetime

dotenv.load_dotenv()
# start a trace session, and print a url for user to check trace
start_trace()

# enable langchain instrumentation
from opentelemetry.instrumentation.langchain import LangchainInstrumentor

instrumentor = LangchainInstrumentor()
if not instrumentor.is_instrumented_by_opentelemetry:
    instrumentor.instrument()

st.title("💬 AI bot that talk to a database")
st.caption("🚀 A Bot that can use tools to answer questions about relational data")

def get_session_id() -> str:
    id = random.randint(0, 1000000)
    return "00000000-0000-0000-0000-" + str(id).zfill(12)

if "session_id" not in st.session_state:
    st.session_state["session_id"] = get_session_id()
    print("started new session: " + st.session_state["session_id"])
    st.write("You are running in session: " + st.session_state["session_id"])

llm: AzureChatOpenAI = None
if "AZURE_OPENAI_API_KEY" in os.environ:
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_VERSION"),
        temperature=0,
        streaming=True
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
        streaming=True
    )

@tool
def get_current_location(input: str) -> str:
    "Get the current timezone location of the user."
    return "Europe/Berlin"

@tool
def get_current_time(location: str) -> str:
    "Get the current time in the given location. The pytz is used to get the timezone for that location. Location names should be in a format like America/New_York, Asia/Bangkok, Europe/London. Anything in Germany should be Europe/Berlin"
    try:
        print("get current time for location: ", location)
        location = str.replace(location, " ", "")
        location = str.replace(location, "\"", "")
        location = str.replace(location, "\n", "")
        # Get the timezone for the city
        timezone = pytz.timezone(location)

        # Get the current time in the timezone
        now = datetime.now(timezone)
        current_time = now.strftime("%I:%M:%S %p")

        return current_time
    except Exception as e:
        print("Error: ", e)
        return "Sorry, I couldn't find the timezone for that location."

tools = [get_current_location, get_current_time]

df = pd.read_csv(
    "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"
)

agent_executor = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS, 
    allow_dangerous_code=True,
    handle_parsing_errors=True,
    # extra_tools=tools
)

if prompt := st.chat_input():

    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor.invoke(
            {"input": prompt}, {"callbacks": [st_callback]}
        )
        st.write(response["output"])
