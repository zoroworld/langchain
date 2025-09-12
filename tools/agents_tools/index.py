from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os


# Step 1 :  setup of hugging face=====================================
# Load environment variables from .env
load_dotenv()
api_key = os.getenv("HF_TOKEN")
# Optional: If you want to use the raw HF InferenceClient
client = InferenceClient(
    provider="auto",
    token=api_key,
)

# Step 2: create build in tools=================================================
search_tool = DuckDuckGoSearchRun()

# Step 3: create custom tools===================================================
@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=4d1d8ae207a8c845a52df8a67bf3623e&query={city}'

  response = requests.get(url)

  return response.json()


# Step 4: create models=================================================
model_id = "moonshotai/Kimi-K2-Instruct"

# HuggingFaceEndpoint for LangChain
llm = HuggingFaceEndpoint(
    repo_id=model_id,
    task="text-generation",
)

# Wrap as Chat Model
model = ChatHuggingFace(llm=llm)


# Step 5: Pull the ReAct prompt from LangChain Hub==================================
prompt = hub.pull("hwchase17/react")  # pulls the standard ReAct agent prompt
# print(prompt)

# Step 6: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=model,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# Step 7: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True
)

# Step 8: Invoke
response = agent_executor.invoke({"input": "Find the capital of Madhya Pradesh, then find it's current weather condition"})
print(response)

# Step 9 : output
print(response['output'])
