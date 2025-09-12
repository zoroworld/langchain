
from langchain_core.tools import InjectedToolArg
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage
from typing import Annotated
import requests
import json
from langchain.agents import initialize_agent, AgentType



from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()


api_key = os.getenv("HF_TOKEN")


# Optional: If you want to use the raw HF InferenceClient
client = InferenceClient(
    provider="auto",
    token=api_key,
)


# tool create===============================================

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
  """
  This function fetches the currency conversion factor between a given base currency and a target currency
  """
  url = f'https://v6.exchangerate-api.com/v6/c754eab14ffab33112e380ca/pair/{base_currency}/{target_currency}'

  response = requests.get(url)

  return response.json()

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
  """
  given a currency conversion rate this function calculates the target currency value from a given base currency value
  """

  return base_currency_value * conversion_rate

# print(convert.args)

# {'base_currency_value': {'title': 'Base Currency Value', 'type': 'integer'},
# 'conversion_rate': {'title': 'Conversion Rate', 'type': 'number'}}


# convert currency usd to inr
convertSchema = get_conversion_factor.invoke({'base_currency':'USD','target_currency':'INR'})
# print(convertSchema)

# {'result': 'success',
#  'documentation': 'https://www.exchangerate-api.com/docs',
#  'terms_of_use': 'https://www.exchangerate-api.com/terms',
#  'time_last_update_unix': 1757635202,
#  'time_last_update_utc': 'Fri, 12 Sep 2025 00:00:02 +0000',
#  'time_next_update_unix': 1757721602,
#  'time_next_update_utc': 'Sat, 13 Sep 2025 00:00:02 +0000',
#  'base_code': 'USD', 'target_code': 'INR',
#  'conversion_rate': 88.3656}

convert_result = convert.invoke({'base_currency_value':10, 'conversion_rate':85.16})
# print(convert_result) #851.5999999999999

# create the model================================================================
model_id = "moonshotai/Kimi-K2-Instruct"

# HuggingFaceEndpoint for LangChain
llm = HuggingFaceEndpoint(
    repo_id=model_id,
    task="text-generation",
)

# Wrap as Chat Model
model = ChatHuggingFace(llm=llm)

# model find tool binding=========================================================================
llm_with_tools = model.bind_tools([get_conversion_factor, convert])

# start conversion
messages = [HumanMessage('What is the conversion factor between INR and USD, and based on that can you convert 10 inr to usd')]
print(messages)

ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)
print(ai_message.tool_calls)



for tool_call in ai_message.tool_calls:
  # execute the 1st tool and get the value of conversion rate
  if tool_call['name'] == 'get_conversion_factor':
    tool_message1 = get_conversion_factor.invoke(tool_call)
    # fetch this conversion rate
    conversion_rate = json.loads(tool_message1.content)['conversion_rate']
    # append this tool message to messages list
    messages.append(tool_message1)
  # execute the 2nd tool using the conversion rate from tool 1
  if tool_call['name'] == 'convert':
    # fetch the current arg
    tool_call['args']['conversion_rate'] = conversion_rate
    tool_message2 = convert.invoke(tool_call)
    messages.append(tool_message2)

print(messages)

print(llm_with_tools.invoke(messages).content)


# Step 5: Initialize the Agent ---
# ---------------- Old-style Agent ----------------
agent_executor = initialize_agent(
    tools=[get_conversion_factor, convert],
    llm=model,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # using ReAct pattern
    verbose=True  # shows internal thinking
)
# new way we have to use langgraph

# --- Step 6: Run the Agent ---
user_query = "Hi how are you?"

response = agent_executor.invoke({"input": user_query})

print(response)