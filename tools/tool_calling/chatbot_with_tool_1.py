# from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
import requests
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


# custom tool create==============================
@tool
def multiply(a: int, b: int) -> int:
  """Given 2 numbers a and b this tool returns their product"""
  return a * b

# print(multiply.invoke({'a':3, 'b':4}))
# print(multiply.name)
# print(multiply.description)
# print(multiply.args)

# create model=========================================

model_id = "moonshotai/Kimi-K2-Instruct"

# HuggingFaceEndpoint for LangChain
llm = HuggingFaceEndpoint(
    repo_id=model_id,
    task="text-generation",
)

# Wrap as Chat Model
model = ChatHuggingFace(llm=llm)


# tool binding=========================================

#AI message
print(model.invoke('hi'))
model_with_tools = model.bind_tools([multiply])
tools_model_result = model_with_tools.invoke('Hi how are you')
print(tools_model_result)

# human message
human_message = HumanMessage('can you multiply 3 with 1000')
messages = [human_message]
result = model_with_tools.invoke(messages)
messages.append(result)
print(messages)

# tools result
tool_result = multiply.invoke(result.tool_calls[0])
print(tool_result)
messages.append(tool_result)


get_answere = model_with_tools.invoke(messages).content
print(get_answere )




