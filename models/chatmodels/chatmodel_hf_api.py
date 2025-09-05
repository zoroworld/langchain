from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()

api_key = os.getenv("HF_TOKEN")

# Optional: If you want to use the raw HF InferenceClient
client = InferenceClient(
    provider="auto",
    token=api_key,
)

model_id = "moonshotai/Kimi-K2-Instruct"

# HuggingFaceEndpoint for LangChain
llm = HuggingFaceEndpoint(
    repo_id=model_id,
    task="text-generation",
)

# Wrap as Chat Model
model = ChatHuggingFace(llm=llm)

query = "What is the national bird of India?"
result = model.invoke(query)
text =  re.sub(r"[*_`#]", "", result.content)
print(text)
