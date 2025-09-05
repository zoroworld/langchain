from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import re

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    google_api_key=os.getenv("GEMINI_API_KEY")
)

result = llm.invoke("What is the capital of India")

print(result)
print(result.content)

cleaned_result = re.sub(r"\*+", "", result.content)
print(cleaned_result)
