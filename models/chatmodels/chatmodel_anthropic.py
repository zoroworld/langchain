from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

input = 'What is the capital of India'
result = model.invoke(input)

print(input)

# input --> model(llm) --> result