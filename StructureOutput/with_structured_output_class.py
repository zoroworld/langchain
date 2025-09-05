from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from huggingface_hub import InferenceClient
import os
import json

# Load environment variables
load_dotenv()

api_key = os.getenv("HF_TOKEN")

client = InferenceClient(
    provider="auto",
    token=api_key,
)

# HuggingFace model
llm = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2-Instruct-0905",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

# Pydantic schema
class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review either negative or positive")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")

# Review text
input_text = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh
"""

# Ask model to output JSON
prompt = f"""
Extract key_themes, summary, sentiment, pros, cons, name from this review
and return it in valid JSON format matching this schema:

# New Pydantic v2 way:
{json.dumps(Review.model_json_schema(), indent=2)}


Review text:
{input_text}
"""

raw_output_text = model.invoke(prompt)

output_text = raw_output_text.content

cleaned_ouput_text = output_text.strip("`").replace("json\n", "")
# print(cleaned_ouput_text)

# Parse JSON into class
data = json.loads(cleaned_ouput_text)
review_instance = Review(**data)

print(review_instance)
