from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from huggingface_hub import InferenceClient
import os
import json

# Load environment
load_dotenv()
api_key = os.getenv("HF_TOKEN")

# HuggingFace model (replace repo_id with the one you want)
api_key = os.getenv("HF_TOKEN")

client = InferenceClient(
    provider="auto",
    token=api_key,
)

# HuggingFace model
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="conversational",
)

model = ChatHuggingFace(llm=llm)

# Schema definition (for typing in Python)
class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative or positive"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]


# Build a real JSON schema (not __annotations__)
review_schema = {
    "type": "object",
    "properties": {
        "key_themes": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
        "sentiment": {"type": "string", "enum": ["pos", "neg"]},
        "pros": {"type": "array", "items": {"type": "string"}},
        "cons": {"type": "array", "items": {"type": "string"}},
        "name": {"type": "string"}
    },
    "required": ["key_themes", "summary", "sentiment"]
}

# Review text
input_text = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! 
The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. 
The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. 
What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. 
Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. 
Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? 
The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Review by Nitish Singh
"""

# Build a prompt to force JSON
prompt = f"""
Extract key_themes, summary, sentiment, pros, cons, and name from this review.
Return ONLY valid JSON in this format:

{json.dumps(review_schema, indent=2)}

Review text:
{input_text}
"""

# Call Hugging Face model
raw_output = model.invoke(prompt)

# Clean code fences if model adds ```json
cleaned_ouput_text = (
    raw_output.content.strip()
    .removeprefix("```json")
    .removeprefix("```")
    .removesuffix("```")
    .strip()
)

data = json.loads(cleaned_ouput_text)
review_instance = Review(**data)

print(review_instance)