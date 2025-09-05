from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Disable symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load environment variables
load_dotenv()
# api_key = os.getenv("HF_TOKEN")  # Not needed for public models

# Set model and kwargs
model_name = "intfloat/e5-small"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}

# Create embeddings object
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# docs
documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

# Get embeddings for a query
result = hf.embed_documents(documents)

print(len(result))
print(result[0])
