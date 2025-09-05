from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Disable symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load environment variables
load_dotenv()
api_key = os.getenv("HF_TOKEN")  # Not needed for public models

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

# Get embeddings for a query
result = hf.embed_query("Delhi is the capital of India")

print(len(result))
print(result[:10])
