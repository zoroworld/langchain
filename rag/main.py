from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import WikipediaRetriever
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Use the correct Gemini embeddings model
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)
