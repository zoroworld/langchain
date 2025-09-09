from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import os

# --- Load environment variables ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")

# --- Initialize Pinecone client ---
pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)

# --- Get the existing Pinecone index ---
index_name = "langchain-genai-index"
index = pc.Index(index_name)

# --- Initialize embeddings ---
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)

# --- Initialize PineconeVectorStore (read-only) ---
vector_store = PineconeVectorStore(
    index=index,        # Pinecone index object
    embedding=embeddings
)

#view
results = vector_store.similarity_search(query="", k=100)

# --- Print results ---
for i, doc in enumerate(results):
    print(f"Result {i+1}:")
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)
    print("-" * 50)

