from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from uuid import uuid4
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)

# Create Pinecone index if it doesn't exist
index_name = "langchain-genai-index"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)

# Create documents
docs = [
    Document(page_content="Virat Kohli is one of the most successful batsmen...", metadata={"team": "RCB"}),
    Document(page_content="Rohit Sharma is the most successful captain...", metadata={"team": "MI"}),
    Document(page_content="MS Dhoni, famously known as Captain Cool...", metadata={"team": "CSK"}),
]

# Initialize PineconeVectorStore
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

# Add documents with UUIDs
uuids = [str(uuid4()) for _ in docs]
vector_store.add_documents(documents=docs, ids=uuids)


