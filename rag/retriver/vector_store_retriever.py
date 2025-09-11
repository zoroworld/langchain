from langchain_chroma import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from uuid import uuid4
import os
from dotenv import load_dotenv

# Load API keys
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Option A: Google embeddings (commented out)
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001",
#     google_api_key=api_key
# )

# Option B: HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Step 1: Define persistence ---
PERSIST_DIR = "./chroma_db"

# --- Step 2: Create / load vector store ---
vector_store = Chroma(
    embedding_function=embeddings,
    collection_name="my_collection",
    persist_directory=PERSIST_DIR
)

# --- Step 3: Create some docs ---
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

# --- Step 4: Insert docs ---
ids = [str(uuid4()) for _ in documents]
vector_store.add_documents(documents=documents, ids=ids)

# --- Step 5: Query / retrieve ---
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
results = retriever.invoke("What is Chroma used for?")
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content, "| Metadata:", doc.metadata)
