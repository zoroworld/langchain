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

# problem:1
# --- Perform similarity search ---
# query = "Best captain in IPL"
# results = vector_store.similarity_search(query, k=3)


# --- Print results ---
# for i, doc in enumerate(results):
#     print(f"Result {i+1}: {doc.page_content} | Metadata: {doc.metadata}")

# problem:2
# # --- Search with similarity score ---
# results = vector_store.similarity_search_with_score(
#     query="Who among these are a bowler?",
#     k=2
# )
#
# # --- Print results with their similarity scores ---
# for i, (doc, score) in enumerate(results):
#     print(f"Result {i+1}:")
#     print("Content:", doc.page_content)
#     print("Metadata:", doc.metadata)
#     print("Score:", score)
#     print("-" * 50)


# problem:3
# # --- Metadata filtering example ---
# results = vector_store.similarity_search_with_score(
#     query="Who is the captain?",
#     k=3,
#     filter={"team": "CSK"}  # Only return documents where metadata 'team' is 'CSK'
# )
#
# # --- Print results with their similarity scores ---
# for i, (doc, score) in enumerate(results):
#     print(f"Result {i+1}:")
#     print("Content:", doc.page_content)
#     print("Metadata:", doc.metadata)
#     print("Score:", score)
#     print("-" * 50)


# problem:4
# # --- Corrected updated document ---
# updated_doc1 = Document(
#     page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli's passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.",
#     metadata={"team": "RCB"}
# )
#
# # --- Recompute embeddings for the updated document ---
# vector = embeddings.embed_documents([updated_doc1.page_content])[0]
#
# # --- Upsert to Pinecone with the same document ID ---
# vector_store.index.upsert(
#     vectors=[(
#         "b1f3d0b3-afa3-4bc9-98e6-fac1cc1ae4f0",  # same ID to update
#         vector,
#         {"text": updated_doc1.page_content, **updated_doc1.metadata}  # include content in metadata
#     )]
# )

# problem:5
# --- Delete a document by ID ---
# vector_store.index.delete(
#     ids=['c73e8fb4-88c9-42e8-ba11-e3d50f8a9c66']
# )

# problem:6

