from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

api_key = os.getenv("HF_TOKEN")

# Optional: If you want to use the raw HF InferenceClient
client = InferenceClient(
    provider="auto",
    token=api_key,
)

video_id = "Gfr50f6ZBvo"

try:
    # Instantiate the API
    ytt_api = YouTubeTranscriptApi()

    # Fetch the transcript (equivalent of old get_transcript)
    transcript_list = ytt_api.fetch(video_id, languages=["en"])

    # Flatten to plain text
    transcript = " ".join(snippet.text for snippet in transcript_list)
    # print(transcript[:500])

except TranscriptsDisabled:
    print("No captions available for this video.")

# Step 1b - Indexing (Text Splitting)============================================================

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# print(len(chunks))
# print(chunks[100])

# Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)

# Option B: HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(
    chunks, embeddings
)

# print(vector_store.index_to_docstore_id)
# print(vector_store.get_by_ids(['4299ecf6-fcd5-422a-894e-a45900eef3f3']))


# Step 2 - Retrieval ===============================================================================

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
# print(retriever)

retriever.invoke('What is deepmind')

# Step 3 - Augmentation================================================================================

model_id = "moonshotai/Kimi-K2-Instruct"

# HuggingFaceEndpoint for LangChain
llm = HuggingFaceEndpoint(
    repo_id=model_id,
    task="text-generation",
)

# Wrap as Chat Model
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)

# print(retrieved_docs)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
# print(context_text)
final_prompt = prompt.invoke({"context": context_text, "question": question})
# print(final_prompt)
# Step 4 - Generation ==========================================================

answer = model.invoke(final_prompt)
print(answer.content)