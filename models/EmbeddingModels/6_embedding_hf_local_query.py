from langchain_huggingface import HuggingFaceEmbeddings
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
local_model_path = "E:/projects/huggingface_cache/e5-small-v2"

hf = HuggingFaceEmbeddings(
    model_name=local_model_path,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)


result = hf.embed_query('Delhi is the capital of India')

print(len(result))
print(result)
