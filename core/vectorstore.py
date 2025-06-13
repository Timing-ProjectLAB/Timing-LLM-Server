from services.chatbot_v3 import load_or_build_vectorstore
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

def load_chroma_if_exists(path, embedding_fn):
    if os.path.exists(path):
        return Chroma(persist_directory=path, embedding_function=embedding_fn)
    else:
        return None

def load_all_vectorstores():
    api_key = os.getenv("OPENAI_API_KEY")
    embedding = OpenAIEmbeddings()
    return {
        "policy": load_or_build_vectorstore("FINAL_key_cat.json", "./chroma_policies", api_key),
        "keyword": load_chroma_if_exists("./kwdb", embedding),
        "category": load_chroma_if_exists("./categorydb", embedding)
    }