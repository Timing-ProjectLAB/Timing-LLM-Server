from services.chatbot_v3 import (
    keyword_vectordb, 
    category_vectordb, 
    get_policy_vectordb
)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
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
        "policy": get_policy_vectordb(), # Lazy loading 함수 호출
        "keyword": keyword_vectordb,   # 함수 호출 X, 객체 바로 사용
        "category": category_vectordb  # 함수 호출 X, 객체 바로 사용
    }