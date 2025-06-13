from utils.exceptions import raise_400, raise_500
from services.chatbot_v3 import (
    extract_user_info, filter_docs, build_query, generate_policy_response
)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI

class ChatbotService:
    def __init__(self, vectordb: Chroma):
        self.vectordb = vectordb
        self.embedding = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name="gpt-4o")
        self.memory_store = {}  # user_id -> memory

    def chat(self, user_id: str, user_input: str) -> dict:
        if not user_input or len(user_input.strip()) < 2:
            raise_400("사용자 입력이 없거나 너무 짧습니다.")

        try:
            return generate_policy_response(user_id, user_input)
        except Exception as e:
            raise_500(f"정책 추천 처리 중 오류가 발생했습니다: {str(e)}")

    def fallback(self) -> list:
        fallback_docs = self.vectordb.similarity_search("전국 공통 정책", k=3)
        return [{
            "policy_id": d.metadata.get("policy_id"),
            "title": d.metadata.get("title"),
            "summary": d.metadata.get("summary"),
            "apply_url": d.metadata.get("apply_url")
        } for d in fallback_docs]