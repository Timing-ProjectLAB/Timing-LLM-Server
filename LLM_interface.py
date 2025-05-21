from LLM.chatbot_v3 import (
    parse_user_input,
    filter_docs,
    load_or_build_vectorstore,
    create_rag_chain,
    combine_prompt
)

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# ─────────────────────────────────── #
# 1. 벡터 DB 및 LLM 초기화 (최초 1회)
# ─────────────────────────────────── #
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY", "")
JSON_PATH = "./LLM/ms_v3_short.json"
PERSIST_DIR = "./chroma_policies"

_vectordb = load_or_build_vectorstore(JSON_PATH, PERSIST_DIR, API_KEY)
rag_chain = create_rag_chain(_vectordb, API_KEY)
_llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# ─────────────────────────────────── #
# 2. API에서 호출할 수 있는 정책 추천 함수
# ─────────────────────────────────── #
def get_policy_answer(question: str) -> str:
    age, region, interests = parse_user_input(question)
    res = rag_chain.invoke({"question": question})

    # 필터링
    filtered_docs = filter_docs(res["source_documents"], question, region, interests)

    # 답변 생성
    if not filtered_docs:
        # 폴백: 유사도 top‑3 재검색
        filtered_docs = rag_chain.retriever.vectorstore.similarity_search(question, k=3)

    return res["answer"]