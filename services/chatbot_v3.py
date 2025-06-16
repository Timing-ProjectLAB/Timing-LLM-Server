# 추가: 코드 맨 위에 clean_text_for_matching 함수 정의
#!/usr/bin/env python3
# chatbot.py  ·  Adaptive Filtering + Keyword·Category Edition
# 실행: python3 chatbot.py
# 필요한 패키지: pip install langchain-openai langchain chromadb python-dotenv

import os, re, json
from typing import List, Tuple, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import (
    ChatPromptTemplate, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from functools import lru_cache
import time

# ─────────────────────────────────── #
# 📌 In‑memory session store (user‑level)
# ─────────────────────────────────── #
from collections import defaultdict
SESSION_STORE = defaultdict(lambda: {"user_info": None, "recommended_ids": set()})

# 특수문자 제거, 소문자화 등을 통해 키워드 매칭에 방해가 되는 요소들을 제거하는 함수
def clean_text_for_matching(text):
    return re.sub(r"[^\w\s]", "", text).replace("에", "").replace("에서", "").replace("인데", "").replace("야", "").strip()
# 사용자가 “다른 정책”, “추가로 보여줘” 같은 추가 추천 요청인지 판별하는 함수
def is_generic_more_request(text: str) -> bool:
    """
    사용자가 '다른 정책', '추가 정책', '더 보여줘' 등
    구체적 조건 없는 추가 추천을 요구하는지 판별.
    """
    text = text.strip()
    if re.search(r"다른\s*정책", text):
        return True
    if "정책" in text and re.search(r"(더|추가|또|없어|있어)", text):
        return True
    # "더 알려줘", "더 보여줘", "더 추천해줘" 처리
    if re.search(r"더\s*(알려줘|보여줘|추천해줘)", text):
        return True
    return False

# ─────────────────────────────────── #
# 정책 관련 질문 여부 판별 함수
# ─────────────────────────────────── #
NON_POLICY_KEYWORDS = [
    "안녕", "하이", "반가워", "잘 지냈어", "뭐해", "심심해", "놀자", "지루해", "고마워", "감사", "잘자", "잘 자", "굿밤",
    "누구야", "너 뭐야", "정체가 뭐야", "자기소개", "이름", "챗지피티", "gpt", "ai야", "로봇이야", "몇 살", "나이",
    "날씨", "온도", "기온", "몇 시", "시간", "오늘 날짜", "지금 몇시", "오늘 뭐야", "요일",
    "기분 어때", "사랑해", "귀여워", "좋아해", "여자친구", "남자친구", "썸", "연애", "이상형",
    "퀴즈", "수수께끼", "농담", "웃겨줘", "재밌는 얘기", "우주", "과학", "역사", "유튜브", "게임", "유머"
]
# 간단한 휴리스틱과 키워드 리스트(NON_POLICY_KEYWORDS)를 기반으로, 입력이 정책 관련 질의인지 1차 검사하는 함수
def is_policy_related_question(text: str) -> bool:
    import re
    # uses global clean_text_for_matching and REVERSE_REGION_LOOKUP
    if len(text.strip()) < 2:
        return False
    cleaned = re.sub(r"[ㅋㅎㅠㅜ]+", "", text.lower())
    for word in NON_POLICY_KEYWORDS:
        if word in cleaned:
            return False
    if re.match(r"^[가-힣]{1,3}(야|이야)?$", text.strip()):
        # Clean conversational endings like '야', '이야'
        clean_key = clean_text_for_matching(text)
        # If cleaned key matches a region in lookup, treat as policy-related (region input)
        if clean_key in REVERSE_REGION_LOOKUP:
            return True
        return False
    return True

# ─────────────────────────────────── #
# LLM 기반 정책 질문 여부 판별 함수
# ─────────────────────────────────── #
# 위 휴리스틱이 모호할 때, GPT-4o-mini에 “Y/N” 분류를 요청해 보다 정확히 판단하고, 실패 시 rule-based로 폴백
@lru_cache(maxsize=1024)           # 같은 문장은 한 번만 문의
def is_policy_related_question_llm(text: str) -> bool:
    """
    GPT-4o-mini로 ‘정책 관련 질문인지’ Y/N 분류.
    - refined heuristic for short/numeric/keyword input
    - LLM 오류 시 rule-based 폴백.
    """
    cleaned = text.strip()
    if not cleaned:
        return False  # 빈 입력

    # '다른 정책', '추가 정책' 등 일반 추가 추천 요청은 정책 관련으로 간주
    if is_generic_more_request(cleaned):
        return True

    # ① 숫자 1~2자리만 입력 → 나이로 간주 → 정책 질문 True
    if re.fullmatch(r"\d{1,2}", cleaned):
        return True

    # ② 단어 1~2자여도 관심사·지역 키워드라면 True
    if cleaned in REVERSE_REGION_LOOKUP:
        return True
    if cleaned in INTEREST_MAPPING:
        return True
    if any(cleaned in kws for kws in INTEREST_MAPPING.values()):
        return True

    # ③ 1글자·특수문자·웃음(ㅋㅎ)만 → False
    if len(cleaned) == 1 or re.fullmatch(r"[ㅋㅎ]+", cleaned):
        return False

    system_msg = (
        "너는 대한민국 청년 정책 상담 챗봇의 분류기야. "
        "아래 사용자 입력이 정책과 *관련된 질문*인지 판단해. "
        "대답은 'Y' 또는 'N' 중 하나로만."
    )
    user_msg = f"사용자 입력: {cleaned}\n\n정책 관련 질문인가?"

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0,
            max_tokens=1,
        )
        return resp.choices[0].message.content.strip().upper().startswith("Y")
    except Exception:
        # 네트워크/쿼터 문제 시 휴리스틱으로 폴백
        return is_policy_related_question(cleaned)

# 토큰 수나 핵심 키워드 포함 여부를 보고 “유효한 정책 질의”인지 추가 검사하는 함수
def is_valid_query(text: str) -> bool:
    # 숫자만 입력되어도 나이로 간주
    if re.search(r"\b\d{1,2}\b", text):
        return True

    tokens = re.findall(r"[가-힣a-zA-Z0-9]+", text)

    # 1) 3단어 이상이면 무조건 정책 관련 질의로 간주
    if len(tokens) >= 3:
        return True

    # 2) 2단어짜리 짧은 질문이라도 핵심 키워드가 포함되면 허용
    if len(tokens) == 2:
        for tok in tokens:
            if (
                tok == "정책" or
                tok in KEYWORDS or
                tok in INTEREST_MAPPING or
                any(tok in kws for kws in INTEREST_MAPPING.values())
            ):
                return True
    return False
# 지역명 키워드 여부 판별 함수
def is_region_keyword(word: str) -> bool:
    return word in REVERSE_REGION_LOOKUP or any(word in names for names in REGION_MAPPING.values())

# 사용자 입력에서 정보 자동 추출 함수
def extract_user_info(user_input: str):
    info = {"age": None, "region": None, "interests": [], "status": None, "income": None}

    # ✅ 전처리: 마침표, 쉼표 등 제거 → '여주에 사는 25살이야'로 만들기
    clean_text = re.sub(r"[^\w가-힣]", " ", user_input)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    # 🔁 정확한 나이/지역/관심사 파싱은 parse_user_input() 재활용
    parsed_age, parsed_region, parsed_interests = parse_user_input(clean_text)
    info["age"] = parsed_age
    info["region"] = parsed_region
    info["interests"] = parsed_interests if parsed_interests else []

    # 상태 추출
    if "대학생" in user_input:
        info["status"] = "대학생"
    elif "취준생" in user_input or "취업 준비" in user_input:
        info["status"] = "취업준비생"

    # 소득
    if "저소득" in user_input:
        info["income"] = "저소득층"
    elif "고소득" in user_input:
        info["income"] = "고소득층"

    return info


def print_result(idx, doc):
    result = {
        "policy_id": doc.metadata.get("policy_id", f"unknown_{idx}"),
        "name":      doc.metadata.get("title"),
        "summary":   doc.metadata.get("summary"),
        "eligibility": f"{doc.metadata.get('min_age','?')}~{doc.metadata.get('max_age','?')}세 / {doc.metadata.get('region','전국')}",
        "period":    doc.metadata.get("apply_period",""),
        # ↓ 디버깅
        # "score":     doc.metadata.get("debug_total_score"),
        # "region":    doc.metadata.get("debug_region_score"),
        # "interest":  doc.metadata.get("debug_interest_score"),
        # "keyword":   doc.metadata.get("debug_keyword_score")
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

# ─────────────────────────────────── #
# 글로벌 임베딩 및 키워드 벡터DB (키워드 전용)
# Load embedding function globally
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "")
import openai
openai.api_key = api_key
embedding = OpenAIEmbeddings()
# Load keyword vectorstore (ensure it's built with keyword terms only)
keyword_vectordb = Chroma(persist_directory="./kwdb", embedding_function=embedding)
category_vectordb = Chroma(persist_directory="./categorydb", embedding_function=embedding)
# Main policy vectorstore
policy_vectordb = Chroma(persist_directory="./chroma_policies", embedding_function=embedding)
# 0. 보조 함수 – 질의 재구성
# ─────────────────────────────────── #

def build_query(base_prompt: str,
                age: Optional[int],
                region: Optional[str],
                interests: Optional[List[str]]) -> str:
    """저장된 정보를 엮어 RAG용 자연어 질의 문자열 생성"""
    parts: List[str] = [base_prompt]
    if region:
        parts.append(f"{region} 거주")
    if age:
        parts.append(f"{age}세")
    if interests:
        parts.append(f"관심사 {', '.join(interests)}")
    return " ".join(parts)
# ─────────────────────────────────── #
# 1. 관심사 · 지역 맵
# ─────────────────────────────────── #
INTEREST_MAPPING = {
    "창업": ["창업", "스타트업", "기업 설립", "벤처", "소상공인", "사업", "자금지원"],
    "취업": ["취업", "일자리", "채용", "고용", "잡페어", "구직활동", "면접", "이력서", "자기소개서", "취업지원", "구직"],
    "운동": ["운동", "스포츠", "체육", "피트니스", "헬스", "헬스케어", "요가", "체육관"],
    "학업": ["학업", "학습", "공부", "교육", "학위", "대학생활", "대학", "연구"],
    "프로그램": ["프로그램", "워크숍", "세미나", "캠프", "연수", "교육프로그램", "훈련프로그램"],
    "장학금": ["장학금", "학비 지원", "등록금 지원", "교육비 지원", "학자금"],
    "해외연수": ["해외연수", "글로벌 연수", "교환학생", "어학연수", "해외교육"],
    "인턴십": ["인턴십", "현장실습", "산학협력", "인턴", "실무경험"],
    "주거": ["주거", "주택", "임대", "전세", "월세", "보증금", "부동산"],
    "복지": ["복지", "사회복지", "지원", "보조금", "바우처", "의료", "건강", "출산", "육아"],
    "참여": ["참여", "권리", "시민", "사회", "봉사", "활동", "동아리"],
    "직업교육": ["직업", "훈련", "기술", "자격증", "교육", "강좌", "직업훈련"],
    "해외취업": ["해외취업", "국외취업", "글로벌취업", "일자리", "진출"],
    "정신건강": ["정신건강", "상담", "심리", "스트레스", "우울증"],
    "금융지원": ["대출", "자금", "지원금", "보조금", "융자"]
}
REGION_KEYWORDS = {
    "서울": ["서울", "서울시"],
    "경기": ["경기", "경기도"],
    "인천": ["인천", "인천시"],
    "부산": ["부산"],
    "대구": ["대구"],
    "광주": ["광주"],
    "대전": ["대전"],
    "울산": ["울산"],
    "세종": ["세종"],
    "강원": ["강원", "강원도", "강원특별자치도"],
    "충북": ["충북", "충청북도"],
    "충남": ["충남", "충청남도"],
    "전북": ["전북", "전라북도"],
    "전남": ["전남", "전라남도"],
    "경북": ["경북", "경상북도"],
    "경남": ["경남", "경상남도"],
    "제주": ["제주", "제주도", "제주특별자치도"]
}

REGION_MAPPING = {
    "서울": [
        "서울특별시 종로구", "서울특별시 중구", "서울특별시 용산구", "서울특별시 성동구",
        "서울특별시 광진구", "서울특별시 동대문구", "서울특별시 중랑구", "서울특별시 성북구",
        "서울특별시 강북구", "서울특별시 도봉구", "서울특별시 노원구", "서울특별시 은평구",
        "서울특별시 서대문구", "서울특별시 마포구", "서울특별시 양천구", "서울특별시 강서구",
        "서울특별시 구로구", "서울특별시 금천구", "서울특별시 영등포구", "서울특별시 동작구",
        "서울특별시 관악구", "서울특별시 서초구", "서울특별시 강남구", "서울특별시 송파구",
        "서울특별시 강동구",
        "서울",
        "서울특별시",
        "서울시"
    ],
    "경기": [
        "경기도 수원시장안구", "경기도 수원시권선구", "경기도 수원시팔달구", "경기도 수원시영통구",
        "경기도 성남시수정구", "경기도 성남시중원구", "경기도 성남시분당구", "경기도 의정부시",
        "경기도 안양시만안구", "경기도 안양시동안구", "경기도 부천시원미구", "경기도 부천시소사구",
        "경기도 부천시오정구", "경기도 광명시", "경기도 평택시", "경기도 동두천시",
        "경기도 안산시상록구", "경기도 안산시단원구", "경기도 고양시덕양구", "경기도 고양시일산동구",
        "경기도 고양시일산서구", "경기도 과천시", "경기도 구리시", "경기도 남양주시",
        "경기도 오산시", "경기도 시흥시", "경기도 군포시", "경기도 의왕시", "경기도 하남시",
        "경기도 용인시처인구", "경기도 용인시기흥구", "경기도 용인시수지구", "경기도 파주시",
        "경기도 이천시", "경기도 안성시", "경기도 김포시", "경기도 화성시", "경기도 광주시",
        "경기도 양주시", "경기도 포천시", "경기도 여주시", "경기도 연천군", "경기도 가평군",
        "경기도 양평군"
    ],
    "인천": [
        "인천광역시 중구", "인천광역시 동구", "인천광역시 미추홀구", "인천광역시 연수구",
        "인천광역시 남동구", "인천광역시 부평구", "인천광역시 계양구", "인천광역시 서구",
        "인천광역시 강화군", "인천광역시 옹진군"
    ],
    "부산": [
        "부산광역시 중구", "부산광역시 서구", "부산광역시 동구", "부산광역시 영도구",
        "부산광역시 부산진구", "부산광역시 동래구", "부산광역시 남구", "부산광역시 북구",
        "부산광역시 해운대구", "부산광역시 사하구", "부산광역시 금정구", "부산광역시 강서구",
        "부산광역시 연제구", "부산광역시 수영구", "부산광역시 사상구", "부산광역시 기장군"
    ],
    "대구": [
        "대구광역시 중구", "대구광역시 동구", "대구광역시 서구", "대구광역시 남구",
        "대구광역시 북구", "대구광역시 수성구", "대구광역시 달서구", "대구광역시 달성군",
        "대구광역시 군위군"
    ],
    "광주": [
        "광주광역시 동구", "광주광역시 서구", "광주광역시 남구", "광주광역시 북구", "광주광역시 광산구"
    ],
    "대전": [
        "대전광역시 동구", "대전광역시 중구", "대전광역시 서구", "대전광역시 유성구", "대전광역시 대덕구"
    ],
    "울산": [
        "울산광역시 중구", "울산광역시 남구", "울산광역시 동구", "울산광역시 북구", "울산광역시 울주군"
    ],
    "세종": [
        "세종특별자치시 세종시"
    ],
    "강원": [
        "강원특별자치도 춘천시", "강원특별자치도 원주시", "강원특별자치도 강릉시", "강원특별자치도 동해시",
        "강원특별자치도 태백시", "강원특별자치도 속초시", "강원특별자치도 삼척시", "강원특별자치도 홍천군",
        "강원특별자치도 횡성군", "강원특별자치도 영월군", "강원특별자치도 평창군", "강원특별자치도 정선군",
        "강원특별자치도 철원군", "강원특별자치도 화천군", "강원특별자치도 양구군", "강원특별자치도 인제군",
        "강원특별자치도 고성군", "강원특별자치도 양양군"
    ],
    "충북": [
        "충청북도 청주시상당구", "충청북도 청주시서원구", "충청북도 청주시흥덕구", "충청북도 청주시청원구",
        "충청북도 충주시", "충청북도 제천시", "충청북도 보은군", "충청북도 옥천군", "충청북도 영동군",
        "충청북도 증평군", "충청북도 진천군", "충청북도 괴산군", "충청북도 음성군", "충청북도 단양군"
    ],
    "충남": [
        "충청남도 천안시동남구", "충청남도 천안시서북구", "충청남도 공주시", "충청남도 보령시", "충청남도 아산시",
        "충청남도 서산시", "충청남도 논산시", "충청남도 계룡시", "충청남도 당진시", "충청남도 금산군",
        "충청남도 부여군", "충청남도 서천군", "충청남도 청양군", "충청남도 홍성군", "충청남도 예산군",
        "충청남도 태안군"
    ],
    "전북": [
        "전북특별자치도 전주시완산구", "전북특별자치도 전주시덕진구", "전북특별자치도 군산시", "전북특별자치도 익산시",
        "전북특별자치도 정읍시", "전북특별자치도 남원시", "전북특별자치도 김제시", "전북특별자치도 완주군",
        "전북특별자치도 진안군", "전북특별자치도 무주군", "전북특별자치도 장수군", "전북특별자치도 임실군",
        "전북특별자치도 순창군", "전북특별자치도 고창군", "전북특별자치도 부안군"
    ],
    "전남": [
        "전라남도 목포시", "전라남도 여수시", "전라남도 순천시", "전라남도 나주시", "전라남도 광양시",
        "전라남도 담양군", "전라남도 곡성군", "전라남도 구례군", "전라남도 고흥군", "전라남도 보성군",
        "전라남도 화순군", "전라남도 장흥군", "전라남도 강진군", "전라남도 해남군", "전라남도 영암군",
        "전라남도 무안군", "전라남도 함평군", "전라남도 영광군", "전라남도 장성군", "전라남도 완도군",
        "전라남도 진도군", "전라남도 신안군"
    ],
    "경북": [
        "경상북도 포항시남구", "경상북도 포항시북구", "경상북도 경주시", "경상북도 김천시", "경상북도 안동시",
        "경상북도 구미시", "경상북도 영주시", "경상북도 영천시", "경상북도 상주시", "경상북도 문경시",
        "경상북도 경산시", "경상북도 의성군", "경상북도 청송군", "경상북도 영양군", "경상북도 영덕군",
        "경상북도 청도군", "경상북도 고령군", "경상북도 성주군", "경상북도 칠곡군", "경상북도 예천군",
        "경상북도 봉화군", "경상북도 울진군", "경상북도 울릉군"
    ],
    "경남": [
        "경상남도 창원시의창구", "경상남도 창원시성산구", "경상남도 창원시마산합포구", "경상남도 창원시마산회원구",
        "경상남도 창원시진해구", "경상남도 진주시", "경상남도 통영시", "경상남도 사천시", "경상남도 김해시",
        "경상남도 밀양시", "경상남도 거제시", "경상남도 양산시", "경상남도 의령군", "경상남도 함안군",
        "경상남도 창녕군", "경상남도 고성군", "경상남도 남해군", "경상남도 하동군", "경상남도 산청군",
        "경상남도 함양군", "경상남도 거창군", "경상남도 합천군"
    ],
    "제주": [
        "제주특별자치도 제주시", "제주특별자치도 서귀포시", "제주도",
        "제주",
        "제주도",
        "제주특별자치도"
    ]
}

# ---- 단일 키워드 및 'OO시' 변형 자동 추가 ----
# 각 표준 지역명 자체(예: '서울', '경기')와 흔히 쓰는 '○○시' 변형을 REGION_MAPPING 리스트에 자동 포함시켜
# 단일 키워드 입력도 인식하도록 확장합니다.
for std_region, names in REGION_MAPPING.items():
    # 1) 단일 표준 지역명 추가
    if std_region not in names:
        names.append(std_region)
    # 2) '○○시' 변형 추가 (도 단위는 제외)
    if not std_region.endswith("도") and not std_region.endswith("시"):
        si_variant = f"{std_region}시"
        if si_variant not in names:
            names.append(si_variant)

# 지역 이름 역매핑 (예: '여주시' → '경기')
REVERSE_REGION_LOOKUP = {}
for std_region, full_names in REGION_MAPPING.items():
    for name in full_names:
        tokens = re.findall(r"[가-힣]{2,}", name)
        for token in tokens:
            if token not in REVERSE_REGION_LOOKUP:
                REVERSE_REGION_LOOKUP[token] = std_region

            # ‘여주시’ → ‘여주’처럼 접미사(시·군·구) 제거 버전도 매핑
            core_tok = re.sub(r"(시|군|구)$", "", token)
            if core_tok and core_tok not in REVERSE_REGION_LOOKUP:
                REVERSE_REGION_LOOKUP[core_tok] = std_region

        # 전체 명칭도 직접 매핑
        if name not in REVERSE_REGION_LOOKUP:
            REVERSE_REGION_LOOKUP[name] = std_region

# 추가: 단일 지명 토큰 매핑
REVERSE_REGION_LOOKUP.setdefault("제주", "제주")
REVERSE_REGION_LOOKUP.setdefault("제주도", "제주")
REVERSE_REGION_LOOKUP.setdefault("서울", "서울")
REVERSE_REGION_LOOKUP.setdefault("서울시", "서울")

# ─────────────────────────────────── #
# 2. 정책 키워드 · 카테고리
# ─────────────────────────────────── #
KEYWORDS = [
    "바우처", "해외진출", "장기미취업청년", "맞춤형상담서비스", "교육지원",
    "출산", "보조금", "중소기업", "벤처", "대출", "금리혜택",
    "인턴", "공공임대주택", "육아", "청년가장", "신용회복"
]

CATEGORIES = ["일자리", "복지문화", "참여권리", "교육", "주거"]

INTEREST_EXPANSION = {
    "운동": ["생활체육", "운동처방", "운동용품 대여", "건강", "체력", "헬스"],
    "창업": ["창업지원", "창업교육", "사업자등록", "스타트업"],
    "취업": ["일자리", "직무교육", "인턴십", "일경험", "청년고용"],
    "주거": ["임대", "청년주택", "보증금지원", "전세", "월세"],
    "복지": ["심리상담", "정신건강", "건강검진", "생활비지원"]
}

def extract_keywords(text: str) -> List[str]:
    """사전 키워드 + 한글 형태소 기반 간이 추출"""
    hits = [kw for kw in KEYWORDS if kw in text]
    # 보강: 2글자 이상 명사 빈도수 상위 5개 자동 추출(간단 regex)
    tokens = re.findall(r"[가-힣]{2,}", text)
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    sorted_extra = sorted((w for w in freq if w not in hits),
                          key=lambda w: freq[w],
                          reverse=True)[:5]
    return hits + sorted_extra

def extract_categories(cat_field: str) -> List[str]:
    """
    정책 JSON의 category 필드(쉼표 구분 텍스트)를 그대로 리스트로 반환.
    사전에 정의된 CATEGORIES에 없더라도 저장해 두고, 필터 단계에서 매칭합니다.
    """
    if not cat_field:
        return []
    return [c.strip() for c in cat_field.split(",") if c.strip()]

# ─────────────────────────────────── #
# 3. 벡터스토어 로드/생성 (강화 버전)
# ─────────────────────────────────── #
def load_or_build_vectorstore(json_path: str,
                              persist_dir: str,
                              api_key: str) -> Chroma:
    os.environ["OPENAI_API_KEY"] = api_key
    embedding = OpenAIEmbeddings()

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=embedding)

    with open(json_path, encoding="utf-8") as f:
        policies = json.load(f)

    def safe_int(val, default=0):
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding)

    for p in tqdm(policies, desc="Vectorizing policies"):
        text = (
            f"정책명: {p['title']}\n"
            f"정책ID: {p.get('policy_id')}\n"
            f"지원대상: {safe_int(p.get('min_age'))}세~{safe_int(p.get('max_age'), 99)}세 / "
            f"지역 {', '.join(p.get('region_name', []))}\n"
            f"소득 분위: {p.get('income_condition', '제한 없음')}\n"
            f"혜택: {p.get('support_content', '')}\n"
            f"신청방법: {p.get('apply_method', '')}\n"
            f"설명: {p.get('description', '')}\n"
            f"링크: {p.get('apply_url', '')}"
        )
    
        existing_keywords = p.get("keywords", "")
        if isinstance(existing_keywords, str):
            existing_keywords = [kw.strip() for kw in existing_keywords.split(",") if kw.strip()]
        merged_keywords = list(set(existing_keywords + extract_keywords(text)))
        metadata = {
            "policy_id":        p.get("policy_id"),
            "title":            p["title"],
            "region":           ", ".join(p.get("region_name", [])),
            "categories":       ", ".join(extract_categories(p.get('category', ''))),
            "keywords":         ", ".join(merged_keywords),
            "min_age":          safe_int(p.get("min_age")),
            "max_age":          safe_int(p.get("max_age"), 99),
            "income_condition": p.get("income_condition", "제한 없음"),
            "summary": (p.get("support_content") or p.get("description", ""))[:200],
            "apply_period":     p.get("apply_period", ""),
            "apply_url":        p.get("apply_url", ""),
        }

        # Ensure metadata values are primitive types for Chroma
        for mk, mv in metadata.items():
            if isinstance(mv, (list, set)):
                metadata[mk] = ", ".join(map(str, mv))

        chunks = splitter.split_text(text)
        documents = [Document(page_content=chunk, metadata=metadata) for chunk in chunks]
        vectordb.add_documents(documents)

    vectordb.persist()
    return vectordb

# ─────────────────────────────────── #
# 4. 사용자 입력 파싱
# ─────────────────────────────────── #
from typing import Tuple, Optional, List

# 사용자 입력 클린업 함수 추가
import re
def clean_user_input(text: str) -> str:
    # Remove common conversational endings and particles that interfere with matching
    return re.sub(r"(에\s*사는?|야|인데|이야|임|입니다|거든|임다|라구|라고)", "", text)

# 조사 등을 제거하고 핵심 단어(예: '여주에' → '여주') 추출
def normalize_korean_tokens(text: str) -> List[str]:
    """
    조사·행정구역 접미사(시·군·구) 제거 후 핵심 단어 추출
    """
    tokens = re.findall(r"[가-힣]{2,}", text)
    normalized = []
    for tok in tokens:
        # 조사 제거
        core = re.sub(r"(에|에서|에게|로|으로|의|를|을|이|가|은|는|도|만|이나|까지|부터)$", "", tok)
        # 행정구역 접미사 제거
        core = re.sub(r"(시|군|구)$", "", core)
        if core and core not in normalized:
            normalized.append(core)
    return normalized

# 지역 추출 보조 함수
def extract_region(user_input: str, REGION_MAPPING: dict) -> str:
    cleaned_input = clean_text_for_matching(user_input)
    for std_region, keywords in REGION_MAPPING.items():
        for keyword in keywords:
            if keyword in cleaned_input:
                return std_region
    return ""

def parse_user_input(text: str) -> Tuple[Optional[int], Optional[str], Optional[List[str]]]:
    # 텍스트 정규화: 앞부분에서 strip 및 특수문자 제거
    text = text.strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    # 텍스트 전처리: 조사 제거 및 공백 정리
    text = re.sub(r"[^\w가-힣]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    age = None
    # ① '26살', '26 세' 형태
    if m := re.search(r"(?:만\s*)?(\d{1,2})\s*(?:세|살)", text):
        age = int(m.group(1))
    # ② 단일 숫자만 있는 경우도 나이로 간주 (15~39세 범위)
    if age is None:
        m2 = re.search(r"\b(\d{1,2})\b", text)
        if m2:
            age_cand = int(m2.group(1))
            if 15 <= age_cand <= 39:
                age = age_cand

    # 지역 추출 부분 교체: REGION_MAPPING의 모든 시/군/구 이름이 포함되도록 확장되어 있어야 함
    region = ""
    for std_region, keywords in REGION_MAPPING.items():
        if any(keyword in text for keyword in keywords):
            region = std_region
            break
    # Fallback: 단일 토큰 기반 지역 추출
    if not region:
        for token in normalize_korean_tokens(text):
            if token in REVERSE_REGION_LOOKUP:
                region = REVERSE_REGION_LOOKUP[token]
                break

    interests = None
    matches = [std_i for std_i, kws in INTEREST_MAPPING.items() if any(k in text for k in kws)]
    if matches:
        interests = matches

    return age, region, interests

# 5. 정보 누락 확인 함수 추가
# ─────────────────────────────────── #
def missing_info(age, region, interests) -> List[str]:
    needs = []
    if age is None:
        needs.append("나이")
    if region is None:
        needs.append("지역")
    if not interests or len(interests) == 0:
        needs.append("관심사")
    return needs

# ─────────────────────────────────── #
# 🔧 추천 가능한 관심사 리스트 헬퍼
# ─────────────────────────────────── #
def suggest_remaining_interests(current: List[str]) -> str:
    """
    현재 stored_interests 를 기준으로 아직 제안하지 않은
    INTEREST_MAPPING 상위 카테고리를 콤마로 나열해 반환.
    5개까지만 보여주고 나머지는 '등'으로 표기.
    """
    remaining = [k for k in INTEREST_MAPPING.keys() if k not in current]
    shown = remaining[:5]
    suggestion = ", ".join(shown)
    if len(remaining) > 5:
        suggestion += " 등"
    return suggestion


def classify_user_type(text: str) -> str:
    known = ["청년내일채움공제", "도약계좌", "구직활동지원금", "국민취업지원제도", "정책명"]
    return "policy_expert" if any(kw in text for kw in known) else "policy_novice"
# ─────────────────────────────────── #

# ─────────────────────────────────── #
# 5. 시스템 프롬프트
# ─────────────────────────────────── #
SYSTEM = SystemMessagePromptTemplate.from_template("""
[ROLE]
당신은 대한민국 만 19~39세 청년을 위한 정책 안내 챗봇입니다. 사용자의 입력과 제공된 context 문서를 바탕으로, 해당 청년에게 가장 적합한 정책을 찾아 안내하는 역할을 수행합니다.

[TASK - Chain of Thought 방식]
사용자의 조건(나이, 지역, 관심사)을 바탕으로 아래 순서대로 추론하며 정책을 추천하세요:

1. 먼저 사용자의 나이가 각 정책의 나이 조건(min_age~max_age)에 부합하는지 확인합니다.
2. 다음으로 지역 조건이 일치하는지 확인합니다. 정확한 지역이 없으면 전국 공통 정책을 포함합니다.
3. 관심사 또는 세부 관심사가 정책 키워드 또는 설명에 포함되어 있는지 판단합니다.
4. 위 조건들에 기반해 적합한 정책을 우선순위로 정렬한 후, 상위 3건을 추천합니다.
5. 각 정책은 추천 이유(나이/지역/관심사 조건에 어떻게 부합하는지)를 한 줄로 설명해 주세요.
6. 조건이 명확하지 않으면 조회량이 많은 전국 공통 정책 3건을 대신 추천하세요.

[OUTPUT FORMAT - MARKDOWN]
- 정책명 (소득: ○○): 지원내용 요약 — 추천 이유 (링크 : apply_url) (정첵ID : policy_id)
- 정책명 (소득: ○○): 지원내용 요약 — 추천 이유 (링크 : apply_url) (정첵ID : policy_id)
- 정책명 (소득: ○○): 지원내용 요약 — 추천 이유 (링크 : apply_url) (정첵ID : policy_id)

[EXCEPTION]
- 조건에 맞는 정책이 없을 경우:
    대신 전국 공통 정책 3건을 출력하세요.

[EXAMPLE - NORMAL]
- 청년내일채움공제 (소득: 제한 없음): 중소기업 근무 청년에게 목돈 마련 지원 — 나이와 소득 조건 모두 부합 (출처: policy_123)
- 국민취업지원제도 (소득: 기준중위소득 100% 이하): 취업준비 중 청년에게 맞춤형 취업지원 — 관심사 '취업'과 일치 (출처: policy_456)
- 청년구직활동지원금 (소득: 기준중위소득 120% 이하): 구직활동비 월 최대 50만원 지원 — 지역, 관심사 모두 일치 (출처: policy_789)

[EXAMPLE - FALLBACK]
해당 조건에 맞는 정책이 없습니다. 대신 전국 공통 정책 3건을 추천합니다.

[EXAMPLE - ASK INFO]
나이 또는 지역 정보를 알려주시면 더욱 정확한 추천이 가능합니다.
""")

combine_prompt = ChatPromptTemplate.from_messages([
    SYSTEM,
    HumanMessagePromptTemplate.from_template(
        "context:\n{context}\n\n질문: {question}\n\n한국어로 간결하게 답변하세요."
    ),
])

# ─────────────────────────────────── #
# 6. RAG 체인
# ─────────────────────────────────── #
def create_rag_chain(vectordb: Chroma, api_key: str) -> ConversationalRetrievalChain:
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    retriever = vectordb.as_retriever(search_kwargs={"k": 30})
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",  
        return_messages=True
    )
    chain =  ConversationalRetrievalChain.from_llm(
        llm, retriever, memory=memory,
        combine_docs_chain_kwargs={"prompt": combine_prompt, "document_variable_name": "context"},
        output_key="answer", return_source_documents=True
    )
    return chain, llm

# ─────────────────────────────────── #
# 7. 가중치 필터 & 폴백
# ─────────────────────────────────── #
# 선형 가중합 모델 기반 필터링
# 가중치: 지역 0.6(전국 포함), 관심사 0.35, 키워드 0.05
MIN_SCORE = 0.3  # 총합 1.0 중 0.3 이상이면 채택

W_REGION   = 0.6
W_INTEREST = 0.35
W_KEYWORD  = 0.05


def jaccard_similarity(a: set, b: set) -> float:
    """두 집합의 자카드 유사도(0~1). 공집합이면 0."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def filter_docs(docs,user_age: int, user_text: str, region: str, interests: List[str]):
    """
    docs        : LangChain Document 리스트
    user_age    : 나이 조건
    user_text   : 사용자가 입력한 원문
    region      : 파싱된 표준 지역(예: '서울')
    interests   : 파싱된 관심사 리스트(예: ['창업', '주거'])
    """
    filtered = []
    kw_hits = extract_keywords(user_text)          # 사용자 문장에서 추출된 키워드 집합
    interests_set = set(interests)

    for d in docs:
        # ─────────────────────── #
        # 0. 나이 필터 : 메타데이터가 없다면 통과
        # ─────────────────────── #
        min_age = d.metadata.get("min_age", 0)
        max_age = d.metadata.get("max_age",999)
        if user_age not in range(min_age, max_age + 1):
            continue

        # ─────────────────────── #
        # 1. 지역 점수 (R: 0 | 0.5 | 1)
        # ─────────────────────── #
        doc_region_str = d.metadata.get("region", "")
        is_nationwide = ("전국" in doc_region_str) or (doc_region_str.strip() == "")
        if is_nationwide:
            region_score = 1.0  # 전국 정책은 동일 가중치
        elif region and any(k in doc_region_str for k in REGION_MAPPING.get(region, [])):
            region_score = 1.0
        elif region and region in doc_region_str:          # 느슨한 포함
            region_score = 0.5
        else:
            region_score = 0.0

        # ─────────────────────── #
        # 2. 관심사 점수 (I: 0~1)
        # ─────────────────────── #
        # 'categories'가 str(list) 형태로 들어올 수도 있어 파싱 진행
        cat_raw = d.metadata.get("categories", [])
        if isinstance(cat_raw, str):
            cat_tokens = [c.strip() for c in re.split(r"[,\[\]'\"\s]+", cat_raw) if c.strip()]
        else:
            cat_tokens = cat_raw
        policy_tags = set(cat_tokens)

        if policy_tags:
            interest_score = jaccard_similarity(interests_set, policy_tags)
        else:
            # 카테고리가 비어 있으면 문서 본문에 관심사 키워드가 직접 포함되어 있는지 계산
            hits = sum(1 for i in interests_set if i in d.page_content)
            interest_score = hits / len(interests_set) if interests_set else 0.0

        # ─────────────────────── #
        # 3. 키워드 점수 (K: 0~1)
        # ─────────────────────── #
        if kw_hits:
            key_raw = d.metadata.get("keywords", [])
            if isinstance(key_raw, str):
                key_tokens = [k.strip() for k in re.split(r"[,\[\]'\"\s]+", key_raw) if k.strip()]
            else:
                key_tokens = key_raw
            doc_keywords = set(key_tokens)
            keyword_score = len(doc_keywords & set(kw_hits)) / len(kw_hits)
        else:
            keyword_score = 0.0

        # ─────────────────────── #
        # 4. 최종 점수 (동적 가중치)
        # ─────────────────────── #
        total_w = 0
        score_sum = 0
        if region:
            total_w += W_REGION
            score_sum += W_REGION * region_score
        if interests_set:
            total_w += W_INTEREST
            score_sum += W_INTEREST * interest_score
        if kw_hits:
            total_w += W_KEYWORD
            score_sum += W_KEYWORD * keyword_score
        # 모든 항목이 비어 있으면 키워드만이라도 사용
        if total_w == 0:
            total_w = W_KEYWORD
            score_sum = W_KEYWORD * keyword_score
        score = score_sum / total_w

        # 디버깅용 점수 메타데이터 저장
        d.metadata["debug_region_score"]   = round(region_score,   3)
        d.metadata["debug_interest_score"] = round(interest_score, 3)
        d.metadata["debug_keyword_score"]  = round(keyword_score,  3)
        d.metadata["debug_total_score"]    = round(score,         3)

        if score >= MIN_SCORE:
            filtered.append((score, d))
        
    # 점수 높은 순 정렬 후 Document 리스트만 반환
    return [d for _, d in sorted(filtered, key=lambda x: x[0], reverse=True)]

# ─────────────────────────────────── #
# 9. 관심사 세부 분류 흐름 유도 (LLM 기반)
# ─────────────────────────────────── #
SUB_INTEREST_MAPPING = {
    "취업": {
        "면접준비": ["모의면접", "면접복장", "이력서 클리닉", "증명사진", "정장 대여"],
        "역량강화": ["직업훈련", "직무교육", "취업기술 향상", "잡케어", "자격증"],
        "현장경험": ["일 경험", "인턴십", "현장실습", "기업 연계 프로젝트"],
        "구직지원금": ["구직촉진수당", "취업성공수당", "취업장려금", "활동비 지원"],
        "고용연계": ["채용연계", "공공기관 채용", "청년채용 연계사업"]
    },
    "창업": {
        "멘토링·상담": ["창업상담", "창업컨설팅", "BM모델", "법률·회계", "세무지원"],
        "사업계획·기획": ["사업계획서 작성", "아이디어 고도화", "창업 R&D", "아이템 발굴"],
        "자금지원": ["금리지원", "보증금", "융자", "창업자금"],
        "창업교육": ["창업 교육", "창업포럼", "창업 아카데미", "네트워킹"]
    },
    "운동": {
        "건강관리": ["헬스케어", "건강검진", "건강서비스", "의료서비스"],
        "체육활동": ["피트니스", "요가", "스포츠센터", "체육관"],
        "정신건강": ["심리상담", "정서지원", "스트레스 완화", "우울증 지원"]
    },
    "주거": {
        "임대료지원": ["월세지원", "임대료 보조", "공공임대주택", "주거바우처"],
        "주택구입·대출": ["주택 대출", "전세 대출", "보증금 지원"],
        "주택개보수": ["주택정비", "리모델링", "빈집 활용"]
    }
}

# 세부 관심사 질문 유도 함수 (대화형 방식, 예시 동적 반영)
def prompt_sub_interest(main_interest: str) -> Optional[str]:
    sub_map = SUB_INTEREST_MAPPING.get(main_interest)
    if not sub_map:
        return None

    print(f"\nBot:\n{main_interest}과 관련해 아래와 같은 지원이 있어요:")
    suggestions = list(sub_map.keys())
    for idx, key in enumerate(suggestions, 1):
        example_keywords = ", ".join(sub_map[key][:2])
        print(f"- {key}: {example_keywords} 관련 지원")

    example_hint = ", ".join(suggestions[:2])
    print(f"\n특별히 궁금한 것이 있으신가요? (예: {example_hint} 등)")
    sel = input("관심 있는 내용을 적어주세요: ").strip()
    for key in suggestions:
        if key in sel:
            return key
    print("입력 내용을 바탕으로 특정 항목을 찾을 수 없었어요. 일반 추천을 진행할게요.")
    return None

# ─────────────────────────────────── #
# 8. 콘솔 채팅
# ─────────────────────────────────── #
def console_chat(rag_chain, llm, keyword_vectordb=None, category_vectordb=None, policy_vectordb=None):
    print("\n챗봇이 시작되었습니다. 종료하려면 '종료'를 입력하세요.\n")

    stored_age = None
    stored_region = None
    stored_interests = []
    # 이미 사용자에게 보여준 정책ID 집합
    recommended_ids = set()

    # Ensure vectordb refers to main policy vectorstore
    vectordb = policy_vectordb if policy_vectordb is not None else policy_vectordb

    # ⏱ total_response_time = 0 # 응답시간 계산 나중에 제거하기
    # ⏱ response_count = 0 # 응답 횟수 나중에 제거
    def is_new_topic(predicted: list[str], stored: list[str]) -> bool:
        return not any(kw in stored for kw in predicted)

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["종료", "exit", "quit"]:
            print("Bot: 이용해 주셔서 감사합니다. 안녕히 가세요!")
            break


        # 사용자가 '다른 정책' 등 추가 추천만 요청했는지 플래그
        force_more_request = is_generic_more_request(user_input)

        # 사용자가 '어떤 분야'를 물으면 현재 가능한 카테고리 제안
        if re.search(r"어떤\s*분야.*(있|야|가)", user_input):
            suggestion = suggest_remaining_interests(stored_interests)
            print(f"Bot:\n현재 선택할 수 있는 분야로는 {suggestion}이 있습니다.\n")
            continue

        if not force_more_request and not is_policy_related_question_llm(user_input):
            print("Bot:\n저는 대한민국 청년 정책 안내를 도와드리는 챗봇이에요! 정책 관련 질문을 해주세요 😊\n")
            continue

        # 유효 질문인지 검사 (단, '다른 정책' 추가 요청은 건너뜀)
        age, region, interests = parse_user_input(user_input)
        if not force_more_request and not is_valid_query(user_input) and not any([age, region, interests]):
            print("Bot:\n안녕하세요! 궁금하신 정책이나 조건을 입력해 주세요 🙂\n")
            continue

        # 사용자 입력에서 자동 정보 추출 및 출력
        user_info = extract_user_info(user_input)
        print(f"[🧠 자동 추출 정보] 나이: {user_info['age']}, 지역: {user_info['region']}, 관심사: {user_info['interests']}, 상태: {user_info['status']}, 소득: {user_info['income']}")

        # extract_user_info의 출력값을 그대로 반영
        if user_info['age']:
            stored_age = user_info['age']
        if user_info['region']:
            stored_region = user_info['region']
        if user_info['interests']:
            stored_interests = user_info['interests']

        # 💡 정보가 모두 없으면 바로 안내하고 다음 입력 대기
        if not any([stored_age, stored_region, stored_interests]):
            print("Bot:\n나이, 지역, 관심사 정보를 알려주시면 맞춤형 정책을 안내해드릴게요 😊\n")
            continue

        # ⏱ start_time = time.time()  # 응답 시간 측정 시작 (위치 이동) # 추후 제거

        # 관심사 추론
        predicted_keywords = None
        embedding = None
        if keyword_vectordb:
            embedding = OpenAIEmbeddings()
            query_vector = embedding.embed_query(user_input)
            docs = keyword_vectordb.similarity_search_by_vector(query_vector, k=3)
            if docs:
                predicted_keywords = [doc.page_content for doc in docs]

        if not predicted_keywords and category_vectordb:
            if embedding is None:
                embedding = OpenAIEmbeddings()
            query_vector = embedding.embed_query(user_input)
            docs = category_vectordb.similarity_search_by_vector(query_vector, k=2)
            if docs:
                predicted_keywords = [doc.page_content for doc in docs]

        if not predicted_keywords:
            from langchain.prompts import PromptTemplate
            prompt = PromptTemplate.from_template("""
            [시스템]
            다음 문장에서 관련 있는 관심사를 추출하세요.
            선택 가능한 항목: 창업, 취업, 금융, 복지, 교육, 공간, 문화예술

            문장:
            {input}

            결과:
            """)
            response = llm.invoke(prompt.format(input=user_input).to_messages())
            predicted_keywords = [i.strip() for i in response.content.split(",") if i.strip()]

        # ─────────────────────────────────────── #
        # 🔧 예측 키워드 → 표준 관심사 매핑 개선
        # - 벡터DB에서 가져온 '제주시', '거주자' 같은 토큰 제거
        # - INTEREST_MAPPING에 정의된 키워드만 표준화해 사용
        # ─────────────────────────────────────── #
        if predicted_keywords:
            std_interests = []
            for kw in predicted_keywords:
                # 먼저, 지역 키워드는 제외
                if is_region_keyword(kw):
                    continue
                # 표준 관심사명 그대로 들어온 경우
                if kw in INTEREST_MAPPING:
                    if kw not in std_interests:
                        std_interests.append(kw)
                    continue
                # 키워드가 INTEREST_MAPPING 하위 키워드에 포함되는지 확인
                for std_i, kws in INTEREST_MAPPING.items():
                    if kw in kws and std_i not in std_interests:
                        std_interests.append(std_i)
                        break
            # 매핑 결과가 없다면 예측 키워드 무시
            predicted_keywords = std_interests if std_interests else None

        # 관심사 초기화 조건 체크 및 저장
        if predicted_keywords:
            if is_new_topic(predicted_keywords, stored_interests):
                print("🧹 기존 관심사를 초기화합니다.")
                new_interests = predicted_keywords
            else:
                new_interests = stored_interests[:]
                for kw in predicted_keywords:
                    if kw not in new_interests:
                        new_interests.append(kw)

            # 지역명을 관심사에서 제거
            filtered_interests = [kw for kw in new_interests if not is_region_keyword(kw)]
            if filtered_interests:
                stored_interests = filtered_interests
            # stored_interests = new_interests  # 기존 직접 대입은 제거/주석 처리

        print(f"[🔍 추론된 관심사] → {predicted_keywords}")
        print(f"[📌 누적 정보] 나이: {stored_age}, 지역: {stored_region}, 관심사: {stored_interests}" )

        # 누적 직후, 사용자 정보가 여전히 모두 비어있다면 추천 차단
        # 수정: 새로 추출된 값이 있으면 추천 흐름 진입하도록 조건 강화
        age = user_info['age']
        region = user_info['region']
        interests = user_info['interests']

        # ------ ① 벡터 검색 & 선별 ------
        filters_keywords_only = {
            "categories": {"$in": stored_interests}
        } if stored_interests else None

        docs = []
        if vectordb is not None:
            try:
                # 만약 '다른 정책' 같은 일반 요청이면 누적 정보를 사용해 질의 재구성
                search_query = user_input
                if force_more_request:
                    search_query = build_query("추천", stored_age, stored_region, stored_interests)
                    # 추가 요청 시 필터 없이 더 많은 후보 검색
                    raw_docs = vectordb.similarity_search(search_query, k=50)
                else:
                    raw_docs = vectordb.similarity_search(search_query, k=50, filter=filters_keywords_only)
            except Exception:
                search_query = user_input
                if force_more_request:
                    search_query = build_query("추천", stored_age, stored_region, stored_interests)
                    raw_docs = vectordb.similarity_search(search_query, k=50)
                else:
                    raw_docs = vectordb.similarity_search(search_query, k=50, filter=filters_keywords_only)

            # ✅ 지역·나이·관심사 기반 스코어링
            user_age_for_score = stored_age if stored_age else 0
            user_region_for_score = stored_region if stored_region else ""
            docs = filter_docs(
                raw_docs,
                user_age_for_score,
                search_query,
                user_region_for_score,
                stored_interests
            )

            docs = docs[:1]  # 상위 1건만

        # ------ ② 출력 로직 ------
        if not docs:
            #print("\nBot:\n현재 조건에 딱 맞는 정책이 보이지 않아요.")

            # 1) 주요/추론 관심사 파악
            main_interest = stored_interests[0] if stored_interests else None
            if not main_interest:
                # 사용자 입력에서 INTEREST_MAPPING 키워드 스캔
                for std_i, kws in INTEREST_MAPPING.items():
                    if any(k in user_input for k in kws):
                        main_interest = std_i
                        break

            # 2) 세부 관심사 질문 (SUB_INTEREST_MAPPING 이용)
            clarified = False
            if main_interest:
                sub_map = SUB_INTEREST_MAPPING.get(main_interest)
                if sub_map:
                    sub_options = ", ".join(sub_map.keys())
                    ask = f"{main_interest}과 관련해 어떤 지원을 찾고 계신가요? (예: {sub_options}) : "
                    sub_choice = input(ask).strip()
                    if sub_choice:
                        if sub_choice not in stored_interests:
                            stored_interests.append(sub_choice)
                        refined_query = f"{user_input} {sub_choice}"
                        raw_docs = vectordb.similarity_search(refined_query, k=50)
                        docs = filter_docs(
                            raw_docs,
                            stored_age if stored_age else 0,
                            refined_query,
                            stored_region if stored_region else "",
                            stored_interests
                        )
                        docs = docs[:3]
                        clarified = True

            # 3) 일반 관심사 질문 (세부 관심사 실패 또는 매핑 없음)
            if not clarified:
                suggestion_list = suggest_remaining_interests(stored_interests)
                prompt_text = f"어떤 분야의 정책을 찾고 계신가요? (예: {suggestion_list}): "
                generic_choice = input(prompt_text).strip()
                if generic_choice:
                    if generic_choice not in stored_interests:
                        stored_interests.append(generic_choice)
                    start_time = time.time()  # ⏱ reset timer after user input
                    refined_query = f"{user_input} {generic_choice}"
                    raw_docs = vectordb.similarity_search(refined_query, k=50)
                    docs = filter_docs(
                        raw_docs,
                        stored_age if stored_age else 0,
                        refined_query,
                        stored_region if stored_region else "",
                        stored_interests
                    )
                    docs = docs[:3]

            # 4) 최종 폴백: 전국 공통 정책
            if not docs:
                print("조건에 맞는 정책이 없어 전국 공통 정책 3건을 대신 보여드릴게요.\n")
                docs = vectordb.similarity_search("청년 정책 전국 공통", k=3)

        # 👉 이미 제시한 정책은 제외하고 최대 3건까지 출력, 중복 제거 강화
        unique_docs = []
        seen_ids = set()              # 중복 제거용(현재 회차)
        for d in docs:
            pid = d.metadata.get("policy_id")
            if not pid:
                continue
            if pid in recommended_ids or pid in seen_ids:
                continue  # 이미 보여줬거나 현재 리스트에 중복
            unique_docs.append(d)
            seen_ids.add(pid)
            break  # ✅ 단 1건만 수집

        # 추가 탐색: 중복 제거로 1건이 안 채워졌을 경우 raw_docs에서 보충 (seen_ids도 체크)
        if len(unique_docs) < 1:
            # raw_docs가 있을 때만
            extra_pool = [rd for rd in raw_docs
                          if rd.metadata.get("policy_id")
                          and rd.metadata.get("policy_id") not in recommended_ids
                          and rd.metadata.get("policy_id") not in seen_ids]
            for rd in extra_pool:
                unique_docs.append(rd)
                seen_ids.add(rd.metadata.get("policy_id"))
                if len(unique_docs) == 1:
                    break
        # ------ ⏱ 응답 시간 측정 및 출력 ------
        # ⏱ end_time = time.time()
        # ⏱ elapsed = end_time - start_time
        # ⏱ total_response_time += elapsed
        # ⏱ response_count += 1

        # ⏱ print(f"\n⏱ 응답 시간: {elapsed:.2f}초")
        # ⏱ print(f"📊 평균 응답 시간: {total_response_time / response_count:.2f}초\n")

        if not unique_docs:
            print("Bot:\n더 이상 새로운 정책을 찾지 못했어요. 다른 조건을 입력해 보실래요?\n")
            continue

        policy_ids = []
        answers    = []
        for doc in unique_docs:
            pid = doc.metadata.get("policy_id", "")
            pname = doc.metadata.get("title", "")
            policy_ids.append(pid)
            answers.append(f"{pname}에 지원할 수 있습니다.")
            recommended_ids.add(pid)          # 기록

        result_obj = {
            "policy_id": policy_ids,
            "answer": answers
        }
        print(json.dumps(result_obj, ensure_ascii=False, indent=2))

# ─────────────────────────────────── #
# Helper: Fallback-based document retrieval
# ─────────────────────────────────── #
def retrieve_with_fallback(query, age, region, interests, vectordb, k=5):
    filters = []

    meta = {}
    if region:
        meta["region"] = {"$contains": region}
    if age:
        meta["min_age"] = {"$lte": age}
        meta["max_age"] = {"$gte": age}
    if interests:
        meta["categories"] = {"$in": interests}
    filters.append(meta)

    if "region" in meta:
        f2 = meta.copy()
        del f2["region"]
        filters.append(f2)

    if "min_age" in meta and "max_age" in meta:
        f3 = meta.copy()
        del f3["min_age"]
        del f3["max_age"]
        filters.append(f3)

    if "categories" in meta:
        filters.append({"categories": {"$in": interests}})

    filters.append({})  # no filters

    for f in filters:
        try:
            docs = vectordb.similarity_search(query, filter=f, k=k)
            if docs:
                return docs
        except Exception as e:
            continue

    return []


# ─────────────────────────────────── #
# 🔗 FastAPI 연동용 단일 요청 처리 함수
# ─────────────────────────────────── #
from typing import Dict

def _compose_reason(doc: Document, user_info: Dict) -> str:
    """
    간단한 추천 사유 문자열 생성
    """
    reasons = []
    age = user_info.get("age")
    region = user_info.get("region")
    interests = user_info.get("interests", [])

    # 나이
    if age is not None:
        min_age = doc.metadata.get("min_age", 0)
        max_age = doc.metadata.get("max_age", 99)
        if min_age <= age <= max_age:
            reasons.append("나이 조건 부합")

    # 지역
    doc_region = doc.metadata.get("region", "")
    if region:
        if "전국" in doc_region or region in doc_region:
            reasons.append("지역 조건 부합")

    # 관심사
    if interests:
        doc_cats = doc.metadata.get("categories", [])
        if isinstance(doc_cats, str):
            doc_cats = [c.strip() for c in doc_cats.split(",") if c.strip()]
        if set(interests) & set(doc_cats):
            reasons.append("관심사 조건 부합")

    return ", ".join(reasons) if reasons else "일부 조건 부합"

def generate_policy_response(
    user_id: str,
    user_input: str,
    *,
    vectordb: Chroma = policy_vectordb,
    keyword_vectordb: Chroma = keyword_vectordb,
    category_vectordb: Chroma = category_vectordb,
) -> dict:
    """
    FastAPI 서버에서 호출 가능한 단일 질의‑응답 함수.
    - user_input: 사용자의 자연어 질문
    - 반환 형식은 업무 요청서에 명시된 JSON 구조를 따른다.
    """

    # ─────────────────────────────── #
    # 👤 세션 메모리 로드 & 머지
    # ─────────────────────────────── #
    session               = SESSION_STORE[user_id]          # dict with 'user_info', 'recommended_ids'
    prev_info             = session["user_info"] or {}
    prev_recommended_ids  = session["recommended_ids"]

    # 1) 새 입력에서 정보 추출
    current_info = extract_user_info(user_input)

    # 2) 이전 정보와 병합 (새 값이 있으면 덮어씀)
    merged_info = prev_info.copy()
    for k, v in current_info.items():
        if v:  # 값이 None/빈 리스트/빈 문자열이 아니면
            merged_info[k] = v

    # 3) 이후 로직은 merged_info 사용
    user_info = merged_info
    age       = user_info.get("age")
    region    = user_info.get("region")
    interests = list(user_info.get("interests", []))  # copy

    # 2) 필수 정보 확인 -----------------------------------------------
    # 👉 누적 정보를 세션에 즉시 저장해 부분 입력도 기억
    session["user_info"] = user_info

    # 누락 항목 식별
    missing = []
    if age is None:
        missing.append("age")
    if region is None:
        missing.append("region")
    if not interests:
        missing.append("interests")

    if missing:
        label_map = {"age": "나이", "region": "지역", "interests": "관심사"}
        missing_kor = [label_map[m] for m in missing]
        prompt_text = f"{', '.join(missing_kor)}를 알려주시면 맞춤형 정책을 추천해드릴게요."
        return {
            "message": prompt_text,
            "missing_info": missing,
        }

    # 3) (선택) 관심사 보강 -------------------------------------------
    #    벡터 DB를 이용해 추가 관심사를 예측하고, 기존 관심사에 병합
    predicted_interests = []
    embedding = None

    if keyword_vectordb:
        embedding = OpenAIEmbeddings()
        qvec = embedding.embed_query(user_input)
        docs = keyword_vectordb.similarity_search_by_vector(qvec, k=3)
        predicted_interests.extend([d.page_content for d in docs])

    if not predicted_interests and category_vectordb:
        if embedding is None:
            embedding = OpenAIEmbeddings()
        qvec = embedding.embed_query(user_input)
        docs = category_vectordb.similarity_search_by_vector(qvec, k=2)
        predicted_interests.extend([d.page_content for d in docs])

    # INTEREST_MAPPING 기반 표준화
    std_preds = []
    for kw in predicted_interests:
        if kw in INTEREST_MAPPING and kw not in std_preds:
            std_preds.append(kw)
            continue
        for std_i, kws in INTEREST_MAPPING.items():
            if kw in kws and std_i not in std_preds:
                std_preds.append(std_i)
                break

    # 병합
    for p in std_preds:
        if p not in interests:
            interests.append(p)

    # 4) 벡터 검색 + 필터링 -------------------------------------------
    search_query = build_query(user_input, age, region, interests)
    try:
        raw_docs = vectordb.similarity_search(search_query, k=50)
    except Exception:
        # 검색 오류 시 최소한의 질의로 재시도
        raw_docs = vectordb.similarity_search(user_input, k=50)

    docs = filter_docs(raw_docs, age, search_query, region, interests)

    # 🔎 이전에 추천했던 정책은 제외 + 중복 응답 차단
    seen_ids = set()
    filtered_docs = []
    for d in docs:
        pid = d.metadata.get("policy_id")
        if not pid or pid in prev_recommended_ids or pid in seen_ids:
            continue
        filtered_docs.append(d)
        seen_ids.add(pid)
        if len(filtered_docs) == 3:
            break
    docs = filtered_docs

    # 5) 결과가 없을 때 폴백 ------------------------------------------
    if not docs:
        fallback_docs = vectordb.similarity_search("청년 정책 전국 공통", k=3)
        policies = []
        for d in fallback_docs:
            policies.append({
                "policy_id": d.metadata.get("policy_id", ""),
                "title":     d.metadata.get("title", ""),
                "summary":   d.metadata.get("summary", "") or d.page_content[:120],
            })
        # 👉 세션 업데이트 (fallback도 기록)
        session["recommended_ids"].update([p["policy_id"] for p in policies])
        session["user_info"] = user_info
        return {
            "message": "조건에 맞는 정책을 찾지 못했어요. 전국 공통 정책을 보여드릴게요.",
            "fallback_policies": policies,
            "user_info": user_info,
        }

    # 6) 정상 추천 -----------------------------------------------------
    policies = []
    for d in docs:
        apply_url = d.metadata.get("apply_url", "")
        if not apply_url:
            # 🔎 페이지 본문이나 요약에서 URL 패턴 추출
            m = re.search(r"https?://[^\s)]+", d.page_content)
            if not m:
                m = re.search(r"https?://[^\s)]+", d.metadata.get("summary", ""))
            if m:
                apply_url = m.group(0)
        policies.append({
            "policy_id": d.metadata.get("policy_id", ""),
            "title":     d.metadata.get("title", ""),
            "summary":   d.metadata.get("summary", "") or d.page_content[:120],
            "apply_url": apply_url,
            "reason":    _compose_reason(d, user_info),
        })

    # 👉 세션 업데이트
    session["recommended_ids"].update([p["policy_id"] for p in policies])
    session["user_info"] = user_info

    # 병합
    user_info["interests"] = interests

    return {
        "message": "추천 정책을 안내드립니다.",
        "policies": policies,
        "user_info": user_info,
    }