from fastapi import APIRouter, Request
from models.schemas import ChatRequest, ChatResponse
from services.chatbot_service import ChatbotService

router = APIRouter()
chatbot_service = None

@router.post("/llm/answers", response_model=ChatResponse, summary="정책 챗봇 질의", description="사용자 입력을 기반으로 정책 챗봇에게 맞춤형 정책을 질의합니다.")
async def chat_endpoint(req: ChatRequest, request: Request):
    global chatbot_service
    if chatbot_service is None:
        chatbot_service = ChatbotService(request.app.state.vectorstores["policy"])
    result = chatbot_service.chat(req.user_id, req.question)
    return result