from fastapi import FastAPI
from models import LLMRequest, LLMResponse
from LLM.LLM_interface import get_policy_answer
from exceptions import raise_400, raise_422, raise_500

app = FastAPI(title="Timing-LLM-Server")

@app.post("/llm/answers", response_model=LLMResponse)
def ask_llm(request: LLMRequest):
    if not request.user_id or not request.question:
        raise_400("Missing required field: 'user_id' or 'question'")
    
    try:
        answer = get_policy_answer(request.question)

        return LLMResponse(answer=answer)
    except Exception as e:
        raise_500(f"LLM processing failed: {str(e)}")
