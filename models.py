from pydantic import BaseModel

class LLMRequest(BaseModel):
    user_id: str
    question: str

class LLMResponse(BaseModel):
    answer: str
