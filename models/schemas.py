from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    user_id: str
    question: str

class PolicyItem(BaseModel):
    policy_id: str
    title: str
    summary: Optional[str]
    apply_url: Optional[str]
    reason: Optional[str]

class ChatResponse(BaseModel):
    message: str
    policies: Optional[List[PolicyItem]] = None
    user_info: Optional[dict] = None
    missing_info: Optional[List[str]] = None
    fallback_policies: Optional[List[PolicyItem]] = None