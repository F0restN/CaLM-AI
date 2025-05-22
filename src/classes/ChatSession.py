from pydantic import BaseModel, Field
from typing import List, Literal

# id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class BaseChatMessage(BaseModel):
    role: str
    content: str
    
class ChatMessage(BaseChatMessage):
    id: str
    timestamp: int

class ChatSession(BaseModel):
    model: str
    messages: List[ChatMessage] = Field(default_factory=list)
    chat_id: str
    session_id: str
    id: str