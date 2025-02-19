from pydantic import BaseModel, Field
from typing import List, Literal

# id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class ChatMessage(BaseModel):
    id: str = None
    role: Literal["user", "assistant", "system"] = None
    content: str = None
    timestamp: int = None

class ChatSession(BaseModel):
    model: str
    messages: List[ChatMessage] = Field(default_factory=list)
    chat_id: str
    session_id: str
    id: str

    # class Config:
    #     validate_assignment = True
    #     json_encoders = {
    #         uuid.UUID: lambda v: str(v)
    #     }