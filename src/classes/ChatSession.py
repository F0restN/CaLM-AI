from pydantic import BaseModel, Field
from typing import List, Literal

# id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class ChatMessage(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    timestamp: int

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