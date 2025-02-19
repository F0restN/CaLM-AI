from pydantic import BaseModel, Field
from classes.ChatSession import ChatSession

class RequestBody(BaseModel):
    user_query: str = Field(
        default="Please introduce yourself",
        description="User's consultation question"
    )
    doc_number: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Maximum number of documents to retrieve"
    )
    threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Document relevance threshold"
    )
    max_retries: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Maximum number of query expansion retries"
    )
    model: str = Field(
        default="phi4:latest",
        description="Main generation model selection"
    )
    intermida_model: str = Field(
        default="qwen2.5:latest",
        description="Intermediate decision model selection"
    )
    chat_session: ChatSession = Field(
        default=[],
        description="Communication history"
    )