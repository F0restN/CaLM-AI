from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from classes.ChatSession import BaseChatMessage


class CurrentSession(BaseModel):
    user_id: Optional[str] = None
    chat_id: Optional[str] = None
    message_id: Optional[str] = None
    session_id: Optional[str] = None
    tool_ids: Optional[str] = None
    files: Optional[str] = None
    features: Dict[str, bool] = Field(default_factory=lambda: {
        "image_generation": False,
        "web_search": False,
        "code_interpreter": False,
    })


class User(BaseModel):
    name: str
    id: str 
    email: str
    role: str


class BodyConfig(BaseModel):
    stream: bool = False
    model: str = "calm_adrd_pipeline"
    messages: List[dict] = []
    current_session: CurrentSession = Field(default_factory=CurrentSession)
    user: Optional[User] = None


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
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Document relevance threshold"
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Model temperature"
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
    intermediate_model: str = Field(
        default="qwen2.5:latest",
        description="Intermediate decision model selection"
    )
    chat_session: List[BaseChatMessage] = Field(
        default=[],
        description="Communication history"
    )
    body_config: BodyConfig = Field(
        default_factory=BodyConfig,
        description="Other configuration"
    )