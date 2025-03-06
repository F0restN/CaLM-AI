from typing import Literal
from pydantic import BaseModel, Field


class AdaptiveDecision(BaseModel):
    require_extra_re: bool = Field(
        description="True if extra retrieval is necessary otherwise False",
        default=False
    )
    knowledge_base: Literal["research", "peer_support", "NA"] = Field(
        description="Knowledge base that is most relevant to current user query or NA if require_extra_re is false",
        default="NA"
    )
    
    class Config:
        arbitrary_types_allowed = True
        
    def to_json(self):
        return self.model_dump_json()
        
    def __str__(self):
        return f"Extra retrieval {self.require_extra_re} necessary, lead to {self.knowledge_base} "