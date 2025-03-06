from typing import List, Dict
from pydantic import BaseModel, Field

class AIGeneration(BaseModel):
    answer: str = Field(description="answer for user's question")
    follow_up_questions: List[str] = Field(description="possible questions that user might ask after reading the answer")

class Generation(AIGeneration):
    sources: List[Dict[str | None, str | None]] = Field(description="list of sources that we use to generate answer")