from typing import List
from pydantic import BaseModel, Field

class Generation(BaseModel):
    answer: str = Field(description="main answer for user's question")
    sources: List[str] = Field(description="list of sources that we use to generate answer")
    follow_up_questions: List[str] = Field(description="possible follow up question according to answer")