
from pydantic import BaseModel, Field


class Source(BaseModel):
    index: int = Field(description="index of the source")
    url: str = Field(description="url of the source")
    title: str = Field(description="title of the source")

class AIGeneration(BaseModel):
    answer: str = Field(description="answer for user's question, use your best knowledge and judgement to answer the question, say 'I'm sorry, I don't know' if you don't know the answer")
    follow_up_questions: list[str] = Field(description="possible questions that user might ask after reading the answer, if there are no follow up questions, return an empty list")

class Generation(AIGeneration):
    sources: list[Source] = Field(description="list of sources that we use to generate answer")
