from pydantic import BaseModel
from langchain_core.documents import Document


class ReasonedDocument(BaseModel):
    document: Document
    relevance_score: float
    reasoning: str
    missing_topics: str

    class Config:
        arbitrary_types_allowed = True

    @property
    def metadata(self):
        return self.document.metadata

    @property
    def page_content(self):
        return self.document.page_content

    def __str__(self):
        return f"Document: {self.document}\n Relevance Score: {self.relevance_score}\n Reasoning: {self.reasoning}\n"

    def __call__(self, *args, **kwargs):
        return self

