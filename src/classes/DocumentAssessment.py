from pydantic import BaseModel, Field
from langchain_core.documents import Document


class DocumentAssessment(BaseModel):
    relevance_score: float = Field(description="how relevant document to user query, scale from 0 to 1 in 3 decimals")
    reasoning: str = Field(description="reasons supporting relevance score")
    missing_topics: list[str] = Field(description="missing topics from user query")

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


class AnnotatedDocumentEvl(DocumentAssessment):    
    document: Document = Field(description="Evaluated document itself")

    class Config:
        arbitrary_types_allowed = True