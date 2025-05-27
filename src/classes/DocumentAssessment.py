from langchain_core.documents import Document
from pydantic import BaseModel, Field


class DocumentAssessment(BaseModel):
    """LLM-as-judge for document relevance evaluation. Given a user query and a document, the LLM will evaluate the relevance, reasoning and missing topics of the document to the query."""

    relevance_score: int = Field(description="how relevant document to user query, scale from 1 to 5")
    reasoning: str = Field(description="reasons supporting relevance score about why so")
    missing_topics: list[str] = Field(description="3 missing topics given document to user query, if any")

    class Config:
        arbitrary_types_allowed = True

class AnnotatedDocumentEvl(DocumentAssessment):
    """Annotated document combined with evaluation result."""

    document: Document = Field(description="Evaluated document itself")

    class Config:
        arbitrary_types_allowed = True

    def __str__(self) -> str:
        return f"Title: {self.document.metadata.get('title', 'Untitled Document')}; Content: {self.document.page_content}"
