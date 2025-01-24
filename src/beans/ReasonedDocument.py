from langchain_core.documents import Document


class ReasonedDocument:
    document: Document
    relevance_score: float
    reasoning: str
    missing_topics: str

    def __init__(self, document, relevance_score, reasoning, missing_topics):
        self.document = document
        self.relevance_score = relevance_score
        self.reasoning = reasoning
        self.missing_topics = missing_topics

    def __str__(self):
        return f"Document: {self.document}\n Relevance Score: {self.relevance_score}\n Reasoning: {self.reasoning}\n"

    def __call__(self, *args, **kwargs):
        return self
