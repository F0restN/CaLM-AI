from typing import Sequence, List, Dict, Any

from typing_extensions import Annotated
from dataclasses import dataclass, field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, AnyMessage
from src.utils.logger import logger
from fastapi import FastAPI

from answer_generation import generate_answer
from src.classes.ChatSession import ChatSession
from classes.ReasonedDocument import ReasonedDocument
from checkpoints.routering import get_routing_decision
from checkpoints.retrieval_grading import grade_retrieval
from embedding.vector_store import get_chroma_vectorstore, retrieve_docs
from embedding.embedding_models import get_bge_embedding

app = FastAPI()

messages: Annotated[Sequence[AnyMessage], "sequence of messages"] = field(
    default_factory=list
)
messages = [SystemMessage(
    content="Your are Calm AI, an consultant for caregiving for ADRD older adults")
]

@app.post("/")
def main(
    user_query: str = "my mom confirmed early on-set alzheimer, what does that means? how should I take care of her ?", 
    doc_number: int = 5,
    threshold: float = 0.8,
    chat_session: ChatSession = ""
):

    latest_human_message = ""

    # Routing
    messages.append(HumanMessage(content=user_query))
    latest_human_message = messages[-1].content
    user_intention = get_routing_decision(latest_human_message)

    # Get relevant document
    embedding_model = get_bge_embedding()
    vectorstore_path = ""
    if user_intention['knowledge_base'] == "peer_support":
        vectorstore_path = "./data/vector_database/peer_kb"
    else:
        vectorstore_path = "./data/vector_database/research_kb"
    relevant_doc = retrieve_docs(
        latest_human_message,
        get_chroma_vectorstore(vectorstore_path, embedding_model),
        k=doc_number
    )

    # Grade documents
    graded_doc = grade_retrieval(latest_human_message, relevant_doc, model="llama3.2", temperature=0)

    # Filter out documents with relevance_score lower than 0.5
    filtered_docs = []
    for doc, grade in zip(relevant_doc, graded_doc):
        if grade["relevance_score"] >= threshold:
            reasoned_doc = ReasonedDocument(
                document=doc,
                relevance_score=grade["relevance_score"],
                reasoning=grade["reasoning"],
                missing_topics=grade["missing_aspects"]
            )
            filtered_docs.append(reasoned_doc)

    # Re-rank the filtered documents based on relevance_score in descending order
    filtered_docs.sort(key=lambda x: x.relevance_score, reverse=True)
    
    res = generate_answer(latest_human_message, filtered_docs, model="llama3.2", temperature=0)
    
    
    
    from pydantic import BaseModel, ValidationError
    import re

    class MarkdownResponse(BaseModel):
        content: str
        
        @classmethod
        def validate_markdown(cls, v):
            # Basic markdown pattern check for headings, lists, or emphasis
            markdown_pattern = r"(^#+\s|^- |\*\*.*\*\*|__.*__|\*.*\*|_.*_|!\[.*\]\(.*\)|\[.*\]\(.*\))"
            if not re.search(markdown_pattern, v, re.MULTILINE):
                raise ValueError("Response is not in valid markdown format")
            return v

        class Config:
            json_schema_extra = {
                "example": {
                    "content": "### Answer\nSample response\n\n**Sources**\n- Doc1-url"
                }
            }

    # Validate and retry logic
    max_retries = 3
    attempt = 0
    
    while attempt < max_retries:
        try:
            validated_response = MarkdownResponse(content=res)
            return validated_response.content
        except ValidationError as e:
            logger.warning(f"Markdown validation failed (attempt {attempt+1}): {str(e)}")
            # Regenerate the answer with stricter formatting instructions
            res = generate_answer(
                question=latest_human_message,
                context_chunks=filtered_docs,
                model="llama3.2",
                temperature=0
            )
            attempt += 1
    
    # If all retries fail, return error with original response
    logger.error("Markdown validation failed after 3 attempts")
    return "Error: Could not format response properly. Please try again."
    
    
    
    
    
    

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    user_query = input("Enter your question \n")
    res = main(user_query)
    print(res)
