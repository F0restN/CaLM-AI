import os
import json

from dotenv import load_dotenv
from typing import Sequence, List, Dict, Any
from typing_extensions import Annotated
from dataclasses import dataclass, field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, AnyMessage
from fastapi import FastAPI

from classes.AdaptiveDecision import AdaptiveDecision
from classes.RequestBody import RequestBody
from answer_generation import generate_answer
from checkpoints.routering import get_routing_decision
from checkpoints.retrieval_grading import grade_retrieval
from checkpoints.query_extander import query_extander
from checkpoints.adaptive_decision import adaptive_rag_decision
from embedding.vector_store import get_connection
from embedding.embedding_models import get_nomic_embedding

app = FastAPI()

load_dotenv()

PGVECTOR_CONN = os.environ.get("PGVECTOR_CONN")
if not PGVECTOR_CONN:
    raise ValueError("PGVECTOR_CONN environment variable is not set")


r_kb = get_connection(connection=PGVECTOR_CONN,embedding_model=get_nomic_embedding(),collection_name='research_kb')
p_kb = get_connection(connection=PGVECTOR_CONN,embedding_model=get_nomic_embedding(),collection_name='peer_support_kb')


@app.post("/")
def main(requestBody: RequestBody):
    
    print(requestBody)
    print("\n")
    
    user_query = requestBody.user_query
    threshold = requestBody.threshold
    max_retries = requestBody.max_retries
    doc_number = requestBody.doc_number
    model = requestBody.model
    intermida_model = requestBody.intermida_model
    chat_session = requestBody.chat_session
    
    if not user_query:
        return "Please type in your question"
    
    latest_human_message = user_query
    
    
    # ============= User Intention Detection and Routing to correspondent KB ============= 
    
    # Routing, answer directly if retrieval is not necessary
    adaptive_decision: AdaptiveDecision = adaptive_rag_decision(latest_human_message, model=intermida_model)
    
    if adaptive_decision.require_extra_re is False:
        return generate_answer(latest_human_message, model=model, temperature=0.6)
    

    # Get relevant document, search professional kb either way 
    # relevant_doc = []    
    # TODO: Format issue, haven't been used for generation
    # relevant_doc.append(
    #     r_kb.similarity_search(latest_human_message)
    # )
    
    ## Retrieve relevant documents according to user's intention
    if adaptive_decision.knowledge_base == "peer_support":
        crs_kb = p_kb
    else:
        crs_kb = r_kb
    
    # ============= Document grading and query expand =============     
    
    retry_count = 0
    filtered_docs = []
    query_messgae = latest_human_message
    
    while True:
        retrieved_docs = crs_kb.similarity_search(query_messgae, k=doc_number)
        graded_doc = grade_retrieval(latest_human_message, retrieved_docs, model=intermida_model, temperature=0)   

        missing_topics = []
        for doc in graded_doc:
            if doc.relevance_score >= threshold:
                filtered_docs.append(doc)
            else:
                missing_topics.append([ mt for mt in doc.missing_topics])
        
        query_messgae = query_extander(latest_human_message, missing_topics, model=intermida_model)[0]
        retry_count = retry_count + 1
        
        if retry_count >= max_retries or len(filtered_docs) >= doc_number:
            break    
    
    filtered_docs.sort(key=lambda x: x.relevance_score, reverse=True) # Sort all documents by relevance score
    
    
    # ============= Generate =============     
    
    return generate_answer(latest_human_message, filtered_docs, model=model, temperature=0)
    
    # from pydantic import BaseModel, ValidationError
    # import re

    # class MarkdownResponse(BaseModel):
    #     content: str
        
    #     @classmethod
    #     def validate_markdown(cls, v):
    #         # Basic markdown pattern check for headings, lists, or emphasis
    #         markdown_pattern = r"(^#+\s|^- |\*\*.*\*\*|__.*__|\*.*\*|_.*_|!\[.*\]\(.*\)|\[.*\]\(.*\))"
    #         if not re.search(markdown_pattern, v, re.MULTILINE):
    #             raise ValueError("Response is not in valid markdown format")
    #         return v

    #     class Config:
    #         json_schema_extra = {
    #             "example": {
    #                 "content": "### Answer Sample response **Sources** - Doc1-url"
    #             }
    #         }


@app.get("/health")
def health_check():
    return {"status": "healthy"}    

if __name__ == "__main__":
    user_query = input("Enter your question \n")
    
    res = main(user_query if user_query else "my mom confirmed early on-set alzheimer, what does that means? how should I take care of her ?")
    
    print(res)
