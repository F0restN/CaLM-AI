# ruff: noqa: ANN201, SIM108

import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from checkpoints.adaptive_decision import adaptive_rag_decision
from checkpoints.answer_generation import generate_answer
from checkpoints.query_extander import query_extander
from checkpoints.retrieval_grading import grade_retrieval_batch
from classes.AdaptiveDecision import AdaptiveDecision
from classes.ChatSession import BaseChatMessage
from classes.RequestBody import RequestBody
from embedding.embedding_models import get_nomic_embedding
from embedding.vector_store import get_connection
from memory.memory_proc import format_conversation_pipeline
from utils.logger import logger

load_dotenv()

fastapi_app = FastAPI()

# Define state machine state structure using Pydantic BaseModel

class GraphState(BaseModel):
    """State machine state structure using Pydantic BaseModel contains necessary fields for CaLM AI ADRD Agent."""

    # Runtime Input parameters
    user_query: str = Field(..., description="User's input query")
    chat_session: list[BaseChatMessage] = Field(default_factory=list, description="Chat session history, used for work memory")

    # Hyperparameters
    model: str = Field(default="deepseek-v3", description="LLM model to use for answer generation")
    intermediate_model: str = Field(default="qwen2.5:14b", description="Intermediate model for auxilary tasks, such as query expansion, document grading, etc.")
    threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Relevance threshold")
    max_retries: int = Field(default=3, ge=1, description="Maximum retry attempts")
    doc_number: int = Field(default=5, ge=1, description="Number of documents to retrieve")
    temperature: float = Field(default=0.3, ge=0.0, le=1.0, description="Model temperature")

    # Running states
    query_message: str = Field(default="", description="Current query message, original from user query modified by query expansion")
    final_answer: str = Field(default="", description="Final generated answer")
    retrieved_docs: list = Field(default_factory=list, description="Retrieved documents")
    filtered_docs: list = Field(default_factory=list, description="Filtered documents")
    missing_topics: list = Field(default_factory=list, description="Missing topics for query expansion")

    # Routing function
    adaptive_decision: Optional[AdaptiveDecision] = Field(default=None, description="Adaptive decision result, whether to use extra knowledge about ADRD. Determined by user's query")  # noqa: UP007
    retry_count: int = Field(default=0, ge=0, description="Current retry count")

    class Config:
        """Pydantic BaseModel Config."""

        # Allow mutation for state updates
        validate_assignment = True
        arbitrary_types_allowed = True
        # Allow extra fields that might be added dynamically
        extra = "allow"


# Initialize knowledge base connections
PGVECTOR_CONN = os.environ.get("PGVECTOR_CONN")

assert PGVECTOR_CONN is not None, "PGVECTOR_CONN is not set"

p_kb = get_connection(connection=PGVECTOR_CONN, embedding_model=get_nomic_embedding(), collection_name="peer_support_kb")
r_kb = get_connection(connection=PGVECTOR_CONN, embedding_model=get_nomic_embedding(), collection_name="research_kb")

def detect_intention(state: GraphState) -> dict:
    """User intention detection node. Determine whether to use extra knowledge about ADRD."""
    logger.info(f"User's query: {state.user_query}")

    decision = adaptive_rag_decision(
        state.user_query,
        model=state.intermediate_model,
        temperature=state.temperature,
    )

    return {"adaptive_decision": decision}


def retrieve_documents(state: GraphState) -> dict:
    """Retrieve documents from knowledge base."""
    if state.adaptive_decision and state.adaptive_decision.knowledge_base == "peer_support":
        cls_kb = p_kb
    else:
        cls_kb = r_kb

    docs = cls_kb.similarity_search(state.query_message, k=state.doc_number, search_type="hybrid")

    logger.success(f"Similarity search retrieved | {len(docs)} | documents")

    return {
        "retrieved_docs": docs,
        "retry_count": state.retry_count + 1,
    }


async def grade_documents(state: GraphState) -> dict:
    """Asynchronously grade documents. Filter out irrelevant documents and identify missing topics for query expansion."""
    graded = await grade_retrieval_batch(
        state.query_message,
        state.retrieved_docs,
        model=state.intermediate_model,
        temperature=state.temperature,
    )

    filtered = state.filtered_docs.copy() if state.filtered_docs else []
    missing = []
    for doc in graded:
        if doc.relevance_score >= state.threshold:
            # Remove duplicates
            if doc not in filtered:
                filtered.append(doc)
        else:
            missing.extend(doc.missing_topics)

    logger.success(
        f"Filtered in {len(filtered)} documents, out of {len(graded)} graded documents",
    )

    return {
        "filtered_docs": sorted(filtered, key=lambda x: x.relevance_score, reverse=True),
        "missing_topics": missing,
    }


def expand_query(state: GraphState) -> dict:
    """Query expansion node."""
    new_query = query_extander(
        state.query_message,
        state.missing_topics,
        model=state.intermediate_model,
        temperature=state.temperature,
    )

    return {
        "query_message": new_query,
    }


def generate_answer_unified(state: GraphState) -> dict:
    """Unified answer generation node - handles both direct and retrieval-based responses."""
    work_memory = format_conversation_pipeline(
        state.chat_session[-6:-1] if len(state.chat_session) >= 4 else state.chat_session,
    )

    # Determine if we have retrieved documents (RAG mode vs direct mode)
    has_context = state.filtered_docs and len(state.filtered_docs) > 0

    # Use retrieved documents if available, otherwise empty list
    context_chunks = state.filtered_docs if has_context else []

    # Set informal tone for direct answers (no retrieval), formal for RAG answers
    is_informal = not has_context

    answer = generate_answer(
        question=state.user_query,
        context_chunks=context_chunks,
        work_memory=work_memory,
        temperature=state.temperature,
        isInformal=is_informal,
    )

    return {"final_answer": answer}


# Build state machine


def setup_workflow():
    """Return the workflow of the Calm ADRD Agent."""
    builder = StateGraph(GraphState)

    # Add nodes
    builder.add_node("detect_intention", detect_intention)
    builder.add_node("retrieve_docs", retrieve_documents)
    builder.add_node("grade_docs", grade_documents)
    builder.add_node("expand_query", expand_query)
    builder.add_node("generate_answer", generate_answer_unified)  # Single unified node

    # Set entry point
    builder.set_entry_point("detect_intention")

    # Add conditional edges
    def should_retrieve(state: GraphState) -> bool:
        return state.adaptive_decision is not None and state.adaptive_decision.require_extra_re

    def should_retry(state: GraphState) -> bool:
        return (state.retry_count < state.max_retries and 
                len(state.filtered_docs) < state.doc_number)

    # Main process routing - both paths now go to the same unified answer node
    builder.add_conditional_edges(
        "detect_intention",
        should_retrieve,
        {
            True: "retrieve_docs",
            False: "generate_answer",  # Direct to unified answer node
        },
    )

    # Retrieval-grading loop process
    builder.add_edge("retrieve_docs", "grade_docs")
    builder.add_conditional_edges(
        "grade_docs",
        should_retry,
        {
            True: "expand_query",
            False: "generate_answer",  # Same unified answer node
        },
    )
    builder.add_edge("expand_query", "retrieve_docs")

    # End node - only one path now
    builder.add_edge("generate_answer", END)

    return builder.compile()


calm_agent = setup_workflow()

# ============== | API Service | ==============


@fastapi_app.post("/ask-calm-adrd-agent")
async def calm_adrd_agent_api(request: RequestBody):
    """Maintain a callable API for the Calm ADRD Agent to pipeline."""
    logger.info(f"request: {request}")

    # Create initial state using Pydantic model
    initial_state = GraphState(
        user_query=request.user_query,
        chat_session=request.chat_session,
        model=request.model,
        intermediate_model=request.intermediate_model,
        threshold=request.threshold,
        max_retries=request.max_retries,
        doc_number=request.doc_number,
        temperature=request.temperature,
        query_message=request.user_query,  # Initialize query_message with user_query
    )

    try:
        # Convert Pydantic model to dict for graph execution
        final_state = None
        async for state_update in calm_agent.astream(initial_state.model_dump(), stream_mode="values"):
            final_state = state_update

        # Make sure final_answer is not empty
        assert final_state is not None, "Final state is None"
        assert final_state.final_answer, "Final answer is empty"

        return final_state.get("final_answer", "")
    except AssertionError as e:
        logger.error(f"Assertion error in calm_agent stream: {e!s}")
        return {
            "answer": f"Sorry, an error occurred while processing your request. Please try again later.{e}",
            "sources": [],
            "follow_up_questions": [],
        }
    except Exception as e:
        logger.error(f"Error in calm_agent stream: {e!s}")
        return {
            "answer": f"Sorry, an error occurred while processing your request. Please try again later.{e}",
            "sources": [],
            "follow_up_questions": [],
        }


@fastapi_app.get("/server-health-check")
def health_check_api():
    """Health check API."""
    return {"status": "CaLM ADRD Agent Server is Healthy"}


def generate_graph_diagram():
    """Generate graph diagram."""
    logger.info("Generating graph diagram")
    return calm_agent.get_graph().draw_mermaid_png(output_file_path="./public/calm_adrd_langgraph_diagram.png")

if __name__ == "__main__":
    generate_graph_diagram()
