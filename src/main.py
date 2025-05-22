# ruff: noqa: ANN201, SIM108

import os
from typing import TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI
from langgraph.graph import END, StateGraph

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

# Define state machine state structure

class GraphState(TypedDict):
    """State machine state structure."""

    user_query: str
    chat_session: list[BaseChatMessage]
    model: str
    intermediate_model: str
    threshold: float
    max_retries: int
    doc_number: int
    temperature: float

    # Running states
    query_message: str

    # Routing function
    adaptive_decision: AdaptiveDecision | None

    # Retrieval related
    retry_count: int
    retrieved_docs: list
    filtered_docs: list
    missing_topics: list

    # Decisive states
    final_answer: str

    def __init__(self, **kwargs: dict) -> None:
        """Initialize the state machine."""
        super().__init__(**kwargs)
        self.retrieved_docs = []
        self.filtered_docs = []
        self.missing_topics = []


# Initialize knowledge base connections
PGVECTOR_CONN = os.environ.get("PGVECTOR_CONN")

assert PGVECTOR_CONN is not None, "PGVECTOR_CONN is not set"

p_kb = get_connection(connection=PGVECTOR_CONN, embedding_model=get_nomic_embedding(
), collection_name="peer_support_kb")
r_kb = get_connection(connection=PGVECTOR_CONN, embedding_model=get_nomic_embedding(
), collection_name="research_kb")

# TODO: Memory procedures - Recall user relevant memories from LTM.

# TODO: Memory - summarize user's memories.

# TODO: Episodic memory.


def detect_intention(state: GraphState):
    """User intention detection node."""
    logger.info(f"state['user_query']: {state['user_query']}")

    decision = adaptive_rag_decision(
        state["user_query"],
        model=state["intermediate_model"],
        temperature=state["temperature"],
    )

    return {"adaptive_decision": decision}


def retrieve_documents(state: GraphState):
    """Retrieve documents from knowledge base."""
    if state["adaptive_decision"] and state["adaptive_decision"].knowledge_base == "peer_support":
        cls_kb = p_kb
    else:
        cls_kb = r_kb

    # TODO: Try to use Hybrid Search
    docs = cls_kb.similarity_search(
        state["query_message"], k=state["doc_number"], search_type="hybrid")

    logger.success(f"Similarity search retrieved | {len(docs)} | documents")

    return {
        "retrieved_docs": docs,
        "retry_count": state["retry_count"] + 1,
    }


async def grade_documents(state: GraphState):
    """Asynchronously grade documents."""
    graded = await grade_retrieval_batch(
        state["query_message"],
        state["retrieved_docs"],
        model=state["intermediate_model"],
        temperature=state["temperature"],
    )

    filtered = state["filtered_docs"]
    missing = []
    for doc in graded:
        if doc.relevance_score >= state["threshold"]:

            # Remove duplicates
            if doc not in filtered:
                filtered.append(doc)

        else:
            missing.extend(doc.missing_topics)

    logger.success(
        f"Filtered in {len(filtered)} documents, out of {len(graded)} graded documents")

    return {
        "filtered_docs": sorted(filtered, key=lambda x: x.relevance_score, reverse=True),
        "missing_topics": missing,
    }


def expand_query(state: GraphState):
    """Query expansion node."""
    new_query = query_extander(
        state["query_message"],
        state["missing_topics"],
        model=state["intermediate_model"],
        temperature=state["temperature"],
    )

    return {
        "query_message": new_query,
    }


def generate_final_answer(state: GraphState):
    """Return final answer generation node."""
    current_chat = [BaseChatMessage(**chat) for chat in state["chat_session"]]
    work_memory = format_conversation_pipeline(
        current_chat[-6:-1] if len(current_chat) >= 4 else current_chat,
    )

    answer = generate_answer(
        question=state["user_query"],
        context_chunks=state["filtered_docs"] if state["filtered_docs"] else [
        ],
        work_memory=work_memory,
        temperature=state["temperature"],
        isInformal=False,
    )
    return {"final_answer": answer}


def direct_answer(state: GraphState):
    """Direct answer node (when retrieval not needed)."""
    current_chat = [BaseChatMessage(**chat) for chat in state["chat_session"]]

    work_memory = format_conversation_pipeline(
        current_chat[-6:-1] if len(current_chat) >= 4 else current_chat,
    )

    answer = generate_answer(
        question=state["user_query"],
        context_chunks=[],
        work_memory=work_memory,
        temperature=state["temperature"],
        isInformal=True,
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
    builder.add_node("generate_answer", generate_final_answer)
    builder.add_node("direct_answer", direct_answer)

    # Set entry point
    builder.set_entry_point("detect_intention")

    # Add conditional edges
    def should_retrieve(state: GraphState) -> bool:
        return state["adaptive_decision"].require_extra_re

    def should_retry(state: GraphState) -> bool:
        return (state["retry_count"] < state["max_retries"] and len(state["filtered_docs"]) < state["doc_number"])

    # Main process routing
    builder.add_conditional_edges(
        "detect_intention",
        should_retrieve,
        {
            True: "retrieve_docs",
            False: "direct_answer",
        },
    )

    # Retrieval-grading loop process
    builder.add_edge("retrieve_docs", "grade_docs")
    builder.add_conditional_edges(
        "grade_docs",
        should_retry,
        {
            True: "expand_query",
            False: "generate_answer",
        },
    )
    builder.add_edge("expand_query", "retrieve_docs")

    # End nodes
    builder.add_edge("direct_answer", END)
    builder.add_edge("generate_answer", END)

    return builder.compile()


calm_agent = setup_workflow()

# ============== | API Service | ==============


@fastapi_app.post("/ask-calm-adrd-agent")
async def calm_adrd_agent_api(request: RequestBody):
    """Maintain a callable API for the Calm ADRD Agent to pipeline."""
    logger.info(f"request: {request}")

    initial_state = {
        "user_query": request.user_query,
        "chat_session": request.chat_session,
        "model": request.model,
        "intermediate_model": request.intermediate_model,
        "threshold": request.threshold,
        "max_retries": request.max_retries,
        "doc_number": request.doc_number,
        "temperature": request.temperature,

        # Running states
        "query_message": request.user_query,

        # Routing function
        "adaptive_decision": None,

        # Retrieval related
        "retry_count": 0,
        "retrieved_docs": [],
        "filtered_docs": [],
        "missing_topics": [],

        # Decisive states
        "final_answer": "",
        **request.model_dump(),
    }

    try:
        async for _ in calm_agent.astream(initial_state, stream_mode="values"):
            continue
        return _.get("final_answer")
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error in calm_agent stream: {e!s}")
        return {
            "answer": "Sorry, an error occurred while processing your request. Please try again later.",
            "sources": [],
            "follow_up_questions": [],
        }

    return "Hello"


@fastapi_app.get("/server-health-check")
def health_check_api():
    """Health check API."""
    return {"status": "CaLM ADRD Agent Server is Healthy"}


def generate_graph_diagram():
    """Generate graph diagram."""
    logger.info("Generating graph diagram")
    return calm_agent.get_graph().draw_mermaid_png(output_file_path="./public/calm_adrd_langgraph_diagram.png")

