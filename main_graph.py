import os
import logging
import json
from typing import TypedDict
from langgraph.graph import StateGraph, END
from fastapi import FastAPI

from classes.AdaptiveDecision import AdaptiveDecision
from classes.RequestBody import RequestBody
from answer_generation import generate_answer
from checkpoints.retrieval_grading import grade_retrieval
from checkpoints.query_extander import query_extander
from checkpoints.adaptive_decision import adaptive_rag_decision
from embedding.vector_store import get_connection
from embedding.embedding_models import get_nomic_embedding
from dotenv import load_dotenv

load_dotenv()
fastapi_app = FastAPI()

logger = logging.getLogger("uvicorn.access")



# Define state machine state structure
class GraphState(TypedDict):
    # Static states
    user_query: str
    chat_session: list
    model: str
    intermida_model: str
    threshold: float
    max_retries: int
    doc_number: int
    temperature: float
    
    # Running states
    
    # Routing function
    adaptive_decision: AdaptiveDecision | None
    
    # Retrieval related
    retry_count: int = 0
    retrieved_docs: list = []
    filtered_docs: list = []
    missing_topics: list = []
    
    # Decisive states
    final_answer: str | None = None

# Initialize knowledge base connections
PGVECTOR_CONN = os.environ.get("PGVECTOR_CONN")
p_kb = get_connection(connection=PGVECTOR_CONN, embedding_model=get_nomic_embedding(), collection_name='peer_support_kb')
r_kb = get_connection(connection=PGVECTOR_CONN, embedding_model=get_nomic_embedding(), collection_name='research_kb')

# Define node functions
def detect_intention(state: GraphState):
    """User intention detection node"""
    
    decision = adaptive_rag_decision(
        state["user_query"],
        model=state["intermida_model"],
        temperature=state["temperature"]
    )
        
    return {"adaptive_decision": decision}



def retrieve_documents(state: GraphState):
    """Document retrieval node"""
    
    if state["adaptive_decision"].knowledge_base == "peer_support":
        cls_kb = p_kb
    else:
        cls_kb = r_kb
    
    docs = cls_kb.similarity_search(state["user_query"], k=state["doc_number"])
    return {
        "retrieved_docs": docs, 
        "retry_count": state["retry_count"] + 1
    }

def grade_documents(state: GraphState):
    """Document grading node"""
    graded = grade_retrieval(
        state["user_query"],
        state["retrieved_docs"],
        model=state["intermida_model"],
        temperature=state["temperature"]
    )
    
    filtered = []
    missing = []
    for doc in graded:
        if doc.relevance_score >= state["threshold"]:
            filtered.append(doc)
        else:
            missing.extend(doc.missing_topics)
    
    return {
        "filtered_docs": sorted(filtered, key=lambda x: x.relevance_score, reverse=True),
        "missing_topics": missing
    }

def expand_query(state: GraphState):
    """Query expansion node"""
    new_query = query_extander(
        state["user_query"],
        state["missing_topics"],
        model=state["intermida_model"],
        temperature=state["temperature"]
    )[0]
    return {
        "query_message": new_query,
        "retry_count": state["retry_count"] + 1
    }

def generate_final_answer(state: GraphState):
    """Final answer generation node"""
    chat_session = state["chat_session"][-4:-1] if len(state["chat_session"]) >=4 else state["chat_session"]
    
    answer = generate_answer(
        question = state["user_query"],
        context_chunks = state["filtered_docs"] if state["filtered_docs"] else [],
        chat_session = chat_session,
        model = state["model"],
        temperature = state["temperature"]
    )
    return {"final_answer": answer}

def direct_answer(state: GraphState):
    """Direct answer node (when retrieval not needed)"""
    answer = generate_answer(
        state["user_query"],
        model=state["model"],
        temperature=0.6
    )
    return {"final_answer": answer}

# Build state machine
def setup_workflow():
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
    def should_retrieve(state: GraphState):
        return state["adaptive_decision"].require_extra_re
    
    def should_retry(state: GraphState):
        return (state["retry_count"] < state["max_retries"] and len(state["filtered_docs"]) < state["doc_number"])
    
    # Main process routing
    builder.add_conditional_edges(
        "detect_intention",
        should_retrieve,
        {
            True: "retrieve_docs",
            False: "direct_answer"
        }
    )
    
    # Retrieval-grading loop process
    builder.add_edge("retrieve_docs", "grade_docs")
    builder.add_conditional_edges(
        "grade_docs",
        should_retry,
        {
            True: "expand_query",
            False: "generate_answer"
        }
    )
    builder.add_edge("expand_query", "retrieve_docs")
    
    # End nodes
    builder.add_edge("direct_answer", END)
    builder.add_edge("generate_answer", END)
    
    return builder.compile()

calm_agent = setup_workflow()

# ===================================================== | API Service | =====================================================

@fastapi_app.post("/ask-calm-adrd-agent")
async def calm_adrd_agent_api(request: RequestBody):
    
    print(f"==== Initial Request from Openweb-UI portal: {request}")
    
    initial_state = {
        **request.model_dump(),
        "adaptive_decision": None,
        "retrieved_docs": [],
        "filtered_docs": [],
        "missing_topics": [],
        "retry_count": 0,
        "final_answer": None
    }
    
    for step in calm_agent.stream(initial_state, stream_mode="values"):
        print(f"Step: {step}")
    
    return step.get("final_answer")


@fastapi_app.get("/server-health-check")
def health_check_api():
    return {"status": "CaLM ADRD Agent Server is Healthy"}


def generate_graph_diagram():
    logger.info("Generating graph diagram")
    return calm_agent.get_graph().draw_mermaid_png(output_file_path="./public/calm_adrd_langgraph_diagram.png")


def test(payload: RequestBody):
    
    initial_state = GraphState(
        **payload.model_dump(),
        adaptive_decision=None,
        retrieved_docs=[],
        filtered_docs=[],
        missing_topics=[],
        retry_count=0,
    )
    
    res = calm_agent.invoke(initial_state)

    return res.get("final_answer")


if __name__ == "__main__":
    
    payload = RequestBody(
        user_query="my mom seems forgetting thinks, what should I do ?",
        threshold=0.65,
        doc_number=4,
        max_retries=1,
        model="phi4:latest",
        intermida_model="qwen2.5:latest",
        temperature=0.65,
        chat_session=[
            {
                "id": "c3db2446-651b-4603-b2a8-2ba558c97ff4",
                "role": "user",
                "content": "What are 5 creative things I could do with my kids' art? I don't want to throw them away",
                "timestamp": 1738345174
            },
            {
                "id": "dcc67334-532e-49c7-9876-996de89ed416",
                "role": "assistant",
                "content": "you are a very helpful ai assistant",
                "timestamp": 1738345287
            }
        ],
    )
    
    r = test(payload)
    
    print(r)