import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from fastapi import FastAPI
from dotenv import load_dotenv

from classes.AdaptiveDecision import AdaptiveDecision
from classes.RequestBody import RequestBody
from answer_generation import generate_answer
from checkpoints.retrieval_grading import grade_retrieval_batch
from checkpoints.query_extander import query_extander
from checkpoints.adaptive_decision import adaptive_rag_decision
from embedding.vector_store import get_connection
from embedding.embedding_models import get_nomic_embedding
from utils.logger import logger

load_dotenv()

fastapi_app = FastAPI()

ls_tracing = {"project_name": os.getenv("LANGSMITH_PROJECT"), "metadata": {"session_id": None}}

tool_call_flag = True

# Define state machine state structure
class GraphState(TypedDict):
    # Static states
    user_query: str
    chat_session: list
    model: str
    intermediate_model: str
    threshold: float
    max_retries: int
    doc_number: int
    temperature: float
    
    # Running states
    query_message: str | None = None
    
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
def detect_intention(state):
    """User intention detection node"""
    
    decision = adaptive_rag_decision(
        state["user_query"],
        model=state["intermediate_model"],
        temperature=state["temperature"],
        langsmith_extra=ls_tracing
    )
    
    return {"adaptive_decision": decision}


def retrieve_documents(state: GraphState):
    """Document retrieval node"""
    
    if state["adaptive_decision"].knowledge_base == "peer_support":
        cls_kb = p_kb
    else:
        cls_kb = r_kb
    
    docs = cls_kb.similarity_search(state["query_message"], k=state["doc_number"])
    
    logger.success(f"Similarity search retrieved | {len(docs)} | documents")
    
    return {
        "retrieved_docs": docs, 
        "retry_count": state["retry_count"] + 1
    }


async def grade_documents(state: GraphState):
    """Document grading node"""
    graded = await grade_retrieval_batch(
        state["query_message"],
        state["retrieved_docs"],
        model=state["intermediate_model"],
        temperature=state["temperature"],
        langsmith_extra=ls_tracing
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
    
    logger.success(f"Filtered in {len(filtered)} documents, out of {len(graded)} graded documents")
    
    return {
        "filtered_docs": sorted(filtered, key=lambda x: x.relevance_score, reverse=True),
        "missing_topics": missing
    }


def expand_query(state: GraphState):
    """Query expansion node"""
    new_query = query_extander(
        state["query_message"],
        state["missing_topics"],
        model=state["intermediate_model"],
        temperature=state["temperature"],
        langsmith_extra=ls_tracing
    )
    
    return {
        "query_message": new_query,
    }


def generate_final_answer(state: GraphState):
    """Final answer generation node"""
    chat_session = state["chat_session"][-4:-1] if len(state["chat_session"]) >=4 else state["chat_session"]
    
    answer = generate_answer(
        question = state["user_query"],
        context_chunks = state["filtered_docs"] if state["filtered_docs"] else [],
        chat_session = chat_session,
        model = state["model"],
        temperature = state["temperature"],
        tool_call_flag = tool_call_flag,
        langsmith_extra=ls_tracing
    )
    return {"final_answer": answer}


def direct_answer(state: GraphState):
    """Direct answer node (when retrieval not needed)"""
    answer = generate_answer(
        state["user_query"],
        model=state["model"],
        temperature=0.6,
        langsmith_extra=ls_tracing
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
    
    logger.info(f"Processing request: {request.user_query}, for user: {request.body_config.current_session.user_id}")
    
    logger.info(f"==== Initial Request from Openweb-UI portal: {request.model_dump_json(indent=4)}")
    
    ls_tracing["metadata"]["session_id"] = request.body_config.current_session.chat_id
    
    ls_tracing["metadata"]["user_id"] = request.body_config.current_session.user_id
    
    initial_state = {
        **request.model_dump(),
        "query_message": request.user_query,
        "adaptive_decision": None,
        "retrieved_docs": [],
        "filtered_docs": [],
        "missing_topics": [],
        "retry_count": 0,
        "final_answer": None 
    }
    
    try:
        async for step in calm_agent.astream(initial_state, stream_mode="values"):
            continue
        
        return step.get("final_answer")
    except Exception as e:
        logger.error(f"Error in calm_agent stream: {str(e)}")
        return {
            "answer": "Sorry, an error occurred while processing your request. Please try again later.", 
            "sources": [],
            "follow_up_questions": []
        }


@fastapi_app.get("/server-health-check")
def health_check_api():
    return {"status": "CaLM ADRD Agent Server is Healthy"}


def generate_graph_diagram():
    logger.info("Generating graph diagram")
    return calm_agent.get_graph().draw_mermaid_png(output_file_path="./public/calm_adrd_langgraph_diagram.png")


def test(payload: RequestBody):
    
    initial_state = GraphState(
        **payload.model_dump(),
        query_message=payload.user_query,
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
        threshold=0.60,
        doc_number=4,
        max_retries=1,
        model="llama3.3:latest",
        intermediate_model="qwen2.5-coder:7b",
        temperature=0.3,
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
    
    # print(r.model_dump_json(indent=2))