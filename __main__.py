import operator
from typing_extensions import TypedDict
from typing import List, Annotated, Union
from IPython.display import display, Image

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from utils.logger import logger
from src.checkpoints.retrieval_grading import grade_retrieval
from src.checkpoints.routering import get_routing_decision
from src.embedding.vector_store import VectorStore
from src.embedding.embedding_models import EmbeddingModels
from beans import GraphState, ReasonedDocument


def retrieve_documents(graph_state):
    """
    Retrieve documents is a function that retrieve documents from vectorestore based on user's intention
    """

    intention = graph_state.get('intention', 'research')
    embedding_model = graph_state.get('embedding_model', 'BAAI/bge-m3')
    number_of_retrieval = graph_state.get('number_of_retrieval', 5)
    vectorstore_path = './data/vector_database/research_kb'

    match intention:
        case 'research': vectorstore_path = './data/vector_database/research_kb'
        case 'peer': vectorstore_path = './data/vector_database/peer_kb'

    chroma_vectorstore = VectorStore().get_chroma_vectorstore(
        vectorstore_path=vectorstore_path,
        embedding=EmbeddingModels().get_bge_embedding(embedding_model)
    )

    documents = VectorStore().retrieve_docs(
        query=graph_state['question'],
        vectorstore=chroma_vectorstore,
        k=number_of_retrieval
    )

    return {
        'retrieved_documents': documents
    }

def graph_retrieval_grading(graph_state: GraphState) -> GraphState:
    """
    Graph retrieval grading is a function that takes in a graph state and returns a graph state
    """
    graded_documents = grade_retrieval(
        question=graph_state['question'],
        retrieved_docs=graph_state['retrieved_documents'],
        model="llama3.2",
    )

    # Filter out documents with relevance_score lower than 0.5 and re-rank relevant documents based on relevance_score descending

    filtered_docs = []

    for doc, grade in zip(graph_state['retrieved_documents'], graded_documents):
        if grade["relevance_score"] >= 0.5:
            reasoned_doc = ReasonedDocument(
                document=doc,
                relevance_score=grade["relevance_score"],
                reasoning=grade["reasoning"]
            )
            filtered_docs.append(reasoned_doc)

    # Re-rank the filtered documents based on relevance_score in descending order
    filtered_docs.sort(key=lambda x: x.relevance_score, reverse=True)

    return {
        'filtered_documents': filtered_docs
    }

def decide_retry_retrieval_v_prompt_modification(graph_state: GraphState):
    number_of_retrieval = graph_state.get('number_of_retrieval', 5)
    current_loop_step = graph_state.get('loop_step', 0)


    if len(graph_state['filtered_documents']) < number_of_retrieval and current_loop_step < number_of_retrieval:
        return {
            "retry_retrieval": True,
            "loop_step": current_loop_step + 1
        }


