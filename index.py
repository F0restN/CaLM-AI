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


class GraphState(TypedDict):
    """
    Graph state is a dictinary that contains information we want to propagte to, and modify in, each graph node
    """
    # User input
    question: str  # User question
    conversation: List[Union[HumanMessage, AIMessage, SystemMessage]]

    # Models
    llm_model: str  # Name of the LLM model
    embedding_model: str  # Name of the embedding model

    # LLM generated variables
    intention: str  # Intention of the user
    retrieved_documents: List[Document]
    filtered_documents: List[Document]
    generated_answers: str

    # Parameters
    number_of_answers: int  # Number of answers to generated
    loop_step: Annotated[int, operator.add]
    max_retries: int  # Maximum number of retries for the LLM
    generation: str  # LLM generation answer


def graph_intention_detection(graph_state: GraphState) -> GraphState:
    """
    Graph intention detection is a function that takes in a graph state and returns a graph state
    """
    intention = get_routing_decision(
        messages=graph_state['conversation'],
        model=graph_state['llm_model'],
        temperature=graph_state['temperature']
    )

    logger.info(f"Detected user intention: {intention['KnowledgeBaseType']}")

    return {
        'intention': intention['KnowledgeBaseType']
    }


def graph_retrieve_documents(graph_state: GraphState) -> GraphState:
    """
    Retrieve documents is a function that takes in a graph state and returns a graph state
    """

    documents = VectorStore().retrieve_docs(
        question=graph_state['question'],
        vectorstore=VectorStore().get_chroma_vectorstore(
            vectorstore_path='./data/vector_database/research_kb' if graph_state[
                'intention'] == 'research' else './data/vector_database/peer_support_kb',
            embedding=graph_state['embedding_model']
        ),
        temperature=graph_state['temperature']
    )

    return {
        'retrieved_documents': documents
    }


def graph_retrieval_grading(graph_state: GraphState) -> GraphState:
    """
    Graph retrieval grading is a function that takes in a graph state and returns a graph state
    """

    filtered_documents = grade_retrieval(
        question=graph_state['question'],
        retrieved_docs=graph_state['documents'],
        model=graph_state['embedding_model'],
        temperature=graph_state['temperature']
    )

    return {
        'filtered_documents': filtered_documents
    }


workflow = StateGraph(GraphState)

workflow.add_node(graph_intention_detection)
workflow.add_node(graph_retrieve_documents)
workflow.add_node(graph_retrieval_grading)

workflow.set_conditional_entry_point(
    graph_intention_detection,
    {
        'research': 'graph_retrieve_documents',
        'peer_support': 'graph_retrieve_documents'
    }
)

workflow.add_conditional_edges('graph_retrieve_documents', graph_retrieval_grading, {
    "filtered_documents": END
})

# workflow.add_edge(retrieve_documents, graph_retrieval_grading)
# workflow.add_edge(graph_retrieval_grading, )

graph = workflow.compile()
display(Image(graph.get_graph().draw_mermaid_png()))
