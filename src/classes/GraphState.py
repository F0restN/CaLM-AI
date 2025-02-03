import operator

from typing_extensions import TypedDict
from typing import List, Annotated, Union

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class GraphState(TypedDict) :
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node
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

    # Flag
    retry_retrieval: bool

    # Parameters
    loop_step: Annotated[int, operator.add]
    max_retries: int  # Maximum number of retries for the LLM
    generation: str  # LLM generation answer
    number_of_answers: int  # Number of answers to generated
    number_of_retrieval: int # Number of retrieval documents
    max_retries_retrieval: int  # Maximum number of retries for the LLM
