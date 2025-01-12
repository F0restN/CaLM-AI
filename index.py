import operator
from typing_extensions import TypedDict
from typing import List, Annotated
from src.checkpoints.retrieval_grading import grade_retrieval
class GraphState(TypedDict):
    """
    Graph state is a dictinary that contains information we want to propagte to, and modify in, each graph node
    """
    question: str # User question
    generation: str # LLM generation answer
    web_search: str # Binary decision to run web search
    max_retries: int # Maximum number of retries for the LLM
    answers: int # Number of answers to generated
    loop_step: Annotated[int, operator.add]
    documents: List[str] # List of documents to be used for the LLM
    
def graph_retrieval_grading(graph_state: GraphState) -> GraphState:
    """
    Graph retrieval grading is a function that takes in a graph state and returns a graph state
    """
    grade_retrieval(
        question=graph_state['question'],
        retrieved_docs=graph_state['documents'],
        model=graph_state['model'],
        temperature=graph_state['temperature']
    )
    
    