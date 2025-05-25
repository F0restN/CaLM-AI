from typing import Optional, Dict

from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException

from utils.logger import logger
from utils.Models import _get_llm
from classes.AdaptiveDecision import AdaptiveDecision




ADAPTIVE_RAG_DECISION_PROMPT = """

You are expert in routing questions to the right knowledge base.

'research' contains documents from PubMed and PubMed Central and other professional journals for research-related and serious and official questions. Practical guidlines from NIH and Family care giver alliance data are included in research data.
'peer_support'  contains data from social media like AgeCare forum and Reddit, they are user shared stories for peer-support related questions. User experience and story sharing data from AgingCare and Alzconnect is included in peer support knowledge base.

Given those information above, you will determine whether extra information from those two knowledge base will help model 
answer question {question}

If user's question: {question} is not related to research or peer_support, response 'False' for 'require_extra_re', otherwise 'True' and also determine which knowledge base is most relevant to user question: {question}, either 'research' or 'peer_support' or `NA` if require_extra_re is False
 
"""


def adaptive_rag_decision(
    query: str,
    model: str = "qwen2.5-coder:7b",
    temperature: float = 0.1,
) -> AdaptiveDecision:
    """
    Decide whether extra retrieval step is necessary for a given query.
    
    Args:
        query (str): The user's input query
        model (str, optional): The model name to use. Defaults to "qwen2.5-coder:7b"
        temperature (float, optional): The sampling temperature. Defaults to 0.1
        
    Returns:
        AdaptiveDecision: A structured decision object containing require_extra_re and knowledge_base
        
    Raises:
        ValueError: If query is empty or temperature is invalid
        OutputParserException: If output parsing fails
    """
    
    prompt = PromptTemplate(
        template=ADAPTIVE_RAG_DECISION_PROMPT,
        input_variables=["question"],
    )    
    
    llm = _get_llm(model, temperature)
    
    structured_llm = prompt | llm.with_structured_output(schema=AdaptiveDecision, method="function_calling", include_raw=False)
    
    try:
        res = structured_llm.invoke({"question": query})
        logger.success(f"Adaptive decision | {res.require_extra_re} | for extra retrieval, with TKB | {res.knowledge_base}")
        return res
    except OutputParserException as ope_err:
        logger.error(f"Output parser exception: {ope_err} for user query: {query}, retry with strict mode")
        return structured_llm.invoke({"question": query}, strict=True)
    except Exception as e:
        logger.error(f"\nError in adaptive decision, for user query: {query} \n\n-------- \nError: {e}")
        raise e
    
if __name__ == "__main__":
    res = adaptive_rag_decision("how's the whether today ?", model="qwen2.5-coder:7b")
    print(type(res))
    print(res.model_dump_json(indent=2))
