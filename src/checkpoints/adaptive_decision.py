from loguru import logger
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langsmith import traceable
from classes.AdaptiveDecision import AdaptiveDecision

ADAPTIVE_RAG_DECISION_PROMPT = """

You are expert in routing questions to the right knowledge base.

'research' contains documents from PubMed and PubMed Central and other professional journals for research-related and serious and official questions. Practical guidlines from NIH and Family care giver alliance data are included in research data.
'peer_support'  contains data from social media like AgeCare forum and Reddit, they are user shared stories for peer-support related questions. User experience and story sharing data from AgingCare and Alzconnect is included in peer support knowledge base.

Given those information above, you will determine whether extra information from those two knowledge base will help model 
answer question {question}

If user's question: {question} is not related to research or peer_support, response 'False' for 'require_extra_re', otherwise 'True' and also determine which knowledge base is most relevant to user question: {question}, either 'research' or 'peer_support'

Response format instruction:
{format_instructions}
 
"""


def adaptive_rag_decision(
    query: str, 
    model: str = "phi4:latest", 
    temperature: float = 0.1,
    langsmith_extra: dict = {}
) -> AdaptiveDecision:
    """
    Decide whether extra retrieval step is necessary
    """
    
    json_parser = JsonOutputParser(pydantic_object=AdaptiveDecision)
    
    prompt = PromptTemplate(
        template=ADAPTIVE_RAG_DECISION_PROMPT,
        input_variables=["question"],
        partial_variables={"format_instructions": json_parser.get_format_instructions}        
    )    
    
    print("====== langsmith log")
    print(langsmith_extra)
    
    llm = ChatOllama(model=model, temperature=temperature, format="json")
    
    chain = prompt | llm | json_parser
    
    try:
        resp = chain.invoke({"question": query})
        
        logger.info(f"Adaptive decision: {resp}")
            
        return AdaptiveDecision(**resp)
    except Exception as e:
        raise e 
    
    
if __name__ == "__main__":
    res = adaptive_rag_decision("how's the whether today ?")
    print(res)
