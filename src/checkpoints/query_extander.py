from typing import List

from loguru import logger
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama

QUERY_EXTAND_PROMPT = """
Exatnd the query below to get more information about the topic:

User Query: {original_query}

Topics that document should cover: {missing_topics}

return a string of the extended query only, do not include other supproting information.
"""



def query_extander(
    original_query: str,
    missing_topics: List[str],
    model: str = "llama3.2",
    temperature: float = 0,
    langsmith_extra: dict = {}
) -> str:
    """
    Extends the original query by incorporating missing topics to create a more comprehensive search query.
    
    Args:
        original_query (str): The initial user query
        missing_topics (List[str]): List of topics that should be covered but are missing from retrieved documents
        model (str, optional): Name of the Ollama model to use. Defaults to "llama3.2"
        temperature (float, optional): Temperature for model generation. Defaults to 0
        langsmith_extra (dict, optional): Extra parameters for langsmith tracing. Defaults to {}
        
    Returns:
        tuple: A tuple containing:
            - str: The extended query string
            - Response: The raw response object from the LLM
            
    Example:
        >>> query_extander(
        ...     "What is Alzheimer's?",
        ...     ["early symptoms", "treatment options"],
        ...     model="deepseek-r1:14b"
        ... )
        ("What is Alzheimer's disease, its early symptoms and available treatment options?", <Response>)
    """
    
    prompt = PromptTemplate(
        template=QUERY_EXTAND_PROMPT,
        input_variables=["original_query", "missing_topics"]
    )

    llm = ChatOllama(model=model, temperature=temperature, max_tokens=200)

    chain = prompt | llm 

    res = chain.invoke({"original_query":original_query, "missing_topics":missing_topics})
    
    logger.info(f"Query extander running using topics: {missing_topics}")
    
    return res.content, res

query_extander_tool = StructuredTool.from_function(
    func = query_extander,
    handle_tool_error=True,
    response_format="content_and_artifact" 
)

if __name__ == "__main__":
    original_query = "What is the capital of France?"
    missing_topics = ["History of France", "Population of France"]

    # res = query_extander_tool.invoke({
    #     "original_query": original_query,
    #     "missing_topics": missing_topics
    # })
    
    res = query_extander(
        original_query, missing_topics, model="qwen2.5"
    )

    print(res)


