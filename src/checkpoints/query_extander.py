from typing import List

from utils.logger import logger
from utils.Models import _get_llm

from langchain_core.prompts import PromptTemplate

QUERY_EXTAND_PROMPT = """
Exatnd the query below to get more information about the topic:

User Query: {original_query}

Topics that document should cover: {missing_topics}

return a string of the extended query only, do not assume other information.
"""

query_json_schema = {
    "title": "QueryExtander",
    "description": "The extended query string",
    "properties": { 
        "query": {
            "type": "string",
            "description": "The extended query string"
        }
    },
    "required": ["query"]
}
    

def query_extander(
    original_query: str,
    missing_topics: List[str],
    model: str = "llama3.2",
    temperature: float = 0,
) -> str:
    """
    Extends query by incorporating missing topics for comprehensive search.
    
    Args:
        original_query (str): User's initial query
        missing_topics (List[str]): Topics missing from retrieved documents
        model (str, optional): Model name. Defaults to "llama3.2"
        temperature (float, optional): Generation temperature. Defaults to 0
        
    Returns:
        str: The extended query string
    """
    
    prompt = PromptTemplate(
        template=QUERY_EXTAND_PROMPT,
        input_variables=["original_query", "missing_topics"]
    )

    llm = _get_llm(model, temperature)
    
    structured_llm = prompt | llm.with_structured_output(schema=query_json_schema, method="function_calling", include_raw=False)

    try:
        res = structured_llm.invoke({"original_query":original_query, "missing_topics":missing_topics})
        logger.success(f"Query expanded to --> {res['query']}")
        return res['query']
    except Exception:
        logger.error(f"Error in query extander, for user query: {original_query}, retry with strict mode")
        return structured_llm.invoke({"original_query":original_query, "missing_topics":missing_topics}, strict=True)['query']

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


