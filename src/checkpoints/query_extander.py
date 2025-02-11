from typing import List

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
) -> str:
    """
    Extand the query to include missing topics.

    Args:
        - original_query: The original query that needs to be extanded.
        - missing_topics: A list of topics that the query should cover.
        - model: The model to use for the generation.
        - temperature: The temperature to use for the generation.
    """
    
    prompt = PromptTemplate(
        template=QUERY_EXTAND_PROMPT,
        input_variables=["original_query", "missing_topics"]
    )

    llm = ChatOllama(model=model, temperature=temperature, max_tokens=100)

    chain = prompt | llm 

    res = chain.invoke({"original_query":original_query, "missing_topics":missing_topics})

    return res

query_extander_tool = StructuredTool.from_function(
    func = query_extander,
    handle_tool_error=True,
)

if __name__ == "__main__":
    original_query = "What is the capital of France?"
    missing_topics = ["History of France", "Population of France"]

    res = query_extander_tool.invoke({
        "original_query": original_query,
        "missing_topics": missing_topics
    })

    print(res.content)


