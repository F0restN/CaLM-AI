
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import PromptTemplate

from classes.AdaptiveDecision import AdaptiveDecision
from utils.logger import logger
from utils.Models import _get_deepseek, _get_llm

ADAPTIVE_RAG_DECISION_PROMPT = """

You are expert in routing questions to the right knowledge base.

'research' contains documents from PubMed and PubMed Central and other professional journals for research-related and serious and official questions. Practical guidelines from NIH and Family caregiver alliance data are included in research data.
'peer_support'  contains data from social media like AgeCare forum and Reddit, they are user shared stories for peer-support related questions. User experience and story sharing data from AgingCare and Alzconnect is included in peer support knowledge base.

Given those information above, you will determine whether extra information from those two knowledge base will help model
answer user's question: {question}

Here is the conversation history between user and assistant, user could use implicit expression refers to words and content in context. So you should properly assess it and use the following context:
{latest_conversation_pair}

Answer format:
1. After your assessment, unless you're very sure that user's question: {question}, is not related to any medical or healthcare related questions about Alzheimer's disease and dementia and nothing related appeared in context, response 'False' for 'require_extra_re', otherwise 'True'.
2. If "require_extra_re" is True, determine which knowledge base is most relevant to user question, either 'research' or 'peer_support'.
3. If "require_extra_re" is False, response 'NA' for 'knowledge_base'.
"""


def adaptive_rag_decision(
    query: str,
    model: str = "qwen3:4b",
    temperature: float = 0.3,
    latest_conversation_pair: str = "",
) -> AdaptiveDecision:
    """Decide whether extra retrieval step is necessary for a given query.

    Args:
        query (str): The user's input query
        model (str, optional): The model name to use. Defaults to "qwen2.5-coder:7b"
        temperature (float, optional): The sampling temperature. Defaults to 0.1
        latest_conversation_pair (str, optional): The latest conversation pair between user and assistant. Defaults to ""

    Returns:
        AdaptiveDecision: A structured decision object containing require_extra_re and knowledge_base

    Raises:
        ValueError: If query is empty or temperature is invalid
        OutputParserException: If output parsing fails

    """
    logger.info(f"Adaptive decision | {query} | {latest_conversation_pair}")

    prompt = PromptTemplate(
        template=ADAPTIVE_RAG_DECISION_PROMPT,
        input_variables=["question", "latest_conversation_pair"],
    )


    # NOTE: Temparary use deepseek-chat due to ollama server issue.
    llm = _get_llm(model, temperature)
    # llm = _get_deepseek(model="deepseek-chat", temperature=temperature)

    structured_llm = prompt | llm.with_structured_output(schema=AdaptiveDecision, method="function_calling", include_raw=False)

    res = structured_llm.invoke({"question": query, "latest_conversation_pair": latest_conversation_pair})

    # Retry
    while not isinstance(res, AdaptiveDecision):
        logger.warning(f"Adaptive decision | {query} | Invalid response type: {type(res)} | Retrying with strict mode")

        # Retry with strict mode if the response is not of type AdaptiveDecision
        # This is to ensure that we get a valid structured output
        res = structured_llm.invoke({"question": query, "latest_conversation_pair": latest_conversation_pair})

        if isinstance(res, AdaptiveDecision):
            return res

    return res

    # try:
    #     res = structured_llm.invoke({"question": query, "latest_conversation_pair": latest_conversation_pair})
    #     logger.success(f"Adaptive decision | {res.require_extra_re} | for extra retrieval, with TKB | {res.knowledge_base}")
    # except OutputParserException as ope_err:
    #     logger.error(f"Output parser exception: {ope_err} for user query: {query}, retry with strict mode")
    #     return structured_llm.invoke({"question": query, "latest_conversation_pair": latest_conversation_pair}, strict=True)
    # except Exception as e:
    #     logger.error(f"\nError in adaptive decision, for user query: {query} \n\n-------- \nError: {e}")
    #     raise
    # else:
    #     return res

if __name__ == "__main__":
    res = adaptive_rag_decision("how's the whether today ?", model="qwen2.5-coder:7b")
    print(type(res))
    print(res.model_dump_json(indent=2))
