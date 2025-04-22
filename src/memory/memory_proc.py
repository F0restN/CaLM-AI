
import os

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from classes.Memory import BaseMemory, BaseEpisodicMemory
from utils.llm_manager import _get_deepseek
from utils.PROMPT import MEMORY_SUMMARIZATION_PROMPT, MEMORY_DETERMINATION_PROMPT_TEMPLATE, EPISODIC_MEMORY_PROMPT_TEMPLATE

PGVECTOR_CONN = os.environ.get("PGVECTOR_CONN")

def format_conversation(chat_history: list[object]) -> str:
    """Format converation into clear and concise conversation list."""
    conversation = [f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history]
    return "\n".join(conversation)


def memory_extract_decision(query: str) -> AIMessage:
    """Decide whether the user's query is about the life situation that is factual and not gonna change for a while."""
    llm = _get_deepseek("deepseek-chat", temperature=0.0)

    prompt = PromptTemplate.from_template(MEMORY_DETERMINATION_PROMPT_TEMPLATE)
    chain = prompt | llm
    
    return chain.invoke({"query": query})
    


def summarize_lstm_from_query(query: str) -> BaseMemory:
    """Summarize the user's query into a memory item."""
    llm = _get_deepseek("deepseek-chat", temperature=0.0)

    # TODO: must well define the prompt to make it work, currently, it just extract some useless information
    
    prompt = PromptTemplate.from_template(MEMORY_SUMMARIZATION_PROMPT)
    structured_llm = prompt | llm.with_structured_output(schema=BaseMemory, method="function_calling", include_raw=False)
    
    res: BaseMemory = structured_llm.invoke({"query": query})
    
    return res


def summarize_episodicM_from_conversation(conversation: str) -> BaseEpisodicMemory:
    """Summarize the conversation into a episodic memory."""
    llm = _get_deepseek("deepseek-chat", temperature=0.0)

    # TODO: must well define the prompt to make it work, currently, it just extract some useless information

    prompt = PromptTemplate.from_template(EPISODIC_MEMORY_PROMPT_TEMPLATE)
    structured_llm = prompt | llm.with_structured_output(schema=BaseEpisodicMemory, method="function_calling", include_raw=False)
    return structured_llm.invoke({"conversation": conversation})


if __name__ == "__main__":

    user_query =  "Can you recommend activities that are suitable for someonewith dementia to engage in and enjoy?"
    
    res = summarize_lstm_from_query(user_query)
    
    print(res)
