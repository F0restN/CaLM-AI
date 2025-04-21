
import os
from classes.Memory import MemoryItem
from langchain_core.prompts import PromptTemplate
from utils.llm_manager import _get_deepseek
from utils.PROMPT import MEMORY_SUMMARIZATION_PROMPT

PGVECTOR_CONN = os.environ.get("PGVECTOR_CONN")

def format_conversation(chat_history: list[object]) -> str:
    """
    Format converation into clear and concise conversation list.
    """
    
    conversation = []
    
    for message in chat_history:
        conversation.append(f"{message['role'].upper()}: {message['content']}")
    
    return "\n".join(conversation)


def summarize_from_chat(chat: str) -> MemoryItem:
    """
    Summarize the chat into a memory item.
    """
    
    llm = _get_deepseek("deepseek-chat", temperature=0.0)
    
    # TODO: must well define the prompt to make it work, currently, it just extract some useless information
    prompt = PromptTemplate.from_template(MEMORY_SUMMARIZATION_PROMPT)
    
    structured_llm = prompt | llm.with_structured_output(schema=MemoryItem, method="function_calling", include_raw=False)
    
    return structured_llm.invoke({"conversation": chat})

    
if __name__ == "__main__":

    chat = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you!"},
        {"role": "user", "content": "Remember my name is Drake"},
        {"role": "assistant", "content": "I will remember that."}
    ]


    # print(format_conversation(chat))
    print(summarize_from_chat(chat))