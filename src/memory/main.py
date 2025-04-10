
from datetime import datetime
from classes.Memory import MemoryItem, Memory
from uuid import uuid4

def summarize_from_chat(chat: str) -> MemoryItem:
    """
    Summarize the chat into a memory item.
    """
    return MemoryItem(content=chat, level="STM", type="chat", source="user", timestamp=datetime.now())


# -- test --

chat = "Hello, how are you?"

memory_item = summarize_from_chat(chat)

print(memory_item.metadata)

# print(uuid4().int)