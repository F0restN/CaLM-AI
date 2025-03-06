from functools import lru_cache
from langchain_ollama import ChatOllama

@lru_cache(maxsize=1000)
def _get_llm(model: str, temperature: float):
    return ChatOllama(model=model, temperature=temperature)

# TODO: add a function to clear the cache
