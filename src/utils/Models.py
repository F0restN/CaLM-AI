import os
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai.chat_models.base import BaseChatOpenAI


@lru_cache(maxsize=1000)
def _get_deepseek(model: str, temperature: float) -> BaseChatOpenAI:
    os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API")
    return ChatDeepSeek(model=model, temperature=temperature)

@lru_cache(maxsize=1000)
def _get_llm(model: str, temperature: float) -> BaseChatModel:
    """Get the LLM model based on the model name and temperature."""
    if model.startswith("deepseek"):
        return _get_deepseek(model, temperature)

    return ChatOllama(model=model, temperature=temperature)

@lru_cache(maxsize=1000)
def get_nomic_embedding() -> OllamaEmbeddings:
    """Get the Nomic embedding model.

    Returns:
        OllamaEmbeddings: The Nomic embedding model.

    """
    return OllamaEmbeddings(model="nomic-embed-text:latest")


# TODO: add a function to clear the cache

if __name__ == "__main__":
    ds_llm = _get_deepseek(model = "deepseek-reasoner", temperature=0.6)

    res:AIMessage = ds_llm.invoke("who are you")

    print(res)
