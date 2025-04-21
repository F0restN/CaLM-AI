from langchain_ollama import OllamaEmbeddings
from functools import lru_cache


@lru_cache(maxsize=1000)
def get_nomic_embedding():
    return OllamaEmbeddings(model="nomic-embed-text:latest")

