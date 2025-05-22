#!/usr/bin/env python3

from functools import lru_cache

from langchain_ollama import OllamaEmbeddings


@lru_cache(maxsize=1000)
def get_nomic_embedding() -> OllamaEmbeddings:
    """Get the Nomic embedding model.

    Returns:
        OllamaEmbeddings: The Nomic embedding model.

    """
    return OllamaEmbeddings(model="nomic-embed-text:latest")
