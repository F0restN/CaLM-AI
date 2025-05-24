#!/usr/bin/env python3

import os

from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector

from classes.Memory import MemoryItem
from embedding.embedding_models import get_nomic_embedding

PGVECTOR_CONN = os.environ.get("PGVECTOR_CONN")


def get_connection(connection: str, embedding_model: Embeddings, collection_name: str) -> PGVector:
    """Get a connection to the vector store.

    Args:
        connection: The connection string for the vector store.
        embedding_model: The embedding model to use.
        collection_name: The name of the collection to use. By default, it is "lstm_memory".

    Returns:
        PGVector: A connection to the vector store.

    """
    if not collection_name:
        msg = "Collection Name can not be empty"
        raise ValueError(msg)

    return PGVector(
        embeddings=embedding_model,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )


def similarity_search(query: str, kb: PGVector = None, k: int = 10) -> list[Document]:
    """Search for similar documents in the vector store.

    Args:
        query: The query to search for.
        kb: The vector store to search in.
        k: The number of results to return.

    Returns:
        list[Document]: The list of documents found.

    """
    if kb is None:
        msg = "KB can not be None"
        raise ValueError(msg)

    return kb.similarity_search(query=query, k=k)
