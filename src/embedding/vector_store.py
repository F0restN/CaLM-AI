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


def add_to_memory(memory: MemoryItem, kb: PGVector = None) -> str:
    """Add a memory to the vector store.

    Args:
        memory: MemoryItem
        kb: PGVector, by default using the lstm_memory collection

    Returns:
        str: result of the add_documents method

    """
    if kb is None:
        kb = get_connection(
            connection=PGVECTOR_CONN,
            embedding_model=get_nomic_embedding(),
            collection_name="lstm_memory",
        )

    doc = Document(
        page_content=memory.content,
        metadata=memory.metadata,
    )

    return kb.add_documents([doc])


def recall_memory(
    query: str,
    user_id: int,
    score: float | None = 0.2,
    kb: PGVector | None = None,
) -> list[Document]:
    """Recall the memory from the vector store. Return document above cut off score if score is provided. Otherwise return all documents using default threshold.

    Args:
        kb: PGVector, by default using the lstm_memory collection
        query: The query to search for.
        score: The score to search for.
        user_id: The user id to search for.

    Returns:
        list[Document]: The list of documents found.

    """
    if kb is None:
        kb = get_connection(
            connection=PGVECTOR_CONN,
            embedding_model=get_nomic_embedding(),
            collection_name="lstm_memory",
        )
    custom_filter = {"user_id": user_id}
    res_unfiltered: list[tuple[Document, float]] = kb.similarity_search_with_score(
        query=query,
        filter=custom_filter,
    )
    if score is not None:
        return [doc for doc, relevancy in res_unfiltered if relevancy >= score]

    return [doc for doc, _ in res_unfiltered]
