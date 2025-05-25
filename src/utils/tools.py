import os

from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector

from utils.Models import get_nomic_embedding


def get_connection(connection: str, embedding_model: Embeddings, collection_name: str) -> PGVector:
    """Get the PGVector connection.

    Args:
        connection: The connection string for the vector store.
        embedding_model: The embedding model to use.
        collection_name: The name of the collection to use.

    Returns:
        PGVector: The vector store connection.

    """
    if not connection:
        connection = os.environ.get("PGVECTOR_CONN")
    if not embedding_model:
        embedding_model = get_nomic_embedding()
    return PGVector(
        embeddings=embedding_model,
        connection=connection,
        collection_name=collection_name,
    )
