#!/usr/bin/env python3

import os

from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector
from pydantic import PrivateAttr

from utils.Models import get_nomic_embedding
from utils.tools import get_connection


class VectorStore:
    """A vector store wrapper for PGVector with embedding capabilities.

    This class encapsulates the PGVector connection and provides a clean interface
    for vector operations using the Facade and Factory Method patterns.
    """

    def __init__(self, collection_name: str, connection: str | None = None, embedding_model: Embeddings | None = None, _kb: PGVector | None = None) -> None:
        """Initialize the vector store.

        Args:
            collection_name: The name of the collection to use.
            connection: [Optional] The connection string for the vector store.
            embedding_model: [Optional] The embedding model to use.
            _kb: [Optional] The PGVector connection to use.

        Raises:
            ValueError: If collection_name is empty or None.

        """
        if not collection_name:
            raise ValueError("Collection name cannot be empty or None")

        self._connection = connection or os.environ.get("PGVECTOR_CONN")
        self._embedding_model = embedding_model or get_nomic_embedding()
        self._collection_name = collection_name
        self._kb: PGVector = get_connection(self._connection, self._embedding_model, self._collection_name)

    def similarity_search(self, query: str, k: int = 10) -> list[Document]:
        """Search for similar documents in the vector store.

        Args:
            query: The query to search for.
            k: The number of results to return.

        Returns:
            list[Document]: The list of documents found.

        """
        return self._kb.similarity_search(query=query, k=k)

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vector store.

        Args:
            documents: List of documents to add.

        Returns:
            list[str]: List of document IDs that were added.

        """
        return self._kb.add_documents(documents)


if __name__ == "__main__":
    kb = VectorStore(collection_name="peer_support_kb")
    print(kb.similarity_search("What is the capital of France?"))
