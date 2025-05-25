#!/usr/bin/env python3

import os

from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector


class VectorStore:
    """A vector store wrapper for PGVector with embedding capabilities.

    This class encapsulates the PGVector connection and provides a clean interface
    for vector operations using the Facade and Factory Method patterns.
    """

    def __init__(self, connection: str, embedding_model: Embeddings, collection_name: str):
        """Initialize the vector store.

        Args:
            connection: The connection string for the vector store.
            embedding_model: The embedding model to use.
            collection_name: The name of the collection to use.

        Raises:
            ValueError: If collection_name is empty or None.

        """
        if not collection_name:
            raise ValueError("Collection name cannot be empty or None")

        self._connection = connection
        self._embedding_model = embedding_model
        self._collection_name = collection_name
        self._kb: PGVector | None = None

    @classmethod
    def from_env(cls, embedding_model: Embeddings, collection_name: str) -> "VectorStore":
        """Create a VectorStore instance using the PGVECTOR_CONN environment variable.
        
        Factory method that creates an instance using environment configuration.

        Args:
            embedding_model: The embedding model to use.
            collection_name: The name of the collection to use.

        Returns:
            VectorStore: A new instance configured with environment connection.

        Raises:
            ValueError: If PGVECTOR_CONN environment variable is not set.

        """
        connection = os.environ.get("PGVECTOR_CONN")
        if not connection:
            raise ValueError("PGVECTOR_CONN environment variable is not set")

        return cls(connection, embedding_model, collection_name)

    @property
    def connection(self) -> PGVector:
        """Get the PGVector connection, creating it lazily if necessary.
        
        Uses lazy initialization pattern to defer connection creation until needed.
        
        Returns:
            PGVector: The vector store connection.

        """
        if self._kb is None:
            self._kb = PGVector(
                embeddings=self._embedding_model,
                collection_name=self._collection_name,
                connection=self._connection,
                use_jsonb=True,
            )
        return self._kb

    def similarity_search(self, query: str, k: int = 10) -> list[Document]:
        """Search for similar documents in the vector store.

        Args:
            query: The query to search for.
            k: The number of results to return.

        Returns:
            list[Document]: The list of documents found.

        """
        return self.connection.similarity_search(query=query, k=k)

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vector store.

        Args:
            documents: List of documents to add.
            
        Returns:
            list[str]: List of document IDs that were added.

        """
        return self.connection.add_documents(documents)

    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> list[str]:
        """Add texts to the vector store.

        Args:
            texts: List of texts to add.
            metadatas: Optional list of metadata dictionaries for each text.
            
        Returns:
            list[str]: List of document IDs that were added.

        """
        return self.connection.add_texts(texts, metadatas=metadatas)

    def delete_collection(self) -> None:
        """Delete the entire collection from the vector store."""
        self.connection.delete_collection()

    def get_collection_name(self) -> str:
        """Get the name of the current collection.
        
        Returns:
            str: The collection name.

        """
        return self._collection_name


# Convenience function for backward compatibility
def create_vector_store(
    connection: str,
    embedding_model: Embeddings,
    collection_name: str,
) -> VectorStore:
    """Create a VectorStore instance.
    
    Convenience function for backward compatibility with the original get_connection function.
    
    Args:
        connection: The connection string for the vector store.
        embedding_model: The embedding model to use.
        collection_name: The name of the collection to use.
        
    Returns:
        VectorStore: A configured vector store instance.

    """
    return VectorStore(connection, embedding_model, collection_name)
