import os
import shutil
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.logger import logger


class VectorStore:
    def __init__(self):
        pass

    def build_chroma_vectorstore(
        self,
        docs: List[Document],
        embedding_model,
        # collection_name: str = "calm_kb",
        vectorstore_path: str = None,
        force_rebuild: bool = False
    ) -> Chroma:
        """
        Build a Chroma vector store from a list of documents

        Args:
            documents: List of documents to be vectorized
            vectorstore_path: Path to the vector store directory
            embedding_model: Embedding model to use
            force_rebuild: Whether to rebuild the vector store if it already exists

        Returns:
            Chroma: Built vector store
        """
        if os.path.isdir(vectorstore_path) and force_rebuild:
            shutil.rmtree(vectorstore_path)
            logger.info(
                f"Vector store {vectorstore_path} already exists and force_rebuild is True. Rebuilding...")

        logger.info(f"Building vector store with {len(docs)} documents")

        try:
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embedding_model,
                # collection_name=collection_name,
                persist_directory=vectorstore_path,
            )
            logger.info(
                f"Vector store built successfully at {vectorstore_path}")
            return vectorstore
        except Exception as e:
            logger.error(f"Failed to build vector store: {str(e)}")
            raise

    def get_chroma_vectorstore(vectorstore_path: str = None, embedding: HuggingFaceEmbeddings = None) -> Chroma:
        """
        Load ChromaDB vector store from persistent directory

        Args:
            vectorstore_path: Path to the vector store directory. If None, uses default path
            embedding: Embedding function to use. If None, uses default BGE embedding

        Returns:
            Chroma: Loaded vector store

        Raises:
            RuntimeError: If ChromaDB cannot be loaded
        """
        if vectorstore_path is None or vectorstore_path == "":
            logger.error("Vector store path is required")
            raise ValueError("Vector store path is required")
        elif not os.path.isdir(vectorstore_path):
            logger.error(
                f"Existing chroma db does not exist on path: {vectorstore_path}")
            raise ValueError(
                f"Existing chroma db does not exist on path: {vectorstore_path}"
            )

        if embedding is None:
            logger.error("Embedding model is required")
            raise ValueError("Embedding model is required")

        try:
            db = Chroma(embedding_function=embedding,
                        persist_directory=vectorstore_path, )
            return db
        except Exception as e:
            logger.error(f"ChromaDB load failed: {str(e)}")
            raise RuntimeError(f"ChromaDB error: {str(e)}")

    def retrieve_docs(query: str, vectorstore: Chroma, k: int = 10) -> List[Document]:
        """
        Retrieve documents from vector store
        """
        try:
            docs = vectorstore.similarity_search(query=query, k=k)
            logger.warning(f"Retrieved {len(docs)} documents")
            return docs
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            raise


if __name__ == "__main__":
    from .embedding_models import EmbeddingModels

    vectorestore_path = "./data/vector_database/peer_kb"
    embedding_model = EmbeddingModels().get_bge_embedding('BAAI/bge-m3')

    chroma_kb = VectorStore.get_chroma_vectorstore(
        vectorstore_path=vectorestore_path,
        embedding=embedding_model
    )

    query = 'what is the specialty for Chinese language. '
    docs = VectorStore.retrieve_docs(query=query, vectorstore=chroma_kb)

    for doc in docs:
        print(doc.page_content)
        print(doc.metadata)
