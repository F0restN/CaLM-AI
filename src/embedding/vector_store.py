import os
from typing import List
from langchain_postgres import PGVector
from langchain.schema import Document
from classes.Memory import MemoryItem
from embedding.embedding_models import get_nomic_embedding
from datetime import datetime

PGVECTOR_CONN = os.environ.get("PGVECTOR_CONN")

def get_connection(connection: str, embedding_model, collection_name: str) -> PGVector:
    if not collection_name:
        raise ("Collection Name can not be empty")
    
    return PGVector(
        embeddings=embedding_model,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )


def similarity_search(query:str, kb:PGVector = None, k:int=10) -> List[Document]:
    if kb is None:
        raise ValueError("KB can not be None")
    
    res = kb.similarity_search(query=query, k=k)
    return res


def add_memory(memory: MemoryItem, kb:PGVector = None) -> str:
    """
    Add a memory to the vector store.
    
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
            collection_name="lstm_memory"
        )
    
    doc = Document(
        page_content=memory.content,
        metadata=memory.metadata
    )
    return kb.add_documents([doc])


if __name__ == "__main__":
    # import os
    # from embedding_models import get_nomic_embedding

    # PGVECTOR_CONN = os.environ.get("PGVECTOR_CONN")
    
    # r_k = get_connection(
    #     connection=PGVECTOR_CONN,
    #     embedding_model=get_nomic_embedding(),
    #     collection_name="lstm_memory"
    # )
    
    # res = r_k.similarity_search("my mom is forgetting things, what should I do ? Is she dimentia ?")
    
    # [print(f"{doc}\n") for doc in res]
    
    add_memory(MemoryItem(content="my mom is forgetting things, what should I do ? Is she dimentia ?", level="LTM", category="ALZ", type="memory", source="user", timestamp=datetime.now()))
