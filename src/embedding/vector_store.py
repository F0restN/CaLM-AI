from typing import List

from langchain_postgres import PGVector
from langchain.schema import Document

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

if __name__ == "__main__":
    import os
    from embedding_models import get_nomic_embedding

    PGVECTOR_CONN = os.environ.get("PGVECTOR_CONN")
    
    r_k = get_connection(
        connection=PGVECTOR_CONN,
        embedding_model=get_nomic_embedding(),
        collection_name="peer_support_kb"
    )
    
    res = r_k.similarity_search("my mom is forgetting things, what should I do ? Is she dimentia ?")
    
    [print(f"{doc}\n") for doc in res]
