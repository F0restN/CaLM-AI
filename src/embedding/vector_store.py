from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector


def get_connection(connection: str, embedding_model, collection_name: str):
    if not collection_name:
        raise ("Collection Name can not be empty")
    
    return PGVector(
        embeddings=embedding_model,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )

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
