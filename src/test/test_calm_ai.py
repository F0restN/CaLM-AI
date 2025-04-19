import os
import pytest
import pandas as pd

from langchain_core.documents import Document
from checkpoints.adaptive_decision import adaptive_rag_decision
from checkpoints.retrieval_grading import grade_retrieval_batch
from classes.AdaptiveDecision import AdaptiveDecision
from classes.DocumentAssessment import AnnotatedDocumentEvl
from embedding.vector_store import get_connection, similarity_search
from embedding.embedding_models import get_nomic_embedding
from generation_rm import generation_with_rm

from main import detect_intention, retrieve_documents

ds_test = pd.read_parquet("src/test/rag-test-dataset.parquet")

# Initialize knowledge base connections
PGVECTOR_CONN = os.environ.get("PGVECTOR_CONN")
p_kb = get_connection(connection=PGVECTOR_CONN, embedding_model=get_nomic_embedding(), collection_name='peer_support_kb')
r_kb = get_connection(connection=PGVECTOR_CONN, embedding_model=get_nomic_embedding(), collection_name='research_kb')

pytest_plugins = ('pytest_asyncio',)

questions = []
for idx, row in ds_test.iterrows():
    questions.append(row["Question"])


# Refined using Langgraph style function
@pytest.mark.parametrize("question", questions)
def test_detect_intention(question):
    
    state = {
        "user_query": question,
        "intermediate_model": "qwen2.5-coder:7b",
        "temperature": 0.3,
        "langsmith_extra": None
    }
    
    res = detect_intention(state)['adaptive_decision']
    
    assert res is not None
    assert isinstance(res, AdaptiveDecision)
    
    # res = adaptive_rag_decision(query=question)

    # assert res is not None
    # assert isinstance(res, AdaptiveDecision)   


@pytest.mark.parametrize("question", questions)
def test_similarity_search(question):
    
    state = {
        "user_query": question,
        "intermediate_model": "qwen2.5-coder:7b",
        "temperature": 0.3,
        "langsmith_extra": None,
        # for retrieval
        "adaptive_decision": None,
        "query_message": question,
        "doc_number": 4,
        "retry_count": 0
    }
    
    state['adaptive_decision'] = detect_intention(state)['adaptive_decision']
    
    res = retrieve_documents(state)['retrieved_docs']
    
    assert res is not None
    assert isinstance(res, list)
    
    for doc in res:
        assert isinstance(doc, Document)
        assert doc.page_content is not None and isinstance(doc.page_content, str)
        assert doc.metadata is not None and isinstance(doc.metadata, dict)
        assert doc.metadata["source"] is not None and isinstance(doc.metadata["source"], str)
        

@pytest.mark.parametrize("question", questions)
@pytest.mark.asyncio
async def test_grading_retrieval(question):
    intention = adaptive_rag_decision(query=question)
    
    if intention.require_extra_re:
        kb = p_kb if intention.knowledge_base == "peer_support" else r_kb
        res = similarity_search(query=question, kb=kb, k=4)
        return await grade_retrieval_batch(question=question, retrieved_docs=res, model="qwen2.5-coder:7b", temperature=0.3)
    else:
        return []
    
    assert res is not None
    assert isinstance(res, list)
    
    
    for doc in res:
        assert isinstance(doc, AnnotatedDocumentEvl)


@pytest.mark.parametrize("question", questions)
def test_reasoning_model_use(question):
    print(generation_with_rm([], question, []))