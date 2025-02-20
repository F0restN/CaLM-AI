import pytest

from fastapi.testclient import TestClient
from main_graph import fastapi_app, GraphState, setup_workflow
from classes.RequestBody import RequestBody
from classes.AdaptiveDecision import AdaptiveDecision
from langchain_core.documents import Document
from unittest.mock import Mock, patch

client = TestClient(fastapi_app)

# Mock 外部依赖
@pytest.fixture(autouse=True)
def mock_dependencies():
    with patch("langgraph_version.get_nomic_embedding") as mock_embedding, \
         patch("langgraph_version.get_connection") as mock_conn:
        
        # 模拟向量数据库连接
        mock_kb = Mock()
        mock_kb.similarity_search.return_value = [Mock(content="test doc")]
        mock_conn.return_value = mock_kb
        
        # 模拟embedding模型
        mock_embedding.return_value = Mock()
        
        yield

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_direct_answer_flow():
    # 测试不需要检索的直接回答流程
    test_payload = {
        "user_query": "Hello",
        "threshold": 0.65,
        "doc_number": 3,
        "max_retries": 2,
        "model": "phi3",
        "intermida_model": "qwen2.5",
        "temperature": 0.7,
        "chat_session": []
    }
    
    with patch("langgraph_version.adaptive_rag_decision") as mock_decision:
        mock_decision.return_value = AdaptiveDecision(
            require_extra_re=False,
            knowledge_base="None"
        )
        
        with patch("langgraph_version.generate_answer") as mock_gen:
            mock_gen.return_value = "Direct answer"
            
            response = client.post("/langgraph", json=test_payload)
            assert response.status_code == 200
            assert response.text == '"Direct answer"'

def test_full_retrieval_flow():
    # 测试完整检索流程
    test_payload = {
        "user_query": "Alzheimer's symptoms",
        "threshold": 0.7,
        "doc_number": 3,
        "max_retries": 2,
        "model": "phi3",
        "intermida_model": "qwen2.5",
        "temperature": 0.7,
        "chat_session": []
    }
    
    workflow = setup_workflow()
    
    # 模拟各节点返回值
    with patch("langgraph_version.adaptive_rag_decision") as mock_decision, \
         patch("langgraph_version.grade_retrieval") as mock_grade, \
         patch("langgraph_version.query_extander") as mock_expander, \
         patch("langgraph_version.generate_answer") as mock_answer:
        
        # 初始决策
        mock_decision.return_value = AdaptiveDecision(
            require_extra_re=True,
            knowledge_base="research"
        )
        
        # 文档评分
        mock_doc = Mock()
        mock_doc.relevance_score = 0.8
        mock_grade.return_value = [mock_doc]
        
        # 查询扩展
        mock_expander.return_value = ["expanded query"]
        
        # 最终答案
        mock_answer.return_value = "Final answer"
        
        # 执行工作流
        initial_state = GraphState(
            **test_payload,
            adaptive_decision=None,
            retrieved_docs=[],
            filtered_docs=[],
            missing_topics=[],
            query_message=test_payload["user_query"],
            retry_count=0,
            final_answer=None
        )
        
        for step in workflow.stream(initial_state):
            if step.current == "grade_docs":
                assert len(step.state["filtered_docs"]) > 0
            if step.current == END:
                break
                
        assert step.state["final_answer"] == "Final answer"

def test_retry_loop():
    # 测试重试机制
    workflow = setup_workflow()
    
    test_state = GraphState(
        user_query="Test query",
        threshold=0.8,
        max_retries=3,
        doc_number=5,
        model="phi3",
        temperature=0.7,
        intermida_model="qwen2.5",
        chat_session=[],
        adaptive_decision=AdaptiveDecision(require_extra_re=True, knowledge_base="research"),
        retrieved_docs=[],
        filtered_docs=[],
        missing_topics=[],
        query_message="Initial query",
        retry_count=0,
        final_answer=None
    )
    
    with patch("langgraph_version.grade_retrieval") as mock_grade:
        # 第一次返回低质量结果
        mock_doc = Mock()
        mock_doc.relevance_score = 0.5
        mock_doc.missing_topics = ["topic1"]
        mock_grade.return_value = [mock_doc]
        
        # 执行工作流
        steps = list(workflow.stream(test_state))
        
        # 验证重试次数
        retry_steps = [s for s in steps if s.current == "expand_query"]
        assert len(retry_steps) == 3

# 单元测试示例
def test_grade_documents_node():
    
    mock_retrieved_docs = [
        Mock(Document(page_content="test doc 1", metadata={"source": "https://www.google.com/test1", "title": "test_title_1"})),
        Mock(Document(page_content="test doc 2", metadata={"source": "https://www.google.com/test2", "title": "test_title_2"})),
        Mock(Document(page_content="test doc 3", metadata={"source": "https://www.google.com/test3", "title": "test_title_3"})),
        Mock(Document(page_content="test doc 4", metadata={"source": "https://www.google.com/test4", "title": "test_title_4"}))
    ]
    
    
    # 测试文档评分节点
    test_state = GraphState(
        user_query="Test",
        threshold=0.7,
        max_retries=2,
        doc_number=3,
        model="qwen2.5:latest",
        temperature=0.5,
        intermida_model="qwen2.5:latest",
        chat_session=[],
        retrieved_docs=mock_retrieved_docs,
        filtered_docs=[],
        missing_topics=[],
        query_message="test",
        retry_count=0,
        final_answer=None
    )
    
    from main_graph import grade_documents
    result = grade_documents(test_state)
    
def test_expand_query_node():
    # 测试查询扩展节点
    test_state = GraphState(
        user_query="Original",
        missing_topics=["topic1", "topic2"],
        retry_count=1,
        intermida_model="qwen2.5",
        temperature=0.7,
        # 其他必要字段
        threshold=0.7,
        max_retries=2,
        doc_number=3,
        model="test",
        chat_session=[],
        adaptive_decision=None,
        retrieved_docs=[],
        filtered_docs=[],
        query_message="test",
        final_answer=None
    )
    
    with patch("langgraph_version.query_extander") as mock_expander:
        mock_expander.return_value = ["Expanded query"]
        from main_graph import expand_query
        result = expand_query(test_state)
        
        assert result["query_message"] == "Expanded query"
        assert result["retry_count"] == 2 