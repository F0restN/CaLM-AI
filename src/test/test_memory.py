import pytest

from classes.Memory import MemoryItem
from embedding.vector_store import add_to_memory, recall_memory
from memory.memory_proc import summarize_from_chat

user_query = [
    "Can you recommend activities that are suitable for someone with dementia to engage in and enjoy?",
    "What are the benefits of hospice care for individuals in the advanced stages of Alzheimer's disease?",
    "How can I navigate difficult family dynamics when caring for a loved one with Alzheimer's?",
    "What are the early signs of dementia?",
    "How does palliative care differ from hospice, and how can it help improve the quality of life for my dad with advanced dementia?"
    "How can I ensure my loved one's safety at home, especially if they have a tendency to wander or become agitated?"
]

user_profile = [
    {
        "user_id": "drake97",
        "type": "preference",
        "content": "step-by-step reasoning with references",
        "category": "preference",
        "level": "STM",
        "source": "user",
        "timestamp": "2025-04-10"
    },
    {
        "user_id": "drake97",
        "type": "care recipient relationship",
        "content": "filiation relationship in caregiving",
        "category": "ALZ INFO",
        "level": "LTM",
        "source": "user query",
        "timestamp": "2025-04-10"
    }
]

@pytest.mark.parametrize("user_profile", user_profile)
def test_construct_memory_item(user_profile):
    res = MemoryItem(**user_profile)
  
    assert res is not None
    assert isinstance(res, MemoryItem)

    print("\n", res)


@pytest.mark.parametrize("user_query", user_query)
def test_summarize_from_chat(user_query):
    res: MemoryItem = summarize_from_chat(user_query)
    
    assert res is not None
    assert isinstance(res, MemoryItem)
    
    print("\n", res)
    
    
@pytest.mark.parametrize("user_profile", user_profile)
def test_add_to_memory(user_profile):
    mi = MemoryItem(**user_profile)
    
    print("\n", mi)
    
    res = add_to_memory(mi)
    
    assert res is not None
    
    
recall_memory_query = [
    "What is the relationship between the user and the care recipient?",
]
    
@pytest.mark.parametrize("recall_memory_query", recall_memory_query)
def test_recall_memory(recall_memory_query):
    res = recall_memory(recall_memory_query, user_id="drake97")
    
    print("\n", res)
    
    assert res is not None
    assert isinstance(res, list)
    assert len(res) > 0
