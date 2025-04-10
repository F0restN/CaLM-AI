import pytest

from classes.Memory import MemoryItem, Memory

user_profile = [
    {
        "type": "preference",
        "content": "step-by-step reasoning with references",
        "category": "USER INFO",
        "level": "STM",
        "source": "user",
        "timestamp": "2025-04-10"
    },
    {
        "type": "care recipient relationship",
        "content": "filiation",
        "category": "ALZ INFO",
        "level": "LTM",
        "source": "user query",
        "timestamp": "2025-04-10"
    }
]

@pytest.mark.parametrize("user_profile", user_profile)
def test_memory_item(user_profile):
    obj = MemoryItem(**user_profile)
    
    res = obj.convert_to_sentence(categories=["USER INFO"])
    print("\n", res)
    return res



