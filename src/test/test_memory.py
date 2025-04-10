import pytest

from classes.Memory import MemoryItem, Memory

user_profile = [
    {
        "type": "name",
        "content": "John",
        "category": "USER INFO",
        "level": "LTM",
        "source": "user",
        "timestamp": "2025-04-10"
    }
]

@pytest.mark.parametrize("user_profile", user_profile)
def test_memory_item(user_profile):
    obj = MemoryItem(**user_profile)
    
    res = obj.convert_to_sentence(categories=["USER INFO"])
    print(obj.metadata)
    print(res)
    return res



