import os

from functools import lru_cache
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from langchain_deepseek import ChatDeepSeek

@lru_cache(maxsize=1000)
def _get_llm(model: str, temperature: float):
    return ChatOllama(model=model, temperature=temperature)

@lru_cache(maxsize=1000)
def _get_deepseek(model, temperature):
    os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API") 
    return ChatDeepSeek(model=model, temperature=temperature)
    

# TODO: add a function to clear the cache

if __name__ == "__main__":
    ds_llm = _get_deepseek(model = "deepseek-reasoner", temperature=0.6)
    
    res:AIMessage = ds_llm.invoke("who are you")
    
    print(res)
