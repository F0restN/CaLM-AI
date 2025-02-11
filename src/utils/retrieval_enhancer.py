from typing import Dict, List, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

def extract_key_concepts(user_message: str, debug_info: Dict) -> Tuple[List[str], str]:
    """
    Extract two key concepts from user message and debug info to enhance document retrieval
    
    Args:
        user_message: Original user query
        debug_info: Debug information from inlet/outlet containing user context
    
    Returns:
        Tuple of (key_concepts, enhanced_query)
    """
    
    # Prompt to extract key concepts
    prompt = f"""
    Extract exactly TWO most important keywords or phrases from the following user message and debug context.
    Focus on medical conditions, symptoms, care needs, or specific concerns.
    
    User message: {user_message}
    Debug context: {debug_info}
    
    Output format:
    1. [first keyword/phrase]
    2. [second keyword/phrase]
    """
    
    llm = ChatOllama(model="llama3.2", temperature=0)
    messages = [
        SystemMessage(content="You are a precise keyword extractor. Only output the two keywords in numbered list format."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    
    # Parse response to get keywords
    lines = response.content.strip().split('\n')
    keywords = [line[3:].strip() for line in lines if line.startswith('1.') or line.startswith('2.')]
    
    # Combine into enhanced query
    enhanced_query = f"{user_message} {' '.join(keywords)}"
    
    return keywords, enhanced_query

def get_enhanced_retrieval_query(
    user_message: str,
    inlet_debug: Dict,
    outlet_debug: Dict
) -> str:
    """
    Create enhanced retrieval query using debug information
    """
    debug_info = {
        "inlet": inlet_debug,
        "outlet": outlet_debug
    }
    
    keywords, enhanced_query = extract_key_concepts(user_message, debug_info)
    print(f"Enhanced retrieval with keywords: {keywords}")
    
    return enhanced_query 