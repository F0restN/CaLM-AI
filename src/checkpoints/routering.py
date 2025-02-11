import json
from typing import Literal, Union
from utils.logger import logger
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import tool, StructuredTool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

ROUTING_PROMPT = """
You are expert in routing questions to the right knowledge base. The knowledge base is either 'research' or 'peer_support'. 
While 'research' contains documents from PubMed and PubMed Central and other professional journals for research-related and serious and official questions, 
'peer_support'  contains data from social media like AgeCare forum and Reddit, they are user shared stories for peer-support related questions.

And the user question is: {question}

Return JSON with a single key "knowledge_base" that is 'research' or 'peer_support' depending on the question.
"""


class UserIntention(BaseModel):
    knowledge_base:  Literal["research", "peer_support"] = Field(description="either 'peer_support' or 'research' ")
    
    class Config:
        arbitrary_types_allowed = True    


# @tool("user intention detection", return_direct=False)
def get_routing_decision(
    message: str,
    model: str = "llama3.2",
    temperature: float = 0,
) -> UserIntention:
    """
    Route a question to the appropriate knowledge base.

    Args:
        messages: List of conversation messages
        model: Name of the Ollama model to use
        temperature: Temperature for model generation

    Returns:
        dict: Contains 'knowledge_base' key with either 'research' or 'peer_support' value

    Raises:
        ValueError: If messages list is empty
    """

    if not message:
        logger.error("Empty messages list provided")
        raise ValueError("Messages list cannot be empty")

    json_parser = JsonOutputParser(pydantic_object=UserIntention)

    prompt = PromptTemplate(
        input_variables=["question"],
        template=ROUTING_PROMPT,
    )

    llm = ChatOllama(model=model, temperature=temperature, format="json")
    
    chain = prompt | llm | json_parser

    result = chain.invoke(
        {"question": message},
    )
    
    return UserIntention(**result)

user_intention_detection_tool = StructuredTool.from_function(
    func = get_routing_decision,
    handle_tool_error=True,
)


# Test the router
if __name__ == "__main__":
    user_input = input("\n Enter your question: ").strip()
    
    result = user_intention_detection_tool.invoke({
        "message": user_input
    })

    # result = get_routing_decision(
    #     user_input, 
    #     model="llama3.2", 
    #     temperature=0
    # )
    
    print(result)
    
    # print(json.dumps(result, indent=2))
