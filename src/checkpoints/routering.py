import json
from typing import Literal, Union
from utils.logger import logger
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

ROUTING_PROMPT = """
You are expert in routing questions to the right knowledge base. The knowledge base is either 'research' or 'peer_support'. 
While 'research' contains documents from PubMed and PubMed Central and other professional journals for research-related and serious and official questions, 
'peer_support'  contains data from social media like AgeCare forum and Reddit, they are user shared stories for peer-support related questions.

Return JSON with a single key "knowledge_base" that is 'research' or 'peer_support' depending on the question.

And the user question is: {question}
"""

KnowledgeBaseType = Literal["research", "peer_support"]

def get_routing_decision(
    messages: list[Union[HumanMessage, AIMessage, SystemMessage]],
    model: str = "llama2",
    temperature: float = 0,
) -> dict[str, KnowledgeBaseType]:
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
    if not messages:
        logger.error("Empty messages list provided")
        raise ValueError("Messages list cannot be empty")
    
    last_message = messages[-1]
    if not isinstance(last_message.content, str):
        logger.error(f"Invalid message content type: {type(last_message.content)}")
        raise ValueError("Message content must be a string")
    
    logger.info(f"Processing question: {last_message.content}")
    
    prompt = PromptTemplate(
        input_variables=["question"],
        template=ROUTING_PROMPT
    )
    
    llm = ChatOllama(model=model, temperature=temperature, format="json")
    chain = prompt | llm
    
    try:
        result = chain.invoke(
            {"question": last_message.content}, 
            config={"response_format": "json"}
        )
        parsed_result = json.loads(result.content)
        logger.info(f"Routing decision: {parsed_result}")
        return parsed_result
    except json.JSONDecodeError:
        logger.warning("Could not parse LLM response as JSON")
        return {"knowledge_base": "research"}
    except Exception as e:
        logger.error(f"Error during routing: {str(e)}")
        return {"knowledge_base": "research"}
    
    
# Test the router
if __name__ == "__main__":
    """Main function to handle user interaction."""
    logger.info("Starting Question Router")
    print("Welcome to the Question Router!")
    print("Type 'quit' to exit")
    
    while True:
        try:
            user_input = input("\nEnter your question: ").strip()
            if not user_input:
                logger.debug("Empty input received")
                print("Please enter a valid question")
                continue
                
            if user_input.lower() == 'quit':
                logger.info("User requested to quit")
                print("Goodbye!")
                break
                
            result = get_routing_decision([HumanMessage(content=user_input)], model="llama3.2", temperature=0)
            print(json.dumps(result, indent=2))
            
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received")
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.exception(f"Unexpected error: {str(e)}")
            print(f"An error occurred: {str(e)}")