import json
from typing import List, Dict, Any
from utils.logger import logger
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

GRADING_PROMPT = """
You are an expert in evaluating the relevance between a user's question and retrieved documents.
Analyze the following user question and retrieved document, then provide a relevance score and explanation.

User Question: {question}

Retrieved Document: {document}

Return a JSON with the following structure:
{{
    "relevance_score": float (0-1),  # How relevant the document is to the question
    "reasoning": string,  # Brief explanation of the score
    "key_matches": list[string],  # Key matching concepts/terms
    "missing_aspects": list[string]  # Important aspects from question not covered in document
}}

Focus on:
1. Semantic relevance, not just keyword matching
2. Whether the document actually answers the question
3. The specificity and completeness of the information
4. The context alignment between question and document
"""

def grade_retrieval(
    question: str,
    retrieved_docs: List[str],
    model: str = "llama3.2",
    temperature: float = 0
) -> List[Dict[str, Any]]:
    """
    Grade the relevance of retrieved documents to a user question.
    
    Args:
        question: User's question
        retrieved_docs: List of retrieved document texts
        model: Name of the Ollama model to use
        temperature: Temperature for model generation
        
    Returns:
        list: List of dictionaries containing grading results for each document
        
    Raises:
        ValueError: If question is empty or retrieved_docs is empty
    """
    if not question or not retrieved_docs:
        logger.error("Empty question or retrieved documents")
        raise ValueError("Question and retrieved documents cannot be empty")
    
    logger.info(f"Grading relevance for question: {question}")
    
    prompt = PromptTemplate(
        input_variables=["question", "document"],
        template=GRADING_PROMPT
    )
    
    llm = ChatOllama(model=model, temperature=temperature, format="json")
    chain = prompt | llm
    
    results = []
    for doc in retrieved_docs:
        try:
            result = chain.invoke(
                {
                    "question": question,
                    "document": doc
                },
                config={"response_format": "json"}
            )
            parsed_result = json.loads(result.content)
            results.append(parsed_result)
            # logger.info(f"Grading result: {parsed_result}")
        except json.JSONDecodeError:
            logger.warning("Could not parse LLM response as JSON")
            results.append({
                "relevance_score": 0.0,
                "reasoning": "Error parsing grading result",
                "key_matches": [],
                "missing_aspects": ["Error in evaluation"]
            })
        except Exception as e:
            logger.error(f"Error during grading: {str(e)}")
            results.append({
                "relevance_score": 0.0,
                "reasoning": f"Error during evaluation: {str(e)}",
                "key_matches": [],
                "missing_aspects": ["Error in evaluation"]
            })
    
    return results

# Test the grader
if __name__ == "__main__":
    """Main function to test the retrieval grading."""
    logger.info("Starting Retrieval Grader")
    print("Welcome to the Retrieval Grader!")
    print("Type 'quit' to exit")
    
    # Sample documents for testing
    test_docs = [
        """
        Alzheimer's disease is a progressive neurologic disorder that causes the brain to shrink (atrophy) 
        and brain cells to die. It is the most common cause of dementia — a continuous decline in thinking, 
        behavioral and social skills that affects a person's ability to function independently.
        """,
        """
        Common symptoms of ADRD (Alzheimer's Disease and Related Dementias) include memory loss,
        difficulty in planning or problem solving, confusion with time or place, and changes in mood
        and personality. Early diagnosis is crucial for better management of the condition.
        """
    ]
    
    while True:
        try:
            print("\nAvailable test documents:")
            for i, doc in enumerate(test_docs, 1):
                print(f"\nDocument {i}:")
                print(doc.strip())
            
            user_input = input("\nEnter your question (or 'quit' to exit): ").strip()
            if not user_input:
                logger.debug("Empty input received")
                print("Please enter a valid question")
                continue
                
            if user_input.lower() == 'quit':
                logger.info("User requested to quit")
                print("Goodbye!")
                break
            
            try:
                results = grade_retrieval(user_input, test_docs, model="llama3.2", temperature=0)
                print("\nGrading Results:")
                for i, result in enumerate(results, 1):
                    print(f"\nDocument {i} Results:")
                    print(json.dumps(result, indent=2, ensure_ascii=False))
            except Exception as e:
                logger.error(f"Error during grading: {str(e)}")
                print(f"Failed to grade documents: {str(e)}")
                
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received")
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.exception(f"Unexpected error: {str(e)}")
            print(f"An error occurred: {str(e)}") 