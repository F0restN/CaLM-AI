from typing import List, Dict, Any

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool, ToolException
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser

from utils.logger import logger
from classes.DocumentAssessment import DocumentAssessment, AnnotatedDocumentEvl


GRADING_PROMPT = """
You are a search relevance expert. Analyze how relevant the given document is to a given query: ({question}) and provide a single numeric score between 0.000 and 1.000. Follow these scoring guidelines:

This is the document you will be grading:
<start_of_document>
{document}
<end_of_document>

Focus on:
1. Semantic relevance, not just keyword matching
2. Whether the document actually answers the question
3. The specificity and completeness of the information
4. The context alignment between question and document
5. Scoring Rubric:

- 0.000: No relevance whatsoever
- 0.001-0.299: Minimal/tangential relevance
- 0.300-0.599: Moderately relevant
- 0.600-0.899: Highly relevant 
- 0.900-1.000: Perfect or near-perfect match

Return a JSON with the following structure:
{format_instructions}
"""


def grade_retrieval(
    question: str,
    retrieved_docs: List[Document],
    model: str = "llama3.2",
    temperature: float = 0,
    langsmith_extra: dict = {}
) -> List[AnnotatedDocumentEvl]:
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
        raise ToolException("Question and retrieved documents cannot be empty")
    logger.info(f"Grading relevance for question: {question}")

    json_parser = JsonOutputParser(pydantic_object=DocumentAssessment)
    
    print(json_parser.get_format_instructions)
    
    prompt = PromptTemplate(
        template=GRADING_PROMPT,
        input_variables=["question", "document"],
        partial_variables={"format_instructions": json_parser.get_format_instructions}
    )
    
    llm = ChatOllama(model=model, temperature=temperature, format="json")
    
    chain = prompt | llm | json_parser
    
    results = []
    
    for doc in retrieved_docs:
        try:
            result: DocumentAssessment = chain.invoke(
                {
                    "question": question,
                    "document": doc.page_content
                },
                config={"response_format": "json"},
                langsmith_extra=langsmith_extra
            )
            
            # For Langsmith tracing render purpose
            doc.metadata["relevance_score"] = result['relevance_score']
            doc.metadata["reasoning"] = result['reasoning']
            doc.metadata["missing_topics"] = result['missing_topics']
            
            results.append(AnnotatedDocumentEvl(
                document=doc,
                **result
            ))
            
            logger.info(f"Document {doc.page_content} is {result}")
        except Exception as e:
            logger.error(f"Error during grading: {str(e)}")
            results.append({
                "relevance_score": 0.0,
                "reasoning": f"Error during evaluation: {str(e)}",
                "missing_aspects": ["Error in evaluation"]
            })
    return results

grading_document = StructuredTool.from_function(
    func=grade_retrieval,
    # response_format="content_and_artifact",
    handle_tool_error=True
)


if __name__ == "__main__":
    
    logger.info("Starting Retrieval Grader")
    print("Welcome to the Retrieval Grader!")
    print("Type 'quit' to exit")

    # Sample documents for testing
    test_docs = [
        Document(
            """
            Drake is the best guy ever!!
            """
        ),
        Document(
            """
            Common symptoms of ADRD (Alzheimer's Disease and Related Dementias) include memory loss,
            difficulty in planning or problem solving, confusion with time or place, and changes in mood
            and personality. Early diagnosis is crucial for better management of the condition.
            """
        )
    ]

    while True:
        user_input = input("\nEnter your question (or 'quit' to exit): ").strip()
        if not user_input:
            print("Please enter a valid question")
            continue

        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        # results = grade_retrieval(
        #     user_input, test_docs, model="llama3.2", temperature=0)
    
        results = grading_document.invoke({
            "question":user_input, 
            "retrieved_docs": test_docs, 
            "model": "phi4:latest"
        })
        
        
        print("\nGrading Results:")
        for i, result in enumerate(results, 1):
            print(f"\nDocument {i} Results:")
            print(result)
