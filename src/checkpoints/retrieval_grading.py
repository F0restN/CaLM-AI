from typing import List, Dict, Any
import asyncio

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException

from utils.logger import logger
from utils.llm_manager import _get_llm
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
"""


async def grade_retrieval(
    question: str,
    retrieved_doc: Document,
    model: str = "llama3.2",
    temperature: float = 0,
    langsmith_extra: dict = {}
) -> AnnotatedDocumentEvl:
    """
    Grade the relevance of retrieved documents to a user question.

    Args:
        question: User's question
        retrieved_doc: Retrieved document
        model: Name of the Ollama model to use
        temperature: Temperature for model generation

    Returns:
        AnnotatedDocumentEvl: Annotated document with grading results

    Raises:
        ValueError: If question is empty or retrieved_docs is empty
    """
    
    logger.info(f"Grading retrieved document relevance")

    prompt = PromptTemplate(
        template=GRADING_PROMPT,
        input_variables=["question", "document"],
    )
    
    llm = _get_llm(model, temperature)
    
    structured_llm = prompt | llm.with_structured_output(schema=DocumentAssessment, method="function_calling", include_raw=False)
    
    try:
        document_assessment: DocumentAssessment = await structured_llm.ainvoke(
            {
                "question": question,
                "document": retrieved_doc.page_content
            }
        )
            
        # For Langsmith tracing render purpose
        # doc.metadata["relevance_score"] = result.relevance_score
        # doc.metadata["reasoning"] = result.reasoning
        # doc.metadata["missing_topics"] = result.missing_topics
        
        return AnnotatedDocumentEvl(
            document=retrieved_doc,
            **document_assessment.model_dump()
        )
        
    except OutputParserException as ope_err:
        logger.error(f"Output parser exception: {ope_err}")
        document_assessment = structured_llm.invoke({"question": question, "document": retrieved_doc.page_content}, strict=True)
        
        return AnnotatedDocumentEvl(
            document=retrieved_doc,
            **document_assessment.model_dump()
        )
        
    except Exception as e:
        logger.error(f"Error: {e}, with document: {retrieved_doc.page_content}")
        return AnnotatedDocumentEvl(
            document=retrieved_doc,
            relevance_score=0.0,
            reasoning=f"Error during evaluation: {str(e)}",
            missing_topics=["Error in evaluation"]
        )
        



async def grade_retrieval_batch(
    question: str,
    retrieved_docs: List[Document],
    **kwargs
) -> List[AnnotatedDocumentEvl]:
    """
    Grade multiple documents in parallel.
    
    Args:
        question: User's question
        retrieved_docs: List of documents to grade
        **kwargs: Additional arguments for grade_retrieval
        
    Returns:
        List[AnnotatedDocumentEvl]: List of graded documents
    """
    results = await asyncio.gather(*[
        grade_retrieval(question, doc, **kwargs)
        for doc in retrieved_docs
    ])
    return results


def grade_retrieval_batch_sync(
    question: str,
    retrieved_docs: List[Document],
    **kwargs
) -> List[AnnotatedDocumentEvl]:
    """
    Synchronous version of grade_retrieval_batch.
    
    Args:
        question: User's question
        retrieved_docs: List of documents to grade
        **kwargs: Additional arguments for grade_retrieval
        
    Returns:
        List[AnnotatedDocumentEvl]: List of graded documents
    """
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(grade_retrieval_batch(question, retrieved_docs, **kwargs))


if __name__ == "__main__":
    async def main():
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
                Hospice services are designed to support individuals towards the end of life. Care can be provided wherever the person resides – including at home or in a care facility. This includes visiting nurses, pain management, and personal care. Hospice can also provide spiritual, grief, and bereavement support as well as respite for family caregivers.
                    Hospice is a Medicare benefit and individuals are eligible when a doctor has determined a patient has 6 months or less to live. Ask your doctor for a referral to begin services, or hospice can assist you in getting a referral if the patient is eligible.
                """
            )
        ]

        user_input = "What are the benefits of hospice care for individuals in the advanced stages of Alzheimer's disease?"
        
        res = await grade_retrieval_batch(
            user_input, test_docs, model="qwen2.5-coder:7b", temperature=0
        )
        
        print("\nGrading Results:")
        for i, result in enumerate(res, 1):
            print(f"\nDocument {i} Results:")
            print(result.model_dump_json(indent=2))
    
    import asyncio
    asyncio.run(main())
