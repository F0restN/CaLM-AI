import asyncio

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import PromptTemplate

from classes.DocumentAssessment import AnnotatedDocumentEvl, DocumentAssessment
from utils.logger import logger
from utils.Models import _get_llm

GRADING_PROMPT = """
You are an expert document relevance evaluator specializing in healthcare and caregiving content. Your task is to analyze how relevant the given document is to a user's query about Alzheimer's disease and dementia caregiving: ({question}). Provide a detailed assessment with a numeric score between from 1 to 5 as the relevance score, a sentence of why you give this score as the reasoning and 3 words summarization of the document is missing from user's question as the missing topics.

This is the document you will be evaluating:
<start_of_document>
{document}
</end_of_document>

Scoring Rubric:
- 1: No relevance to user's query
- 2: Minimal relevance, lacks practical caregiving and healthcare guidance
- 3: Partial relevance, contains some useful caregiving and healthcare information
- 4: Strong relevance, provides comprehensive caregiving and healthcare guidance
- 5: Perfect match, offers complete and actionable caregiving and healthcare solutions

Follow the schema of DocumentAssessment to structure your response.
"""


async def grade_retrieval(
    question: str,
    retrieved_doc: Document,
    model: str = "qwen3:4b",
    temperature: float = 0.3,
) -> AnnotatedDocumentEvl:
    """Grade the relevance of retrieved documents to a user question.

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
    logger.info("Grading retrieved document relevance")

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
                "document": retrieved_doc.page_content,
            },
        )

        return AnnotatedDocumentEvl(
            document=retrieved_doc,
            **document_assessment.model_dump(),
        )

    except OutputParserException as ope_err:
        logger.error(f"Output parser exception: {ope_err}")
        document_assessment = structured_llm.invoke({"question": question, "document": retrieved_doc.page_content}, strict=True)

        return AnnotatedDocumentEvl(
            document=retrieved_doc,
            **document_assessment.model_dump(),
        )

    except Exception as e:
        logger.error(f"Error: {e}, with document: {retrieved_doc.page_content}")
        return AnnotatedDocumentEvl(
            document=retrieved_doc,
            relevance_score=1,
            reasoning=f"Error during evaluation: {e!s}",
            missing_topics=["Error in evaluation"],
        )




async def grade_retrieval_batch(
    question: str,
    retrieved_docs: list[Document],
    **kwargs,
) -> list[AnnotatedDocumentEvl]:
    """Grade multiple documents in parallel.

    Args:
        question: User's question
        retrieved_docs: List of documents to grade
        **kwargs: Additional arguments for grade_retrieval

    Returns:
        List[AnnotatedDocumentEvl]: List of graded documents

    """
    return await asyncio.gather(*[
        grade_retrieval(question, doc, **kwargs)
        for doc in retrieved_docs
    ])


# def grade_retrieval_batch_sync(
#     question: str,
#     retrieved_docs: list[Document],
#     **kwargs,
# ) -> list[AnnotatedDocumentEvl]:
#     """Synchronous version of grade_retrieval_batch.

#     Args:
#         question: User's question
#         retrieved_docs: List of documents to grade
#         **kwargs: Additional arguments for grade_retrieval

#     Returns:
#         List[AnnotatedDocumentEvl]: List of graded documents

#     """
#     loop = asyncio.get_event_loop()
#     return loop.run_until_complete(grade_retrieval_batch(question, retrieved_docs, **kwargs))


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
                """,
            ),
            Document(
                """
                Hospice services are designed to support individuals towards the end of life. Care can be provided wherever the person resides – including at home or in a care facility. This includes visiting nurses, pain management, and personal care. Hospice can also provide spiritual, grief, and bereavement support as well as respite for family caregivers.
                    Hospice is a Medicare benefit and individuals are eligible when a doctor has determined a patient has 6 months or less to live. Ask your doctor for a referral to begin services, or hospice can assist you in getting a referral if the patient is eligible.
                """,
            ),
        ]

        user_input = "What are the benefits of hospice care for individuals in the advanced stages of Alzheimer's disease?"

        res = await grade_retrieval_batch(
            user_input, test_docs, model="qwen3:14b", temperature=0.3,
        )

        print("\nGrading Results:")
        for i, result in enumerate(res, 1):
            print(f"\nDocument {i} Results:")
            print(result.model_dump_json(indent=2))

    import asyncio
    asyncio.run(main())
