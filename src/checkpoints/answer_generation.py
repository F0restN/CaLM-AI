from langchain_core.prompts import PromptTemplate

from classes.ChatSession import ChatSessionFactory
from classes.DocumentAssessment import AnnotatedDocumentEvl
from classes.Generation import AIGeneration, Generation
from utils.logger import logger
from utils.Models import _get_deepseek, _get_llm
from utils.PROMPT import BASIC_PROMPT, CALM_ADRD_PROMPT


def generate_answer(
    question: str,
    context_chunks: list[AnnotatedDocumentEvl] | None = None,
    work_memory: ChatSessionFactory | None = None,
    temperature: float = 0.3,
    model: str = "qwen3:30b-a3b",
    *,
    isInformal: bool = False,
) -> Generation:
    """Generate answer from context documents using LLM.

    Args:
        question: User's question
        context_chunks: List of Langchain Document objects
        work_memory: User conversation history
        temperature: Model temperature
        isInformal: Whether the question is Alezhimer's disease related, yes if it is related and vise versa.

    Returns:
        Generation: Generated answer

    """
    # Validation
    if context_chunks is None:
        context_chunks = []
    if not question:
        raise ValueError("Question and context required")

    # Craft context in RAG
    working_memory_content = work_memory.get_formatted_conversation("messages")
    context_page_content: str = ""
    source_list = []
    seen_urls = set()
    for i, doc in enumerate(context_chunks):
        title = doc.document.metadata.get("title", "Untitled Document")
        content = doc.document.page_content
        url = doc.document.metadata.get("url", "") or doc.document.metadata.get("source", "")

        if url not in seen_urls:
            seen_urls.add(url)
            source_list.append({
                "index": i + 1,
                "url": url,
                "title": title,
            })
            context_page_content += (f"Index: {i + 1}; Title: {title}; Content: {content} \n")

    # Initialize LLM
    llm = _get_llm(model, temperature)

    prompt = PromptTemplate(
        input_variables=["context", "question", "work_memory"],
        template=BASIC_PROMPT if isInformal else CALM_ADRD_PROMPT,
    )

    structured_llm = prompt | llm.with_structured_output(
        schema=AIGeneration,
        method="function_calling",
        include_raw=False,
    )


    # Generate answer
    try:
        response = structured_llm.invoke(
            {
                "context": context_page_content,
                "question": question,
                "work_memory": working_memory_content,
            },
        )

        assert isinstance(response, AIGeneration), "Response is not a Generation object"

        response = Generation(
            **response.model_dump(),
            sources=source_list,
        )

        # if not work_memory or len(work_memory.messages) <= 1:
        #     answer_with_greeting = f"Hi, I'm your Caregiving Assistant. I hope you have a wonderful day! \n {response.answer}"
        #     response.answer = answer_with_greeting

        logger.info(
            f"Answer generation completed for question: {question}, using model: deepseek-v3-0324, temperature: {temperature}")
        logger.info(f"Appendix documents: \n {context_page_content}")
        logger.info(f"Work memory: {work_memory}")
    except Exception as e:
        logger.error(f"Answer generation failed: {e!s}")
        return Generation(
            answer="Sorry, I couldn't generate an answer to your question. Please try again. Error: {e!s}",
            follow_up_questions=[],
            sources=[],
        )
    else:
        return response
