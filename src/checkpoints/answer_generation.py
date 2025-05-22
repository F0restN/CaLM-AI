
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate

from classes.ChatSession import ChatMessage
from classes.Generation import AIGeneration, Generation
from utils.llm_manager import _get_deepseek
from utils.logger import logger
from utils.PROMPT import BASIC_PROMPT, CLAUDE_EMOTIONAL_SUPPORT_PROMPT


def generate_answer(
    question: str,
    context_chunks: list[Document] = [],
    work_memory: str = "",
    temperature: float = 0,
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
    if not question:
        raise ValueError("Question and context required")

    context_page_content = "\n".join(
        doc.page_content for doc in context_chunks)

    llm = _get_deepseek("deepseek-chat", temperature)

    prompt = PromptTemplate(
        input_variables=["context", "question", "work_memory"],
        template=BASIC_PROMPT if isInformal else CLAUDE_EMOTIONAL_SUPPORT_PROMPT,
    )

    structured_llm = prompt | llm.with_structured_output(
        schema=AIGeneration,
        method="function_calling",
        include_raw=False,
    )

    try:
        # Create a list of unique sources
        seen_urls = set()
        source_list = []

        for doc in context_chunks:
            url = doc.metadata.get("url", "") or doc.metadata.get("source", "")
            title = doc.metadata.get("title", "")

            # Only add if URL is not in seen_urls
            if url not in seen_urls:
                source_list.append({
                    "url": url,
                    "title": title,
                })
                seen_urls.add(url)

        response = structured_llm.invoke(
            {
                "context": context_page_content,
                "question": question,
                "work_memory": work_memory,
            },
        )

        if not isinstance(response, AIGeneration):
            return Generation(
                answer="Sorry, I couldn't generate an answer to your question. Please try again.",
                follow_up_questions=[],
                sources=[],
            )

        response = Generation(
            **response.model_dump(),
            sources=source_list,
        )

        logger.info(
            f"Answer generation completed for question: {question}, using model: deepseek-v3-0324, temperature: {temperature}")
        logger.info(f"Appendix documents: {context_chunks}")
        logger.info(f"Work memory: {work_memory}")
        logger.info(f"Answer: {response.answer}")

        return response

    except Exception as e:
        logger.error(f"Answer generation failed: {e!s}")
        raise
