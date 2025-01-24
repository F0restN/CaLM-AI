from typing import List

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from utils.logger import logger
from utils.PROMPT import CLAUDE_EMOTIONAL_SUPPORT_PROMPT


def generate_answer(
    question: str,
    context_chunks: List[Document],
    model: str = "llama3.2",
    temperature: float = 0
) -> str:
    """
    Generate answer from context documents using LLM.

    Args:
        question: User's question
        context_chunks: List of Langchain Document objects
        model: Name of the Ollama model
        temperature: Model temperature
    """

    MEW_PROMPT = """
    Answers user's question: {question}
    
    Summary user's story and sharing to provide peer support (put result under section "### Peer support" ) according to:
    {context}
    """

    if not question:
        logger.error("Missing question or context")
        raise ValueError("Question and context required")

    context = "\n\n".join(doc.page_content for doc in context_chunks)
    logger.info(f"Generating answer for: {question}")

    try:
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=MEW_PROMPT
        )

        llm = ChatOllama(model=model, temperature=temperature)
        chain = prompt | llm

        response = chain.invoke({
            "context": context,
            "question": question,
            "memory": "NAN",
            "user_profile": "NAN",
        })

        return response.content
    except Exception as e:
        logger.error(f"Answer generation failed: {str(e)}")
        raise


if __name__ == "__main__":
    test_chunks = [
        Document(
            page_content="""Alzheimer's disease is a progressive neurologic disorder that causes the brain to shrink (atrophy) 
            and brain cells to die. It is the most common cause of dementia.""",
            metadata={"source": "medical_text_1"}
        ),
        Document(
            page_content="""Common symptoms include memory loss, confusion, and changes in thinking abilities. 
            Early diagnosis and management can help improve quality of life.""",
            metadata={"source": "medical_text_2"}
        )
    ]

    answer = generate_answer(
        question="My parent is suffering from Alzheimer's disease, what should I do?",
        context_chunks=test_chunks
    )
    print(f"\nGenerated Answer:\n{answer}")
