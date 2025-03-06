from typing import List

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langsmith import traceable

from classes.Generation import AIGeneration, Generation
from classes.ChatSession import ChatMessage
from utils.logger import logger
from utils.PROMPT import CLAUDE_EMOTIONAL_SUPPORT_PROMPT
from utils.llm_manager import _get_llm


def generate_answer(
    question: str,
    context_chunks: List[Document] = [],
    chat_session: List[ChatMessage] = [],
    model: str = "llama3.2",
    temperature: float = 0,
    tool_call_flag: bool = False,
    langsmith_extra: dict = {}
) -> Generation:
    """
    Generate answer from context documents using LLM.

    Args:
        question: User's question
        context_chunks: List of Langchain Document objects
        chat_session: User conversation history
        model: Name of the Ollama model
        temperature: Model temperature
    """

    if not question:
        raise ValueError("Question and context required")

    context_page_content = "\n".join(doc.page_content for doc in context_chunks)

    llm = _get_llm(model, temperature)
    structured_llm = None
    
    if tool_call_flag:
        prompt = PromptTemplate(
            input_variables=["context", "question", "chat_session"],
            template=CLAUDE_EMOTIONAL_SUPPORT_PROMPT
        )
        structured_llm = prompt | llm.with_structured_output(schema=AIGeneration, method="function_calling", include_raw=False)
    else:
        json_parser = JsonOutputParser(pydantic_object=AIGeneration)
        prompt = PromptTemplate(
            input_variables=["context", "question", "chat_session"],
            partial_variables={"format_instructions": json_parser.get_format_instructions()},
            template=CLAUDE_EMOTIONAL_SUPPORT_PROMPT + "\n\nOutput Format: {format_instructions}"
        )
        structured_llm = prompt | llm | json_parser
    
    
    try:
        source_list = [
            {
                "url": doc.metadata.get("url", "") or doc.metadata.get("source", ""),
                "title": doc.metadata.get("title", "")
            }
            for doc in context_chunks
        ]
        
        response = structured_llm.invoke(
            {
                "context": context_page_content,
                "question": question,
                "chat_session": []
                # "chat_session": chat_session # Not for now
            }
        )
        
        if tool_call_flag:
            response = Generation(
                **response.model_dump(),
                sources=source_list
            )
        else:
            response = Generation(
                **response,
                sources=source_list
            )
        
        logger.info(f"Answer generation completed for question: {question}, using model: {model}, temperature: {temperature}")
        logger.info(f"Appendix documents: {context_chunks}")
        logger.info(f"Answer: {response.answer}")

        return response
        
    except Exception as e:
        logger.error(f"Answer generation failed: {str(e)}")
        raise


if __name__ == "__main__":
    test_chunks = [
        Document(
            page_content="""Alzheimer's disease is a progressive neurologic disorder that causes the brain to shrink (atrophy) 
            and brain cells to die. It is the most common cause of dementia.""",
            metadata={"source": "medical_text_1", "title": "ADRD"}
        ),
        Document(
            page_content="""Common symptoms include memory loss, confusion, and changes in thinking abilities. 
            Early diagnosis and management can help improve quality of life.""",
            metadata={"source": "medical_text_2", "title": "Alzheimer's Disease"}
        )
    ]
    
    test_chat_session = [
        ChatMessage(
            id="c3db2446-651b-4603-b2a8-2ba558c97ff4",
            role="user",
            content="What are 5 creative things I could do with my kids' art? I don't want to throw them away, but it's also so much clutter.",
            timestamp=1738345174
        ),
        ChatMessage(
            id="f5ecfd01-c78a-4f2e-8495-ea0c9996a0bc",
            role="assistant",
            content="base_pipeline_scaffold response to: What are 5 creative things I could do with my kids' art? I don't want to throw them away, but it's also so much clutter.",
            timestamp=1738345174
        ),
        ChatMessage(
            id="5a013a73-ff22-4e88-a69a-e09a28217331",
            role="user",
            content="what do you mean",
            timestamp=1738345287
        ),
        ChatMessage(
            id="dcc67334-532e-49c7-9876-996de89ed416",
            role="assistant",
            content="base_pipeline_scaffold response to: what do you mean",
            timestamp=1738345287
        ),
    ]

    answer = generate_answer(
        question="My parent is suffering from Alzheimer's disease, what should I do?",
        context_chunks=test_chunks,
        chat_session=test_chat_session,
        model="qwen2.5",
        tool_call_flag=False
    )
    
    print(answer.model_dump_json(indent=4))
    
    # print(f"\nGenerated Answer:\n{answer}")
