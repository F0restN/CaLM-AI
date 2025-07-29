import asyncio

from main import GraphState, calm_agent
from classes.ChatSession import ChatSessionFactory
from langgraph.pregel.io import AddableValuesDict
from classes.Generation import Generation

def format_response(response: Generation) -> str:
    """Format answer and sources into a string."""
    res = response.answer

    res += "\nReferences: \n"
    for source in response.sources:
        res += f"- [{source.index}] {source.url} \n"

    return res

async def call_api(query: str, model: str = "qwen3:30b-a3b", intermediate_model: str = "qwen3:4b") -> str:
    """Test the basic workflow functionality."""
    # Create test state
    test_state = GraphState(
        user_query=query,
        query_message=query,
        chat_session=ChatSessionFactory(
            messages=[],
            max_messages=6,
        ),
        model=model,
        intermediate_model=intermediate_model,
        threshold=3,
        max_retries=1,
        doc_number=3,
        temperature=0.3,
    )

    try:
        # Run the workflow
        final_state: AddableValuesDict
        async for state_update in calm_agent.astream(test_state.model_dump(), stream_mode="values"):
            assert isinstance(state_update, AddableValuesDict), "State update is not a dictionary"
            final_state = state_update

        res = final_state.get("final_answer", "")
        return format_response(res)

    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        return ""



async def call_api_batch(queries: list[str], model: str, intermediate_model: str) -> list[str]:
    return await asyncio.gather(*[
        call_api(query=query, model=model, intermediate_model=intermediate_model)
        for query in queries
    ])
