import asyncio
import pprint

from main import GraphState, calm_agent


async def test_basic_workflow():
    """Test the basic workflow functionality."""
    # Create test state
    test_state = GraphState(
        user_query="What are the symptoms of dementia?",
        chat_session=[],
        model="deepseek-v3",
        intermediate_model="qwen2.5:14b",
        threshold=0.7,
        max_retries=2,
        doc_number=3,
        temperature=0.1,
        query_message="What are the symptoms of dementia?",
    )

    print("🚀 Starting workflow test...")
    print(f"Initial query: {test_state.user_query}")

    try:
        # Run the workflow
        final_state = None
        async for state_update in calm_agent.astream(test_state.model_dump(), stream_mode="values"):
            print(f"📝 State update: {list(state_update.keys())}")
            final_state = state_update

        print("✅ Workflow completed successfully!")
        print(f"Final answer: {final_state}")

        return final_state

    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(test_basic_workflow())
