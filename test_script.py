import asyncio
from agents.agent_orchestrator import MasterOrchestrator
import os

async def test_workflow():
    orchestrator = MasterOrchestrator(
        project_id=os.getenv("PROJECT_ID"),  # dummy project id
        config_path="config/agent_config.yaml",
        location="us-central1",
        model_name="gemini-2.0-flash-001"
    )

    user_query = "How much did I spend on general itmes in the past 2 months?"

    # Provide dummy user_id and optional context
    result = await orchestrator.process_query(
        query=user_query,
        user_id="a73ff731-9018-45ed-86ff-214e91baf702",
        additional_context={
            "currency": "USD",
            "timezone": "America/New_York",
            "preferred_categories": ["groceries", "food", "transport"]
        }
    )

    print("----- Final Orchestrator Response -----")
    print(result)

if __name__ == "__main__":
    asyncio.run(test_workflow())