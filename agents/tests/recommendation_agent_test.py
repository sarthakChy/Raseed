import asyncio
import os
import sys
import json
import logging
from dotenv import load_dotenv
import vertexai
from vertexai.language_models import TextEmbeddingModel

# Configure logging early to avoid conflicts
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # This ensures we override any existing configuration
)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the agent we want to test
from agents.recommendation_engine_agent import RecommendationEngineAgent

async def main():
    """
    Main function to initialize and run the agent test.
    """
    print("--- Raseed Agent Test Script (Fixed) ---")
    
    # --- 1. Setup Environment and Clients ---
    load_dotenv()
    try:
        PROJECT_ID = os.getenv("GCP_PROJECT_ID")
        LOCATION = os.getenv("GCP_LOCATION", "us-central1")
        
        if not PROJECT_ID:
            raise ValueError("GCP_PROJECT_ID environment variable is required")
            
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        
        embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        print("Successfully initialized clients.")
        
    except Exception as e:
        print(f"Error during initialization: {e}")
        return

    # --- 2. Initialize the Agent to be Tested ---
    try:
        recommendation_agent = RecommendationEngineAgent(
            agent_name="recommendation_engine_agent_test",
            project_id=PROJECT_ID,
            location=LOCATION
        )
        
        # Initialize the agent's database connector
        if hasattr(recommendation_agent, 'db_connector') and recommendation_agent.db_connector:
            await recommendation_agent.db_connector.initialize()
        else:
            print("Warning: Agent doesn't have a db_connector attribute")
        
        print(f"Successfully initialized agent: {recommendation_agent.agent_name}")
        
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return

    # --- 3. Simulate Input Data ---
    high_cost_item_name = "Amul Gold Milk 1L"
    print(f"\nGenerating embedding for test item: '{high_cost_item_name}'...")
    
    try:
        # Generate embedding
        embedding_response = embedding_model.get_embeddings([high_cost_item_name])
        item_embedding = embedding_response[0].values
        
        mock_input_data = {
            "user_id": "user_alice_123",
            "spending_analysis": {
                "high_cost_item_for_recommendation": {
                    "name": high_cost_item_name,
                    "price": 72.00,
                    "category": "Groceries",
                    "embedding": item_embedding
                }
            }
        }
        
        print("Simulated input data prepared.")
        
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return

    # --- 4. Execute the Agent's Method ---
    print("\n--- Calling agent.find_alternatives() ---")
    
    try:
        # Call the specific method we want to test
        result = await recommendation_agent.find_alternatives(mock_input_data)
        
        print("\n--- Agent Execution Complete ---")
        print("Result:")
        print(json.dumps(result, indent=2, default=str))
        
        # Test other functions as well
        print("\n--- Testing behavioral recommendations ---")
        behavioral_input = {
            "user_id": "user_alice_123",
            "lookback_months": 3
        }
        behavioral_result = await recommendation_agent.generate_behavioral_recommendations(behavioral_input)
        print("Behavioral Result:")
        print(json.dumps(behavioral_result, indent=2, default=str))
        
        print("\n--- Testing budget optimization ---")
        budget_input = {
            "user_id": "user_alice_123",
            "target_category": "Groceries",
            "savings_goal": 0.15
        }
        budget_result = await recommendation_agent.optimize_budget_allocation(budget_input)
        print("Budget Result:")
        print(json.dumps(budget_result, indent=2, default=str))
        
    except Exception as e:
        print(f"\nAn error occurred during agent execution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # --- 5. Clean Up ---
        try:
            if hasattr(recommendation_agent, 'db_connector') and recommendation_agent.db_connector:
                await recommendation_agent.db_connector.close()
            print("\nDatabase connection closed. Test finished.")
        except Exception as e:
            print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    # This allows us to run the async main function
    asyncio.run(main())