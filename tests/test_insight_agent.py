import os
import sys
import json
import logging
import asyncio
from dotenv import load_dotenv

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

# --- Add project root to sys.path ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import the Agent ---
from agents.insight_synthesis_agent import InsightSynthesisAgent

# --- Import the model class (adjust if you're using a different backend) ---
from vertexai.language_models import ChatModel

def main():
    print("\n🧪 Running InsightSynthesisAgent Test")

    # --- 1. Load Environment ---
    load_dotenv()
    try:
        PROJECT_ID = os.getenv("GCP_PROJECT_ID")
        if not PROJECT_ID:
            raise ValueError("GCP_PROJECT_ID environment variable is required")
        print("✅ Environment loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load environment: {e}")
        return

    # --- 2. Instantiate the Model and Agent ---
    try:
        # chat_model = ChatModel.from_pretrained("chat-bison")
        # chat = chat_model.start_chat()

        agent = InsightSynthesisAgent(
            project_id=PROJECT_ID,
            user_id="test-user",
            model="gemini-2.0-flash-001"  # ✅ Pass shared model to agent
        )
        print(f"✅ Initialized agent: {agent.agent_name}")
    except Exception as e:
        print(f"❌ Failed to initialize agent or model: {e}")
        return

    # --- 4. Prepare Mock Input ---
    mock_input = [ 
#         {
#     "user_id": "user_john_456",
#     "original_query": "How did my spending vary across categories in the last 3 months?",
#     "analysis_results": {
#         "categories": [
#             {"name": "Groceries", "total": 8320.75},
#             {"name": "Dining", "total": 6925.50},
#             {"name": "Utilities", "total": 4210.20},
#             {"name": "Entertainment", "total": 5380.00},
#             {"name": "Subscriptions", "total": 1599.99},
#             {"name": "Healthcare", "total": 2890.00}
#         ],
#         "monthly_spending": {
#             "April": 8475.22,
#             "May": 9420.88,
#             "June": 11740.34
#         },
#         "raw_items": [
#             {"name": "Swiggy - Dinner", "price": 1125.00, "category": "Dining"},
#             {"name": "Amazon Prime Video", "price": 149.00, "category": "Subscriptions"},
#             {"name": "Spotify Family", "price": 199.00, "category": "Subscriptions"},
#             {"name": "Health Checkup", "price": 1850.00, "category": "Healthcare"},
#             {"name": "BigBasket - Monthly Essentials", "price": 3400.50, "category": "Groceries"},
#             {"name": "Inox Movie", "price": 1200.00, "category": "Entertainment"},
#             {"name": "Netflix", "price": 499.00, "category": "Subscriptions"},
#             {"name": "BESCOM Bill", "price": 2250.00, "category": "Utilities"},
#             {"name": "Swiggy - Friends Night", "price": 2450.50, "category": "Dining"},
#             {"name": "Apple One", "price": 749.99, "category": "Subscriptions"}
#         ]
#     }
# },
# {
#     "user_id": "user_traveler_789",
#     "original_query": "Were there any spikes or unusual patterns in my spending during my Europe trip?",
#     "analysis_results": {
#         "categories": [
#             {"name": "Flights", "total": 45000.00},
#             {"name": "Dining", "total": 18500.00},
#             {"name": "Accommodation", "total": 36500.00},
#             {"name": "Local Transport", "total": 4200.00},
#             {"name": "Entertainment", "total": 7400.00},
#             {"name": "Shopping", "total": 9800.00}
#         ],
#         "monthly_spending": {
#             "June": 39500.00,
#             "July": 117900.00
#         },
#         "raw_items": [
#             {"name": "Air France - Return Ticket", "price": 45000.00, "category": "Flights"},
#             {"name": "Airbnb Paris", "price": 19500.00, "category": "Accommodation"},
#             {"name": "Uber Rome", "price": 1100.00, "category": "Local Transport"},
#             {"name": "Prague Museum Pass", "price": 2200.00, "category": "Entertainment"},
#             {"name": "H&M Milan", "price": 3400.00, "category": "Shopping"},
#             {"name": "Dining - Florence Restaurants", "price": 8650.00, "category": "Dining"},
#             {"name": "Berlin Metro Pass", "price": 650.00, "category": "Local Transport"},
#             {"name": "Dinner - Eiffel Tower", "price": 3250.00, "category": "Dining"},
#             {"name": "Zara Spain", "price": 6400.00, "category": "Shopping"},
#             {"name": "Hilton Rome", "price": 17000.00, "category": "Accommodation"}
#         ]
#     }
# },
# {
#     "user_id": "user_family_001",
#     "original_query": "Can you summarize my regular vs unexpected spending?",
#     "analysis_results": {
#         "categories": [
#             {"name": "Groceries", "total": 7200.00},
#             {"name": "Utilities", "total": 3850.00},
#             {"name": "Dining", "total": 2100.00},
#             {"name": "Impulse Purchases", "total": 8700.00},
#             {"name": "Education", "total": 4500.00},
#             {"name": "Medical", "total": 2300.00}
#         ],
#         "monthly_spending": {
#             "June": 15520.00,
#             "July": 15130.00
#         },
#         "raw_items": [
#             {"name": "BigBasket Weekly", "price": 1800.00, "category": "Groceries"},
#             {"name": "Electricity Bill", "price": 1900.00, "category": "Utilities"},
#             {"name": "Swiggy Pizza", "price": 950.00, "category": "Dining"},
#             {"name": "Lenskart Sunglasses (Offer)", "price": 2800.00, "category": "Impulse Purchases"},
#             {"name": "Udemy Python Course", "price": 499.00, "category": "Education"},
#             {"name": "Hospital Visit", "price": 2300.00, "category": "Medical"},
#             {"name": "Flipkart Big Sale - Air Fryer", "price": 5900.00, "category": "Impulse Purchases"},
#             {"name": "Water Bill", "price": 1950.00, "category": "Utilities"},
#             {"name": "Zomato Family Dinner", "price": 1150.00, "category": "Dining"},
#             {"name": "Groceries - Fresh Produce", "price": 1400.00, "category": "Groceries"}
#         ]
#     }
# },
{"query" : "I want to optimise my budget by 15-20% especially for entertainment",
"user_id" : "user_1234",
"analysis_results" : """
{
  "category_spending": {
    "Entertainment": 12000,
    "Groceries": 15000,
    "Transport": 5000,
    "Utilities": 3000,
    "Dining Out": 8000
  },
  "recommended_cuts": {
    "Entertainment": {
      "current": 12000,
      "target": 9600,
      "suggested_reduction": 2400,
      "justification": "Frequent OTT subscriptions and weekend outings. Cutting down on non-essential subscriptions and limiting outings to twice a month can save significantly."
    },
    "Dining Out": {
      "current": 8000,
      "target": 6800,
      "suggested_reduction": 1200,
      "justification": "Switch to home-cooked meals for weekday dinners. Use dining coupons if eating out."
    }
  },
  "total_monthly_budget": 43000,
  "goal_reduction_percent": 18
}
"""}




    ]


    print("\n📄 Mock input prepared.")

    # --- 5. Run Agent ---
    try:
        for mi in mock_input:
            result = asyncio.run(agent.process(mi))
            print("\n💬 Agent Response:")
            print(json.dumps(result, indent=2, default=str))
    except Exception as e:
        print(f"\n❌ Agent execution failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n✅ Test Complete.")

if __name__ == "__main__":
    main()
