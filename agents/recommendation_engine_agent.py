import json
from typing import Dict, Any, List
from vertexai.generative_models import GenerativeModel
import traceback
import logging
from .base_agent import BaseAgent

class RecommendationEngineAgent(BaseAgent):
    """
    An agent specializing in generating personalized recommendations by directly
    querying the database using its inherited tools.
    """
    def __init__(self, agent_name: str, project_id: str, location: str, **kwargs):
        # Initialize the BaseAgent, which provides self.db_connector, self.logger, etc.
        super().__init__(agent_name, project_id, location, **kwargs)
        
        # Create a unique logger for this agent to avoid conflicts
        self.logger = logging.getLogger(f"raseed.{agent_name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # This agent can use a more cost-effective model for synthesizing results
        self.synthesis_model = GenerativeModel("gemini-1.5-flash-001")
        
        # System instructions for different recommendation types
        self.behavioral_system_instruction = """
You are the "Behavioral Recommendation Agent" for Raseed. Your task is to analyze spending patterns and provide behavioral change recommendations.
**Your Rules:**
1. Analyze the provided spending patterns to identify behavioral changes that could reduce costs.
2. Your response MUST be a single, valid JSON object with keys: `behavioral_insights` (a string) and `recommendations` (a list of strings).
"""
        self.alternatives_system_instruction = """
You are the "Alternatives Discovery Agent" for Raseed. Your task is to find cheaper alternatives to products and services.
**Your Rules:**
1. Analyze the provided high-cost item and the list of cheaper alternatives found.
2. Your response MUST be a single, valid JSON object with keys: `summary` (a string) and `alternatives` (a list of objects).
"""
        self.budget_optimization_system_instruction = """
You are the "Budget Optimization Agent" for Raseed. Your task is to optimize budget allocation across categories.
**Your Rules:**
1. Analyze the user's current budget allocation and suggest optimizations to meet their savings goal.
2. Your response MUST be a single, valid JSON object with keys: `current_allocation` (a string summary), `optimized_allocation` (a list of suggestions), and `projected_savings` (a number).
"""

    async def generate_behavioral_recommendations(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        user_id = input_data.get('user_id')
        self.logger.info(f"Generating behavioral recommendations for user: {user_id}")
        
        try:
            # The agent now defines and runs its own SQL query using the inherited connector
            query = """
                SELECT EXTRACT(DOW FROM transaction_date) AS purchase_day, merchant_name, category,
                       COUNT(id) AS visit_count, AVG(total_amount) AS average_spend
                FROM receipts
                WHERE user_id = $1 AND transaction_date >= NOW() - INTERVAL '%s days'
                GROUP BY purchase_day, merchant_name, category ORDER BY visit_count DESC;
            """
            lookback_days = input_data.get('lookback_months', 6) * 30
            formatted_query = query.replace('%s', str(lookback_days))
            params = [user_id]
            
            # Use the inherited db_connector
            behavioral_result = await self.db_connector.execute_query(formatted_query, params)
            behavioral_data = behavioral_result.data
            
            if not behavioral_data:
                return {
                    "behavioral_insights": "Insufficient transaction data for analysis.",
                    "recommendations": ["Start tracking more transactions for better insights."],
                    "recommendation_type": "behavioral"
                }
            
            prompt = f"Analyze these spending patterns: {json.dumps(behavioral_data, indent=2)}\nGenerate behavioral recommendations."
            
            synthesis_model = GenerativeModel("gemini-2.0-flash-001", system_instruction=self.behavioral_system_instruction)
            response = await synthesis_model.generate_content_async(prompt, generation_config={"response_mime_type": "application/json"})
            result = json.loads(response.text)
            result["recommendation_type"] = "behavioral"
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate behavioral recommendations: {e}")
            traceback.print_exc()
            return {"behavioral_insights": "Error analyzing patterns.", "recommendations": [], "recommendation_type": "behavioral"}

    async def find_alternatives(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        user_id = input_data.get("user_id")
        self.logger.info(f"Finding alternatives for user: {user_id}")
        
        high_cost_item = input_data.get("spending_analysis", {}).get("high_cost_item_for_recommendation")
        if not high_cost_item:
            return {"summary": "No high-cost items were identified for analysis.", "alternatives": []}

        try:
            # Convert embedding list to PostgreSQL vector format
            embedding_vector = high_cost_item['embedding']
            embedding_str = '[' + ','.join(map(str, embedding_vector)) + ']'
            
            # The agent defines its own vector search query
            # Fixed the vector similarity search syntax
            query = """
                SELECT i.name, i.price, r.merchant_name, 
                    1 - (i.embedding <=> $1::vector(768)) AS similarity
                FROM items i 
                JOIN receipts r ON i.receipt_id = r.id
                WHERE r.category = $2 
                AND i.price < $3 
                AND r.user_id != $4
                AND i.embedding IS NOT NULL
                ORDER BY similarity DESC 
                LIMIT 5;
            """
            
            params = [
                embedding_str,
                high_cost_item.get('category'),
                high_cost_item['price'],
                user_id
            ]
            
            self.logger.info("Executing vector similarity search")
            
            # Use the inherited db_connector
            alternatives_result = await self.db_connector.execute_query(query, params)
            alternatives_found = alternatives_result.data
            
            self.logger.info(f"Found {len(alternatives_found)} alternatives")
            
            if not alternatives_found:
                return {
                    "summary": "No cheaper alternatives found in the database.",
                    "alternatives": [],
                    "recommendation_type": "alternatives"
                }
            
            prompt = f"High-cost item: {json.dumps(high_cost_item, indent=2)}\nAlternatives found: {json.dumps(alternatives_found, indent=2)}\nGenerate recommendations."
           
            synthesis_model = GenerativeModel("gemini-2.0-flash-001", system_instruction=self.alternatives_system_instruction)
            response = await synthesis_model.generate_content_async(prompt, generation_config={"response_mime_type": "application/json"})
            result = json.loads(response.text)
            result["recommendation_type"] = "alternatives"

            self.logger.info(f"Generated recommendations: {len(result.get('alternatives', []))} alternatives")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to find alternatives: {e}")
            traceback.print_exc()
            return {"summary": "An error occurred while analyzing alternatives.", "alternatives": []}

    async def optimize_budget_allocation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        user_id = input_data.get("user_id")
        self.logger.info(f"Optimizing budget for user: {user_id}")
        
        try:
            # The agent defines its own budget analysis query
            query = """
                SELECT category, SUM(total_amount) as total_spent,
                       COUNT(*) as transaction_count,
                       AVG(total_amount) as avg_transaction
                FROM receipts
                WHERE user_id = $1 AND transaction_date >= NOW() - INTERVAL '30 days'
                GROUP BY category 
                ORDER BY total_spent DESC;
            """
            params = [user_id]
            
            # Use the inherited db_connector
            current_allocation_result = await self.db_connector.execute_query(query, params)
            
            if not current_allocation_result.data:
                return {
                    "current_allocation": "No recent transaction data available.",
                    "optimized_allocation": [],
                    "projected_savings": 0,
                    "recommendation_type": "budget_optimization"
                }
            
            prompt = f"""
Current spending allocation: {json.dumps(current_allocation_result.data, indent=2)}
Target category for savings: {input_data.get("target_category", "all categories")}
Savings goal percentage: {input_data.get("savings_goal", 0.1) * 100}%
Provide specific budget reallocation recommendations.
"""
            
            synthesis_model = GenerativeModel("gemini-2.0-flash-001", system_instruction=self.budget_optimization_system_instruction)
            response = await synthesis_model.generate_content_async(prompt, generation_config={"response_mime_type": "application/json"})
            result = json.loads(response.text)
            result["recommendation_type"] = "budget_optimization"
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to optimize budget: {e}")
            traceback.print_exc()
            return {"optimized_allocation": [], "projected_savings": 0, "recommendation_type": "budget_optimization"}

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        The orchestrator will call specific functions directly. This process method
        satisfies the BaseAgent's abstract method requirement and can act as a fallback.
        """
        function_name = input_data.get("function_name", "find_alternatives")
        self.logger.info(f"Generic 'process' called. Routing to function: {function_name}")
        
        if function_name == "generate_behavioral_recommendations":
            return await self.generate_behavioral_recommendations(input_data)
        elif function_name == "optimize_budget_allocation":
            return await self.optimize_budget_allocation(input_data)
        else: # Default to find_alternatives
            return await self.find_alternatives(input_data)