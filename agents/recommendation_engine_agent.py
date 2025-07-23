import logging
import asyncio
import json
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import random
from agents.base_agent import BaseAgent
from core.recommendation_agent_tools.tools_intructions import behavioral_synthesis_instruction,alternatives_synthesis_instruction,budget_optimization_synthesis_instruction,cost_benefit_synthesis_instruction,goal_alignment_synthesis_instruction
from vertexai.generative_models import GenerativeModel, FunctionDeclaration, Tool
from datetime import datetime, timedelta


class RecommendationEngineAgent(BaseAgent):
    """
    An agent specializing in generating personalized recommendations for budgeting,
    cost savings, and financial optimization. It exposes and leverages the following tools:
    - Behavioral Analysis Tool: analyze_behavioral_spending_patterns
    - Alternative Discovery Tool: discover_cheaper_alternatives
    - Budget Optimization Tool: optimize_budget_allocation
    - Cost-Benefit Analysis Tool: perform_cost_benefit_analysis
    - Goal Alignment Tool: align_recommendation_to_financial_goals

    This agent receives prompts from the Master Orchestrator and uses its internal
    LLM (inherited from BaseAgent) to intelligently call these specialized tools.
    """
    def __init__(self, agent_name: str = "recommendation_engine_agent", project_id: str = "massive-incline-466204-t5", location: str = "us-central1", model_name: str = "gemini-2.0-flash-001", user_id: Optional[str] = None):
        """
        Initializes the RecommendationEngineAgent.

        Args:
            project_id: Google Cloud project ID.
            location: Vertex AI location (default: "us-central1").
            model_name: Generative model to use for this agent (default: "gemini-2.0-flash-001").
            user_id: Current user identifier, if available.
        """

        super().__init__(
            agent_name="RecommendationEngineAgent",
            project_id=project_id,
            location=location,
            model_name=model_name,
            user_id=user_id
        )

        # Store System Instructions for synthesis phase
        # These will be used to initialize *new* GenerativeModel instances for synthesis.
        self.behavioral_synthesis_instruction = behavioral_synthesis_instruction
        self.alternatives_synthesis_instruction = alternatives_synthesis_instruction
        self.budget_optimization_synthesis_instruction = budget_optimization_synthesis_instruction
        self.cost_benefit_synthesis_instruction = cost_benefit_synthesis_instruction
        self.goal_alignment_synthesis_instruction = goal_alignment_synthesis_instruction

        # Register the specialized tools. This call will internally trigger self._initialize_model()
        # from BaseAgent, making the main self.model aware of these new tools.
        self._register_recommendation_tools()

        print(f"{self.agent_name} initialized and specialized tools registered.")

    def _register_recommendation_tools(self):
        """
            Registers the five specialized tools for the RecommendationEngineAgent as per the
            architecture document. These FunctionDeclarations allow the main GenerativeModel
            (self.model) to call the agent's internal _execute_* methods.
        """
            # --- 1. Behavioral Analysis Tool ---
        behavioral_analysis_tool = FunctionDeclaration(
                name="analyze_behavioral_spending_patterns",
                # Corrected Description: Focus on data identification and pattern extraction, not recommendations
                description="Analyzes a user's historical spending habits and preferences from transactional data to identify spending patterns, frequencies, and lifestyle factors. Provides raw data on observed behaviors for further analysis.",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "The unique identifier for the user whose spending patterns are to be analyzed."
                        },
                        "lookback_months": {
                            "type": "integer",
                            "description": "The number of months to look back for spending data analysis (e.g., 3, 6, 12). Defaults to 6 months.",
                            "default": 3,
                            "minimum": 1
                        },
                        "category_filter": {
                            "type": "string",
                            "description": "Optional: Focus analysis on a specific spending category (e.g., 'Dining Out', 'Shopping').",
                            "nullable": True
                        }
                    },
                    "required": ["user_id"]
                }
            )

            # --- 2. Alternative Discovery Tool ---
        
        alternative_discovery_tool = FunctionDeclaration(
            name="discover_cheaper_alternatives",
            description=(
                "Finds cheaper alternatives for a product using vector similarity search. "
                "Identifies substitute products or services that offer better prices "
                "compared to a specified high-cost item. This tool requires the embedding "
                "of the item for accurate similarity matching." # Added clarity to description
            ),
            parameters={ 
                "type": "object", 
                "properties": { 
                    "user_id": {
                        "type": "string",
                        "description": "The unique identifier for the user."
                    },
                    "item_description": {
                        "type": "string",
                        "description": "The name or detailed description of the item for which alternatives are sought."
                    },
                    "item_category": {
                        "type": "string",
                        "description": "The spending category of the item (e.g., 'Electronics', 'Groceries', 'Dining Out')."
                    },
                    "price": { 
                        "type": "number", 
                        "format": "float", 
                        "description": "The price of the item to find cheaper alternatives for."
                    },
                    "item_embedding": { 
                        "type": "string",
                        "description": "The pre-generated embedding vector represented as a JSON-formatted string (e.g., '[0.1, -0.2, ...]') for similarity search."
                    },
                    "location_preference": {
                        "type": "string",
                        "description": "Optional: User's preferred geographical area for alternatives (e.g., 'nearby', 'city_name').",
                        "nullable": True # Add nullable if it's optional
                    }
                },
                "required": ["user_id", "item_description", "item_category", "price", "item_embedding"]
            }
        )


            # --- 3. Budget Optimization Tool ---
        budget_optimization_tool = FunctionDeclaration(
                name="optimize_budget_allocation",
                # Corrected Description: Focus on generating allocation data, not "recommendations" as a final output
                description="Analyzes a user's spending history and financial goals to generate optimized budget allocations across spending categories. Identifies areas of over or underspending and proposes adjusted budget figures.",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "The unique identifier for the user whose budget is to be optimized."
                        },
                        "financial_goal": {
                            "type": "string",
                            "description": "A specific financial goal to align the budget optimization with (e.g., 'save for down payment', 'pay off debt').",
                            
                        },
                        "target_amount": {
                            "type": "number",
                            "format": "float",
                            "description": "A specific amount the user aims to save monthly/annually. Provide either this or `savings_target_percentage`.",
                            
                        },
                        "current_amount": {
                            "type": "number",
                            "format": "float",
                            "description": "Amount saved till now.",
                            
                        },
                        "focus_category": {
                            "type": "string",
                            "description": "Optional: A specific category to focus budget adjustments on (e.g., 'Groceries', 'Entertainment').",
                            "nullable": True
                        }
                    },
                    "required": ["user_id","financial_goal","target_amount","current_amount"]
                }
            )

            # --- 4. Cost-Benefit Analysis Tool ---
        cost_benefit_analysis_tool = FunctionDeclaration(
                name="perform_cost_benefit_analysis",
                # Corrected Description: Focus on calculation and analysis, not "assessing impact of adopting a recommendation" (that's synthesis)
                description="Calculates the potential financial savings and costs associated with a proposed financial change or action. Provides quantitative metrics like net savings, payback period, and lists qualitative trade-offs.",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "The unique identifier for the user for whom the analysis is being performed."
                        },
                        "recommendation_summary": {
                            "type": "string",
                            "description": "A brief description of the recommendation or change being analyzed (e.g., 'cancel streaming service', 'switch to generic brands')."
                        },
                        "current_cost_per_period": {
                            "type": "number",
                            "format": "float",
                            "description": "The current cost associated with the item/behavior being analyzed (e.g., 15.99 for a subscription, 500 for monthly dining out)."
                        },
                        "estimated_new_cost_per_period": {
                            "type": "number",
                            "format": "float",
                            "description": "The estimated cost after adopting the recommendation (e.g., 0 after cancellation, 300 after reducing dining out).",
                            "nullable": True
                        },
                        "initial_investment_required": {
                            "type": "number",
                            "format": "float",
                            "description": "Any one-time upfront cost required to adopt the recommendation (e.g., buying new equipment). Defaults to 0.",
                            "default": 0.0
                        },
                        "period_unit": {
                            "type": "string",
                            "description": "The unit of the period for costs/savings (e.g., 'month', 'year', 'week').",
                            "enum": ["day", "week", "month", "quarter", "year"],
                            "default": "month"
                        },
                        "analysis_duration_periods": {
                            "type": "integer",
                            "description": "The number of periods over which to perform the analysis (e.g., 12 for 12 months). Defaults to 12.",
                            "default": 12,
                            "minimum": 1
                        },
                        "qualitative_trade_offs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional: Key qualitative factors or trade-offs to consider (e.g., 'loss of convenience', 'improved health').",
                            "default": []
                        }
                    },
                    "required": ["user_id", "recommendation_summary", "current_cost_per_period"]
                }
            )

            # --- 5. Goal Alignment Tool ---
        goal_alignment_tool = FunctionDeclaration(
                name="align_recommendation_to_financial_goals",
                # Corrected Description: Focus on assessing alignment and providing data points, not "balancing savings with lifestyle" or "tracking progress" (which are synthesis tasks)
                description="Evaluates how a given financial action or recommendation aligns with a user's stated financial goals. Provides data points on goal contribution and suggests specific steps to improve alignment.",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "The unique identifier for the user whose goals are to be aligned with recommendations."
                        },
                        "recommendation_details": {
                            "type": "string",
                            "description": "A clear description of the recommendation or financial action being assessed (e.g., 'reducing dining out by $50/month', 'investing $200 in a retirement fund')."
                        },
                        "impact_estimate": {
                            "type": "string",
                            "description": "Optional: An estimated financial impact of the recommendation (e.g., '$50 monthly savings', '$1000 annual growth').",
                            "nullable": True
                        },
                        "relevant_goals": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional: Specific financial goals to evaluate against (e.g., 'down payment', 'debt repayment'). If not provided, general goals from user profile will be used.",
                            "nullable": True
                        },
                        "goal_timeframe": {
                            "type": "string",
                            "description": "Optional: The timeframe associated with the relevant goals (e.g., 'short-term', 'long-term', 'next 5 years').",
                            "nullable": True
                        }
                    },
                    "required": ["user_id", "recommendation_details"]
                }
            )

        # Register these declarations with the base agent's tool registry.
        # This will internally cause self.model to be re-initialized with these new tools.
        self.register_tool(behavioral_analysis_tool, self._execute_analyze_behavioral_spending_patterns)
        self.register_tool(alternative_discovery_tool, self._execute_discover_cheaper_alternatives)
        self.register_tool(budget_optimization_tool, self._execute_optimize_budget_allocation)

        # maybe later
        #self.register_tool(cost_benefit_analysis_tool, self._execute_perform_cost_benefit_analysis)
        #self.register_tool(goal_alignment_tool, self._execute_align_recommendation_to_financial_goals)

    async def _execute_analyze_behavioral_spending_patterns(
        self, 
        user_id: str, 
        lookback_months: int = 6, 
        category_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes the SQL query for behavioral spending patterns based on tool parameters,
        then synthesizes insights with a dedicated LLM instance using a specific system instruction.
        """
        print(f"Executing _execute_analyze_behavioral_spending_patterns for user: {user_id}, lookback: {lookback_months} months, category: {category_filter}")
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=lookback_months * 30)
            # Base query
            query = """
                SELECT
                EXTRACT(DOW FROM t.transaction_date) AS purchase_day, -- Day of Week (0=Sunday, 6=Saturday)
                COALESCE(m.name, 'Unknown Merchant') AS merchant_name,
                t.category,
                COUNT(t.transaction_id) AS visit_count,
                AVG(t.amount) AS average_spend,
                SUM(t.amount) AS total_category_spend
            FROM transactions t
            LEFT JOIN merchants m ON t.merchant_id = m.merchant_id
            WHERE
                t.user_id = $1
                AND t.transaction_date >= $2
            
            GROUP BY
                purchase_day,
                m.name,
                t.category
            ORDER BY
                total_category_spend DESC,
                visit_count DESC;
            """
            
            # Set up parameters
            params = [user_id,cutoff_date]
            
            behavioral_result = await self.db_connector.execute_query(query, params)
            behavioral_data = getattr(behavioral_result, 'data', [])
            
            # Check if we have data
            if not behavioral_data:
                return {
                    "behavioral_patterns": "No spending patterns found for the specified period",
                    "raw_data_summary": f"No transactions found for user {user_id} in the last {lookback_months} months",
                    "potential_areas_for_recommendation": ["Insufficient data for analysis"]
                }
            
            # Prepare prompt with correct variable name
            prompt = f"""
            Raw behavioral spending data for user {user_id}:
            ```json
            {json.dumps(behavioral_data, indent=2, default=str)}
            ```
            """
            
            # Create synthesis model instance 
            synthesis_model = GenerativeModel(
                self.model_name,
                system_instruction=self.behavioral_synthesis_instruction
            )
            
            response = await synthesis_model.generate_content_async(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                }
            )
            # Parse and return result
            result = json.loads(response.text)
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return {
                "behavioral_patterns": "Error processing behavioral analysis",
                "raw_data_summary": "Failed to parse LLM response",
                "key_metrics": {"error": str(e)},
                "potential_areas_for_recommendation": ["Analysis unavailable due to processing error"]
            }
        except Exception as e:
            print(f"Error in behavioral spending analysis: {e}")
            return {
                "behavioral_patterns": "Error analyzing spending patterns",
                "raw_data_summary": f"Analysis failed: {str(e)}",
                "key_metrics": {"error": str(e)},
                "potential_areas_for_recommendation": ["Analysis unavailable due to system error"]
            }

    async def _execute_discover_cheaper_alternatives(
            self,
            user_id: str,
            item_description: str,
            item_category: str,
            price: float,
            item_embedding: str,
            location_preference: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            Executes the vector similarity search for alternatives based on tool parameters,
            then synthesizes recommendations with a dedicated LLM instance.
            """
            print(f"Executing _execute_discover_cheaper_alternatives for user: {user_id}, item: {item_description}, category: {item_category}")

            try:
                
                if isinstance(item_embedding, str):
                    item_embedding = item_embedding
                else:
                    item_embedding = '[' + ','.join(map(str, item_embedding)) + ']'
                
                # Prepare query
                query = """
                    SELECT DISTINCT ON (ti.name, ti.total_price, m.name)
                    ti.name AS item_name,
                    ti.total_price AS alternative_price,
                    m.name AS merchant_name,
                    (1 - (ti.item_embedding <=> $1::vector(768))) AS similarity_score -- Cosine similarity
                FROM
                    transaction_items ti
                JOIN
                    transactions t ON ti.transaction_id = t.transaction_id
                JOIN
                    merchants m ON t.merchant_id = m.merchant_id
                WHERE
                    t.category ILIKE $2          -- Filter by category (case-insensitive)
                    AND ti.unit_price < $3   
                    AND t.user_id != $4
                    AND ti.item_embedding IS NOT NULL -- Ensure the item has an embedding
                ORDER BY
                    ti.name, ti.total_price, m.name,
                    similarity_score DESC,        -- Most similar items first
                    alternative_price ASC         -- Then by lowest price
                LIMIT 10; 
                """
                
                # Set params
                params = [item_embedding, item_category, price, user_id]
                
                # Execute query
                alternatives_result = await self.db_connector.execute_query(query, params)
                alternatives_found = getattr(alternatives_result,"data")

                if not alternatives_found:
                    return {
                        "raw_data_summary": f"No cheaper alternatives found for user {user_id}",
                        "potential_areas_for_recommendation": ["No similar but cheaper items found in category"]
                    }

                print(f"Found {len(alternatives_found)} alternatives for {item_description}")

                # Prompt
                prompt = (
                    f"High-cost item:\n```json\n{json.dumps({'item_description': item_description, 'category': item_category, 'price': price}, indent=2)}\n```\n"
                    f"Alternatives found:\n```json\n{json.dumps(alternatives_found, indent=2)}\n```\n"
                )

                # Initialize model and get synthesis
                synthesis_model = GenerativeModel(
                    self.model_name,
                    system_instruction=self.alternatives_synthesis_instruction
                )
                response = await synthesis_model.generate_content_async(
                    prompt,
                    generation_config={"response_mime_type": "application/json"}
                )

                result = json.loads(response.text)
                return result

            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                return {
                    "raw_data_summary": "Failed to parse LLM response",
                    "potential_areas_for_recommendation": ["Analysis unavailable due to processing error"]
                }

            except Exception as e:
                print(f"Unexpected error during alternative discovery: {e}")
                return {
                    "raw_data_summary": "Error occurred during recommendation discovery",
                    "potential_areas_for_recommendation": ["System error"]
                }

    async def _execute_optimize_budget_allocation(
        self, 
        user_id: str, 
        financial_goal: str, 
        target_amount: float, 
        current_amount: float, 
        focus_category: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes budget analysis based on tool parameters, then synthesizes optimization
        recommendations with a dedicated LLM instance.
        """
        print(f"Executing _execute_optimize_budget_allocation for user: {user_id}, goal: {financial_goal}, target: {target_amount}")

        try:

            query = """
                SELECT category, SUM(amount) as total_spent,
                       COUNT(*) as transaction_count,
                       AVG(amount) as avg_transaction
                FROM transactions
                WHERE user_id = $1
                GROUP BY category
                ORDER BY total_spent DESC;
            """
            params = [user_id]
            current_allocation_result = await self.db_connector.execute_query(query, params)
            current_spending_data = getattr(current_allocation_result,'data')

            # Fetch user goals if applicable, using UserProfileManager tool (a BaseAgent tool)
            user_profile = await self.user_profile_manager.get_profile(user_id, ["financial_goals"])
            user_financial_goals = getattr(user_profile,"financial_goals",[])

            prompt = f"""
                Current spending allocation for user {user_id}:\n```json\n{json.dumps(current_spending_data, indent=2)}\n```
                Financial goal: {financial_goal or 'N/A'} (User's broader goals: {user_financial_goals})
                Savings target amount: {target_amount or 'N/A'}
                Savings current amount: {current_amount or 'N/A'}
                Focus category for optimization: {focus_category or 'all categories'}
                Generate specific budget reallocation recommendations based on the system instructions.
                """

            print(prompt)
            # Create a NEW GenerativeModel instance for this specific synthesis task
            synthesis_model = GenerativeModel(
                self.model_name,
                system_instruction=self.budget_optimization_synthesis_instruction
            )
            response = await synthesis_model.generate_content_async(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )

            # Parse and Return Result
            result = json.loads(response.text)
            return result

        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error from synthesis model for budget optimization: {e}\nResponse text: '{response.text}'")
            self.error_handler.handle_error(
                    error=e,
                    context=f"Failed to parse LLM synthesis response for budget optimization for user {user_id}",
                    user_id=user_id
                )
            return {
                    "current_allocation_summary": "Analysis unavailable due to an internal processing error.",
                    "optimized_allocation_suggestions": [],
                    "projected_savings": 0.0,
                    "status": "error",
                    "message": "Failed to parse LLM synthesis response."
                }
        except Exception as e:
            logging.error(f"Unexpected error during budget optimization: {traceback.format_exc()}")
            self.error_handler.handle_error(
                    error=e,
                    context=f"Unexpected error during budget optimization for user {user_id}",
                    user_id=user_id
                )
            return {
                    "current_allocation_summary": "An unexpected system error occurred during budget optimization.",
                    "optimized_allocation_suggestions": [],
                    "projected_savings": 0.0,
                    "status": "error",
                    "message": f"An unexpected error occurred: {str(e)}"
                }


    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a recommendation request by leveraging the generative model for tool calling
        and subsequent synthesis. This method acts as the entry point for the orchestrator.

        Args:
            request: A dictionary containing the user's query ('prompt') and any relevant
                     context for tool parameters, including 'user_id'. This 'prompt'
                     will be interpreted by the agent's LLM to determine tool usage.

        Returns:
            Dictionary with processing results, including the synthesized recommendation.
        """
        user_id = request.get("user_id")
        prompt_text = request.get("prompt")
        context_data = request.get("context", {}) # Additional data the model might need for tool parameters

        if user_id:
            context_data['user_id'] = user_id
        if not user_id or not prompt_text:
            ("Missing 'user_id' or 'prompt' in request to RecommendationEngineAgent.process")
            return {"status": "error", "message": "Missing 'user_id' or 'prompt' in request."}

        # Set user context for base agent's error handling and user profile manager
        self.set_user_context(user_id)

        print(f"RecommendationEngineAgent processing request for user {self.user_id} with prompt: '{prompt_text[:100]}...'")

        try:

            input_content = [
                {"role": "user", 
                "parts": [{"text": prompt_text,}]
                }
            ]
            if context_data:
                input_content[0]["parts"].append({"text": f"\n\nAdditional context for analysis:\n```json\n{json.dumps(context_data, indent=2)}\n```"})

            initial_response = await self.model.generate_content_async(input_content)

            # 2. Handle potential tool calls from the model's response
            if initial_response.candidates and initial_response.candidates[0].content.parts:
                for part in initial_response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_call = part.function_call
                        print(f"RecommendationEngineAgent: Model called tool: {function_call.name} with args: {function_call.args}")

                        tool_output = await self.execute_tool_call(function_call)

                        if tool_output.get("success"):
                            print(f"RecommendationEngineAgent: Tool {function_call.name} executed successfully.")

                            return {
                                "status": "success",
                                "tool_executed": function_call.name,
                                "tool_raw_output": tool_output
                            }
                        else:
                            (f"RecommendationEngineAgent: Tool {function_call.name} execution failed: {tool_output.get('error')}")
                            return {
                                "status": "error",
                                "message": f"Failed to get recommendation: Tool '{function_call.name}' failed with error: {tool_output.get('error')}",
                                "tool_attempted": function_call.name
                            }

            # If no tool was called, the main model generated a direct text response.
            print("RecommendationEngineAgent: Model generated direct response (no tool call).")
            return {
                "status": "success",
                "recommendation_text": initial_response.text if initial_response.text else "No specific recommendation generated.",
                "tool_executed": None
            }

        except Exception as e:
            self.error_handler.handle_error(
                error=e,
                context=f"Error in RecommendationEngineAgent.process for user {user_id} with prompt: {prompt_text}",
                user_id=user_id
            )
            return {"status": "error", "message": "An internal error occurred while processing your request."}