import logging
import asyncio
import asyncpg
import json
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import random
from agents.base_agent import BaseAgent
from core.recommendation_agent_tools.tools_intructions import behavioral_synthesis_instruction,alternatives_synthesis_instruction,budget_optimization_synthesis_instruction,cost_benefit_synthesis_instruction,goal_alignment_synthesis_instruction
from vertexai.generative_models import GenerativeModel, FunctionDeclaration, Tool
from datetime import datetime, timedelta
from google.cloud import secretmanager

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
        self.connection_pool: Optional[asyncpg.Pool] = None
        self.behavioral_synthesis_instruction = behavioral_synthesis_instruction
        self.alternatives_synthesis_instruction = alternatives_synthesis_instruction
        self.budget_optimization_synthesis_instruction = budget_optimization_synthesis_instruction
        self.cost_benefit_synthesis_instruction = cost_benefit_synthesis_instruction
        self.goal_alignment_synthesis_instruction = goal_alignment_synthesis_instruction
        self.secret_client = secretmanager.SecretManagerServiceClient()
        secret_name = f"projects/{self.project_id}/secrets/postgres-config/versions/latest"
        response = self.secret_client.access_secret_version(request={"name": secret_name})
        secret_data = json.loads(response.payload.data.decode("UTF-8"))
        self.db_config = {
            'host': secret_data["host"],
            'port': secret_data.get("port", 5432),
            'database': secret_data["database"],
            'user': secret_data["user"],
            'password': secret_data["password"],
            'min_size': 5,
            'max_size': 20,
            'command_timeout': 60
        }
        # Register the specialized tools. This call will internally trigger self._initialize_model()
        # from BaseAgent, making the main self.model aware of these new tools.
        self._register_recommendation_tools()

        print(f"{self.agent_name} initialized and specialized tools registered.")
    
    async def initialize_connection_pool(self):
        """Initialize the database connection pool."""
        try:
            if not self.connection_pool:
                self.connection_pool = await asyncpg.create_pool(**self.db_config)
                self.logger.info("PostgreSQL connection pool initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    async def close_connection_pool(self):
        """Close the database connection pool."""
        if self.connection_pool:
            await self.connection_pool.close()
            self.connection_pool = None
            self.logger.info("PostgreSQL connection pool closed")

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
                    name = "cost_benefit_analysis",
                    description = "Quantifies financial impact of cost-saving recommendations using structured analysis",
                    parameters = {
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "User identifier for personalized analysis"
                            },
                            "spending_analysis": {
                                "type": "object",
                                "description": "Financial analysis results from previous workflow step containing spending patterns and category breakdowns"
                            },
                            "user_goals": {
                                "type": "object",
                                "description": "User's financial goals and preferences for recommendation targeting",
                                "nullable": True
                            },
                            "original_query": {
                                "type": "string", 
                                "description": "Original user query for context-aware recommendations",
                                "nullable": True
                            }
                        },
                        "required": ["user_id", "spending_analysis"]
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
        self.register_tool(cost_benefit_analysis_tool, self._execute_cost_benefit_analysis)

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
            async with self.connection_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
            behavioral_result = rows
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
                async with self.connection_pool.acquire() as conn:
                    rows = await conn.fetch(query, *params)
                alternatives_result = rows
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
            async with self.connection_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
            current_allocation_result = rows
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

    async def _execute_cost_benefit_analysis(
        self, 
        user_id: str, 
        spending_analysis: Dict[str, Any],
        user_goals: Optional[Dict[str, Any]] = None,
        original_query: str = "") -> Dict[str, Any]:
        """
        Executes cost-benefit analysis within the generate_recommendations step.
        Processes spending analysis data to generate financially quantified recommendations.
        """
        print(f"Executing cost-benefit analysis for recommendations - user: {user_id}")

        try:
            # Extract transaction data from financial_analysis_agent output
            transaction_data = spending_analysis.get('data', [])
            
            # Aggregate transactions by category
            category_totals = {}
            total_spending = 0
            
            for transaction in transaction_data:
                category = transaction.get('category', 'uncategorized')
                amount = float(transaction.get('amount', 0))
                
                if category in category_totals:
                    category_totals[category]['total'] += amount
                    category_totals[category]['count'] += 1
                    category_totals[category]['transactions'].append(transaction)
                else:
                    category_totals[category] = {
                        'total': amount,
                        'count': 1,
                        'transactions': [transaction]
                    }
                total_spending += amount
            
            # Convert to list and identify high-spend categories
            categories = [
                {
                    'name': cat, 
                    'amount': data['total'],
                    'transaction_count': data['count'],
                    'avg_transaction': data['total'] / data['count'],
                    'transactions': data['transactions']
                } 
                for cat, data in category_totals.items()
            ]
            
            # Identify potential cost-saving opportunities (>15% of total spending)
            high_spend_categories = [cat for cat in categories 
                                if cat.get('amount', 0) > total_spending * 0.10]
            
            # Generate cost-benefit scenarios for each opportunity
            cost_benefit_scenarios = []
            
            for category in high_spend_categories:
                category_name = category.get('name', 'Unknown')
                current_monthly_spend = category.get('amount', 0)
                transaction_count = category.get('transaction_count', 0)
                avg_transaction = category.get('avg_transaction', 0)
                
                # Generate optimization scenarios based on spending patterns
                scenarios = self._generate_optimization_scenarios(
                    category_name, 
                    current_monthly_spend, 
                    transaction_count,
                    avg_transaction
                )
                
                for scenario in scenarios:
                    scenario_data = {
                        "analysis_target": scenario['description'],
                        "current_cost_per_period": current_monthly_spend,
                        "estimated_new_cost_per_period": scenario['projected_cost'],
                        "initial_investment_required": scenario.get('setup_cost', 0),
                        "analysis_duration_months": 12,
                        "cost_period_type": "monthly",
                        "qualitative_factors": scenario.get('benefits', [])
                    }
                    
                    cost_benefit_scenarios.append(scenario_data)

            # Process each scenario through cost-benefit analysis
            analyzed_recommendations = []
            
            for scenario in cost_benefit_scenarios:
                prompt = f"""
                Process this cost-saving scenario for financial quantification:
                
                Raw Input Data:
                ```json
                {json.dumps(scenario, indent=2)}
                ```
                
                User Context:
                - User ID: {user_id}
                - Original Query: {original_query}
                - Financial Goals: {user_goals or 'Not specified'}
                
                Apply cost-benefit analysis processing to generate structured recommendation data.
                """

                # Use cost-benefit synthesis model
                synthesis_model = GenerativeModel(
                    self.model_name,
                    system_instruction=self.cost_benefit_synthesis_instruction
                )
                
                response = await synthesis_model.generate_content_async(
                    prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                
                cost_benefit_result = json.loads(response.text)
                
                # Enhance with recommendation metadata
                recommendation = {
                    "id": f"rec_{user_id}_{len(analyzed_recommendations)}",
                    "type": "cost_optimization",
                    "priority": self._calculate_priority(cost_benefit_result),
                    "cost_benefit_analysis": cost_benefit_result,
                    "financial_impact": cost_benefit_result.get('financial_metrics', {}),
                    "implementation_complexity": scenario.get('complexity', 'medium'),
                    "category": scenario.get('category', 'general')
                }
                
                analyzed_recommendations.append(recommendation)

            # Sort recommendations by financial impact
            sorted_recommendations = sorted(
                analyzed_recommendations,
                key=lambda x: x['financial_impact'].get('net_financial_impact_over_duration', 0),
                reverse=True
            )

            return {
                "recommendations": sorted_recommendations[:7],  # Top 5 recommendations
                "total_scenarios_analyzed": len(cost_benefit_scenarios),
                "aggregate_potential_savings": sum(
                    rec['financial_impact'].get('net_financial_impact_over_duration', 0)
                    for rec in sorted_recommendations
                ),
                "analysis_metadata": {
                    "user_id": user_id,
                    "based_on_spending_analysis": True,
                    "scenarios_generated": len(cost_benefit_scenarios),
                    "high_impact_recommendations": len([r for r in sorted_recommendations 
                                                    if r['financial_impact'].get('net_financial_impact_over_duration', 0) > 500])
                }
            }

        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error in cost-benefit recommendations for user {user_id}: {e}")
            return self._return_error_response(user_id, "Failed to parse cost-benefit analysis")

        except Exception as e:
            logging.error(f"Error in cost-benefit recommendations: {traceback.format_exc()}")
            return self._return_error_response(user_id, f"Cost-benefit analysis failed: {str(e)}")


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
        prompt_text = request.get("query")
        context_data = request.get("spending_analysis", {})  # Additional data the model might need for tool parameters

        if user_id:
            context_data['user_id'] = user_id
        if not user_id or not prompt_text:
            print("Missing 'user_id' or 'prompt' in request to RecommendationEngineAgent.process")
            return {"status": "error", "message": "Missing 'user_id' or 'prompt' in request."}

        # Set user context for base agent's error handling and user profile manager
        self.set_user_context(user_id)

        print(f"RecommendationEngineAgent processing request for user {self.user_id} with prompt: '{prompt_text[:100]}...'")

        try:
            # Fix: Include context data directly in the text prompt instead of as a separate part
            full_prompt = prompt_text
            if context_data:
                # Handle UUID serialization by using default=str
                full_prompt += f"\n\nAdditional Context: {json.dumps(context_data, indent=2, default=str)}"

            input_content = [
                {"role": "user", 
                "parts": [{"text": full_prompt}]
                }
            ]

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
                            print(f"RecommendationEngineAgent: Tool {function_call.name} execution failed: {tool_output.get('error')}")
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

    #helper cost_benefit and budget
    def _generate_optimization_scenarios(self, category: str, current_spend: float, transaction_count: int, avg_transaction: float) -> List[Dict]:
        """Generate potential cost-saving scenarios based on spending category and transaction patterns"""
        scenarios = []
        
        category_lower = category.lower()
        
        if 'food' in category_lower or 'dining' in category_lower:
            scenarios.extend([
                {
                    "description": f"Reduce {category} spending through meal planning and home cooking",
                    "projected_cost": current_spend * 0.7,
                    "setup_cost": 0,
                    "benefits": ["healthier eating", "reduced food waste", "cooking skills"],
                    "complexity": "low",
                    "category": category
                },
                {
                    "description": f"Switch to bulk buying and wholesale for {category}",
                    "projected_cost": current_spend * 0.8,
                    "setup_cost": 50,
                    "benefits": ["cost savings", "fewer shopping trips", "bulk discounts"],
                    "complexity": "medium",
                    "category": category
                }
            ])
        
        elif 'shopping' in category_lower:
            # High frequency = impulse buying opportunity
            if transaction_count > 10:
                scenarios.append({
                    "description": f"Implement 24-hour waiting period for {category} purchases",
                    "projected_cost": current_spend * 0.6,
                    "setup_cost": 0,
                    "benefits": ["reduced impulse buying", "more thoughtful purchases"],
                    "complexity": "low",
                    "category": category
                })
            
            # High average transaction = look for alternatives  
            if avg_transaction > 1000:
                scenarios.append({
                    "description": f"Research alternatives and compare prices for {category}",
                    "projected_cost": current_spend * 0.85,
                    "setup_cost": 0,
                    "benefits": ["better value", "price awareness"],
                    "complexity": "medium",
                    "category": category
                })
        
        elif 'transport' in category_lower or 'gas' in category_lower or 'fuel' in category_lower:
            scenarios.extend([
                {
                    "description": f"Optimize {category} through carpooling and public transport",
                    "projected_cost": current_spend * 0.6,
                    "setup_cost": 0,
                    "benefits": ["environmental impact", "reduced stress", "social connections"],
                    "complexity": "medium",
                    "category": category
                },
                {
                    "description": f"Consolidate trips and improve route planning for {category}",
                    "projected_cost": current_spend * 0.8,
                    "setup_cost": 0,
                    "benefits": ["time savings", "reduced wear and tear"],
                    "complexity": "low",
                    "category": category
                }
            ])
        
        elif 'subscription' in category_lower or 'entertainment' in category_lower:
            scenarios.extend([
                {
                    "description": f"Audit and cancel unused {category} services",
                    "projected_cost": current_spend * 0.5,
                    "setup_cost": 0,
                    "benefits": ["simplified finances", "reduced digital clutter", "focus on used services"],
                    "complexity": "low",
                    "category": category
                },
                {
                    "description": f"Switch to family/shared plans for {category}",
                    "projected_cost": current_spend * 0.7,
                    "setup_cost": 0,
                    "benefits": ["cost sharing", "family coordination"],
                    "complexity": "medium",
                    "category": category
                }
            ])
        
        # Generic scenarios based on transaction patterns
        if transaction_count > 10:  # High frequency spending
            scenarios.append({
                "description": f"Set monthly budget limit for {category} spending",
                "projected_cost": current_spend * 0.8,
                "setup_cost": 0,
                "benefits": ["budget discipline", "spending awareness"],
                "complexity": "low",
                "category": category
            })
        
        if avg_transaction > 500:  # High-value transactions
            scenarios.append({
                "description": f"Implement approval process for {category} purchases over ₹500",
                "projected_cost": current_spend * 0.85,
                "setup_cost": 0,
                "benefits": ["thoughtful spending", "avoiding buyer's remorse"],
                "complexity": "low",
                "category": category
            })
        
        # Always include a generic optimization scenario
        scenarios.append({
            "description": f"General optimization of {category} spending habits",
            "projected_cost": current_spend * 0.85,
            "setup_cost": 0,
            "benefits": ["improved budgeting", "increased savings awareness"],
            "complexity": "low",
            "category": category
        })
        
        return scenarios

    def _calculate_priority(self, cost_benefit_result: Dict) -> str:
        """Calculate recommendation priority based on cost-benefit metrics"""
        financial_metrics = cost_benefit_result.get('financial_metrics', {})
        net_impact = financial_metrics.get('net_financial_impact_over_duration', 0)
        payback_months = financial_metrics.get('payback_period_months', float('inf'))
        
        if net_impact > 1000 and payback_months < 6:
            return "high"
        elif net_impact > 500 and payback_months < 12:
            return "medium"
        else:
            return "low"

    def _return_error_response(self, user_id: str, message: str) -> Dict:
        """Return standardized error response for recommendations"""
        return {
            "recommendations": [],
            "total_scenarios_analyzed": 0,
            "aggregate_potential_savings": 0,
            "analysis_metadata": {
                "user_id": user_id,
                "error": True,
                "error_message": message
            }
        }