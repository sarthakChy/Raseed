import logging
import asyncio
import json
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import random
import numpy as np
from agents.base_agent import BaseAgent
from vertexai.generative_models import GenerativeModel, FunctionDeclaration, Tool
from utils.database_connector import DatabaseConnector


class RecommendationEngineAgent(BaseAgent):
    """
    Main recommendation engine agent that coordinates different recommendation tools.
    """
    def __init__(self, agent_name: str = "recommendation_engine_agent", project_id: str = "massive-incline-466204-t5", location: str = "us-central1", model_name: str = "gemini-2.0-flash-001", user_id: Optional[str] = None):
        super().__init__(
            agent_name="RecommendationEngineAgent",
            project_id=project_id,
            location=location,
            model_name=model_name,
            user_id=user_id
        )

        self.logger.info(f"Initializing RecommendationEngineAgent with project_id={project_id}")
        
        # Initialize specialized tools
        self.behavioral_analysis_tool = BehavioralAnalysisTool(project_id=project_id, logger=self.logger)
        self.alternative_discovery_tool = AlternativeDiscoveryTool(project_id=project_id, logger=self.logger)
        self.budget_optimization_tool = BudgetOptimizationTool(project_id=project_id, logger=self.logger)
        self.cost_benefit_analysis_tool = CostBenefitAnalysisTool(project_id=project_id, logger=self.logger)
        self.goal_alignment_tool = GoalAlignmentTool(project_id=project_id, logger=self.logger)

        self._register_recommendation_tools()
        self.analysis_cache = {}

    def _register_recommendation_tools(self):
        """Register all recommendation tools with Vertex AI."""
        function_declarations = [
            FunctionDeclaration(
                name="analyze_behavioral_spending_patterns",
                description="Analyzes user's historical spending habits to identify patterns and optimization opportunities.",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "User identifier"},
                        "lookback_months": {"type": "integer", "description": "Months to analyze", "default": 6},
                        "category_filter": {"type": "string", "description": "Optional category filter", "nullable": True}
                    },
                    "required": ["user_id"]
                }
            ),
            FunctionDeclaration(
                name="discover_cheaper_alternatives",
                description="Finds cheaper alternatives using vector similarity search.",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "item_description": {"type": "string"},
                        "item_category": {"type": "string"},
                        "price": {"type": "number"},
                        "location_preference": {"type": "string", "nullable": True}
                    },
                    "required": ["user_id", "item_description", "item_category", "price"]
                }
            ),
            FunctionDeclaration(
                name="optimize_budget_allocation",
                description="Optimizes budget allocation based on financial goals.",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "financial_goal": {"type": "string"},
                        "target_amount": {"type": "number"},
                        "current_amount": {"type": "number"},
                        "focus_category": {"type": "string", "nullable": True}
                    },
                    "required": ["user_id", "financial_goal", "target_amount", "current_amount"]
                }
            ),
            FunctionDeclaration(
                name="perform_cost_benefit_analysis",
                description="Performs comprehensive cost-benefit analysis with embedding-based item matching.",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "spending_analysis": {"type": "object"},
                        "user_goals": {"type": "object", "nullable": True},
                        "original_query": {"type": "string", "nullable": True}
                    },
                    "required": ["user_id", "spending_analysis"]
                }
            ),
            FunctionDeclaration(
                name="align_recommendation_to_financial_goals",
                description="Aligns recommendations with user's financial goals.",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "recommendation_details": {"type": "string"},
                        "impact_estimate": {"type": "string", "nullable": True},
                        "relevant_goals": {"type": "array", "items": {"type": "string"}, "nullable": True}
                    },
                    "required": ["user_id", "recommendation_details"]
                }
            )
        ]

        # Create a single tool with all function declarations
        tool = Tool(function_declarations=function_declarations)
        
        # Configure the model with the tool
        self.model = GenerativeModel(
            model_name=self.model_name,
            tools=[tool],
            system_instruction="""You are a financial recommendation assistant. Analyze spending patterns and provide personalized recommendations for budget optimization. 

When you receive spending data, analyze it and call the appropriate functions to provide insights. Start with behavioral analysis, then consider alternatives and optimizations."""
        )

        # Map function names to their executors
        self.function_executors = {
            "analyze_behavioral_spending_patterns": self.behavioral_analysis_tool.analyze_patterns,
            "discover_cheaper_alternatives": self.alternative_discovery_tool.discover_alternatives,
            "optimize_budget_allocation": self.budget_optimization_tool.optimize_allocation,
            "perform_cost_benefit_analysis": self.cost_benefit_analysis_tool.analyze_recommendations,
            "align_recommendation_to_financial_goals": self.goal_alignment_tool.align_with_goals
        }

    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing method for recommendation requests."""
        try:
            user_id = request.get("user_id")
            prompt_text = request.get("query", "")
            context_data = request.get("spending_analysis", {})

            if not user_id:
                return {"status": "error", "message": "Missing 'user_id' in request."}

            self.set_user_context(user_id)

            # Prepare the input content with spending analysis
            analysis_summary = self._prepare_spending_summary(context_data)
            
            input_prompt = f"""
            Analyze the following spending data and provide recommendations:
            
            User Query: {prompt_text}
            
            Spending Summary:
            {analysis_summary}
            
            Please analyze this data and call the appropriate functions to provide insights and recommendations.
            """

            # Generate content with function calling
            response = await self.model.generate_content_async(input_prompt)
            
            # Process function calls if any
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        # Execute the function call
                        function_name = part.function_call.name
                        function_args = {}
                        
                        # Extract arguments from function call
                        for key, value in part.function_call.args.items():
                            function_args[key] = value
                        
                        # Add user_id if not present
                        if 'user_id' not in function_args:
                            function_args['user_id'] = user_id
                        
                        # Add spending_analysis if calling cost-benefit analysis
                        if function_name == "perform_cost_benefit_analysis" and 'spending_analysis' not in function_args:
                            function_args['spending_analysis'] = context_data
                        
                        # Execute the function
                        if function_name in self.function_executors:
                            try:
                                tool_output = await self.function_executors[function_name](**function_args)
                                
                                return {
                                    "status": "success",
                                    "tool_executed": function_name,
                                    "tool_raw_output": tool_output,
                                    "recommendations": tool_output.get("recommendations", []) if isinstance(tool_output, dict) else [],
                                    "insights": tool_output.get("insights", "") if isinstance(tool_output, dict) else str(tool_output)
                                }
                            except Exception as e:
                                self.logger.error(f"Error executing function {function_name}: {e}")
                                return {
                                    "status": "error",
                                    "message": f"Tool execution failed: {str(e)}",
                                    "tool_attempted": function_name
                                }

            # If no function calls, provide direct analysis
            return await self._provide_direct_analysis(context_data, user_id, prompt_text)

        except Exception as e:
            self.logger.error(f"Error in recommendation processing: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {"status": "error", "message": f"Internal error occurred while processing request: {str(e)}"}

    def _prepare_spending_summary(self, spending_data: Dict[str, Any]) -> str:
        """Prepare a summary of spending data for the model."""
        try:
            if not spending_data or 'data' not in spending_data:
                return "No spending data available."
            
            transactions = spending_data['data']
            if not transactions:
                return "No transactions found."
            
            # Calculate category totals
            category_totals = {}
            total_amount = 0
            
            for transaction in transactions:
                category = transaction.get('category', 'unknown')
                amount = float(transaction.get('amount', 0))
                
                if category not in category_totals:
                    category_totals[category] = 0
                category_totals[category] += amount
                total_amount += amount
            
            # Sort categories by spending
            sorted_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
            
            summary = f"Total Spending: ₹{total_amount:,.2f}\n"
            summary += f"Number of Transactions: {len(transactions)}\n"
            summary += "\nSpending by Category:\n"
            
            for category, amount in sorted_categories:
                percentage = (amount / total_amount) * 100 if total_amount > 0 else 0
                summary += f"- {category.title()}: ₹{amount:,.2f} ({percentage:.1f}%)\n"
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error preparing spending summary: {e}")
            return "Error processing spending data."

    async def _provide_direct_analysis(self, spending_data: Dict[str, Any], user_id: str, query: str) -> Dict[str, Any]:
        """Provide direct analysis when no function calls are made."""
        try:
            # Perform basic cost-benefit analysis
            tool_output = await self.cost_benefit_analysis_tool.analyze_recommendations(
                user_id=user_id,
                spending_analysis=spending_data,
                original_query=query
            )
            
            return {
                "status": "success",
                "tool_executed": "perform_cost_benefit_analysis",
                "tool_raw_output": tool_output,
                "recommendations": tool_output.get("recommendations", []) if isinstance(tool_output, dict) else [],
                "insights": tool_output.get("analysis_metadata", {}).get("summary", "Analysis completed") if isinstance(tool_output, dict) else "Analysis completed"
            }
            
        except Exception as e:
            self.logger.error(f"Error in direct analysis: {e}")
            return {
                "status": "error",
                "message": f"Direct analysis failed: {str(e)}"
            }


# The tool classes remain the same, but let's fix a few key issues in CostBenefitAnalysisTool

class BehavioralAnalysisTool:
    """Tool for analyzing user spending behavioral patterns."""
    
    def __init__(self, project_id: str, logger):
        self.project_id = project_id
        self.logger = logger
        self.db_connector = None

    async def _get_db_connector(self):
        """Get database connector instance."""
        if not self.db_connector:
            self.db_connector = await DatabaseConnector.get_instance(self.project_id)
        return self.db_connector

    async def analyze_patterns(self, user_id: str, lookback_months: int = 6, category_filter: Optional[str] = None) -> Dict[str, Any]:
        """Analyze user's spending behavioral patterns."""
        try:
            db = await self._get_db_connector()
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30 * lookback_months)
            
            # Base query for spending patterns
            query = """
                SELECT 
                    t.category,
                    COUNT(*) as transaction_count,
                    SUM(t.amount) as total_spent,
                    AVG(t.amount) as avg_transaction,
                    MIN(t.amount) as min_transaction,
                    MAX(t.amount) as max_transaction,
                    EXTRACT(dow FROM t.transaction_date) as day_of_week,
                    EXTRACT(hour FROM t.transaction_date) as hour_of_day,
                    m.name as merchant_name,
                    COUNT(DISTINCT m.merchant_id) as unique_merchants
                FROM transactions t
                LEFT JOIN merchants m ON t.merchant_id = m.merchant_id
                WHERE t.user_id = $1 
                AND t.transaction_date >= $2 
                AND t.transaction_date <= $3
            """
            
            params = [user_id, start_date, end_date]
            
            if category_filter:
                query += " AND t.category = $4"
                params.append(category_filter)
            
            query += """
                GROUP BY t.category, EXTRACT(dow FROM t.transaction_date), 
                         EXTRACT(hour FROM t.transaction_date), m.name
                ORDER BY total_spent DESC
            """
            
            results = await db.execute_query(query, *params)
            
            # Process behavioral patterns
            patterns = self._process_behavioral_data(results)
            
            return {
                "success": True,
                "behavioral_patterns": patterns,
                "analysis_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "lookback_months": lookback_months
                },
                "category_filter": category_filter,
                "raw_data_summary": f"Analyzed {len(results)} transaction patterns"
            }
            
        except Exception as e:
            self.logger.error(f"Error in behavioral analysis: {e}")
            return {
                "success": False,
                "error": str(e),
                "behavioral_patterns": "Analysis failed due to system error"
            }

    def _process_behavioral_data(self, results: List) -> Dict[str, Any]:
        """Process raw behavioral data into insights."""
        category_patterns = {}
        temporal_patterns = {"by_day": {}, "by_hour": {}}
        merchant_preferences = {}
        
        for row in results:
            category = row.get('category', 'unknown')
            
            # Category spending patterns
            if category not in category_patterns:
                category_patterns[category] = {
                    "total_spent": 0,
                    "transaction_count": 0,
                    "avg_transaction": 0,
                    "spending_frequency": "unknown"
                }
            
            category_patterns[category]["total_spent"] += float(row.get('total_spent', 0))
            category_patterns[category]["transaction_count"] += int(row.get('transaction_count', 0))
            
            # Temporal patterns
            dow = int(row.get('day_of_week', 0))
            hour = int(row.get('hour_of_day', 0))
            
            if dow not in temporal_patterns["by_day"]:
                temporal_patterns["by_day"][dow] = 0
            temporal_patterns["by_day"][dow] += float(row.get('total_spent', 0))
            
            if hour not in temporal_patterns["by_hour"]:
                temporal_patterns["by_hour"][hour] = 0
            temporal_patterns["by_hour"][hour] += float(row.get('total_spent', 0))
        
        return {
            "category_patterns": category_patterns,
            "temporal_patterns": temporal_patterns,
            "insights": self._generate_behavioral_insights(category_patterns, temporal_patterns)
        }

    def _generate_behavioral_insights(self, category_patterns: Dict, temporal_patterns: Dict) -> List[str]:
        """Generate actionable insights from behavioral patterns."""
        insights = []
        
        # Top spending categories
        sorted_categories = sorted(category_patterns.items(), key=lambda x: x[1]["total_spent"], reverse=True)
        if sorted_categories:
            top_category = sorted_categories[0]
            insights.append(f"Primary spending category: {top_category[0]} (₹{top_category[1]['total_spent']:.2f})")
        
        # Day of week patterns
        if temporal_patterns["by_day"]:
            peak_day = max(temporal_patterns["by_day"].items(), key=lambda x: x[1])
            day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
            insights.append(f"Highest spending day: {day_names[peak_day[0]]} (₹{peak_day[1]:.2f})")
        
        return insights


class AlternativeDiscoveryTool:
    """Tool for discovering cheaper alternatives using vector similarity."""
    
    def __init__(self, project_id: str, logger):
        self.project_id = project_id
        self.logger = logger
        self.db_connector = None

    async def _get_db_connector(self):
        if not self.db_connector:
            self.db_connector = await DatabaseConnector.get_instance(self.project_id)
        return self.db_connector

    async def discover_alternatives(self, user_id: str, item_description: str, item_category: str, 
                                  price: float, location_preference: Optional[str] = None) -> Dict[str, Any]:
        """Discover cheaper alternatives using vector similarity search."""
        try:
            # For now, provide mock alternatives since vector search might not be available
            mock_alternatives = [
                {
                    "name": f"Budget alternative to {item_description}",
                    "price": price * 0.8,
                    "savings": price * 0.2,
                    "savings_percentage": 20.0,
                    "merchant": "Alternative Store",
                    "similarity_score": 0.85,
                    "purchase_history": 0
                },
                {
                    "name": f"Generic version of {item_description}",
                    "price": price * 0.7,
                    "savings": price * 0.3,
                    "savings_percentage": 30.0,
                    "merchant": "Budget Store",
                    "similarity_score": 0.75,
                    "purchase_history": 0
                }
            ]
            
            return {
                "success": True,
                "original_item": {
                    "description": item_description,
                    "price": price,
                    "category": item_category
                },
                "alternatives": mock_alternatives,
                "total_alternatives_found": len(mock_alternatives),
                "potential_savings": sum(alt["savings"] for alt in mock_alternatives)
            }
            
        except Exception as e:
            self.logger.error(f"Error in alternative discovery: {e}")
            return {
                "success": False,
                "error": str(e),
                "alternatives": []
            }


class BudgetOptimizationTool:
    """Tool for optimizing budget allocation based on financial goals."""
    
    def __init__(self, project_id: str, logger):
        self.project_id = project_id
        self.logger = logger
        self.db_connector = None

    async def _get_db_connector(self):
        if not self.db_connector:
            self.db_connector = await DatabaseConnector.get_instance(self.project_id)
        return self.db_connector

    async def optimize_allocation(self, user_id: str, financial_goal: str, target_amount: float, 
                                current_amount: float, focus_category: Optional[str] = None) -> Dict[str, Any]:
        """Optimize budget allocation based on spending patterns and goals."""
        try:
            # Calculate required savings
            required_savings = target_amount - current_amount
            
            # Mock spending data for demonstration
            mock_spending = [
                {"category": "shopping", "total_spent": 150000, "transaction_count": 45},
                {"category": "electronics", "total_spent": 200000, "transaction_count": 25},
                {"category": "food", "total_spent": 35000, "transaction_count": 60},
                {"category": "groceries", "total_spent": 24000, "transaction_count": 30}
            ]
            
            total_monthly_spending = sum(float(row['total_spent']) for row in mock_spending) / 3
            
            # Generate optimization suggestions
            optimizations = self._generate_optimization_suggestions(
                mock_spending, required_savings, focus_category, financial_goal
            )
            
            return {
                "success": True,
                "current_allocation": {
                    "total_monthly_spending": total_monthly_spending,
                    "category_breakdown": [
                        {
                            "category": row['category'],
                            "monthly_amount": float(row['total_spent']) / 3,
                            "percentage": (float(row['total_spent']) / 3) / total_monthly_spending * 100 if total_monthly_spending > 0 else 0
                        }
                        for row in mock_spending
                    ]
                },
                "goal_analysis": {
                    "financial_goal": financial_goal,
                    "target_amount": target_amount,
                    "current_amount": current_amount,
                    "required_savings": required_savings
                },
                "optimized_allocation_suggestions": optimizations,
                "projected_monthly_savings": sum(opt.get("monthly_savings", 0) for opt in optimizations)
            }
            
        except Exception as e:
            self.logger.error(f"Error in budget optimization: {e}")
            return {
                "success": False,
                "error": str(e),
                "current_allocation_summary": "Budget optimization failed due to system error"
            }

    def _generate_optimization_suggestions(self, current_spending: List, required_savings: float, 
                                         focus_category: Optional[str], goal: str) -> List[Dict[str, Any]]:
        """Generate budget optimization suggestions."""
        suggestions = []
        
        for spending_row in current_spending[:5]:  # Top 5 categories
            category = spending_row['category']
            monthly_amount = float(spending_row['total_spent']) / 3
            
            # Skip if focus category is specified and this isn't it
            if focus_category and category != focus_category:
                continue
            
            # Calculate potential reduction (10-30% based on category)
            reduction_percentage = self._get_reduction_percentage(category)
            potential_savings = monthly_amount * (reduction_percentage / 100)
            
            suggestions.append({
                "category": category,
                "current_monthly_spending": monthly_amount,
                "suggested_reduction_percentage": reduction_percentage,
                "monthly_savings": potential_savings,
                "new_monthly_budget": monthly_amount - potential_savings,
                "rationale": f"Reduce {category} spending by {reduction_percentage}% to support {goal}"
            })
        
        return suggestions

    def _get_reduction_percentage(self, category: str) -> float:
        """Get suggested reduction percentage by category."""
        reduction_map = {
            "food": 15.0,
            "shopping": 20.0,
            "entertainment": 25.0,
            "groceries": 10.0,
            "electronics": 30.0,
            "general": 15.0
        }
        return reduction_map.get(category.lower(), 15.0)


class CostBenefitAnalysisTool:
    """Tool for performing comprehensive cost-benefit analysis."""
    
    def __init__(self, project_id: str, logger):
        self.project_id = project_id
        self.logger = logger
        self.db_connector = None

    async def _get_db_connector(self):
        if not self.db_connector:
            self.db_connector = await DatabaseConnector.get_instance(self.project_id)
        return self.db_connector

    async def analyze_recommendations(self, user_id: str, spending_analysis: Dict[str, Any], 
                                    user_goals: Optional[Dict[str, Any]] = None, 
                                    original_query: str = "") -> Dict[str, Any]:
        """Perform cost-benefit analysis with spending data."""
        try:
            # Extract transaction data from spending analysis
            transaction_data = spending_analysis.get('data', []) if spending_analysis else []
            
            if not transaction_data:
                return {
                    "success": False,
                    "error": "No transaction data available for analysis",
                    "recommendations": []
                }
            
            # Analyze spending patterns
            category_spending = self._analyze_category_spending(transaction_data)
            
            # Generate recommendations based on spending patterns
            recommendations = self._generate_recommendations(category_spending, user_id, original_query)
            
            return {
                "success": True,
                "recommendations": recommendations,
                "total_scenarios_analyzed": len(recommendations),
                "embedding_matched_recommendations": 0,  # Not using embeddings in this simplified version
                "aggregate_potential_savings": sum(r['financial_impact']['net_annual_impact'] for r in recommendations),
                "analysis_metadata": {
                    "user_id": user_id,
                    "original_query": original_query,
                    "categories_analyzed": len(category_spending),
                    "total_transactions": len(transaction_data)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in cost-benefit analysis: {e}")
            return {
                "success": False,
                "error": str(e),
                "recommendations": [],
                "analysis_metadata": {"error": True, "error_message": str(e)}
            }

    def _analyze_category_spending(self, transactions: List[Dict]) -> Dict[str, Dict]:
        """Analyze spending by category."""
        category_spending = {}
        
        for transaction in transactions:
            category = transaction.get('category', 'unknown')
            amount = float(transaction.get('amount', 0))
            
            if category not in category_spending:
                category_spending[category] = {
                    'total': 0,
                    'count': 0,
                    'avg': 0,
                    'transactions': []
                }
            
            category_spending[category]['total'] += amount
            category_spending[category]['count'] += 1
            category_spending[category]['transactions'].append(transaction)
        
        # Calculate averages
        for category in category_spending:
            data = category_spending[category]
            data['avg'] = data['total'] / data['count'] if data['count'] > 0 else 0
        
        return category_spending

    def _generate_recommendations(self, category_spending: Dict, user_id: str, query: str) -> List[Dict]:
        """Generate recommendations based on category spending."""
        recommendations = []
        
        # Sort categories by spending amount
        sorted_categories = sorted(category_spending.items(), key=lambda x: x[1]['total'], reverse=True)
        
        for i, (category, data) in enumerate(sorted_categories[:5]):  # Top 5 categories
            monthly_spending = data['total'] / 6  # Assuming 6 months of data
            
            # Different recommendation types based on category
            if category in ['shopping', 'electronics']:
                potential_savings = monthly_spending * 0.25  # 25% potential savings
                complexity = 'medium'
                description = f"Optimize {category} purchases by comparing prices and waiting for sales"
            elif category in ['food', 'groceries']:
                potential_savings = monthly_spending * 0.15  # 15% potential savings
                complexity = 'low'
                description = f"Reduce {category} expenses through meal planning and bulk buying"
            else:
                potential_savings = monthly_spending * 0.20  # 20% potential savings
                complexity = 'medium'
                description = f"Review and optimize {category} spending patterns"
            
            recommendation = {
                "id": f"rec_{user_id}_{i}",
                "type": "category_optimization",
                "description": description,
                "financial_impact": {
                    "potential_monthly_savings": potential_savings,
                    "implementation_cost": 0,
                    "net_annual_impact": potential_savings * 12
                },
                "implementation_complexity": complexity,
                "category": category,
                "similarity_score": 0,
                "embedding_matched": False,
                "current_monthly_spending": monthly_spending,
                "transaction_count": data['count']
            }
            recommendations.append(recommendation)
        
        return recommendations


class GoalAlignmentTool:
    """Tool for aligning recommendations with user's financial goals."""
    
    def __init__(self, project_id: str, logger):
        self.project_id = project_id
        self.logger = logger
        self.db_connector = None

    async def _get_db_connector(self):
        if not self.db_connector:
            self.db_connector = await DatabaseConnector.get_instance(self.project_id)
        return self.db_connector

    async def align_with_goals(self, user_id: str, recommendation_details: str, 
                              impact_estimate: Optional[str] = None, 
                              relevant_goals: Optional[List[str]] = None) -> Dict[str, Any]:
        """Align recommendations with user's financial goals."""
        try:
            # Mock goals for demonstration - in production, fetch from database
            mock_goals = [
                {
                    "goal_id": "goal_1",
                    "goal_type": "emergency_fund",
                    "target_amount": 100000,
                    "current_amount": 25000,
                    "priority": 1,
                    "description": "Build emergency fund"
                },
                {
                    "goal_id": "goal_2", 
                    "goal_type": "investment",
                    "target_amount": 500000,
                    "current_amount": 150000,
                    "priority": 2,
                    "description": "Investment portfolio growth"
                }
            ]
            
            if not mock_goals:
                return {
                    "success": True,
                    "alignment_score": 0.5,  # Neutral if no goals
                    "aligned_goals": [],
                    "alignment_analysis": "No active financial goals found for alignment analysis",
                    "recommendations": [{
                        "description": recommendation_details,
                        "alignment_notes": "General recommendation - consider setting financial goals for better alignment"
                    }]
                }
            
            # Analyze alignment with each goal
            goal_alignments = []
            for goal in mock_goals:
                alignment = self._analyze_goal_alignment(
                    goal, recommendation_details, impact_estimate
                )
                goal_alignments.append(alignment)
            
            # Calculate overall alignment score
            overall_score = self._calculate_overall_alignment_score(goal_alignments)
            
            # Generate aligned recommendations
            aligned_recommendations = self._generate_aligned_recommendations(
                goal_alignments, recommendation_details, impact_estimate
            )
            
            return {
                "success": True,
                "alignment_score": overall_score,
                "aligned_goals": [
                    {
                        "goal_id": goal['goal_id'],
                        "goal_type": goal['goal_type'],
                        "target_amount": float(goal['target_amount']),
                        "current_amount": float(goal['current_amount']),
                        "alignment_score": alignment['score'],
                        "contribution_potential": alignment['contribution_potential']
                    }
                    for goal, alignment in zip(mock_goals, goal_alignments)
                    if alignment['score'] > 0.3
                ],
                "alignment_analysis": self._generate_alignment_analysis(goal_alignments, mock_goals),
                "recommendations": aligned_recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Error in goal alignment: {e}")
            return {
                "success": False,
                "error": str(e),
                "alignment_score": 0,
                "aligned_goals": [],
                "recommendations": []
            }

    def _analyze_goal_alignment(self, goal: Dict, recommendation: str, impact_estimate: Optional[str]) -> Dict[str, Any]:
        """Analyze how a recommendation aligns with a specific goal."""
        goal_type = goal.get('goal_type', '').lower()
        target_amount = float(goal.get('target_amount', 0))
        current_amount = float(goal.get('current_amount', 0))
        remaining_amount = target_amount - current_amount
        
        # Extract potential savings from impact estimate
        potential_savings = self._extract_savings_from_impact(impact_estimate)
        
        # Calculate alignment score based on goal type and recommendation content
        alignment_score = self._calculate_goal_specific_alignment(goal_type, recommendation, potential_savings)
        
        # Calculate contribution potential
        contribution_potential = min(potential_savings / remaining_amount, 1.0) if remaining_amount > 0 else 0
        
        return {
            'score': alignment_score,
            'contribution_potential': contribution_potential,
            'potential_monthly_contribution': potential_savings,
            'goal_type': goal_type,
            'remaining_amount': remaining_amount
        }

    def _extract_savings_from_impact(self, impact_estimate: Optional[str]) -> float:
        """Extract savings amount from impact estimate string."""
        if not impact_estimate:
            return 1000.0  # Default assumption for mock data
        
        # Simple regex to find dollar amounts
        import re
        amounts = re.findall(r'[₹$]?(\d+(?:\.\d{2})?)', impact_estimate)
        
        if amounts:
            return float(amounts[0])
        
        return 1000.0  # Default fallback

    def _calculate_goal_specific_alignment(self, goal_type: str, recommendation: str, potential_savings: float) -> float:
        """Calculate alignment score based on goal type and recommendation."""
        recommendation_lower = recommendation.lower()
        
        # Goal type alignment weights
        alignment_weights = {
            'emergency_fund': 0.9 if 'save' in recommendation_lower or 'reduce' in recommendation_lower else 0.3,
            'debt_payoff': 0.8 if 'reduce' in recommendation_lower or 'optimize' in recommendation_lower else 0.4,
            'investment': 0.7 if 'save' in recommendation_lower or 'alternative' in recommendation_lower else 0.3,
            'vacation': 0.6 if 'save' in recommendation_lower else 0.2,
            'home_purchase': 0.8 if 'save' in recommendation_lower or 'reduce' in recommendation_lower else 0.4,
            'retirement': 0.7 if 'optimize' in recommendation_lower or 'reduce' in recommendation_lower else 0.3
        }
        
        base_score = alignment_weights.get(goal_type, 0.5)
        
        # Boost score if potential savings are significant
        if potential_savings > 5000:
            base_score *= 1.2
        elif potential_savings > 2000:
            base_score *= 1.1
        
        return min(base_score, 1.0)

    def _calculate_overall_alignment_score(self, goal_alignments: List[Dict]) -> float:
        """Calculate overall alignment score across all goals."""
        if not goal_alignments:
            return 0.5
        
        # Weighted average based on alignment scores and contribution potential
        total_weight = 0
        weighted_sum = 0
        
        for alignment in goal_alignments:
            weight = alignment['contribution_potential'] + 0.1  # Minimum weight
            weighted_sum += alignment['score'] * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def _generate_alignment_analysis(self, goal_alignments: List[Dict], user_goals: List[Dict]) -> str:
        """Generate human-readable alignment analysis."""
        if not goal_alignments:
            return "No goals available for alignment analysis."
        
        high_alignment_goals = [
            (goal, alignment) for goal, alignment in zip(user_goals, goal_alignments)
            if alignment['score'] > 0.6
        ]
        
        if high_alignment_goals:
            goal_types = [goal['goal_type'].replace('_', ' ') for goal, _ in high_alignment_goals]
            return f"This recommendation strongly aligns with your {', '.join(goal_types)} goals. " \
                   f"It could contribute significantly to {len(high_alignment_goals)} of your financial objectives."
        else:
            return "This recommendation has moderate alignment with your current financial goals. " \
                   "Consider how it fits into your overall financial strategy."

    def _generate_aligned_recommendations(self, goal_alignments: List[Dict], 
                                        recommendation_details: str, 
                                        impact_estimate: Optional[str]) -> List[Dict[str, Any]]:
        """Generate recommendations aligned with user goals."""
        recommendations = []
        
        # Primary recommendation
        recommendations.append({
            "description": recommendation_details,
            "alignment_notes": "Primary recommendation based on spending analysis",
            "estimated_impact": impact_estimate or "Impact analysis pending",
            "priority": "high"
        })
        
        # Goal-specific recommendations
        for alignment in goal_alignments:
            if alignment['score'] > 0.5:
                goal_type = alignment['goal_type']
                monthly_contribution = alignment['potential_monthly_contribution']
                
                recommendations.append({
                    "description": f"Allocate savings from this optimization toward your {goal_type.replace('_', ' ')} goal",
                    "alignment_notes": f"Could contribute ₹{monthly_contribution:.2f}/month toward this goal",
                    "estimated_impact": f"₹{monthly_contribution * 12:.2f} annual contribution",
                    "priority": "medium"
                })
        
        return recommendations[:3]  # Limit to top 3 recommendations