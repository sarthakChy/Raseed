import json
import logging
from typing import Dict, Any, Optional
from vertexai.generative_models import GenerativeModel
from core.recommendation_agent_tools.tools_instructions import budget_optimization_synthesis_instruction


class BudgetOptimizationTool:
    """
    Tool for optimizing budget allocation across spending categories.
    Analyzes spending history and financial goals to suggest budget adjustments.
    """
    
    def __init__(self, project_id: str, logger: logging.Logger):
        """
        Initialize the Budget Optimization Tool.
        
        Args:
            project_id: Google Cloud project ID
            logger: Logger instance for this tool
        """
        self.project_id = project_id
        self.logger = logger
        self.model_name = "gemini-2.0-flash-001"
        
        # Initialize synthesis model with specific instructions
        self.synthesis_model = GenerativeModel(
            self.model_name,
            system_instruction=budget_optimization_synthesis_instruction
        )
        
        self.logger.info("BudgetOptimizationTool initialized")
    
    async def optimize_allocation(
        self,
        user_id: str,
        financial_goal: str,
        target_amount: float,
        current_amount: float,
        focus_category: Optional[str] = None,
        db_connector = None,
        user_profile_manager = None
    ) -> Dict[str, Any]:
        """
        Optimize budget allocation for a user based on their goals and spending history.
        
        Args:
            user_id: User identifier
            financial_goal: The financial goal to optimize for
            target_amount: Target savings/goal amount
            current_amount: Current saved amount
            focus_category: Optional category to focus optimization on
            db_connector: Database connection (injected by agent)
            user_profile_manager: User profile manager (injected by agent)
            
        Returns:
            Dictionary containing budget optimization recommendations
        """
        self.logger.info(f"Optimizing budget for user: {user_id}, goal: {financial_goal}")
        
        try:
            # Query current spending allocation
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
            
            # Execute query
            if db_connector:
                current_allocation_result = await db_connector.execute_query(query, params)
                current_spending_data = getattr(current_allocation_result, 'data', [])
            else:
                # Mock data for testing
                current_spending_data = []
            
            # Fetch user financial goals if profile manager is available
            user_financial_goals = []
            if user_profile_manager:
                try:
                    user_profile = await user_profile_manager.get_profile(user_id, ["financial_goals"])
                    user_financial_goals = getattr(user_profile, "financial_goals", [])
                except Exception as e:
                    self.logger.warning(f"Could not fetch user profile: {e}")
            
            # Prepare analysis data
            analysis_data = {
                "user_id": user_id,
                "financial_goal": financial_goal,
                "target_amount": target_amount,
                "current_amount": current_amount,
                "focus_category": focus_category,
                "current_spending": current_spending_data,
                "user_goals": user_financial_goals
            }
            
            # Calculate key metrics
            total_spending = sum(float(item.get('total_spent', 0)) for item in current_spending_data)
            gap_amount = target_amount - current_amount
            
            # Create synthesis prompt
            prompt = f"""
            Current spending allocation for user {user_id}:
            ```json
            {json.dumps(current_spending_data, indent=2, default=str)}
            ```
            
            Optimization Parameters:
            - Financial goal: {financial_goal}
            - Target amount: ${target_amount}
            - Current amount: ${current_amount}
            - Gap to target: ${gap_amount}
            - Focus category: {focus_category or 'all categories'}
            - Total current spending: ${total_spending}
            - User's broader goals: {user_financial_goals}
            
            Please generate specific budget reallocation recommendations based on the system instructions.
            """

            # Generate synthesis
            response = await self.synthesis_model.generate_content_async(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )

            # Parse and return result
            result = json.loads(response.text)
            result["success"] = True
            result["optimization_metadata"] = {
                "total_current_spending": total_spending,
                "gap_to_target": gap_amount,
                "optimization_potential": self._calculate_optimization_potential(current_spending_data, gap_amount)
            }
            return result

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error in budget optimization: {e}")
            return {
                "success": False,
                "error": str(e),
                "current_allocation_summary": "Analysis unavailable due to an internal processing error.",
                "optimized_allocation_suggestions": [],
                "projected_savings": 0.0,
                "status": "error",
                "message": "Failed to parse LLM synthesis response."
            }
        except Exception as e:
            self.logger.error(f"Unexpected error during budget optimization: {e}")
            return {
                "success": False,
                "error": str(e),
                "current_allocation_summary": "An unexpected system error occurred during budget optimization.",
                "optimized_allocation_suggestions": [],
                "projected_savings": 0.0,
                "status": "error",
                "message": f"An unexpected error occurred: {str(e)}"
            }
    
    def _calculate_optimization_potential(self, spending_data: list, gap_amount: float) -> Dict[str, Any]:
        """
        Calculate the optimization potential based on current spending patterns.
        
        Args:
            spending_data: Current spending data by category
            gap_amount: Amount needed to reach target
            
        Returns:
            Dictionary with optimization potential metrics
        """
        if not spending_data:
            return {"optimization_score": 0, "high_potential_categories": []}
        
        try:
            total_spending = sum(float(item.get('total_spent', 0)) for item in spending_data)
            
            # Identify high-spending categories (>15% of total)
            high_spend_categories = []
            for item in spending_data:
                spent = float(item.get('total_spent', 0))
                if spent > total_spending * 0.15:
                    high_spend_categories.append({
                        "category": item.get('category'),
                        "amount": spent,
                        "percentage": (spent / total_spending) * 100
                    })
            
            # Calculate optimization score based on how achievable the gap is
            potential_reduction = total_spending * 0.1  # Assume 10% reduction is achievable
            optimization_score = min(100, (potential_reduction / gap_amount) * 100) if gap_amount > 0 else 100
            
            return {
                "optimization_score": optimization_score,
                "high_potential_categories": high_spend_categories,
                "total_spending": total_spending,
                "potential_monthly_reduction": potential_reduction,
                "achievability": "high" if optimization_score > 80 else "medium" if optimization_score > 50 else "low"
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating optimization potential: {e}")
            return {"error": str(e)}
    
    def _generate_category_recommendations(self, spending_data: list, gap_amount: float) -> list:
        """
        Generate specific recommendations for each spending category.
        
        Args:
            spending_data: Current spending data by category
            gap_amount: Amount needed to reach financial goal
            
        Returns:
            List of category-specific recommendations
        """
        recommendations = []
        
        try:
            total_spending = sum(float(item.get('total_spent', 0)) for item in spending_data)
            
            for item in spending_data:
                category = item.get('category', 'Unknown')
                spent = float(item.get('total_spent', 0))
                percentage = (spent / total_spending) * 100 if total_spending > 0 else 0
                
                # Generate recommendations based on category and spending level
                if percentage > 20:  # High spending category
                    potential_reduction = spent * 0.15  # 15% reduction
                    recommendations.append({
                        "category": category,
                        "current_amount": spent,
                        "suggested_amount": spent - potential_reduction,
                        "potential_savings": potential_reduction,
                        "confidence": "high",
                        "rationale": f"High spending category ({percentage:.1f}% of total) with good reduction potential"
                    })
                elif percentage > 10:  # Medium spending category
                    potential_reduction = spent * 0.10  # 10% reduction
                    recommendations.append({
                        "category": category,
                        "current_amount": spent,
                        "suggested_amount": spent - potential_reduction,
                        "potential_savings": potential_reduction,
                        "confidence": "medium",
                        "rationale": f"Medium spending category with moderate reduction potential"
                    })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating category recommendations: {e}")
            return []