import json
import logging
from typing import Dict, Any, Optional, List
from vertexai.generative_models import GenerativeModel
from core.recommendation_agent_tools.tools_instructions import cost_benefit_synthesis_instruction


class CostBenefitAnalysisTool:
    """
    Tool for performing cost-benefit analysis on financial recommendations.
    Quantifies financial impact and generates structured recommendations.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the Cost-Benefit Analysis Tool.
        
        Args:
            logger: Logger instance for this tool
        """
        self.logger = logger
        self.model_name = "gemini-2.0-flash-001"
        
        # Initialize synthesis model with specific instructions
        self.synthesis_model = GenerativeModel(
            self.model_name,
            system_instruction=cost_benefit_synthesis_instruction
        )
        
        self.logger.info("CostBenefitAnalysisTool initialized")
    
    async def analyze_recommendations(
        self,
        user_id: str,
        spending_analysis: Dict[str, Any],
        user_goals: Optional[Dict[str, Any]] = None,
        original_query: str = ""
    ) -> Dict[str, Any]:
        """
        Analyze spending data and generate cost-benefit recommendations.
        
        Args:
            user_id: User identifier
            spending_analysis: Financial analysis results from previous workflow step
            user_goals: User's financial goals and preferences
            original_query: Original user query for context
            
        Returns:
            Dictionary containing cost-benefit analysis and recommendations
        """
        self.logger.info(f"Performing cost-benefit analysis for user: {user_id}")
        
        try:
            # Extract transaction data from spending analysis
            transaction_data = spending_analysis.get('data', [])
            
            if not transaction_data:
                return {
                    "success": True,
                    "recommendations": [],
                    "total_scenarios_analyzed": 0,
                    "aggregate_potential_savings": 0,
                    "analysis_metadata": {
                        "user_id": user_id,
                        "error": False,
                        "message": "No transaction data available for analysis"
                    }
                }
            
            # Aggregate transactions by category
            category_analysis = self._aggregate_by_category(transaction_data)
            
            # Identify high-spend categories for optimization
            total_spending = sum(cat['total'] for cat in category_analysis.values())
            high_spend_categories = self._identify_optimization_opportunities(category_analysis, total_spending)
            
            # Generate optimization scenarios
            cost_benefit_scenarios = []
            for category_data in high_spend_categories:
                scenarios = self._generate_optimization_scenarios(category_data)
                cost_benefit_scenarios.extend(scenarios)
            
            # Process each scenario through cost-benefit analysis
            analyzed_recommendations = []
            
            for scenario in cost_benefit_scenarios:
                try:
                    # Prepare prompt for cost-benefit synthesis
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

                    # Generate synthesis
                    response = await self.synthesis_model.generate_content_async(
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
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse cost-benefit result for scenario: {e}")
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing scenario: {e}")
                    continue

            # Sort recommendations by financial impact
            sorted_recommendations = sorted(
                analyzed_recommendations,
                key=lambda x: x['financial_impact'].get('net_financial_impact_over_duration', 0),
                reverse=True
            )

            return {
                "success": True,
                "recommendations": sorted_recommendations[:7],  # Top 7 recommendations
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
                                                    if r['financial_impact'].get('net_financial_impact_over_duration', 0) > 500]),
                    "categories_analyzed": len(high_spend_categories)
                }
            }

        except Exception as e:
            self.logger.error(f"Error in cost-benefit analysis: {e}")
            return {
                "success": False,
                "error": str(e),
                "recommendations": [],
                "total_scenarios_analyzed": 0,
                "aggregate_potential_savings": 0,
                "analysis_metadata": {
                    "user_id": user_id,
                    "error": True,
                    "error_message": str(e)
                }
            }
    
    def _aggregate_by_category(self, transaction_data: list) -> Dict[str, Dict]:
        """
        Aggregate transaction data by category.
        
        Args:
            transaction_data: List of transaction records
            
        Returns:
            Dictionary with category aggregations
        """
        category_totals = {}
        
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
        
        return category_totals
    
    def _identify_optimization_opportunities(self, category_analysis: Dict, total_spending: float) -> List[Dict]:
        """
        Identify categories with high optimization potential.
        
        Args:
            category_analysis: Aggregated spending by category
            total_spending: Total spending amount
            
        Returns:
            List of high-potential categories for optimization
        """
        high_spend_categories = []
        
        for category, data in category_analysis.items():
            category_total = data['total']
            # Focus on categories that are >10% of total spending
            if category_total > total_spending * 0.10:
                high_spend_categories.append({
                    'name': category,
                    'amount': category_total,
                    'transaction_count': data['count'],
                    'avg_transaction': category_total / data['count'],
                    'transactions': data['transactions'],
                    'percentage_of_total': (category_total / total_spending) * 100
                })
        
        # Sort by spending amount (highest first)
        return sorted(high_spend_categories, key=lambda x: x['amount'], reverse=True)
    
    def _generate_optimization_scenarios(self, category_data: Dict) -> List[Dict]:
        """
        Generate potential cost-saving scenarios for a category.
        
        Args:
            category_data: Category spending data
            
        Returns:
            List of optimization scenarios
        """
        scenarios = []
        category_name = category_data['name']
        current_spend = category_data['amount']
        transaction_count = category_data['transaction_count']
        avg_transaction = category_data['avg_transaction']
        
        category_lower = category_name.lower()
        
        # Category-specific scenarios
        if 'food' in category_lower or 'dining' in category_lower:
            scenarios.extend([
                {
                    "description": f"Reduce {category_name} spending through meal planning and home cooking",
                    "projected_cost": current_spend * 0.7,
                    "setup_cost": 0,
                    "benefits": ["healthier eating", "reduced food waste", "cooking skills"],
                    "complexity": "low",
                    "category": category_name,
                    "analysis_target": f"Optimize {category_name} spending",
                    "current_cost_per_period": current_spend,
                    "estimated_new_cost_per_period": current_spend * 0.7,
                    "initial_investment_required": 0,
                    "analysis_duration_months": 12,
                    "cost_period_type": "monthly",
                    "qualitative_factors": ["healthier eating", "reduced food waste"]
                },
                {
                    "description": f"Switch to bulk buying and wholesale for {category_name}",
                    "projected_cost": current_spend * 0.8,
                    "setup_cost": 50,
                    "benefits": ["cost savings", "fewer shopping trips"],
                    "complexity": "medium",
                    "category": category_name,
                    "analysis_target": f"Bulk purchasing for {category_name}",
                    "current_cost_per_period": current_spend,
                    "estimated_new_cost_per_period": current_spend * 0.8,
                    "initial_investment_required": 50,
                    "analysis_duration_months": 12,
                    "cost_period_type": "monthly",
                    "qualitative_factors": ["bulk discounts", "fewer shopping trips"]
                }
            ])
        
        elif 'shopping' in category_lower:
            if transaction_count > 10:  # High frequency = impulse buying
                scenarios.append({
                    "description": f"Implement 24-hour waiting period for {category_name} purchases",
                    "projected_cost": current_spend * 0.6,
                    "setup_cost": 0,
                    "benefits": ["reduced impulse buying", "more thoughtful purchases"],
                    "complexity": "low",
                    "category": category_name,
                    "analysis_target": f"Reduce impulse {category_name} purchases",
                    "current_cost_per_period": current_spend,
                    "estimated_new_cost_per_period": current_spend * 0.6,
                    "initial_investment_required": 0,
                    "analysis_duration_months": 12,
                    "cost_period_type": "monthly",
                    "qualitative_factors": ["better purchase decisions", "reduced regret"]
                })
        
        elif 'transport' in category_lower or 'gas' in category_lower:
            scenarios.extend([
                {
                    "description": f"Optimize {category_name} through carpooling and public transport",
                    "projected_cost": current_spend * 0.6,
                    "setup_cost": 0,
                    "benefits": ["environmental impact", "reduced stress"],
                    "complexity": "medium",
                    "category": category_name,
                    "analysis_target": f"Alternative transportation for {category_name}",
                    "current_cost_per_period": current_spend,
                    "estimated_new_cost_per_period": current_spend * 0.6,
                    "initial_investment_required": 0,
                    "analysis_duration_months": 12,
                    "cost_period_type": "monthly",
                    "qualitative_factors": ["environmental benefits", "social connections"]
                }
            ])
        
        # Generic optimization scenario
        scenarios.append({
            "description": f"General optimization of {category_name} spending habits",
            "projected_cost": current_spend * 0.85,
            "setup_cost": 0,
            "benefits": ["improved budgeting", "increased savings awareness"],
            "complexity": "low",
            "category": category_name,
            "analysis_target": f"General {category_name} optimization",
            "current_cost_per_period": current_spend,
            "estimated_new_cost_per_period": current_spend * 0.85,
            "initial_investment_required": 0,
            "analysis_duration_months": 12,
            "cost_period_type": "monthly",
            "qualitative_factors": ["better spending awareness", "budget discipline"]
        })
        
        return scenarios
    
    def _calculate_priority(self, cost_benefit_result: Dict) -> str:
        """
        Calculate recommendation priority based on cost-benefit metrics.
        
        Args:
            cost_benefit_result: Cost-benefit analysis results
            
        Returns:
            Priority level (high/medium/low)
        """
        try:
            financial_metrics = cost_benefit_result.get('financial_metrics', {})
            net_impact = financial_metrics.get('net_financial_impact_over_duration', 0)
            payback_months = financial_metrics.get('payback_period_months', float('inf'))
            
            if net_impact > 1000 and payback_months < 6:
                return "high"
            elif net_impact > 500 and payback_months < 12:
                return "medium"
            else:
                return "low"
        except Exception:
            return "low"