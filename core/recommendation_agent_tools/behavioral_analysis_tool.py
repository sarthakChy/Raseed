import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from vertexai.generative_models import GenerativeModel
from core.recommendation_agent_tools.tools_instructions import behavioral_synthesis_instruction


class BehavioralAnalysisTool:
    """
    Tool for analyzing user behavioral spending patterns.
    Executes SQL queries to gather spending data and synthesizes insights.
    """
    
    def __init__(self, project_id: str, logger: logging.Logger):
        """
        Initialize the Behavioral Analysis Tool.
        
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
            system_instruction=behavioral_synthesis_instruction
        )
        
        self.logger.info("BehavioralAnalysisTool initialized")
    
    async def analyze_patterns(
        self, 
        user_id: str, 
        lookback_months: int = 6, 
        category_filter: Optional[str] = None,
        db_connector = None
    ) -> Dict[str, Any]:
        """
        Analyze behavioral spending patterns for a user.
        
        Args:
            user_id: User identifier
            lookback_months: Number of months to analyze
            category_filter: Optional category to focus on
            db_connector: Database connection (injected by agent)
            
        Returns:
            Dictionary containing behavioral analysis results
        """
        self.logger.info(f"Analyzing behavioral patterns for user: {user_id}, lookback: {lookback_months} months")
        
        try:
            # Calculate cutoff date
            cutoff_date = datetime.utcnow() - timedelta(days=lookback_months * 30)
            
            # Base query for behavioral analysis
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
            """
            
            # Add category filter if specified
            params = [user_id, cutoff_date]
            if category_filter:
                query += " AND t.category ILIKE $3"
                params.append(f"%{category_filter}%")
            
            query += """
                GROUP BY
                    purchase_day,
                    m.name,
                    t.category
                ORDER BY
                    total_category_spend DESC,
                    visit_count DESC;
            """
            
            # Execute query (db_connector would be injected by the agent)
            if db_connector:
                behavioral_result = await db_connector.execute_query(query, params)
                behavioral_data = getattr(behavioral_result, 'data', [])
            else:
                # Mock data for testing when db_connector is not available
                behavioral_data = []
            
            # Check if we have data
            if not behavioral_data:
                return {
                    "success": True,
                    "behavioral_patterns": "No spending patterns found for the specified period",
                    "raw_data_summary": f"No transactions found for user {user_id} in the last {lookback_months} months",
                    "key_metrics": {},
                    "potential_areas_for_recommendation": ["Insufficient data for analysis"]
                }
            
            # Prepare data for synthesis
            analysis_data = {
                "user_id": user_id,
                "lookback_months": lookback_months,
                "category_filter": category_filter,
                "raw_data": behavioral_data
            }
            
            # Create synthesis prompt
            prompt = f"""
            Raw behavioral spending data for user {user_id}:
            ```json
            {json.dumps(behavioral_data, indent=2, default=str)}
            ```
            
            Analysis Parameters:
            - Lookback period: {lookback_months} months
            - Category filter: {category_filter or 'All categories'}
            
            Please analyze this data and provide structured insights following the system instructions.
            """
            
            # Generate synthesis
            response = await self.synthesis_model.generate_content_async(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                }
            )
            
            # Parse and return result
            result = json.loads(response.text)
            result["success"] = True
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error in behavioral analysis: {e}")
            return {
                "success": False,
                "error": str(e),
                "behavioral_patterns": "Error processing behavioral analysis",
                "raw_data_summary": "Failed to parse LLM response",
                "key_metrics": {"error": str(e)},
                "potential_areas_for_recommendation": ["Analysis unavailable due to processing error"]
            }
        except Exception as e:
            self.logger.error(f"Error in behavioral spending analysis: {e}")
            return {
                "success": False,
                "error": str(e),
                "behavioral_patterns": "Error analyzing spending patterns",
                "raw_data_summary": f"Analysis failed: {str(e)}",
                "key_metrics": {"error": str(e)},
                "potential_areas_for_recommendation": ["Analysis unavailable due to system error"]
            }
    
    def _process_behavioral_data(self, raw_data: list) -> Dict[str, Any]:
        """
        Process raw behavioral data to extract key metrics.
        
        Args:
            raw_data: Raw transaction data from database
            
        Returns:
            Processed metrics dictionary
        """
        if not raw_data:
            return {}
        
        try:
            # Calculate key metrics
            total_transactions = len(raw_data)
            total_spending = sum(float(item.get('total_category_spend', 0)) for item in raw_data)
            
            # Group by day of week
            day_spending = {}
            for item in raw_data:
                day = item.get('purchase_day', 0)
                amount = float(item.get('total_category_spend', 0))
                day_spending[day] = day_spending.get(day, 0) + amount
            
            # Group by merchant
            merchant_spending = {}
            for item in raw_data:
                merchant = item.get('merchant_name', 'Unknown')
                amount = float(item.get('total_category_spend', 0))
                merchant_spending[merchant] = merchant_spending.get(merchant, 0) + amount
            
            # Group by category
            category_spending = {}
            for item in raw_data:
                category = item.get('category', 'Unknown')
                amount = float(item.get('total_category_spend', 0))
                category_spending[category] = category_spending.get(category, 0) + amount
            
            return {
                "total_transactions": total_transactions,
                "total_spending": total_spending,
                "average_transaction": total_spending / total_transactions if total_transactions > 0 else 0,
                "day_of_week_spending": day_spending,
                "top_merchants": dict(sorted(merchant_spending.items(), key=lambda x: x[1], reverse=True)[:5]),
                "category_breakdown": category_spending
            }
            
        except Exception as e:
            self.logger.error(f"Error processing behavioral data: {e}")
            return {"error": str(e)}