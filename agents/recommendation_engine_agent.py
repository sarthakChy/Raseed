import logging
import asyncio
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import json
import time
import uuid
from vertexai.generative_models import FunctionDeclaration, Tool

from agents.base_agent import BaseAgent
from core.recommendation_agent_tools.alternative_discovery_tool import AlternativeDiscoveryTool


class RecommendationEngineAgent(BaseAgent):
    """
    Specialized agent for generating financial recommendations including merchant alternatives,
    product suggestions, and spending optimization recommendations.
    """
    
    def __init__(
        self,
        agent_name: str = "recommendation_engine_agent",
        project_id: str = "massive-incline-466204-t5",
        location: str = "us-central1",
        model_name: str = "gemini-2.0-flash-001",
        user_id: Optional[str] = None,
    ):
        """
        Initialize the Recommendation Agent.
        
        Args:
            agent_name: Name identifier for this agent
            project_id: Google Cloud project ID
            location: Vertex AI location
            model_name: Model for analysis and responses
            user_id: Current user identifier
        """
        super().__init__(agent_name, project_id, location, model_name, user_id)
        
        # Initialize alternative discovery tool
        self.logger.info(f"Initializing RecommendationEngineAgent with project_id={project_id}, location={location}")
        self.alternative_discovery_tool = AlternativeDiscoveryTool(
            project_id=project_id,
            logger=self.logger
        )
        
        # Register recommendation tools
        self._register_recommendation_tools()
        
        # Recommendation cache for performance optimization
        self.recommendation_cache = {}
        
        self.logger.info("Recommendation Agent initialized with alternative discovery tools")
    
    def _register_recommendation_tools(self):
        """Register recommendation specific tools using Vertex AI FunctionDeclaration."""
        
        # Step 1: Define all FunctionDeclaration instances
        function_declarations = [
            FunctionDeclaration(
                name="find_merchant_alternatives",
                description="Find alternative merchants based on user's transaction history and spending patterns",
                parameters={
                    "type": "object",
                    "properties": {
                        "criteria": {
                            "type": "string",
                            "enum": ["cost_savings", "value", "convenience", "quality"],
                            "description": "Criteria for finding alternatives",
                            "default": "cost_savings"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of alternatives per merchant",
                            "default": 5
                        },
                        "focus_merchants": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific merchants to find alternatives for (optional)"
                        }
                    }
                }
            ),
            FunctionDeclaration(
                name="find_category_alternatives",
                description="Find alternatives within a specific spending category for optimization",
                parameters={
                    "type": "object",
                    "properties": {
                        "target_category": {
                            "type": "string",
                            "description": "Category to find alternatives for (e.g., 'food', 'transportation')"
                        },
                        "optimization_type": {
                            "type": "string",
                            "enum": ["cost_reduction", "quality_improvement", "convenience"],
                            "description": "Type of optimization to focus on",
                            "default": "cost_reduction"
                        }
                    },
                    "required": ["target_category"]
                }
            ),
            FunctionDeclaration(
                name="find_product_alternatives",
                description="Find alternative products based on transaction items and user preferences",
                parameters={
                    "type": "object",
                    "properties": {
                        "focus_items": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific items to find alternatives for (optional)"
                        }
                    }
                }
            ),
            FunctionDeclaration(
                name="generate_spending_recommendations",
                description="Generate comprehensive spending optimization recommendations",
                parameters={
                    "type": "object",
                    "properties": {
                        "recommendation_type": {
                            "type": "string",
                            "enum": ["savings", "optimization", "diversification", "quality", "convenience"],
                            "description": "Type of recommendations to generate",
                            "default": "savings"
                        },
                        "priority_level": {
                            "type": "string",
                            "enum": ["high", "medium", "low", "all"],
                            "description": "Priority level of recommendations to return",
                            "default": "all"
                        }
                    }
                }
            )
        ]

        # Step 2: Register each FunctionDeclaration with its executor
        for function_decl in function_declarations:
            raw_name = function_decl._raw_function_declaration.name
            executor = self._get_tool_executor(raw_name)
            self.register_tool(function_decl, executor)

        self.logger.info("Registered recommendation tools and initialized model with tool support.")
    
    def _get_tool_executor(self, tool_name: str):
        """Get the appropriate executor function for a tool."""
        executor_map = {
            "find_merchant_alternatives": self._execute_merchant_alternatives,
            "find_category_alternatives": self._execute_category_alternatives,
            "find_product_alternatives": self._execute_product_alternatives,
            "generate_spending_recommendations": self._execute_spending_recommendations,
        }
        return executor_map.get(tool_name)
    
    def _extract_transaction_data(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract and normalize transaction data from various possible formats.
        
        Args:
            request: The request containing transaction data in various possible formats
            
        Returns:
            List of normalized transaction dictionaries
        """
        transaction_data = []
        
        # Check multiple possible locations for transaction data
        possible_data_keys = [
            'spending_analysis',
            'transaction_data', 
            'transactions',
            'data',
            'financial_data'
        ]
        
        raw_data = None
        for key in possible_data_keys:
            if key in request:
                raw_data = request[key]
                break
        
        if raw_data is None:
            self.logger.warning("No transaction data found in request")
            return []
        
        # Handle different data formats
        if isinstance(raw_data, list):
            # Direct list of transactions
            transaction_data = raw_data
        elif isinstance(raw_data, dict):
            # Check if it's a nested structure
            if 'data' in raw_data:
                transaction_data = raw_data['data']
            elif 'transactions' in raw_data:
                transaction_data = raw_data['transactions']
            else:
                # Assume the dict itself contains transaction fields
                transaction_data = [raw_data]
        elif isinstance(raw_data, str):
            # Try to parse as JSON
            try:
                parsed_data = json.loads(raw_data)
                if isinstance(parsed_data, list):
                    transaction_data = parsed_data
                elif isinstance(parsed_data, dict):
                    transaction_data = parsed_data.get('data', [parsed_data])
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse transaction data as JSON: {raw_data}")
                return []
        
        # Validate and normalize transaction data
        normalized_transactions = []
        for item in transaction_data:
            if isinstance(item, dict):
                normalized_transactions.append(item)
            else:
                self.logger.warning(f"Skipping non-dict transaction item: {type(item)}")
        
        self.logger.info(f"Extracted {len(normalized_transactions)} valid transactions")
        return normalized_transactions
    
    # Tool executor methods
    async def _execute_merchant_alternatives(
        self,
        user_id: str,
        transaction_data: List[Dict[str, Any]],
        criteria: str = "cost_savings",
        limit: int = 5,
        focus_merchants: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute merchant alternatives finding.
        """
        try:
            self.logger.info(f"Finding merchant alternatives for user {user_id} with criteria: {criteria}")
            
            result = await self.alternative_discovery_tool.find_merchant_alternatives(
                transaction_data=transaction_data,
                user_id=user_id,
                criteria=criteria,
                limit=limit
            )
            
            # Filter by focus merchants if specified
            if focus_merchants and result.get("success"):
                filtered_alternatives = {}
                for merchant_name, alternatives in result.get("alternatives", {}).items():
                    if merchant_name in focus_merchants:
                        filtered_alternatives[merchant_name] = alternatives
                result["alternatives"] = filtered_alternatives
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute merchant alternatives: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_category_alternatives(
        self,
        user_id: str,
        transaction_data: List[Dict[str, Any]],
        target_category: str,
        optimization_type: str = "cost_reduction"
    ) -> Dict[str, Any]:
        """
        Execute category alternatives finding.
        """
        try:
            self.logger.info(f"Finding category alternatives for {target_category} with optimization: {optimization_type}")
            
            result = await self.alternative_discovery_tool.find_category_alternatives(
                transaction_data=transaction_data,
                user_id=user_id,
                target_category=target_category,
                optimization_type=optimization_type
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute category alternatives: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_product_alternatives(
        self,
        user_id: str,
        transaction_data: List[Dict[str, Any]],
        focus_items: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute product alternatives finding.
        """
        try:
            self.logger.info(f"Finding product alternatives for user {user_id}")
            
            result = await self.alternative_discovery_tool.find_product_alternatives(
                transaction_data=transaction_data,
                user_id=user_id,
                focus_items=focus_items
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute product alternatives: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_spending_recommendations(
        self,
        user_id: str,
        transaction_data: List[Dict[str, Any]],
        recommendation_type: str = "savings",
        priority_level: str = "all"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive spending recommendations.
        """
        try:
            self.logger.info(f"Generating spending recommendations for user {user_id}")
            
            # Get merchant alternatives
            merchant_alternatives = await self.alternative_discovery_tool.find_merchant_alternatives(
                transaction_data=transaction_data,
                user_id=user_id,
                criteria=recommendation_type if recommendation_type in ["cost_savings", "quality", "convenience"] else "cost_savings"
            )
            
            # Analyze categories for recommendations
            categories = list(set(t.get('category', 'other') for t in transaction_data))
            category_recommendations = []
            
            for category in categories[:3]:  # Limit to top 3 categories
                category_result = await self.alternative_discovery_tool.find_category_alternatives(
                    transaction_data=transaction_data,
                    user_id=user_id,
                    target_category=category,
                    optimization_type="cost_reduction" if recommendation_type == "savings" else "quality_improvement"
                )
                if category_result.get("success"):
                    category_recommendations.append(category_result)
            
            # Compile comprehensive recommendations
            comprehensive_recommendations = {
                "success": True,
                "user_id": user_id,
                "recommendation_type": recommendation_type,
                "priority_level": priority_level,
                "merchant_alternatives": merchant_alternatives,
                "category_recommendations": category_recommendations,
                "summary": self._generate_recommendation_summary(
                    merchant_alternatives, 
                    category_recommendations, 
                    recommendation_type
                ),
                "timestamp": datetime.now().isoformat()
            }
            
            # Filter by priority if specified
            if priority_level != "all":
                comprehensive_recommendations = self._filter_by_priority(
                    comprehensive_recommendations, 
                    priority_level
                )
            
            return comprehensive_recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate spending recommendations: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_recommendation_summary(
        self,
        merchant_alternatives: Dict[str, Any],
        category_recommendations: List[Dict[str, Any]],
        recommendation_type: str
    ) -> Dict[str, Any]:
        """Generate a summary of all recommendations."""
        summary = {
            "total_merchants_analyzed": 0,
            "total_alternatives_found": 0,
            "categories_analyzed": len(category_recommendations),
            "potential_monthly_savings": 0,
            "top_recommendations": [],
            "quick_wins": []
        }
        
        # Process merchant alternatives
        if merchant_alternatives.get("success"):
            alternatives_data = merchant_alternatives.get("alternatives", {})
            summary["total_merchants_analyzed"] = len(alternatives_data)
            
            total_alternatives = sum(len(alts) for alts in alternatives_data.values())
            summary["total_alternatives_found"] = total_alternatives
            
            # Calculate potential savings
            total_savings = merchant_alternatives.get("total_potential_savings", {})
            summary["potential_monthly_savings"] = total_savings.get("monthly_potential", 0)
            
            # Extract top recommendations
            for merchant, alternatives in alternatives_data.items():
                if alternatives:
                    best_alt = alternatives[0]  # First one is usually best scored
                    if best_alt.get("financial_impact", {}).get("savings_percentage", 0) > 10:
                        summary["top_recommendations"].append({
                            "current_merchant": merchant,
                            "recommended_alternative": best_alt.get("name"),
                            "savings_percentage": best_alt.get("financial_impact", {}).get("savings_percentage", 0),
                            "reason": best_alt.get("recommendation_reason", "")
                        })
        
        # Process category recommendations for quick wins
        for cat_rec in category_recommendations:
            if cat_rec.get("success") and cat_rec.get("alternatives"):
                category = cat_rec.get("target_category", "Unknown")
                alternatives = cat_rec.get("alternatives", [])
                
                if alternatives:
                    best_alt = alternatives[0]
                    savings_pct = best_alt.get("financial_impact", {}).get("savings_percentage", 0)
                    
                    if savings_pct > 15:  # High savings threshold for quick wins
                        summary["quick_wins"].append({
                            "category": category,
                            "recommended_merchant": best_alt.get("name"),
                            "savings_percentage": savings_pct,
                            "action": f"Try {best_alt.get('name')} for your next {category} purchase"
                        })
        
        return summary
    
    def _filter_by_priority(
        self,
        recommendations: Dict[str, Any],
        priority_level: str
    ) -> Dict[str, Any]:
        """Filter recommendations by priority level."""
        if priority_level == "high":
            # Keep only high-impact recommendations
            summary = recommendations.get("summary", {})
            high_priority_recs = []
            
            for rec in summary.get("top_recommendations", []):
                if rec.get("savings_percentage", 0) > 15:
                    high_priority_recs.append(rec)
            
            summary["top_recommendations"] = high_priority_recs
            recommendations["summary"] = summary
            
        elif priority_level == "medium":
            # Keep medium-impact recommendations
            summary = recommendations.get("summary", {})
            medium_priority_recs = []
            
            for rec in summary.get("top_recommendations", []):
                savings_pct = rec.get("savings_percentage", 0)
                if 5 <= savings_pct <= 15:
                    medium_priority_recs.append(rec)
            
            summary["top_recommendations"] = medium_priority_recs
            recommendations["summary"] = summary
            
        elif priority_level == "low":
            # Keep low-impact but still valuable recommendations
            summary = recommendations.get("summary", {})
            low_priority_recs = []
            
            for rec in summary.get("top_recommendations", []):
                if rec.get("savings_percentage", 0) < 5:
                    low_priority_recs.append(rec)
            
            summary["top_recommendations"] = low_priority_recs
            recommendations["summary"] = summary
        
        return recommendations
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing entry point for recommendation queries.
        Expects transaction data from financial analysis results.
        """
        start_time = time.time()
        try:
            self.logger.info("Processing recommendation request")

            # Extract user ID
            user_id = request.get("user_id")
            if not user_id:
                return {
                    "success": False,
                    "error": "User ID is required for recommendations"
                }

            # Extract and normalize transaction data using the new method
            transaction_data = self._extract_transaction_data(request)
            
            if not transaction_data:
                return {
                    "success": False,
                    "error": "No valid transaction data found in request",
                    "debug_info": {
                        "request_keys": list(request.keys()),
                        "extracted_transactions": len(transaction_data)
                    }
                }
            
            self.logger.info(f"Processing {len(transaction_data)} transactions for recommendations")
            
            # Determine recommendation type from request
            recommendation_type = request.get("recommendation_type", "savings")
            analysis_type = request.get("analysis_type", "general")
            
            # Map analysis types to recommendation approaches
            if analysis_type in ["spending_analysis", "category_breakdown"]:
                recommendation_type = "cost_savings"
            elif analysis_type in ["merchant_analysis"]:
                recommendation_type = "diversification"
            
            # Generate comprehensive recommendations
            recommendations = await self._execute_spending_recommendations(
                user_id=user_id,
                transaction_data=transaction_data,
                recommendation_type=recommendation_type,
                priority_level="all"
            )
            
            # Enhance with specific insights based on transaction data
            enhanced_recommendations = await self._enhance_recommendations_with_insights(
                recommendations,
                transaction_data,
                user_id
            )
            
            return {
                "success": True,
                "recommendations": enhanced_recommendations,
                "metadata": {
                    "transactions_analyzed": len(transaction_data),
                    "recommendation_type": recommendation_type,
                    "execution_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat()
                }
            }

        except Exception as e:
            self.logger.error(f"Error during recommendation processing: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "debug_info": {
                    "request_keys": list(request.keys()) if isinstance(request, dict) else "Not a dict",
                    "request_type": type(request).__name__
                }
            }
    
    async def _enhance_recommendations_with_insights(
        self,
        recommendations: Dict[str, Any],
        transaction_data: List[Dict[str, Any]],
        user_id: str
    ) -> Dict[str, Any]:
        """Enhance recommendations with additional insights from transaction analysis."""
        try:
            # Analyze transaction patterns
            patterns = self._analyze_transaction_patterns(transaction_data)
            
            # Add pattern-based insights
            recommendations["transaction_insights"] = {
                "spending_patterns": patterns,
                "frequent_merchants": self._get_frequent_merchants(transaction_data),
                "category_distribution": self._get_category_distribution(transaction_data),
                "spending_trends": self._analyze_spending_trends(transaction_data)
            }
            
            # Add personalized recommendations based on patterns
            personalized_recs = self._generate_personalized_recommendations(
                patterns, 
                transaction_data
            )
            recommendations["personalized_recommendations"] = personalized_recs
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to enhance recommendations: {e}")
            return recommendations  # Return original if enhancement fails
    
    def _analyze_transaction_patterns(self, transaction_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in transaction data."""
        if not transaction_data:
            return {}
        
        # Calculate basic statistics
        amounts = [float(t.get('amount', 0)) for t in transaction_data]
        total_spent = sum(amounts)
        avg_transaction = total_spent / len(amounts) if amounts else 0
        
        # Analyze by payment method
        payment_methods = {}
        for t in transaction_data:
            method = t.get('payment_method', 'Unknown')
            payment_methods[method] = payment_methods.get(method, 0) + float(t.get('amount', 0))
        
        # Analyze by date patterns
        dates = [t.get('transaction_date') for t in transaction_data if t.get('transaction_date')]
        date_range = {
            "earliest": min(dates) if dates else None,
            "latest": max(dates) if dates else None,
            "days_span": len(set(dates)) if dates else 0
        }
        
        return {
            "total_transactions": len(transaction_data),
            "total_amount": round(total_spent, 2),
            "average_transaction": round(avg_transaction, 2),
            "payment_method_breakdown": payment_methods,
            "date_analysis": date_range
        }
    
    def _get_frequent_merchants(self, transaction_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get most frequent merchants from transaction data."""
        if not transaction_data:
            return []
        
        merchant_stats = {}
        for t in transaction_data:
            merchant = t.get('merchant_name', 'Unknown')
            if merchant not in merchant_stats:
                merchant_stats[merchant] = {
                    "name": merchant,
                    "normalized": t.get('merchant_normalized', merchant),
                    "transaction_count": 0,
                    "total_spent": 0,
                    "categories": set()
                }
            
            merchant_stats[merchant]["transaction_count"] += 1
            merchant_stats[merchant]["total_spent"] += float(t.get('amount', 0))
            merchant_stats[merchant]["categories"].add(t.get('category', 'other'))
        
        # Convert to list and sort by frequency
        frequent_merchants = []
        for merchant, stats in merchant_stats.items():
            stats["categories"] = list(stats["categories"])
            stats["avg_transaction"] = stats["total_spent"] / stats["transaction_count"] if stats["transaction_count"] > 0 else 0
            frequent_merchants.append(stats)
        
        return sorted(frequent_merchants, key=lambda x: x["transaction_count"], reverse=True)[:10]
    
    def _get_category_distribution(self, transaction_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze spending distribution by category."""
        if not transaction_data:
            return {}
        
        category_stats = {}
        total_spent = 0
        
        for t in transaction_data:
            category = t.get('category', 'other')
            amount = float(t.get('amount', 0))
            total_spent += amount
            
            if category not in category_stats:
                category_stats[category] = {
                    "total_spent": 0,
                    "transaction_count": 0,
                    "merchants": set()
                }
            
            category_stats[category]["total_spent"] += amount
            category_stats[category]["transaction_count"] += 1
            category_stats[category]["merchants"].add(t.get('merchant_name', 'Unknown'))
        
        # Calculate percentages and format
        for category, stats in category_stats.items():
            stats["percentage"] = (stats["total_spent"] / total_spent * 100) if total_spent > 0 else 0
            stats["avg_transaction"] = stats["total_spent"] / stats["transaction_count"] if stats["transaction_count"] > 0 else 0
            stats["merchants"] = list(stats["merchants"])
            stats["unique_merchants"] = len(stats["merchants"])
        
        return category_stats
    
    def _analyze_spending_trends(self, transaction_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze spending trends over time."""
        if not transaction_data:
            return {}
        
        # Group by date
        daily_spending = {}
        for t in transaction_data:
            date = t.get('transaction_date')
            if date:
                amount = float(t.get('amount', 0))
                daily_spending[date] = daily_spending.get(date, 0) + amount
        
        if not daily_spending:
            return {}
        
        # Calculate trend metrics
        spending_values = list(daily_spending.values())
        dates = sorted(daily_spending.keys())
        
        trend_analysis = {
            "total_days": len(dates),
            "daily_average": sum(spending_values) / len(spending_values) if spending_values else 0,
            "highest_spending_day": {
                "date": max(daily_spending, key=daily_spending.get),
                "amount": max(spending_values)
            } if spending_values else None,
            "lowest_spending_day": {
                "date": min(daily_spending, key=daily_spending.get),
                "amount": min(spending_values)
            } if spending_values else None,
            "date_range": {
                "start": dates[0] if dates else None,
                "end": dates[-1] if dates else None
            }
        }
        
        return trend_analysis
    
    def _generate_personalized_recommendations(
        self,
        patterns: Dict[str, Any],
        transaction_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate personalized recommendations based on transaction patterns."""
        recommendations = []
        
        # High spending merchant recommendations
        frequent_merchants = self._get_frequent_merchants(transaction_data)
        if frequent_merchants:
            top_merchant = frequent_merchants[0]
            if top_merchant["transaction_count"] > 2:
                recommendations.append({
                    "type": "merchant_optimization",
                    "priority": "high",
                    "title": f"Optimize spending at {top_merchant['name']}",
                    "description": f"You've spent ₹{top_merchant['total_spent']:.0f} across {top_merchant['transaction_count']} transactions. Consider exploring alternatives.",
                    "action": f"Look for alternatives to {top_merchant['name']} in the {', '.join(top_merchant['categories'])} category"
                })
        
        # Category diversification recommendations  
        category_dist = self._get_category_distribution(transaction_data)
        if category_dist:
            dominant_category = max(category_dist.items(), key=lambda x: x[1]["percentage"])
            if dominant_category[1]["percentage"] > 60:  # If one category dominates
                recommendations.append({
                    "type": "category_diversification",
                    "priority": "medium",
                    "title": f"High concentration in {dominant_category[0]} spending",
                    "description": f"{dominant_category[1]['percentage']:.1f}% of your spending is in {dominant_category[0]}. Consider exploring cost-effective alternatives.",
                    "action": f"Find cheaper alternatives in the {dominant_category[0]} category"
                })
        
        # Payment method optimization
        if patterns.get("payment_method_breakdown"):
            cash_spending = patterns["payment_method_breakdown"].get("CASH", 0)
            total_spending = patterns.get("total_amount", 0)
            
            if cash_spending > total_spending * 0.3:  # More than 30% cash
                recommendations.append({
                    "type": "payment_optimization",
                    "priority": "low",
                    "title": "Consider digital payment methods",
                    "description": f"₹{cash_spending:.0f} spent in cash. Digital payments often offer cashback and better tracking.",
                    "action": "Try UPI or card payments to earn rewards and better expense tracking"
                })
        
        return recommendations
    
    def get_supported_recommendation_types(self) -> List[str]:
        """Return list of supported recommendation types."""
        return [
            "merchant_alternatives",
            "category_optimization", 
            "product_alternatives",
            "cost_savings",
            "quality_improvement",
            "convenience_optimization",
            "spending_diversification",
            "payment_optimization"
        ]