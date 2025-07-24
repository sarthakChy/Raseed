import logging
import asyncio
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import json

from vertexai.generative_models import FunctionDeclaration, Tool

from agents.base_agent import BaseAgent
from core.financial_analysis_tools.postgresql_query_tool import PostgreSQLQueryTool
from core.financial_analysis_tools.vector_search_tool import VectorSearchTool
from core.financial_analysis_tools.statistical_analysis_tool import StatisticalAnalysisTool
from core.financial_analysis_tools.pattern_recognition_tool import PatternRecognitionTool
# from core.financial_analysis_tools.external_data_integration_tool import ExternalDataIntegrationTool


class FinancialAnalysisAgent(BaseAgent):
    """
    Specialized agent for complex financial analysis including SQL queries,
    vector searches, statistical analysis, and pattern recognition.
    """
    
    def __init__(
        self,
        agent_name: str = "financial_analysis_agent",
        project_id: str = "massive-incline-466204-t5",
        location: str = "us-central1",
        model_name: str = "gemini-2.0-flash-001",
        user_id: Optional[str] = None,
    ):
        """
        Initialize the Financial Analysis Agent.
        
        Args:
            agent_name: Name identifier for this agent
            project_id: Google Cloud project ID
            location: Vertex AI location
            model_name: Model for analysis and responses
            user_id: Current user identifier
        """
        super().__init__(agent_name, project_id, location, model_name, user_id)
        
        # Initialize specialized tools
        self.logger.info(f"Initializing FinancialAnalysisAgent with project_id={project_id}, location={location}")
        self.postgresql_tool = PostgreSQLQueryTool(
            project_id=project_id,
            logger=self.logger
        )
        self.vector_search_tool = VectorSearchTool(
            project_id=project_id,
            location=location,
            logger=self.logger
        )
        self.statistical_tool = StatisticalAnalysisTool(self.logger)
        self.pattern_recognition_tool = PatternRecognitionTool(self.logger)
        # self.external_data_tool = ExternalDataIntegrationTool(self.logger)
        
        # Register specialized tools
        self._register_financial_tools()
        
        # Analysis cache for expensive operations
        self.analysis_cache = {}
        
        self.logger.info("Financial Analysis Agent initialized with specialized tools")
    
    # def _register_financial_tools(self):
    #     """Register financial analysis specific tools."""
        
    #     # PostgreSQL Query Tool
    #     postgresql_query_tool = FunctionDeclaration(
    #         name="execute_financial_query",
    #         description="Execute complex PostgreSQL queries for financial data analysis",
    #         parameters={
    #             "type": "object",
    #             "properties": {
    #                 "query_type": {
    #                     "type": "string",
    #                     "enum": ["transactions", "aggregations", "comparisons", "trends", "custom"],
    #                     "description": "Type of financial query to execute"
    #                 },
    #                 "sql_query": {
    #                     "type": "string",
    #                     "description": "Raw SQL query to execute (for custom queries)"
    #                 },
    #                 "filters": {
    #                     "type": "object",
    #                     "properties": {
    #                         "date_range": {
    #                             "type": "object",
    #                             "properties": {
    #                                 "start_date": {"type": "string", "format": "date"},
    #                                 "end_date": {"type": "string", "format": "date"}
    #                             }
    #                         },
    #                         "categories": {"type": "array", "items": {"type": "string"}},
    #                         "accounts": {"type": "array", "items": {"type": "string"}},
    #                         "amount_range": {
    #                             "type": "object",
    #                             "properties": {
    #                                 "min_amount": {"type": "number"},
    #                                 "max_amount": {"type": "number"}
    #                             }
    #                         }
    #                     }
    #                 },
    #                 "aggregation": {
    #                     "type": "string",
    #                     "enum": ["sum", "avg", "count", "min", "max", "group_by"],
    #                     "description": "Type of aggregation to perform"
    #                 },
    #                 "group_by": {
    #                     "type": "array",
    #                     "items": {"type": "string"},
    #                     "description": "Fields to group results by"
    #                 }
    #             },
    #             "required": ["query_type"]
    #         }
    #     )
        
    #     # Vector Search Tool
    #     vector_search_tool = FunctionDeclaration(
    #         name="search_similar_transactions",
    #         description="Find similar transactions using vector search based on descriptions and patterns",
    #         parameters={
    #             "type": "object",
    #             "properties": {
    #                 "query_text": {
    #                     "type": "string",
    #                     "description": "Transaction description or pattern to search for"
    #                 },
    #                 "search_type": {
    #                     "type": "string",
    #                     "enum": ["description", "merchant", "category", "amount_pattern"],
    #                     "description": "Type of similarity search to perform"
    #                 },
    #                 "limit": {
    #                     "type": "integer",
    #                     "default": 10,
    #                     "description": "Maximum number of similar transactions to return"
    #                 },
    #                 "threshold": {
    #                     "type": "number",
    #                     "default": 0.7,
    #                     "description": "Similarity threshold (0-1)"
    #                 },
    #                 "filters": {
    #                     "type": "object",
    #                     "description": "Additional filters to apply to search results"
    #                 }
    #             },
    #             "required": ["query_text"]
    #         }
    #     )
        
    #     # Statistical Analysis Tool
    #     statistical_analysis_tool = FunctionDeclaration(
    #         name="perform_statistical_analysis",
    #         description="Perform statistical analysis on financial data including trends, correlations, and forecasts",
    #         parameters={
    #             "type": "object",
    #             "properties": {
    #                 "analysis_type": {
    #                     "type": "string",
    #                     "enum": ["trend_analysis", "correlation", "forecast", "distribution", "outlier_detection", "seasonality"],
    #                     "description": "Type of statistical analysis to perform"
    #                 },
    #                 "data_source": {
    #                     "type": "string",
    #                     "enum": ["transactions", "budgets", "accounts", "categories"],
    #                     "description": "Source of data for analysis"
    #                 },
    #                 "metrics": {
    #                     "type": "array",
    #                     "items": {"type": "string"},
    #                     "description": "Financial metrics to analyze (e.g., spending, income, balance)"
    #                 },
    #                 "time_period": {
    #                     "type": "string",
    #                     "enum": ["daily", "weekly", "monthly", "quarterly", "yearly"],
    #                     "description": "Time period granularity for analysis"
    #                 },
    #                 "parameters": {
    #                     "type": "object",
    #                     "description": "Additional parameters specific to the analysis type"
    #                 }
    #             },
    #             "required": ["analysis_type", "data_source"]
    #         }
    #     )
        
    #     # Pattern Recognition Tool
    #     pattern_recognition_tool = FunctionDeclaration(
    #         name="recognize_financial_patterns",
    #         description="Identify patterns in financial behavior including spending habits and anomalies",
    #         parameters={
    #             "type": "object",
    #             "properties": {
    #                 "pattern_type": {
    #                     "type": "string",
    #                     "enum": ["spending_habits", "income_patterns", "category_trends", "anomaly_detection", "recurring_transactions"],
    #                     "description": "Type of pattern to recognize"
    #                 },
    #                 "time_window": {
    #                     "type": "string",
    #                     "enum": ["1_month", "3_months", "6_months", "1_year", "all_time"],
    #                     "description": "Time window for pattern analysis"
    #                 },
    #                 "sensitivity": {
    #                     "type": "number",
    #                     "default": 0.5,
    #                     "description": "Pattern detection sensitivity (0-1)"
    #                 },
    #                 "categories": {
    #                     "type": "array",
    #                     "items": {"type": "string"},
    #                     "description": "Specific categories to analyze (optional)"
    #                 }
    #             },
    #             "required": ["pattern_type"]
    #         }
    #     )
        
    #     # External Data Integration Tool
    #     # external_data_tool = FunctionDeclaration(
    #     #     name="integrate_external_data",
    #     #     description="Integrate external financial data sources for enhanced analysis",
    #     #     parameters={
    #     #         "type": "object",
    #     #         "properties": {
    #     #             "data_source": {
    #     #                 "type": "string",
    #     #                 "enum": ["market_data", "economic_indicators", "inflation_rates", "interest_rates", "industry_benchmarks"],
    #     #                 "description": "Type of external data to integrate"
    #     #             },
    #     #             "data_points": {
    #     #                 "type": "array",
    #     #                 "items": {"type": "string"},
    #     #                 "description": "Specific data points to retrieve"
    #     #             },
    #     #             "time_range": {
    #     #                 "type": "object",
    #     #                 "properties": {
    #     #                     "start_date": {"type": "string", "format": "date"},
    #     #                     "end_date": {"type": "string", "format": "date"}
    #     #                 }
    #     #             },
    #     #             "context": {
    #     #                 "type": "string",
    #     #                 "description": "Context for how this data relates to user's financial analysis"
    #     #             }
    #     #         },
    #     #         "required": ["data_source"]
    #     #     }
    #     # )
        
    #     # Register tools with the base agent
    #     financial_tools = [
    #         postgresql_query_tool,
    #         vector_search_tool,
    #         statistical_analysis_tool,
    #         pattern_recognition_tool,
    #         # external_data_tool
    #     ]

    #     for tool in financial_tools:
    #         tool_name = tool.name
    #         self.register_tool(tool, self._get_tool_executor(tool_name))

    def _register_financial_tools(self):
        """Register financial analysis specific tools using Vertex AI FunctionDeclaration."""
        
        # Step 1: Define all FunctionDeclaration instances
        function_declarations = [
            FunctionDeclaration(
                name="execute_financial_query",
                description="Execute complex PostgreSQL queries for financial data analysis",
                parameters={
                    "type": "object",
                    "properties": {
                        "query_type": {
                            "type": "string",
                            "enum": ["transactions", "aggregations", "comparisons", "trends", "custom"],
                            "description": "Type of financial query to execute"
                        },
                        "sql_query": {
                            "type": "string",
                            "description": "Raw SQL query to execute (for custom queries)"
                        },
                        "filters": {
                            "type": "object",
                            "properties": {
                                "date_range": {
                                    "type": "object",
                                    "properties": {
                                        "start_date": {"type": "string", "format": "date"},
                                        "end_date": {"type": "string", "format": "date"}
                                    }
                                },
                                "categories": {"type": "array", "items": {"type": "string"}},
                                "accounts": {"type": "array", "items": {"type": "string"}},
                                "amount_range": {
                                    "type": "object",
                                    "properties": {
                                        "min_amount": {"type": "number"},
                                        "max_amount": {"type": "number"}
                                    }
                                }
                            }
                        },
                        "aggregation": {
                            "type": "string",
                            "enum": ["sum", "avg", "count", "min", "max", "group_by"],
                            "description": "Type of aggregation to perform"
                        },
                        "group_by": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Fields to group results by"
                        }
                    },
                    "required": ["query_type"]
                }
            ),
            FunctionDeclaration(
                name="search_similar_transactions",
                description="Find similar transactions using vector search based on descriptions and patterns",
                parameters={
                    "type": "object",
                    "properties": {
                        "query_text": {
                            "type": "string",
                            "description": "Transaction description or pattern to search for"
                        },
                        "search_type": {
                            "type": "string",
                            "enum": ["description", "merchant", "category", "amount_pattern"],
                            "description": "Type of similarity search to perform"
                        },
                        "limit": {
                            "type": "integer",
                            "default": 10,
                            "description": "Maximum number of similar transactions to return"
                        },
                        "threshold": {
                            "type": "number",
                            "default": 0.7,
                            "description": "Similarity threshold (0-1)"
                        },
                        "filters": {
                            "type": "object",
                            "description": "Additional filters to apply to search results"
                        }
                    },
                    "required": ["query_text"]
                }
            ),
            FunctionDeclaration(
                name="perform_statistical_analysis",
                description="Perform statistical analysis on financial data including trends, correlations, and forecasts",
                parameters={
                    "type": "object",
                    "properties": {
                        "analysis_type": {
                            "type": "string",
                            "enum": ["trend_analysis", "correlation", "forecast", "distribution", "outlier_detection", "seasonality"],
                            "description": "Type of statistical analysis to perform"
                        },
                        "data_source": {
                            "type": "string",
                            "enum": ["transactions", "budgets", "accounts", "categories"],
                            "description": "Source of data for analysis"
                        },
                        "metrics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Financial metrics to analyze (e.g., spending, income, balance)"
                        },
                        "time_period": {
                            "type": "string",
                            "enum": ["daily", "weekly", "monthly", "quarterly", "yearly"],
                            "description": "Time period granularity for analysis"
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Additional parameters specific to the analysis type"
                        }
                    },
                    "required": ["analysis_type", "data_source"]
                }
            ),
            FunctionDeclaration(
                name="recognize_financial_patterns",
                description="Identify patterns in financial behavior including spending habits and anomalies",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern_type": {
                            "type": "string",
                            "enum": ["spending_habits", "income_patterns", "category_trends", "anomaly_detection", "recurring_transactions"],
                            "description": "Type of pattern to recognize"
                        },
                        "time_window": {
                            "type": "string",
                            "enum": ["1_month", "3_months", "6_months", "1_year", "all_time"],
                            "description": "Time window for pattern analysis"
                        },
                        "sensitivity": {
                            "type": "number",
                            "default": 0.5,
                            "description": "Pattern detection sensitivity (0-1)"
                        },
                        "categories": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific categories to analyze (optional)"
                        }
                    },
                    "required": ["pattern_type"]
                }
            ),
        ]

        # Step 2: Register each FunctionDeclaration with its executor
        for function_decl in function_declarations:
            raw_name = function_decl._raw_function_declaration.name
            executor = self._get_tool_executor(raw_name)
            self.register_tool(function_decl, executor)

        self.logger.info("Registered financial tools and initialized model with tool support.")
    
    def _get_tool_executor(self, tool_name: str):
        """Get the appropriate executor function for a tool."""
        executor_map = {
            "execute_financial_query": self._execute_financial_query,
            "search_similar_transactions": self._execute_vector_search,
            "perform_statistical_analysis": self._execute_statistical_analysis,
            "recognize_financial_patterns": self._execute_pattern_recognition,
            # "integrate_external_data": self._execute_external_data_integration
        }
        return executor_map.get(tool_name)
    
    # Tool executor methods
    async def _execute_financial_query(
        self,
        query_type: str,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None,
        aggregation: Optional[str] = None,
        group_by: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        time_period: Optional[str] = "month",
        recent_days: Optional[int] = None,
        lookback_days: Optional[int] = None,
        sql_query: Optional[str] = None,
        analysis_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes a financial query using the PostgreSQLQueryTool.
        """
        # Ensure filters is a standard dictionary and handle nested MapComposite objects
        if filters is not None:
            try:
                # Convert top-level MapComposite to dict
                if not isinstance(filters, dict):
                    filters = dict(filters)
                
                # Recursively convert nested MapComposite objects
                def convert_nested_map_composite(obj):
                    if hasattr(obj, 'items') and not isinstance(obj, dict):
                        return {k: convert_nested_map_composite(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_nested_map_composite(elem) for elem in obj]
                    else:
                        return obj

                filters = convert_nested_map_composite(filters)

            except Exception as e:
                self.logger.error(f"Failed to convert filters object to dict: {e}")
                return {"success": False, "error": f"Invalid filters format: {e}"}

        self.logger.info(f"Executing query type: {query_type}, user_id: {user_id}, filters: {filters}, kwargs: {{'aggregation': {aggregation}, 'group_by': {group_by}, 'limit': {limit}, 'offset': {offset}, 'sort_by': {sort_by}, 'sort_order': {sort_order}, 'start_date': {start_date}, 'end_date': {end_date}, 'time_period': {time_period}, 'recent_days': {recent_days}, 'lookback_days': {lookback_days}, 'sql_query': {sql_query}, 'analysis_params': {analysis_params}}}")
        try:
            result = await self.postgresql_tool.execute_query(
                query_type=query_type,
                user_id=user_id,
                filters=filters,
                aggregation=aggregation,
                group_by=group_by,
                limit=limit,
                offset=offset,
                sort_by=sort_by,
                sort_order=sort_order,
                start_date=start_date,
                end_date=end_date,
                time_period=time_period,
                recent_days=recent_days,
                lookback_days=lookback_days,
                sql_query=sql_query,
                analysis_params=analysis_params,
            )
            return result
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_vector_search(self, query_text: str, **kwargs) -> Dict[str, Any]:
        """Execute vector search for similar transactions."""
        return await self.vector_search_tool.search_similar_transactions(query_text, **kwargs)
    
    async def _execute_statistical_analysis(self, analysis_type: str, data_source: str, **kwargs) -> Dict[str, Any]:
        """Execute statistical analysis."""
        return await self.statistical_tool.perform_analysis(analysis_type, data_source, **kwargs)
    
    async def _execute_pattern_recognition(self, pattern_type: str, **kwargs) -> Dict[str, Any]:
        """Execute pattern recognition."""
        return await self.pattern_recognition_tool.recognize_patterns(pattern_type, **kwargs)
    
    # async def _execute_external_data_integration(self, data_source: str, **kwargs) -> Dict[str, Any]:
    #     """Execute external data integration."""
    #     return await self.external_data_tool.integrate_data(data_source, **kwargs)
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing entry point for financial analysis queries.
        Expects a structured_query and user_id in the request.
        """
        try:
            self.logger.info("Processing financial analysis request")

            structured_query = request.get("structured_query")
            user_id = request.get("user_id")

            if structured_query is None or user_id is None:
                raise ValueError("Missing required input: structured_query and user_id must be provided")

            # Handle structured_query if it's a pydantic model
            if hasattr(structured_query, "model_dump"):
                structured_query = structured_query.model_dump()

            # Extract time range
            time_range = structured_query.get("time_range", {})
            if hasattr(time_range, "model_dump"):
                time_range = time_range.model_dump()

            start_date = time_range.get("start_date")
            end_date = time_range.get("end_date")

            # Extract filters and ensure it's a dictionary
            filters = structured_query.get("filters", {})
            if hasattr(filters, "model_dump"):
                filters = filters.model_dump()

            # Inject date_range into filters
            if start_date and end_date:
                filters["date_range"] = {
                    "start_date": start_date,
                    "end_date": end_date
                }

            analysis_params = structured_query.get("analysis_parameters", {})
            if hasattr(analysis_params, "model_dump"):
                analysis_params = analysis_params.model_dump()

            aggregation = analysis_params.get("aggregation_level", "sum")
            group_by = analysis_params.get("grouping", [])
            query_type = structured_query.get("query_type")

            # Map external query types to internal ones
            query_type_mapping = {
                "spending_analysis": "transactions",
                "category_breakdown": "aggregations",
                "trend_analysis": "trends",
                "comparison": "comparisons",
                "budget_check": "budget_analysis",
                "anomaly_detection": "anomalies",
                "goal_tracking": "goal_progress",
                "merchant_analysis": "aggregations",  # or a separate handler if needed
                "forecast": "trends",  # depending on what you're forecasting
                "general_inquiry": "transactions"
            }

            original_query_type = structured_query.get("query_type")
            query_type_key = original_query_type.lower() if isinstance(original_query_type, str) else original_query_type.value

            query_type = query_type_mapping.get(query_type_key)
            if not query_type:
                raise ValueError(f"Unsupported query type: {original_query_type}")


            # Run the PostgreSQL query using extracted params
            result = await self._execute_financial_query(
                query_type=query_type,
                user_id=user_id,
                filters=filters,
                aggregation=aggregation,
                group_by=group_by,
                start_date=start_date,
                end_date=end_date,
                analysis_params=analysis_params
            )

            return {
                "success": True,
                "analysis": result
            }

        except Exception as e:
            self.logger.error(f"Error during financial analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _build_analysis_prompt(
        self, 
        query: str, 
        user_profile: Dict[str, Any], 
        context: Dict[str, Any], 
        analysis_type: str
    ) -> str:
        """Build comprehensive analysis prompt."""
        
        # Base prompt template
        prompt_template = """
You are a Financial Analysis Agent specialized in complex financial data analysis. 
Analyze the user's query and provide comprehensive insights using available tools.

User Query: {query}

Analysis Type: {analysis_type}

User Context:
{user_context}

Available Capabilities:
- PostgreSQL queries for transaction and account data
- Vector search for finding similar transactions
- Statistical analysis for trends and forecasts  
- Pattern recognition for spending habits and anomalies
- External data integration for market context

Instructions:
1. Analyze the query to determine what financial insights are needed
2. Use appropriate tools to gather and analyze relevant data
3. Provide actionable insights based on the analysis
4. Consider the user's profile and preferences in your recommendations
5. Be specific with numbers, dates, and concrete findings

Focus on providing valuable, data-driven insights that help the user understand their financial situation better.
"""
        
        # Format user context
        user_context_str = ""
        if user_profile:
            user_context_str = f"User Profile: {json.dumps(user_profile, indent=2)}\n"
        if context:
            user_context_str += f"Additional Context: {json.dumps(context, indent=2)}"
        
        return prompt_template.format(
            query=query,
            analysis_type=analysis_type,
            user_context=user_context_str or "No additional context provided"
        )
    
    async def _generate_comprehensive_analysis(
        self,
        original_query: str,
        initial_response: str,
        tool_results: Dict[str, Any],
        user_profile: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Generate comprehensive financial analysis incorporating all results."""
        
        if not tool_results:
            return initial_response if initial_response else "No analysis data available."
        
        # Extract meaningful data from tool results for analysis
        analysis_data = {}
        for tool_name, result in tool_results.items():
            if isinstance(result, dict) and result.get("success"):
                analysis_data[tool_name] = result.get("data", result)
            else:
                self.logger.warning(f"Tool {tool_name} failed or returned no data")
        
        # Build synthesis prompt focusing on the actual data
        synthesis_prompt = f"""
    Based on the financial data retrieved, provide a comprehensive analysis for this query: "{original_query}"

    Retrieved Financial Data:
    {json.dumps(analysis_data, indent=2, default=str)}

    User Profile Context:
    {json.dumps(user_profile, indent=2) if user_profile else "No profile data available"}

    Please provide a clear, comprehensive financial analysis that:

    1. **Direct Answer**: Start with a direct answer to the user's question
    2. **Key Findings**: Highlight the most important insights from the data
    3. **Detailed Breakdown**: Provide specific numbers, dates, and transaction details
    4. **Patterns & Trends**: Identify any notable patterns in the spending data
    5. **Actionable Insights**: Offer practical insights or recommendations

    Format your response in a clear, easy-to-read structure. Use specific numbers and dates from the actual data.

    Focus on being helpful and informative while maintaining accuracy with the retrieved data.
    """
        
        try:
            synthesis_response = await self.model.generate_content_async(synthesis_prompt)
            return synthesis_response.text if synthesis_response.text else "Analysis completed but response generation failed."
        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive analysis: {e}")
            # Fallback: provide basic analysis based on tool results
            return self._create_fallback_analysis(original_query, analysis_data)

    def _create_fallback_analysis(self, query: str, analysis_data: Dict[str, Any]) -> str:
        """Create a basic fallback analysis when synthesis fails."""
        try:
            analysis_parts = [f"Analysis for: {query}\n"]
            
            # Check for transaction data
            if "execute_financial_query" in analysis_data:
                query_data = analysis_data["execute_financial_query"]
                if isinstance(query_data, dict) and "data" in query_data:
                    transactions = query_data["data"]
                    if transactions:
                        total_amount = sum(float(t.get("amount", 0)) for t in transactions)
                        transaction_count = len(transactions)
                        
                        analysis_parts.extend([
                            f"Found {transaction_count} transactions",
                            f"Total amount: ${total_amount:.2f}",
                            f"Date range: {transactions[-1].get('transaction_date')} to {transactions[0].get('transaction_date')}"
                        ])
                        
                        # Group by merchant
                        merchants = {}
                        for t in transactions:
                            merchant = t.get("merchant_name", "Unknown")
                            merchants[merchant] = merchants.get(merchant, 0) + float(t.get("amount", 0))
                        
                        analysis_parts.append("\nBreakdown by merchant:")
                        for merchant, amount in sorted(merchants.items(), key=lambda x: x[1], reverse=True):
                            analysis_parts.append(f"- {merchant}: ${amount:.2f}")
            
            return "\n".join(analysis_parts)
        except Exception as e:
            return f"Basic analysis failed: {str(e)}"
    
    def _get_date_range(self, period: str) -> Dict[str, str]:
        """Convert period string to date range."""
        end_date = datetime.now().date()
        
        period_map = {
            "1_month": timedelta(days=30),
            "3_months": timedelta(days=90),
            "6_months": timedelta(days=180),
            "1_year": timedelta(days=365)
        }
        
        start_date = end_date - period_map.get(period, timedelta(days=30))
        
        return {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
    
    def get_supported_analysis_types(self) -> List[str]:
        """Return list of supported analysis types."""
        return [
            "spending_analysis",
            "income_analysis", 
            "budget_analysis",
            "trend_analysis",
            "comparative_analysis",
            "predictive_analysis",
            "pattern_analysis",
            "anomaly_detection",
            "category_analysis",
            "account_analysis"
        ]