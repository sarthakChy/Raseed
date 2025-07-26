import logging
import asyncio
import re
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import uuid

from pydantic import BaseModel, Field, validator
from vertexai.generative_models import FunctionDeclaration, GenerationConfig

from agents.base_agent import BaseAgent


class QueryType(str, Enum):
    """Types of queries the agent can process."""
    SPENDING_ANALYSIS = "spending_analysis"
    TREND_ANALYSIS = "trend_analysis"
    COMPARISON = "comparison"
    BUDGET_CHECK = "budget_check"
    CATEGORY_BREAKDOWN = "category_breakdown"
    MERCHANT_ANALYSIS = "merchant_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    FORECAST = "forecast"
    GOAL_TRACKING = "goal_tracking"
    GENERAL_INQUIRY = "general_inquiry"


class TimeReference(str, Enum):
    """Standard time references for queries."""
    TODAY = "today"
    YESTERDAY = "yesterday"
    THIS_WEEK = "this_week"
    LAST_WEEK = "last_week"
    THIS_MONTH = "this_month"
    LAST_MONTH = "last_month"
    THIS_QUARTER = "this_quarter"
    LAST_QUARTER = "last_quarter"
    THIS_YEAR = "this_year"
    LAST_YEAR = "last_year"
    CUSTOM_RANGE = "custom_range"


class ParsedEntity(BaseModel):
    """Represents a parsed entity from the query."""
    entity_type: str = Field(..., description="Type of entity: amount, category, merchant, date, etc.")
    value: Union[str, float, int, dict] = Field(..., description="Raw extracted value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for extraction")
    original_text: str = Field(..., description="Original text where entity was found")
    normalized_value: Optional[Union[str, float, int, dict]] = Field(None, description="Normalized/standardized value")


class TimeRange(BaseModel):
    """Represents a time range specification."""
    type: TimeReference = Field(..., description="Type of time reference")
    start_date: Optional[str] = Field(None, description="Start date in ISO format (for custom ranges)")
    end_date: Optional[str] = Field(None, description="End date in ISO format (for custom ranges)")
    relative_period: Optional[str] = Field(None, description="Relative period description")


class QueryFilters(BaseModel):
    """Represents filters to apply to the query."""
    categories: List[str] = Field(default_factory=list)
    merchants: List[str] = Field(default_factory=list)
    amount_min: Optional[float] = Field(default=None)
    amount_max: Optional[float] = Field(default=None)
    tags: List[str] = Field(default_factory=list)
    exclude_categories: List[str] = Field(default_factory=list)

class AnalysisParameters(BaseModel):
    """Parameters for the requested analysis."""
    aggregation_level: str = Field(default="monthly")
    comparison_baseline: Optional[str] = Field(default=None)
    metrics: List[str] = Field(default_factory=list)
    grouping: List[str] = Field(default_factory=list)
    sorting: Optional[str] = Field(default=None)
    limit: Optional[int] = Field(default=None)
    
    @validator('aggregation_level', pre=True)
    def validate_aggregation_level(cls, v):
        """Ensure aggregation_level is never None."""
        if v is None or v == "":
            return "monthly"
        return v


class StructuredQuery(BaseModel):
    """Structured representation of a parsed query."""
    query_type: QueryType = Field(..., description="Primary type of query")
    entities: List[ParsedEntity] = Field(default_factory=list, description="Extracted entities")
    time_range: TimeRange = Field(..., description="Time range for the query")
    filters: QueryFilters = Field(default_factory=QueryFilters, description="Query filters")
    analysis_parameters: AnalysisParameters = Field(default_factory=AnalysisParameters, description="Analysis configuration")
    context_requirements: List[str] = Field(default_factory=list, description="Required context for execution")
    original_query: str = Field(..., description="Original user query")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall parsing confidence")
    requires_clarification: bool = Field(default=False, description="Whether query needs clarification")
    clarification_questions: List[str] = Field(default_factory=list, description="Questions for clarification")
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {uuid.UUID: lambda u: str(u)}

    @validator('confidence_score')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence score must be between 0.0 and 1.0')
        return v


class QueryComplexity(BaseModel):
    """Assessment of query complexity."""
    level: str = Field(..., description="Complexity level: low, medium, high")
    factors: List[str] = Field(default_factory=list, description="Factors contributing to complexity")
    estimated_execution_time: Optional[float] = Field(None, description="Estimated execution time in seconds")
    requires_decomposition: bool = Field(default=False, description="Whether query should be broken down")


class SubQuery(BaseModel):
    """Represents a sub-component of a complex query."""
    query_id: str = Field(..., description="Unique identifier for this sub-query")
    structured_query: StructuredQuery = Field(..., description="Structured representation")
    dependencies: List[str] = Field(default_factory=list, description="IDs of queries this depends on")
    priority: int = Field(default=1, description="Execution priority")


class QueryDecomposition(BaseModel):
    """Represents a decomposed complex query."""
    main_query_id: str = Field(..., description="ID of the main query")
    sub_queries: List[SubQuery] = Field(default_factory=list, description="Sub-queries")
    execution_order: List[str] = Field(default_factory=list, description="Order of execution")
    aggregation_method: str = Field(default="simple", description="How to combine results")


class ValidationResult(BaseModel):
    """Result of query validation."""
    is_valid: bool = Field(..., description="Whether the query is valid")
    issues: List[str] = Field(default_factory=list, description="Validation issues found")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    required_data: List[str] = Field(default_factory=list, description="Required data sources")
    estimated_complexity: QueryComplexity = Field(..., description="Complexity assessment")


class QueryTranslationResult(BaseModel):
    """Complete result of query translation."""
    success: bool = Field(..., description="Whether translation was successful")
    structured_query: Optional[StructuredQuery] = Field(None, description="Parsed structured query")
    decomposition: Optional[QueryDecomposition] = Field(None, description="Query decomposition if needed")
    validation: Optional[ValidationResult] = Field(None, description="Validation results")
    processing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")
    error: Optional[str] = Field(None, description="Error message if translation failed")
    analysis_type: Optional[str] = Field(None, description="Type of analysis to perform")

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            uuid.UUID: lambda u: str(u)
        }

    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for workflow consumption."""
        result_dict = self.model_dump()
        
        # Ensure structured_query is properly serialized
        if self.structured_query:
            result_dict['structured_query'] = self.structured_query.dict()
            result_dict['analysis_type'] = self.structured_query.query_type.value
        
        return result_dict


class QueryTranslationAgent(BaseAgent):
    """
    Agent responsible for converting natural language queries into structured
    analytical requests using LLM with structured outputs.
    """

    def __init__(
        self,
        agent_name: str = "query_translation_agent",
        project_id: str = None,
        location: str = "us-central1",
        model_name: str = "gemini-2.0-flash-001",
        user_id: Optional[str] = None
    ):
        """Initialize the Query Translation Agent."""
        super().__init__(agent_name, project_id, location, model_name, user_id)

        # Context management
        self.conversation_context: Dict[str, Any] = {}
        self.session_queries: List[Dict[str, Any]] = []

        # Generation configuration for structured outputs
        self.generation_config = GenerationConfig(
            temperature=0.1,  # Low temperature for consistent parsing
            top_p=0.8,
            top_k=20,
            max_output_tokens=2048
        )

        self.logger.info("Query Translation Agent initialized successfully")

    def _calculate_default_date_range(self, months: int = 6) -> Tuple[str, str]:
        """
        Calculate default date range for queries without specific dates.
        
        Args:
            months: Number of months to go back from today
            
        Returns:
            Tuple of (start_date, end_date) in ISO format
        """
        today = datetime.now().date()
        start_date = today - timedelta(days=months * 30)  # Approximate month calculation
        return start_date.isoformat(), today.isoformat()

    def _parse_relative_time_period(self, query: str, relative_period: str = None) -> Tuple[Optional[str], Optional[str], str]:
        """
        Parse relative time periods from query text and return appropriate date range.
        Enhanced to handle comparison queries dynamically.
        """
        text_to_analyze = (relative_period or query).lower()
        today = datetime.now().date()
        
        # Enhanced patterns for comparison queries
        comparison_patterns = [
            # More flexible patterns to catch "this month vs past 3 months"
            (r'compar(?:e|ison).*(?:between|of)\s+([^,\s]+(?:\s+\d+\s+\w+)?)\s*(?:and|vs|versus)\s+([^,\s]+(?:\s+\d+\s+\w+)?)', 'comparison'),
            (r'([^,\s]+(?:\s+\d+\s+\w+)?)\s+(?:vs|versus)\s+([^,\s]+(?:\s+\d+\s+\w+)?)', 'comparison'),
            (r'(?:between|compare)\s+([^,\s]+(?:\s+\d+\s+\w+)?)\s+(?:and|vs|versus)\s+([^,\s]+(?:\s+\d+\s+\w+)?)', 'comparison'),
            # "difference between X and Y"
            (r'difference.*between\s+([^,\s]+(?:\s+\d+\s+\w+)?)\s*(?:and|vs)\s+([^,\s]+(?:\s+\d+\s+\w+)?)', 'comparison'),
        ]
        
        # Check for comparison patterns first
        for pattern, pattern_type in comparison_patterns:
            match = re.search(pattern, text_to_analyze)
            if match:
                period1, period2 = match.groups()
                # For comparisons, we need a broader range that covers both periods
                start_date, end_date = self._calculate_comparison_range(period1.strip(), period2.strip())
                if start_date and end_date:
                    return start_date, end_date, f"comparison between {period1.strip()} and {period2.strip()}"
        
        # Existing patterns for single time periods...
        comparison_patterns = [
        # More flexible patterns to catch various comparison formats
        (r'compar(?:e|ison).*(?:between|of)\s+([^,\s]+(?:\s+\d+\s+\w+)?)\s*(?:and|vs|versus)\s+([^,\s]+(?:\s+\d+\s+\w+)?)', 'comparison'),
        (r'([^,\s]+(?:\s+\d+\s+\w+)?)\s+(?:vs|versus)\s+([^,\s]+(?:\s+\d+\s+\w+)?)', 'comparison'),
        (r'(?:between|compare)\s+([^,\s]+(?:\s+\d+\s+\w+)?)\s+(?:and|vs|versus)\s+([^,\s]+(?:\s+\d+\s+\w+)?)', 'comparison'),
        # "difference between X and Y"
        (r'difference.*between\s+([^,\s]+(?:\s+\d+\s+\w+)?)\s*(?:and|vs)\s+([^,\s]+(?:\s+\d+\s+\w+)?)', 'comparison'),
        # Enhanced patterns for year/week/day comparisons
        (r'(this\s+year)\s+(?:vs|versus)\s+(last\s+year)', 'comparison'),
        (r'(past\s+\d+\s+months?)\s+(?:vs|versus)\s+(past\s+year)', 'comparison'),
        (r'(last\s+week)\s+(?:vs|versus)\s+(past\s+\d+\s+days?)', 'comparison'),
        ]
        
        for pattern, pattern_type in patterns:
            match = re.search(pattern, text_to_analyze)
            if match:
                if pattern_type == 'this_month':
                    # Current month
                    start_date = today.replace(day=1)
                    return start_date.isoformat(), today.isoformat(), "this month"
                elif pattern_type == 'last_month':
                    # Previous month
                    last_month_start, last_month_end = self._get_last_month_range()
                    return last_month_start.isoformat(), last_month_end.isoformat(), "last month"
                elif 'number' in pattern_type:
                    value = int(match.group(1))
                    unit = match.group(2)
                else:
                    value = 1
                    unit = match.group(1)
                
                # Calculate start date based on unit (existing logic)
                if unit in ['day', 'days']:
                    start_date = today - timedelta(days=value)
                elif unit in ['week', 'weeks']:
                    start_date = today - timedelta(weeks=value)
                elif unit in ['month', 'months']:
                    start_date = today - timedelta(days=value * 30)
                elif unit in ['quarter', 'quarters']:
                    start_date = today - timedelta(days=value * 90)
                elif unit in ['year', 'years']:
                    start_date = today - timedelta(days=value * 365)
                else:
                    continue
                
                period_desc = f"past {value} {unit}{'s' if value > 1 else ''}"
                return start_date.isoformat(), today.isoformat(), period_desc
        
        # If no pattern matched, return None values
        return None, None, relative_period or "unspecified period"

    async def process(self, request: Dict[str, Any]) -> QueryTranslationResult:
        """
        Main processing method for the Query Translation Agent.

        Args:
            request: Request containing query and context

        Returns:
            QueryTranslationResult with structured query and processing metadata
        """
        try:
            query = request.get("query", "")
            user_context = request.get("user_context", {})
            user_id = request.get("user_id", self.user_id)

            # Set user context if provided
            if user_id:
                self.set_user_context(user_id)

            self.logger.info(f"Processing query translation for: {query[:100]}...")

            # Step 1: Parse query using LLM with structured output
            structured_query = await self._parse_query_with_llm(query, user_context)

            if not structured_query:
                return QueryTranslationResult(
                    success=False,
                    error="Failed to parse natural language query",
                    processing_metadata={
                        "agent": self.agent_name,
                        "timestamp": datetime.now().isoformat()
                    }
                )

            # Step 2: Assess complexity and determine if decomposition is needed
            complexity = await self._assess_query_complexity(structured_query)

            # Step 3: Decompose query if needed
            decomposition = None
            if complexity.requires_decomposition:
                decomposition = await self._decompose_query(structured_query, complexity)

            # Step 4: Validate query feasibility
            validation = await self._validate_query(structured_query, complexity)

            # Step 5: Update conversation context
            await self._update_conversation_context(query, structured_query)

            # Prepare response
            result = QueryTranslationResult(
                success=True,
                structured_query=structured_query,
                analysis_type=structured_query.query_type.value,
                decomposition=decomposition,
                validation=validation,
                processing_metadata={
                    "agent": self.agent_name,
                    "timestamp": datetime.now().isoformat(),
                    "confidence": structured_query.confidence_score,
                    "complexity": complexity.level
                }
            )

            # Store successful translation in session
            self.session_queries.append({
                "original_query": query,
                "structured_query": structured_query.model_dump(),
                "timestamp": datetime.now().isoformat()
            })

            return result

        except Exception as e:
            self.logger.error(f"Error in query translation: {str(e)}")
            return QueryTranslationResult(
                success=False,
                error=str(e),
                processing_metadata={
                    "agent": self.agent_name,
                    "timestamp": datetime.now().isoformat(),
                    "original_query": request.get("query", "")
                }
            )

    async def _assess_query_complexity(self, structured_query: StructuredQuery) -> QueryComplexity:
        """
        Assess the complexity of a structured query.

        Args:
            structured_query: The parsed query structure

        Returns:
            QueryComplexity assessment
        """
        complexity_factors = []

        # Check various complexity indicators
        if len(structured_query.entities) > 5:
            complexity_factors.append("high_entity_count")

        if structured_query.query_type in [QueryType.COMPARISON, QueryType.TREND_ANALYSIS, QueryType.FORECAST]:
            complexity_factors.append("complex_analysis_type")

        if structured_query.time_range.type == TimeReference.CUSTOM_RANGE:
            complexity_factors.append("custom_time_range")

        if len(structured_query.filters.categories) > 3:
            complexity_factors.append("multiple_category_filters")

        if structured_query.analysis_parameters.grouping:
            complexity_factors.append("grouping_required")

        # Determine complexity level
        if len(complexity_factors) == 0:
            level = "low"
            estimated_time = 1.0
            requires_decomposition = False
        elif len(complexity_factors) <= 2:
            level = "medium"
            estimated_time = 3.0
            requires_decomposition = False
        else:
            level = "high"
            estimated_time = 10.0
            requires_decomposition = True

        return QueryComplexity(
            level=level,
            factors=complexity_factors,
            estimated_execution_time=estimated_time,
            requires_decomposition=requires_decomposition
        )

    async def _decompose_query(self, structured_query: StructuredQuery, complexity: QueryComplexity) -> Optional[QueryDecomposition]:
        """
        Decompose a complex query into simpler sub-queries.

        Args:
            structured_query: The complex query to decompose
            complexity: Complexity assessment

        Returns:
            QueryDecomposition or None if decomposition isn't needed
        """
        if not complexity.requires_decomposition:
            return None

        try:
            prompt = f"""
Given this complex financial query, break it down into simpler sub-queries that can be executed independently.

Original Query: {structured_query.original_query}
Query Type: {structured_query.query_type}
Complexity Factors: {complexity.factors}

Create sub-queries that:
1. Can be executed in parallel where possible
2. Have clear dependencies where sequential execution is needed
3. Can be aggregated back into a final result

Return a JSON structure with:
- main_query_id: identifier for the main query
- sub_queries: list of simpler queries with dependencies
- execution_order: optimal order of execution
- aggregation_method: how to combine results

Each sub-query should be a complete, executable query on its own.
"""

            response = await self.model.generate_content_async(
                prompt,
                generation_config=self.generation_config
            )

            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]

            decomposition_data = json.loads(response_text)

            # Convert to QueryDecomposition object
            # Note: This is simplified - you'd need to properly convert sub-queries
            return QueryDecomposition(**decomposition_data)

        except Exception as e:
            self.logger.error(f"Error in query decomposition: {str(e)}")
            return None

    async def _update_conversation_context(self, original_query: str, structured_query: StructuredQuery):
        """Update the conversation context with the new query."""
        self.conversation_context.update({
            "last_query": original_query,
            "last_query_type": structured_query.query_type.value,
            "last_time_range": structured_query.time_range.model_dump(),
            "query_count": self.conversation_context.get("query_count", 0) + 1,
            "updated_at": datetime.now().isoformat()
        })

    def _build_context_info(self, user_context: Dict[str, Any]) -> str:
        """Build context information string for the prompt."""
        if not user_context:
            return "No user context provided. Using default assumptions for currency (USD) and timezone (UTC)."

        context_parts = []

        if user_context.get("timezone"):
            context_parts.append(f"User timezone: {user_context['timezone']}")

        if user_context.get("currency"):
            context_parts.append(f"Currency: {user_context['currency']}")

        if user_context.get("preferred_categories"):
            context_parts.append(f"User's common categories: {', '.join(user_context['preferred_categories'])}")

        return "\n".join(context_parts) if context_parts else "No specific user context available."

    def _get_recent_conversation_context(self) -> str:
        """Get recent conversation context for the prompt."""
        if not self.session_queries:
            return "No previous queries in this session."

        recent_queries = self.session_queries[-3:]  # Last 3 queries
        context_parts = []

        for i, query_info in enumerate(recent_queries, 1):
            context_parts.append(f"{i}. {query_info['original_query']} (Type: {query_info['structured_query'].get('query_type', 'unknown')})")

        return "Recent queries:\n" + "\n".join(context_parts)

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session queries."""
        return {
            "total_queries": len(self.session_queries),
            "conversation_context": self.conversation_context,
            "recent_queries": self.session_queries[-5:] if self.session_queries else []
        }

    def _get_last_month_range(self) -> Tuple[datetime.date, datetime.date]:
        """Get the start and end dates for last month."""
        today = datetime.now().date()
        if today.month == 1:
            last_month_start = today.replace(year=today.year-1, month=12, day=1)
            last_month_end = today.replace(year=today.year-1, month=12, day=31)
        else:
            import calendar
            last_month_start = today.replace(month=today.month-1, day=1)
            last_month_end = today.replace(month=today.month-1, day=calendar.monthrange(today.year, today.month-1)[1])
        
        return last_month_start, last_month_end

    def _calculate_comparison_range(self, period1: str, period2: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Calculate date range that encompasses both periods for comparison.
        Enhanced to handle mixed period types.
        """
        today = datetime.now().date()
        
        # Enhanced period mapping with calculated ranges
        def get_period_range(period: str) -> Tuple[Optional[datetime.date], Optional[datetime.date]]:
            period = period.lower().strip()
            
            # Direct period mapping
            if period == 'this month':
                return today.replace(day=1), today
            elif period == 'last month':
                return self._get_last_month_range()
            elif period == 'this year':
                return today.replace(month=1, day=1), today
            elif period == 'last year':
                return (today.replace(year=today.year-1, month=1, day=1), 
                    today.replace(year=today.year-1, month=12, day=31))
            elif period == 'this week':
                # Monday to today
                monday = today - timedelta(days=today.weekday())
                return monday, today
            elif period == 'last week':
                # Previous Monday to Sunday
                monday_this_week = today - timedelta(days=today.weekday())
                sunday_last_week = monday_this_week - timedelta(days=1)
                monday_last_week = sunday_last_week - timedelta(days=6)
                return monday_last_week, sunday_last_week
            
            # Pattern-based parsing for "past X months/weeks/days/year"
            patterns = [
                (r'past\s+(\d+)\s*(month|week|day)s?', 'past_number'),
                (r'last\s+(\d+)\s*(month|week|day)s?', 'past_number'),
                (r'past\s+year', 'past_year'),  # Handle "past year" specifically
            ]
            
            for pattern, pattern_type in patterns:
                match = re.search(pattern, period)
                if match:
                    if pattern_type == 'past_year':
                        # Past year = 365 days ago to today
                        start_date = today - timedelta(days=365)
                        return start_date, today
                    else:
                        value = int(match.group(1))
                        unit = match.group(2)
                        
                        if unit in ['month', 'months']:
                            start_date = today - timedelta(days=value * 30)
                        elif unit in ['week', 'weeks']:
                            start_date = today - timedelta(weeks=value)
                        elif unit in ['day', 'days']:
                            start_date = today - timedelta(days=value)
                        else:
                            return None, None
                        
                        return start_date, today
            
            return None, None
        
        # Fallback to default range if periods not recognized
        self.logger.warning(f"Could not parse comparison periods: '{period1}' vs '{period2}', using default range")
        return self._calculate_default_date_range(months=4)  # 4 months to be safe for most comparisons
    
    async def _parse_query_with_llm(self, query: str, user_context: Dict[str, Any]) -> Optional[StructuredQuery]:
        try:
            # Handle empty or None user_context
            if not user_context:
                user_context = {}

            context_info = self._build_context_info(user_context)
            conversation_history = self._get_recent_conversation_context()

            prompt = f"""
You are a financial query parsing expert. Parse the following natural language query into a structured format.

Context Information:
{context_info}

Recent Conversation Context:
{conversation_history}

User Query: "{query}"

IMPORTANT INSTRUCTIONS FOR TIME PARSING:
- For queries containing "past year", "last year", set time_range.type to "custom_range" and calculate dates for exactly 1 year ago from today
- For queries containing "past X months/weeks/days", set time_range.type to "custom_range" and calculate the exact date range
- For standard references like "last month", "this year", use the appropriate enum value
- When setting custom_range, ALWAYS provide both start_date and end_date in ISO format (YYYY-MM-DD)
- Current date context: Today is {datetime.now().date().isoformat()}
TIME RANGE GUIDELINES:
- If no specific time period is mentioned in the query (e.g., "what did I spend most on"), use "custom_range"
- For open-ended queries without time context, set type to "custom_range" with start_date and end_date as null
- The system will apply a reasonable default range for data processing
- For queries like "what did I spend most on" without time context, use "custom_range" with null dates

VALID QUERY TYPES (use EXACTLY these values):
- "spending_analysis" - for analyzing spending patterns, amounts, totals
- "trend_analysis" - for trends over time, growth, changes
- "comparison" - for comparing periods, categories, merchants
- "budget_check" - for budget-related queries, budget vs actual, budget status
- "category_breakdown" - for category-wise analysis
- "merchant_analysis" - for merchant-specific analysis  
- "anomaly_detection" - for unusual spending detection
- "forecast" - for future predictions
- "goal_tracking" - for financial goal progress
- "general_inquiry" - for general questions

Default assumptions for missing context:
- Default currency: USD
- Default timezone: UTC
- Categories should be normalized to common financial categories (food, groceries, dining, transportation, etc.)

IMPORTANT: Return a complete JSON object matching this exact structure:
{{
    "query_type": "budget_check",
    "entities": [
        {{
            "entity_type": "budget",
            "value": "monthly_budget",
            "confidence": 0.9,
            "original_text": "budget",
            "normalized_value": "monthly_budget"
        }}
    ],
    "time_range": {{
        "type": "this_month",
        "start_date": null,
        "end_date": null,
        "relative_period": "this month"
    }},
    "filters": {{
        "categories": [],
        "merchants": [],
        "amount_min": null,
        "amount_max": null,
        "tags": [],
        "exclude_categories": []
    }},
    "analysis_parameters": {{
        "aggregation_level": "monthly",
        "comparison_baseline": "budget",
        "metrics": ["total_amount", "budget_remaining"],
        "grouping": [],
        "sorting": null,
        "limit": null
    }},
    "context_requirements": ["transactions", "budgets"],
    "original_query": "{query}",
    "confidence_score": 0.85,
    "requires_clarification": false,
    "clarification_questions": []
}}

CRITICAL: 
1. Use ONLY the exact query_type values listed above
2. For budget-related queries, use "budget_check" (NOT "budget_analysis")
3. For time periods like "past year", "last 12 months", etc., you MUST:
   - Set type to "custom_range"
   - Calculate exact start_date (1 year ago from today)
   - Set end_date to today's date
   - Set relative_period to describe the period

Return ONLY the JSON, no other text.
"""
            response = await self.model.generate_content_async(
                prompt,
                generation_config=self.generation_config
            )

            # Parse JSON response into StructuredQuery
            response_text = response.text.strip()

            # Clean up response text (remove markdown formatting if present)
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]

            # Parse JSON
            parsed_json = json.loads(response_text.strip())

            # POST-PROCESSING: Fix common LLM mistakes with query_type
            query_type_mapping = {
                "budget_analysis": "budget_check",
                "spending_breakdown": "category_breakdown",
                "expense_analysis": "spending_analysis",
                "transaction_analysis": "spending_analysis",
                "financial_analysis": "spending_analysis",
                "cost_analysis": "spending_analysis"
            }
            
            current_query_type = parsed_json.get('query_type')
            if current_query_type in query_type_mapping:
                self.logger.info(f"Mapping query_type from '{current_query_type}' to '{query_type_mapping[current_query_type]}'")
                parsed_json['query_type'] = query_type_mapping[current_query_type]

            # Clean up None values that should have defaults
            if parsed_json.get('analysis_parameters', {}).get('aggregation_level') is None:
                parsed_json['analysis_parameters']['aggregation_level'] = 'monthly'

            # Ensure other required fields have defaults if None
            if not parsed_json.get('entities'):
                parsed_json['entities'] = []
            if not parsed_json.get('context_requirements'):
                parsed_json['context_requirements'] = []
            if not parsed_json.get('clarification_questions'):
                parsed_json['clarification_questions'] = []

            # Enhanced post-processing for time ranges
            time_range = parsed_json.get('time_range', {})
            start_date, end_date, period_desc = None, None, None
            # If LLM didn't properly handle the time range, apply our logic
            if time_range.get('type') == TimeReference.CUSTOM_RANGE.value:
        
                # Case 1: Custom range with missing dates (open-ended query)
                if (not time_range.get('start_date') or not time_range.get('end_date')):
                    
                    # Try to parse the time range from the original query first
                    start_date, end_date, period_desc = self._parse_relative_time_period(
                        query, time_range.get('relative_period')
                    )
                
                if start_date and end_date:
                    # Found specific time reference in query
                    parsed_json['time_range']['start_date'] = start_date
                    parsed_json['time_range']['end_date'] = end_date
                    parsed_json['time_range']['relative_period'] = period_desc
                    self.logger.info(f"Applied parsed time range: {start_date} to {end_date} for '{period_desc}'")
                else:
                    # Open-ended query without specific time context - apply default range
                    self.logger.info("Open-ended query detected, applying default range for data processing")
                    default_start, default_end = self._calculate_default_date_range(months=12)  # Use 12 months for open-ended
                    parsed_json['time_range']['start_date'] = default_start
                    parsed_json['time_range']['end_date'] = default_end
                    parsed_json['time_range']['relative_period'] = "all available data (last 12 months default)"
                    # Create StructuredQuery object
                    structured_query = StructuredQuery(**parsed_json)

                    self.logger.info(f"Successfully parsed query with confidence: {structured_query.confidence_score}")
                    self.logger.info(f"Time range: {structured_query.time_range.type.value} ({structured_query.time_range.start_date} to {structured_query.time_range.end_date})")
                    return structured_query

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {str(e)}")
            self.logger.error(f"Response text: {response_text[:500] if 'response_text' in locals() else 'No response'}")
            return None
        except Exception as e:
            self.logger.error(f"Error in LLM query parsing: {str(e)}")
            return None

    async def _validate_query(self, structured_query: StructuredQuery, complexity: QueryComplexity) -> ValidationResult:
        """
        Validate query feasibility and provide suggestions.

        Args:
            structured_query: Query to validate
            complexity: Complexity assessment

        Returns:
            ValidationResult
        """
        issues = []
        suggestions = []
        required_data = ["transactions"]  # Base requirement

        
        # In the _validate_query method, replace the custom_range validation:
        # Check time range validity - improved validation for default ranges
        if structured_query.time_range.type == TimeReference.CUSTOM_RANGE:
            if not structured_query.time_range.start_date or not structured_query.time_range.end_date:
                # Check if this is an open-ended query (our system applied defaults)
                if (structured_query.time_range.relative_period and 
                    "default" in structured_query.time_range.relative_period.lower()):
                    # This is fine - system applied reasonable defaults for open-ended query
                    suggestions.append("Using default time range for open-ended query")
                else:
                    # Truly missing dates that should have been provided
                    issues.append("Custom date range specified but dates are missing")
                    suggestions.append("Please provide specific start and end dates")
            else:
                # Existing validation logic for explicit custom ranges...
                try:
                    start = datetime.fromisoformat(structured_query.time_range.start_date)
                    end = datetime.fromisoformat(structured_query.time_range.end_date)
                    if start > end:
                        issues.append("Start date is after end date")
                        suggestions.append("Please ensure start date is before end date")
                    
                    # Check if date range is too large (more than 3 years)
                    if (end - start).days > 1095:  # 3 years
                        suggestions.append("Date range is quite large, consider narrowing it for better performance")
                        
                except ValueError:
                    issues.append("Invalid date format in time range")
                    suggestions.append("Please use ISO format (YYYY-MM-DD) for dates")

        # Check for data requirements based on query type
        if structured_query.query_type == QueryType.BUDGET_CHECK:
            required_data.append("budgets")
        elif structured_query.query_type == QueryType.GOAL_TRACKING:
            required_data.append("financial_goals")
        elif structured_query.query_type == QueryType.FORECAST:
            required_data.append("historical_data")
            # Check if we have enough historical data for forecasting
            if structured_query.time_range.type != TimeReference.CUSTOM_RANGE:
                suggestions.append("Forecasting typically requires specific historical data range")

        # Confidence-based validation
        if structured_query.confidence_score < 0.6:
            issues.append("Query parsing confidence is low")
            suggestions.append("Consider rephrasing the query for better clarity")
        elif structured_query.confidence_score < 0.8:
            suggestions.append("Query could be more specific for better results")

        # Check for potential issues with filters
        if (structured_query.filters.amount_min is not None and 
            structured_query.filters.amount_max is not None and
            structured_query.filters.amount_min > structured_query.filters.amount_max):
            issues.append("Minimum amount is greater than maximum amount")
            suggestions.append("Please check your amount range filters")

        # Check for conflicting categories
        if (structured_query.filters.categories and 
            structured_query.filters.exclude_categories):
            common_categories = set(structured_query.filters.categories) & set(structured_query.filters.exclude_categories)
            if common_categories:
                issues.append(f"Categories appear in both include and exclude filters: {', '.join(common_categories)}")
                suggestions.append("Remove conflicting categories from either include or exclude filters")

        # Validate entity consistency
        for entity in structured_query.entities:
            if entity.confidence < 0.5:
                suggestions.append(f"Low confidence entity detected: {entity.original_text}")

        # Check if clarification is needed
        if structured_query.requires_clarification and not structured_query.clarification_questions:
            issues.append("Query requires clarification but no questions provided")

        is_valid = len(issues) == 0

        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            suggestions=suggestions,
            required_data=required_data,
            estimated_complexity=complexity
        )

async def main():
    agent = QueryTranslationAgent()
    request = {
        "query": "Help me save money on shopping",  # Test query without explicit dates
        "user_context": {
            "timezone": "Asia/Kolkata",
            "currency": "INR",
            "preferred_categories": ["groceries", "food", "transport", "electronics"]
        },
        "user_id": "a73ff731-9018-45ed-86ff-214e91baf702"
    }
    result = await agent.process(request)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())