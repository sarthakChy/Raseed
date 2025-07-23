import logging
import asyncio
import re
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

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
    aggregation_level: str = Field(default="monthly")  # Changed from "daily"
    comparison_baseline: Optional[str] = Field(default=None)
    metrics: List[str] = Field(default_factory=list)
    grouping: List[str] = Field(default_factory=list)
    sorting: Optional[str] = Field(default=None)
    limit: Optional[int] = Field(default=None)


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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.dict()

    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.json()


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

Since limited context is available, make reasonable assumptions:
- Default currency: USD
- Default timezone: UTC
- Time references like "last month", "in the past 2 months", or explicit date ranges should be interpreted relative to current date or as custom ranges.
    - If a relative period (e.g., "past 2 months") is detected, set `time_range.type` to "custom_range" and calculate `start_date` and `end_date`. The `end_date` should be today's date, and `start_date` should be calculated by subtracting the specified period from today's date.
    - For explicit date ranges (e.g., "between 2024-01-15 and 2025-07-23"), set `time_range.type` to "custom_range" and populate `start_date` and `end_date` directly.
- Categories should be normalized to common financial categories.

IMPORTANT: Return a complete JSON object matching this exact structure:
{{
    "query_type": "spending_analysis",
    "entities": [
        {{
            "entity_type": "category",
            "value": "groceries",
            "confidence": 0.9,
            "original_text": "groceries",
            "normalized_value": "groceries"
        }}
    ],
    "time_range": {{
        "type": "last_month",
        "start_date": null,
        "end_date": null,
        "relative_period": "previous month"
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
        "comparison_baseline": null,
        "metrics": [],
        "grouping": [],
        "sorting": null,
        "limit": null
    }},
    "context_requirements": [],
    "original_query": "{query}",
    "confidence_score": 0.85,
    "requires_clarification": false,
    "clarification_questions": []
}}

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

            # --- Start of new logic for handling custom ranges based on relative periods ---
            # If the LLM identifies a relative period, ensure time_range.type is 'custom_range'
            # and calculate start/end dates.
            if parsed_json.get('time_range', {}).get('relative_period'):
                relative_period = parsed_json['time_range']['relative_period'].lower()
                today = datetime.now().date()
                start_date = None
                end_date = today

                # Simple parsing for "past X [days/weeks/months/years]"
                match_past_period = re.match(r'past (\d+) (day|week|month|year)s?', relative_period)
                if match_past_period:
                    value = int(match_past_period.group(1))
                    unit = match_past_period.group(2)
                    if unit == 'day':
                        start_date = today - timedelta(days=value)
                    elif unit == 'week':
                        start_date = today - timedelta(weeks=value)
                    elif unit == 'month':
                        # This is a simplification; for exact months, you might need calendar logic
                        start_date = today - timedelta(days=value * 30)
                    elif unit == 'year':
                        start_date = today - timedelta(days=value * 365)
                    
                    parsed_json['time_range']['type'] = TimeReference.CUSTOM_RANGE.value
                    parsed_json['time_range']['start_date'] = start_date.isoformat()
                    parsed_json['time_range']['end_date'] = end_date.isoformat()
                    parsed_json['time_range']['relative_period'] = relative_period # Keep original relative period description
                elif parsed_json['time_range']['type'] != TimeReference.CUSTOM_RANGE.value:
                    # If it's a relative period but not recognized by our simple regex,
                    # and it's not already 'custom_range', force it to 'custom_range'
                    # and leave start/end date for further processing or clarification
                    parsed_json['time_range']['type'] = TimeReference.CUSTOM_RANGE.value
                    parsed_json['time_range']['start_date'] = None
                    parsed_json['time_range']['end_date'] = None


            # If the LLM has already provided explicit start_date and end_date,
            # ensure time_range.type is set to CUSTOM_RANGE if not already.
            if parsed_json.get('time_range', {}).get('start_date') and \
               parsed_json.get('time_range', {}).get('end_date') and \
               parsed_json['time_range']['type'] != TimeReference.CUSTOM_RANGE.value:
                parsed_json['time_range']['type'] = TimeReference.CUSTOM_RANGE.value
            # --- End of new logic ---

            # Create StructuredQuery object
            structured_query = StructuredQuery(**parsed_json)

            self.logger.info(f"Successfully parsed query with confidence: {structured_query.confidence_score}")
            return structured_query

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {str(e)}")
            self.logger.error(f"Response text: {response_text[:500] if 'response_text' in locals() else 'No response'}")
            return None
        except Exception as e:
            self.logger.error(f"Error in LLM query parsing: {str(e)}")
            return None

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

        # Check time range validity
        if structured_query.time_range.type == TimeReference.CUSTOM_RANGE:
            if not structured_query.time_range.start_date or not structured_query.time_range.end_date:
                issues.append("Custom date range specified but dates are missing")
                suggestions.append("Please provide specific start and end dates")

        # Check for data requirements
        if structured_query.query_type == QueryType.BUDGET_CHECK:
            required_data.append("budgets")
        elif structured_query.query_type == QueryType.GOAL_TRACKING:
            required_data.append("financial_goals")

        # Confidence-based validation
        if structured_query.confidence_score < 0.6:
            issues.append("Query parsing confidence is low")
            suggestions.append("Consider rephrasing the query for better clarity")

        is_valid = len(issues) == 0

        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            suggestions=suggestions,
            required_data=required_data,
            estimated_complexity=complexity
        )

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

async def main():
    agent = QueryTranslationAgent()
    request = {
        "query": "How much did I spend on electronics in the past 2 years?", # Changed query
        "user_context": {
            "timezone": "Asia/Kolkata",
            "currency": "INR",
            "preferred_categories": ["groceries", "food", "transport", "electronics"] # Added electronics
        },
        "user_id": "a73ff731-9018-45ed-86ff-214e91baf702"
    }
    result = await agent.process(request)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())