import asyncio
import asyncpg
import logging
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import json
from decimal import Decimal
import pandas as pd
from google.cloud import secretmanager


class PostgreSQLQueryTool:
    def __init__(
        self,
        project_id: str,
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.project_id = project_id
        self.connection_pool: Optional[asyncpg.Pool] = None
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

        self.query_templates = self._initialize_query_templates()
        
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
    
    def _initialize_query_templates(self) -> Dict[str, str]:
        """Initialize query templates for different analysis types."""
        return {
            # Basic transaction queries
            'transactions_basic': """
                SELECT 
                    t.transaction_id,
                    t.amount,
                    t.category,
                    t.subcategory,
                    t.transaction_date,
                    m.name as merchant_name,
                    m.normalized_name as merchant_normalized,
                    t.payment_method
                FROM transactions t
                LEFT JOIN merchants m ON t.merchant_id = m.merchant_id
                WHERE t.user_id = $1 
                AND t.deleted_at IS NULL
                {date_filter}
                {category_filter}
                {amount_filter}
                ORDER BY t.transaction_date DESC
                LIMIT $2
            """,
            
            # Category aggregations with time series
            'category_aggregations': """
                SELECT 
                    t.category,
                    t.subcategory,
                    COUNT(*) as transaction_count,
                    SUM(t.amount) as total_amount,
                    AVG(t.amount) as avg_amount,
                    MIN(t.amount) as min_amount,
                    MAX(t.amount) as max_amount,
                    STDDEV(t.amount) as amount_stddev,
                    DATE_TRUNC('{time_period}', t.transaction_date) as period
                FROM transactions t
                WHERE t.user_id = $1 
                AND t.deleted_at IS NULL
                {date_filter}
                {category_filter}
                GROUP BY t.category, t.subcategory, DATE_TRUNC('{time_period}', t.transaction_date)
                ORDER BY period DESC, total_amount DESC
            """,
            
            # Merchant analysis
            'merchant_analysis': """
                SELECT 
                    m.name as merchant_name,
                    m.normalized_name,
                    m.category as merchant_category,
                    COUNT(t.transaction_id) as visit_count,
                    SUM(t.amount) as total_spent,
                    AVG(t.amount) as avg_transaction,
                    MIN(t.transaction_date) as first_visit,
                    MAX(t.transaction_date) as last_visit,
                    DATE_TRUNC('month', t.transaction_date) as month
                FROM transactions t
                JOIN merchants m ON t.merchant_id = m.merchant_id
                WHERE t.user_id = $1 
                AND t.deleted_at IS NULL
                {date_filter}
                GROUP BY m.merchant_id, m.name, m.normalized_name, m.category, DATE_TRUNC('month', t.transaction_date)
                ORDER BY total_spent DESC
            """,
            
            # Time series trend analysis
            'time_series_trends': """
                WITH daily_spending AS (
                    SELECT 
                        t.transaction_date,
                        t.category,
                        SUM(t.amount) as daily_amount,
                        COUNT(*) as transaction_count
                    FROM transactions t
                    WHERE t.user_id = $1 
                    AND t.deleted_at IS NULL
                    {date_filter}
                    {category_filter}
                    GROUP BY t.transaction_date, t.category
                ),
                spending_with_trends AS (
                    SELECT *,
                        AVG(daily_amount) OVER (
                            PARTITION BY category 
                            ORDER BY transaction_date 
                            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                        ) as moving_avg_7day,
                        LAG(daily_amount, 7) OVER (
                            PARTITION BY category 
                            ORDER BY transaction_date
                        ) as amount_7days_ago,
                        PERCENT_RANK() OVER (
                            PARTITION BY category 
                            ORDER BY daily_amount
                        ) as percentile_rank
                    FROM daily_spending
                )
                SELECT 
                    transaction_date,
                    category,
                    daily_amount,
                    transaction_count,
                    moving_avg_7day,
                    CASE 
                        WHEN amount_7days_ago IS NOT NULL 
                        THEN ((daily_amount - amount_7days_ago) / amount_7days_ago) * 100
                        ELSE NULL
                    END as week_over_week_change,
                    percentile_rank
                FROM spending_with_trends
                ORDER BY transaction_date DESC, category
            """,
            
            # Budget comparison analysis
            'budget_analysis': """
                WITH category_spending AS (
                    SELECT 
                        t.category,
                        SUM(t.amount) as spent_amount,
                        COUNT(*) as transaction_count
                    FROM transactions t
                    WHERE t.user_id = $1 
                    AND t.deleted_at IS NULL
                    AND t.transaction_date >= DATE_TRUNC('{period}', CURRENT_DATE)
                    GROUP BY t.category
                ),
                budget_limits AS (
                    SELECT 
                        bl.category,
                        bl.limit_amount,
                        bl.period_type,
                        bl.current_spent
                    FROM budget_limits bl
                    WHERE bl.user_id = $1
                    AND bl.period_type = '{period}'
                    AND (bl.effective_to IS NULL OR bl.effective_to >= CURRENT_DATE)
                    AND bl.effective_from <= CURRENT_DATE
                )
                SELECT 
                    COALESCE(cs.category, bl.category) as category,
                    COALESCE(cs.spent_amount, 0) as actual_spent,
                    COALESCE(bl.limit_amount, 0) as budget_limit,
                    COALESCE(cs.transaction_count, 0) as transaction_count,
                    CASE 
                        WHEN bl.limit_amount > 0 
                        THEN (COALESCE(cs.spent_amount, 0) / bl.limit_amount) * 100
                        ELSE NULL
                    END as budget_utilization_percent,
                    CASE
                        WHEN bl.limit_amount > 0 AND cs.spent_amount > bl.limit_amount
                        THEN cs.spent_amount - bl.limit_amount
                        ELSE 0
                    END as over_budget_amount
                FROM category_spending cs
                FULL OUTER JOIN budget_limits bl ON cs.category = bl.category
                ORDER BY budget_utilization_percent DESC NULLS LAST
            """,
            
            # Spending pattern analysis
            'spending_patterns': """
                WITH spending_by_day_of_week AS (
                    SELECT 
                        EXTRACT(DOW FROM t.transaction_date) as day_of_week,
                        t.category,
                        AVG(t.amount) as avg_amount,
                        COUNT(*) as transaction_count,
                        SUM(t.amount) as total_amount
                    FROM transactions t
                    WHERE t.user_id = $1 
                    AND t.deleted_at IS NULL
                    {date_filter}
                    GROUP BY EXTRACT(DOW FROM t.transaction_date), t.category
                ),
                spending_by_hour AS (
                    SELECT 
                        EXTRACT(HOUR FROM t.transaction_time) as hour_of_day,
                        t.category,
                        AVG(t.amount) as avg_amount,
                        COUNT(*) as transaction_count
                    FROM transactions t
                    WHERE t.user_id = $1 
                    AND t.deleted_at IS NULL
                    AND t.transaction_time IS NOT NULL
                    {date_filter}
                    GROUP BY EXTRACT(HOUR FROM t.transaction_time), t.category
                )
                SELECT 
                    'day_of_week' as pattern_type,
                    dow.day_of_week::text as pattern_value,
                    dow.category,
                    dow.avg_amount,
                    dow.transaction_count,
                    dow.total_amount
                FROM spending_by_day_of_week dow
                UNION ALL
                SELECT 
                    'hour_of_day' as pattern_type,
                    h.hour_of_day::text as pattern_value,
                    h.category,
                    h.avg_amount,
                    h.transaction_count,
                    NULL as total_amount
                FROM spending_by_hour h
                ORDER BY pattern_type, pattern_value::int
            """,
            
            # Anomaly detection query
            'anomaly_detection': """
                WITH transaction_stats AS (
                    SELECT 
                        t.category,
                        AVG(t.amount) as mean_amount,
                        STDDEV(t.amount) as stddev_amount,
                        COUNT(*) as total_transactions
                    FROM transactions t
                    WHERE t.user_id = $1 
                    AND t.deleted_at IS NULL
                    AND t.transaction_date >= CURRENT_DATE - INTERVAL '{lookback_days} days'
                    GROUP BY t.category
                    HAVING COUNT(*) >= 5  -- Minimum transactions for statistical significance
                ),
                recent_transactions AS (
                    SELECT 
                        t.transaction_id,
                        t.amount,
                        t.category,
                        t.transaction_date,
                        m.name as merchant_name,
                        ts.mean_amount,
                        ts.stddev_amount,
                        ABS(t.amount - ts.mean_amount) / NULLIF(ts.stddev_amount, 0) as z_score
                    FROM transactions t
                    LEFT JOIN merchants m ON t.merchant_id = m.merchant_id
                    JOIN transaction_stats ts ON t.category = ts.category
                    WHERE t.user_id = $1 
                    AND t.deleted_at IS NULL
                    AND t.transaction_date >= CURRENT_DATE - INTERVAL '{recent_days} days'
                )
                SELECT 
                    transaction_id,
                    amount,
                    category,
                    transaction_date,
                    merchant_name,
                    mean_amount,
                    z_score,
                    CASE 
                        WHEN z_score > 2.5 THEN 'high_anomaly'
                        WHEN z_score > 1.5 THEN 'moderate_anomaly'
                        ELSE 'normal'
                    END as anomaly_level
                FROM recent_transactions
                WHERE z_score > 1.5
                ORDER BY z_score DESC
            """,
            
            # Financial goal progress tracking
            'goal_progress': """
                WITH goal_spending AS (
                    SELECT 
                        fg.goal_id,
                        fg.title,
                        fg.goal_type,
                        fg.target_amount,
                        fg.current_amount,
                        fg.target_date,
                        fg.category,
                        COALESCE(SUM(t.amount), 0) as recent_spending
                    FROM financial_goals fg
                    LEFT JOIN transactions t ON (
                        fg.category = t.category 
                        AND t.user_id = fg.user_id
                        AND t.deleted_at IS NULL
                        AND t.transaction_date >= CURRENT_DATE - INTERVAL '30 days'
                    )
                    WHERE fg.user_id = $1
                    AND fg.status = 'active'
                    GROUP BY fg.goal_id, fg.title, fg.goal_type, fg.target_amount, 
                             fg.current_amount, fg.target_date, fg.category
                )
                SELECT 
                    goal_id,
                    title,
                    goal_type,
                    target_amount,
                    current_amount,
                    target_date,
                    category,
                    recent_spending,
                    (current_amount / NULLIF(target_amount, 0)) * 100 as progress_percentage,
                    CASE 
                        WHEN target_date IS NOT NULL 
                        THEN target_date - CURRENT_DATE
                        ELSE NULL
                    END as days_remaining,
                    CASE 
                        WHEN goal_type = 'budget_reduction' AND recent_spending > 0
                        THEN ((target_amount - recent_spending) / NULLIF(target_amount, 0)) * 100
                        ELSE NULL
                    END as reduction_progress
                FROM goal_spending
                ORDER BY progress_percentage DESC
            """
        }
    
    def _build_filters(self, filters: Dict[str, Any], initial_param_idx: int) -> Tuple[str, List[Any], int]:
        """Build SQL filter clauses and parameters from filter dictionary.
        Returns (filter_clause_string, parameters_list, next_param_index)."""
        all_filter_parts = []
        params = []
        param_count = initial_param_idx # Start parameter indexing from here

        if not filters:
            return "", [], initial_param_idx

        if 'date_range' in filters and filters['date_range']:
            date_range = filters['date_range']
            if 'start_date' in date_range:
                # Convert date string to datetime.date object
                start_date_obj = datetime.strptime(date_range['start_date'], '%Y-%m-%d').date()
                all_filter_parts.append(f"t.transaction_date >= ${param_count}")
                params.append(start_date_obj)
                param_count += 1
            if 'end_date' in date_range:
                # Convert date string to datetime.date object
                end_date_obj = datetime.strptime(date_range['end_date'], '%Y-%m-%d').date()
                all_filter_parts.append(f"t.transaction_date <= ${param_count}")
                params.append(end_date_obj)
                param_count += 1

        # Category filter
        if 'categories' in filters and filters['categories']:
            categories = filters['categories']
            all_filter_parts.append(f"t.category = ANY(${param_count})")
            params.append(categories)
            param_count += 1

        # Payment method filter
        if 'payment_methods' in filters and filters['payment_methods']:
            payment_methods = filters['payment_methods']
            all_filter_parts.append(f"t.payment_method = ANY(${param_count})")
            params.append(payment_methods)
            param_count += 1

        # Amount range filter
        if 'amount_range' in filters and filters['amount_range']:
            amount_range = filters['amount_range']
            if 'min_amount' in amount_range:
                all_filter_parts.append(f"t.amount >= ${param_count}")
                params.append(amount_range['min_amount'])
                param_count += 1
            if 'max_amount' in amount_range:
                all_filter_parts.append(f"t.amount <= ${param_count}")
                params.append(amount_range['max_amount'])
                param_count += 1
        
        filter_clause_string = " AND " + " AND ".join(all_filter_parts) if all_filter_parts else ""
        # FIXED: Return the tuple correctly - the current code returns 3 values, but some callers expect only 2
        return filter_clause_string, params, param_count
    
    async def execute_query(
        self, 
        query_type: str, 
        user_id: str = None,
        sql_query: str = None,
        filters: Dict[str, Any] = None,
        aggregation: str = None,
        group_by: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        try:
            await self.initialize_connection_pool()
            
            self.logger.info(f"Executing query type: {query_type}, user_id: {user_id}, filters: {filters}, kwargs: {kwargs}")
            
            if user_id is None and query_type != "custom":
                self.logger.error("No valid user_id provided")
                return {
                    "success": False,
                    "error": "No valid user_id provided",
                    "query_type": query_type
                }
            
            if query_type == "custom" and sql_query:
                self.logger.info(f"Custom query: {sql_query}")
                return await self._execute_custom_query(sql_query, user_id)
            
            # Execute predefined query types
            if query_type == "transactions":
                result = await self._execute_transactions_query(user_id, filters, **kwargs)
            elif query_type == "aggregations":
                result = await self._execute_aggregations_query(user_id, filters, aggregation, group_by, **kwargs)
            elif query_type == "trends":
                result = await self._execute_trends_query(user_id, filters, **kwargs)
            elif query_type == "comparisons":
                result = await self._execute_comparisons_query(user_id, filters, **kwargs)
            elif query_type == "patterns":
                result = await self._execute_patterns_query(user_id, filters, **kwargs)
            elif query_type == "anomalies":
                result = await self._execute_anomaly_detection(user_id, **kwargs)
            elif query_type == "budget_analysis":
                result = await self._execute_budget_analysis(user_id, **kwargs)
            elif query_type == "goal_progress":
                result = await self._execute_goal_progress(user_id)
            else:
                raise ValueError(f"Unsupported query type: {query_type}")
            
            self.logger.info(f"Query result: {result}")
            return result
                
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query_type": query_type
            }
    
    async def _execute_transactions_query(
        self,
        user_id: str,
        filters: Dict[str, Any] = None,
        limit: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        filters = filters or {}
        
        # user_id is $1. So filters and the LIMIT clause start from $2.
        # This will correctly build the filter clause with parameters starting from $2
        filter_clause, filter_params, next_param_idx = self._build_filters(filters, initial_param_idx=2)

        # The LIMIT parameter will be the very next parameter after all filters.
        limit_param_idx = next_param_idx

        base_query_template_parts = self.query_templates['transactions_basic'].split('LIMIT $2')
        query_before_limit = base_query_template_parts[0].strip()

        base_query_parts = self.query_templates['transactions_basic'].split('{date_filter}')
        select_from_where_fixed = base_query_parts[0] # Contains up to AND t.deleted_at IS NULL

        # Get filter clauses and their parameters
        # `initial_param_idx=2` because user_id is $1.
        filter_clause, filter_params, next_param_idx_after_filters = self._build_filters(filters, initial_param_idx=2)

        # The LIMIT parameter will be the next one after all filter parameters.
        limit_param_index = next_param_idx_after_filters

        # Construct the final query string.
        # It's crucial that the `filter_clause` is inserted correctly, and `LIMIT` is at the correct index.
        final_query = f"""
            {select_from_where_fixed.strip()}
            {filter_clause.strip()}
            ORDER BY t.transaction_date DESC
            LIMIT ${limit_param_index}
        """

        # Assemble all parameters: user_id ($1), then filter parameters, then limit.
        params = [user_id] + filter_params + [limit]

        self.logger.info(f"Executing transactions query: {final_query}")
        self.logger.info(f"Parameters: {params}")

        if user_id is None:
            self.logger.error("No valid user_id provided")
            return {
                "success": False,
                "error": "No valid user_id provided",
                "query_type": "transactions"
            }

        async with self.connection_pool.acquire() as conn:
            # When executing, make sure the number of parameters matches the placeholders.
            rows = await conn.fetch(final_query, *params)

            return {
                "success": True,
                "data": [dict(row) for row in rows],
                "count": len(rows),
                "query_type": "transactions"
            }
    
    async def _execute_aggregations_query(
        self,
        user_id: str,
        filters: Dict[str, Any] = None,
        aggregation: str = "sum",
        group_by: List[str] = None,
        time_period: str = "month",
        **kwargs
    ) -> Dict[str, Any]:
        """Execute category aggregations query."""
        filters = filters or {}
        filter_clause, filter_params, _ = self._build_filters(filters, initial_param_idx=2)
        
        query = self.query_templates['category_aggregations'].format(
            time_period=time_period,
            date_filter=filter_clause.replace('AND t.transaction_date', 'AND t.transaction_date') if 'date_range' in str(filter_clause) else '',
            category_filter=filter_clause.replace('AND t.category', 'AND t.category') if 'categories' in str(filter_clause) else ''
        )
        
        params = [user_id] + filter_params
        
        async with self.connection_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            
            return {
                "success": True,
                "data": [dict(row) for row in rows],
                "aggregation_type": aggregation,
                "time_period": time_period,
                "query_type": "aggregations"
            }
    
    async def _execute_trends_query(
        self,
        user_id: str,
        filters: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute time series trends analysis."""
        filters = filters or {}
        filter_clause, filter_params, _ = self._build_filters(filters, initial_param_idx=2)
        
        query = self.query_templates['time_series_trends'].format(
            date_filter=filter_clause.replace('AND t.transaction_date', 'AND t.transaction_date') if 'date_range' in str(filter_clause) else '',
            category_filter=filter_clause.replace('AND t.category', 'AND t.category') if 'categories' in str(filter_clause) else ''
        )
        
        params = [user_id] + filter_params
        
        async with self.connection_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            
            return {
                "success": True,
                "data": [dict(row) for row in rows],
                "query_type": "trends"
            }

    async def _execute_comparisons_query(
        self,
        user_id: str,
        filters: Dict[str, Any] = None,
        comparison_type: str = "merchant",
        **kwargs
    ) -> Dict[str, Any]:
        """Execute merchant or category comparison analysis."""
        filters = filters or {}
        filter_clause, filter_params, _ = self._build_filters(filters, initial_param_idx=2)
        
        if comparison_type == "merchant":
            query = self.query_templates['merchant_analysis'].format(
                date_filter=filter_clause.replace('AND t.transaction_date', 'AND t.transaction_date') if 'date_range' in str(filter_clause) else ''
            )
        else:
            # Default to category comparison
            query = self.query_templates['category_aggregations'].format(
                time_period='month',
                date_filter=filter_clause.replace('AND t.transaction_date', 'AND t.transaction_date') if 'date_range' in str(filter_clause) else '',
                category_filter=''
            )
        
        params = [user_id] + filter_params
        
        async with self.connection_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            
            return {
                "success": True,
                "data": [dict(row) for row in rows],
                "comparison_type": comparison_type,
                "query_type": "comparisons"
            }

    async def _execute_patterns_query(
        self,
        user_id: str,
        filters: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute spending patterns analysis."""
        filters = filters or {}
        filter_clause, filter_params, _ = self._build_filters(filters, initial_param_idx=2)
        
        query = self.query_templates['spending_patterns'].format(
            date_filter=filter_clause.replace('AND t.transaction_date', 'AND t.transaction_date') if 'date_range' in str(filter_clause) else ''
        )
        
        params = [user_id] + filter_params
        
        async with self.connection_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            
            return {
                "success": True,
                "data": [dict(row) for row in rows],
                "query_type": "patterns"
            }
    
    async def _execute_anomaly_detection(
        self,
        user_id: str,
        lookback_days: int = 90,
        recent_days: int = 7,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute anomaly detection analysis."""
        query = self.query_templates['anomaly_detection'].format(
            lookback_days=lookback_days,
            recent_days=recent_days
        )
        
        async with self.connection_pool.acquire() as conn:
            rows = await conn.fetch(query, user_id)
            
            return {
                "success": True,
                "data": [dict(row) for row in rows],
                "lookback_days": lookback_days,
                "recent_days": recent_days,
                "query_type": "anomalies"
            }
    
    async def _execute_budget_analysis(
        self,
        user_id: str,
        period: str = "monthly",
        **kwargs
    ) -> Dict[str, Any]:
        """Execute budget vs actual spending analysis."""
        query = self.query_templates['budget_analysis'].format(period=period)
        
        async with self.connection_pool.acquire() as conn:
            rows = await conn.fetch(query, user_id)
            
            return {
                "success": True,
                "data": [dict(row) for row in rows],
                "period": period,
                "query_type": "budget_analysis"
            }
    
    async def _execute_goal_progress(self, user_id: str, **kwargs) -> Dict[str, Any]:
        """Execute financial goal progress analysis."""
        query = self.query_templates['goal_progress']
        
        async with self.connection_pool.acquire() as conn:
            rows = await conn.fetch(query, user_id)
            
            return {
                "success": True,
                "data": [dict(row) for row in rows],
                "query_type": "goal_progress"
            }
    
    async def _execute_custom_query(self, sql_query: str, user_id: str = None) -> Dict[str, Any]:
        """Execute custom SQL query with safety checks."""
        # Basic safety checks
        sql_lower = sql_query.lower().strip()
        
        # Prevent dangerous operations
        dangerous_keywords = ['drop', 'delete', 'truncate', 'alter', 'create', 'insert', 'update']
        if any(keyword in sql_lower for keyword in dangerous_keywords):
            raise ValueError("Custom query contains potentially dangerous operations")
        
        # Ensure it's a SELECT query
        if not sql_lower.startswith('select'):
            raise ValueError("Only SELECT queries are allowed for custom queries")
        
        async with self.connection_pool.acquire() as conn:
            rows = await conn.fetch(sql_query)
            
            return {
                "success": True,
                "data": [dict(row) for row in rows],
                "query_type": "custom"
            }
    
    async def get_user_financial_summary(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive financial summary for a user."""
        try:
            # Execute multiple queries in parallel for comprehensive summary
            tasks = [
                self.execute_query("aggregations", user_id=user_id, 
                                 filters={"date_range": {"start_date": (datetime.now() - timedelta(days=days)).date()}},
                                 aggregation="sum", group_by=["category"]),
                self.execute_query("trends", user_id=user_id,
                                 filters={"date_range": {"start_date": (datetime.now() - timedelta(days=days)).date()}}),
                self.execute_query("budget_analysis", user_id=user_id),
                self.execute_query("goal_progress", user_id=user_id),
                self.execute_query("anomalies", user_id=user_id, recent_days=7, lookback_days=30)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return {
                "success": True,
                "summary": {
                    "spending_by_category": results[0] if not isinstance(results[0], Exception) else None,
                    "trends": results[1] if not isinstance(results[1], Exception) else None,
                    "budget_status": results[2] if not isinstance(results[2], Exception) else None,
                    "goal_progress": results[3] if not isinstance(results[3], Exception) else None,
                    "recent_anomalies": results[4] if not isinstance(results[4], Exception) else None
                },
                "period_days": days,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate financial summary: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def __del__(self):
        """Cleanup connection pool on object destruction."""
        if self.connection_pool:
            try:
                asyncio.create_task(self.close_connection_pool())
            except:
                pass