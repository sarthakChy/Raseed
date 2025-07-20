import asyncio
import asyncpg
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import json
import hashlib
from dataclasses import dataclass
from contextlib import asynccontextmanager

from google.cloud import secretmanager
import redis.asyncio as redis


@dataclass
class QueryResult:
    """Structured result from database queries."""
    data: List[Dict[str, Any]]
    row_count: int
    execution_time: float
    cached: bool = False
    query_hash: Optional[str] = None


class DatabaseConnector:
    """
    Database connector for financial analysis agents.
    Handles PostgreSQL connections, query execution, caching, and optimization
    specifically for financial data operations.
    """
    
    def __init__(
        self,
        project_id: str,
        connection_pool_size: int = 20,
        redis_url: Optional[str] = None,
        cache_ttl: int = 300  # 5 minutes default cache
    ):
        """
        Initialize database connector.
        
        Args:
            project_id: Google Cloud project ID for secret management
            connection_pool_size: PostgreSQL connection pool size
            redis_url: Redis URL for caching (optional)
            cache_ttl: Default cache TTL in seconds
        """
        self.project_id = project_id
        self.connection_pool_size = connection_pool_size
        self.cache_ttl = cache_ttl
        
        # Connection pool and cache
        self.pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        
        # Setup logging
        self.logger = logging.getLogger("financial_agent.database_connector")
        
        # Secret manager client
        self.secret_client = secretmanager.SecretManagerServiceClient()
        
        # Query cache (in-memory fallback)
        self.memory_cache: Dict[str, Tuple[Any, datetime]] = {}
        
        # Financial data specific configurations
        self.financial_tables = {
            'transactions': 'user_transactions',
            'categories': 'spending_categories',
            'merchants': 'merchants',
            'budgets': 'user_budgets',
            'goals': 'financial_goals',
            'accounts': 'user_accounts'
        }
        
        # Common financial queries (pre-optimized)
        self.prepared_queries = {
            'monthly_spending': """
                SELECT 
                    DATE_TRUNC('month', transaction_date) as month,
                    category,
                    SUM(amount) as total_amount,
                    COUNT(*) as transaction_count,
                    AVG(amount) as avg_amount
                FROM user_transactions 
                WHERE user_id = $1 
                    AND transaction_date >= $2 
                    AND transaction_date <= $3
                GROUP BY month, category
                ORDER BY month DESC, total_amount DESC
            """,
            
            'spending_trends': """
                WITH monthly_totals AS (
                    SELECT 
                        DATE_TRUNC('month', transaction_date) as month,
                        SUM(amount) as monthly_total
                    FROM user_transactions 
                    WHERE user_id = $1 
                        AND transaction_date >= $2
                    GROUP BY month
                ),
                trend_analysis AS (
                    SELECT 
                        month,
                        monthly_total,
                        LAG(monthly_total) OVER (ORDER BY month) as prev_month,
                        AVG(monthly_total) OVER (
                            ORDER BY month 
                            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                        ) as three_month_avg
                    FROM monthly_totals
                )
                SELECT 
                    month,
                    monthly_total,
                    prev_month,
                    three_month_avg,
                    CASE 
                        WHEN prev_month IS NOT NULL 
                        THEN ((monthly_total - prev_month) / prev_month) * 100 
                        ELSE NULL 
                    END as month_over_month_change
                FROM trend_analysis
                ORDER BY month DESC
            """,
            
            'category_analysis': """
                SELECT 
                    c.category_name,
                    c.category_type,
                    COUNT(t.id) as transaction_count,
                    SUM(t.amount) as total_amount,
                    AVG(t.amount) as avg_amount,
                    MIN(t.amount) as min_amount,
                    MAX(t.amount) as max_amount,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY t.amount) as median_amount
                FROM user_transactions t
                JOIN spending_categories c ON t.category_id = c.id
                WHERE t.user_id = $1 
                    AND t.transaction_date >= $2 
                    AND t.transaction_date <= $3
                GROUP BY c.category_name, c.category_type
                ORDER BY total_amount DESC
            """,
            
            'merchant_analysis': """
                SELECT 
                    m.merchant_name,
                    m.merchant_category,
                    COUNT(t.id) as visit_count,
                    SUM(t.amount) as total_spent,
                    AVG(t.amount) as avg_transaction,
                    MIN(t.transaction_date) as first_transaction,
                    MAX(t.transaction_date) as last_transaction
                FROM user_transactions t
                JOIN merchants m ON t.merchant_id = m.id
                WHERE t.user_id = $1 
                    AND t.transaction_date >= $2 
                    AND t.transaction_date <= $3
                GROUP BY m.merchant_name, m.merchant_category
                ORDER BY total_spent DESC
            """,
            
            'budget_performance': """
                SELECT 
                    b.category,
                    b.monthly_limit,
                    COALESCE(SUM(t.amount), 0) as actual_spending,
                    b.monthly_limit - COALESCE(SUM(t.amount), 0) as remaining_budget,
                    CASE 
                        WHEN b.monthly_limit > 0 
                        THEN (COALESCE(SUM(t.amount), 0) / b.monthly_limit) * 100 
                        ELSE 0 
                    END as budget_utilization
                FROM user_budgets b
                LEFT JOIN user_transactions t ON b.user_id = t.user_id 
                    AND b.category = t.category
                    AND DATE_TRUNC('month', t.transaction_date) = DATE_TRUNC('month', CURRENT_DATE)
                WHERE b.user_id = $1 AND b.active = true
                GROUP BY b.category, b.monthly_limit
                ORDER BY budget_utilization DESC
            """
        }
    
    async def initialize(self):
        """Initialize database connections and cache."""
        try:
            # Get database credentials from Secret Manager
            db_config = await self._get_database_credentials()
            
            # Initialize PostgreSQL connection pool
            self.pool = await asyncpg.create_pool(
                **db_config,
                min_size=5,
                max_size=self.connection_pool_size,
                command_timeout=30
            )
            
            # Initialize Redis cache if URL provided
            if hasattr(self, 'redis_url') and self.redis_url:
                self.redis_client = redis.from_url(self.redis_url)
                await self.redis_client.ping()
                self.logger.info("Redis cache initialized")
            
            # Prepare common queries
            await self._prepare_queries()
            
            self.logger.info("Database connector initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database connector: {e}")
            raise
    
    async def _get_database_credentials(self) -> Dict[str, str]:
        """Retrieve database credentials from Google Secret Manager."""
        try:
            secret_name = f"projects/{self.project_id}/secrets/postgres-config/versions/latest"
            response = self.secret_client.access_secret_version(request={"name": secret_name})
            secret_data = json.loads(response.payload.data.decode("UTF-8"))
            
            return {
                "host": secret_data["host"],
                "port": secret_data.get("port", 5432),
                "database": secret_data["database"],
                "user": secret_data["user"],
                "password": secret_data["password"],
                "ssl": "require"
            }
        except Exception as e:
            self.logger.error(f"Failed to retrieve database credentials: {e}")
            raise
    
    async def _prepare_queries(self):
        """Prepare common financial queries for better performance."""
        async with self.pool.acquire() as conn:
            for query_name, query_sql in self.prepared_queries.items():
                try:
                    await conn.prepare(query_sql)
                    self.logger.debug(f"Prepared query: {query_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to prepare query {query_name}: {e}")
    
    def _generate_cache_key(self, query: str, parameters: Optional[Dict] = None) -> str:
        """Generate cache key for query and parameters."""
        cache_data = {"query": query, "parameters": parameters or {}}
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    async def _get_cached_result(self, cache_key: str) -> Optional[QueryResult]:
        """Retrieve cached query result."""
        try:
            # Try Redis first
            if self.redis_client:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    result_data = json.loads(cached_data)
                    return QueryResult(**result_data, cached=True)
            
        except Exception as e:
            self.logger.warning(f"Cache retrieval error: {e}")
        
        return None
    
    async def _cache_result(self, cache_key: str, result: QueryResult):
        """Cache query result."""
        try:
            cache_data = {
                "data": result.data,
                "row_count": result.row_count,
                "execution_time": result.execution_time,
                "query_hash": result.query_hash
            }
            
            # Cache in Redis
            if self.redis_client:
                await self.redis_client.setex(
                    cache_key, 
                    self.cache_ttl, 
                    json.dumps(cache_data, default=str)
                )
            
        except Exception as e:
            self.logger.warning(f"Cache storage error: {e}")
    
    async def execute_query(
        self, 
        query: str, 
        parameters: Optional[Union[Dict, List]] = None,
        cache_key: Optional[str] = None,
        use_cache: bool = True
    ) -> QueryResult:
        """
        Execute a database query with caching and error handling.
        
        Args:
            query: SQL query string
            parameters: Query parameters
            cache_key: Custom cache key (optional)
            use_cache: Whether to use caching
            
        Returns:
            QueryResult object with data and metadata
        """
        start_time = datetime.now()
        
        # Generate cache key
        if not cache_key:
            cache_key = self._generate_cache_key(query, parameters)
        
        # Check cache if enabled
        if use_cache:
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                self.logger.debug(f"Cache hit for query: {cache_key[:16]}...")
                return cached_result
        
        # Execute query
        try:
            async with self.pool.acquire() as conn:
                if parameters:
                    if isinstance(parameters, dict):
                        # Named parameters - convert to positional
                        param_values = list(parameters.values())
                        rows = await conn.fetch(query, *param_values)
                    else:
                        # Positional parameters
                        rows = await conn.fetch(query, *parameters)
                else:
                    rows = await conn.fetch(query)
                
                # Convert to dictionaries
                data = [dict(row) for row in rows]
                execution_time = (datetime.now() - start_time).total_seconds()
                
                result = QueryResult(
                    data=data,
                    row_count=len(data),
                    execution_time=execution_time,
                    query_hash=hashlib.md5(query.encode()).hexdigest()[:16]
                )
                
                # Cache the result
                if use_cache and len(data) > 0:
                    await self._cache_result(cache_key, result)
                
                self.logger.info(
                    f"Query executed: {result.row_count} rows, "
                    f"{result.execution_time:.3f}s"
                )
                
                return result
                
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            raise
    
    async def execute_prepared_query(
        self, 
        query_name: str, 
        parameters: List[Any],
        use_cache: bool = True
    ) -> QueryResult:
        """
        Execute a prepared financial query.
        
        Args:
            query_name: Name of prepared query
            parameters: Query parameters
            use_cache: Whether to use caching
            
        Returns:
            QueryResult object
        """
        if query_name not in self.prepared_queries:
            raise ValueError(f"Unknown prepared query: {query_name}")
        
        query = self.prepared_queries[query_name]
        return await self.execute_query(query, parameters, use_cache=use_cache)
    
    async def get_user_spending_summary(
        self, 
        user_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get comprehensive spending summary for a user."""
        try:
            # Execute multiple queries in parallel
            tasks = [
                self.execute_prepared_query(
                    'monthly_spending', 
                    [user_id, start_date, end_date]
                ),
                self.execute_prepared_query(
                    'category_analysis', 
                    [user_id, start_date, end_date]
                ),
                self.execute_prepared_query(
                    'merchant_analysis', 
                    [user_id, start_date, end_date]
                ),
                self.execute_prepared_query(
                    'budget_performance', 
                    [user_id]
                )
            ]
            
            monthly_data, category_data, merchant_data, budget_data = await asyncio.gather(*tasks)
            
            return {
                "user_id": user_id,
                "period": {"start": start_date, "end": end_date},
                "monthly_spending": monthly_data.data,
                "category_analysis": category_data.data,
                "merchant_analysis": merchant_data.data,
                "budget_performance": budget_data.data,
                "metadata": {
                    "total_queries": 4,
                    "total_execution_time": sum([
                        monthly_data.execution_time,
                        category_data.execution_time,
                        merchant_data.execution_time,
                        budget_data.execution_time
                    ]),
                    "cached_results": sum([
                        monthly_data.cached,
                        category_data.cached,
                        merchant_data.cached,
                        budget_data.cached
                    ])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get spending summary for user {user_id}: {e}")
            raise
    
    async def invalidate_cache(self, pattern: Optional[str] = None):
        """Invalidate cached queries."""
        try:
            if pattern:
                # Clear specific pattern
                if self.redis_client:
                    keys = await self.redis_client.keys(f"*{pattern}*")
                    if keys:
                        await self.redis_client.delete(*keys)
                
                # Clear from memory cache
                keys_to_delete = [k for k in self.memory_cache.keys() if pattern in k]
                for key in keys_to_delete:
                    del self.memory_cache[key]
            else:
                # Clear all cache
                if self.redis_client:
                    await self.redis_client.flushdb()
                self.memory_cache.clear()
                
            self.logger.info(f"Cache invalidated: {pattern or 'all'}")
            
        except Exception as e:
            self.logger.error(f"Failed to invalidate cache: {e}")
    
    @asynccontextmanager
    async def transaction(self):
        """Database transaction context manager."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                yield conn
    
    async def health_check(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            start_time = datetime.now()
            
            # Test basic connectivity
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Test cache connectivity
            cache_status = "disabled"
            if self.redis_client:
                try:
                    await self.redis_client.ping()
                    cache_status = "connected"
                except:
                    cache_status = "error"
            
            return {
                "database": "connected" if result == 1 else "error",
                "response_time": response_time,
                "cache": cache_status,
                "pool_size": self.pool.get_size(),
                "active_connections": len(self.pool._holders),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "database": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def close(self):
        """Close database connections and cache."""
        try:
            if self.pool:
                await self.pool.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            self.logger.info("Database connector closed")
            
        except Exception as e:
            self.logger.error(f"Error closing database connector: {e}")