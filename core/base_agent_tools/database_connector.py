import asyncpg
import json
import logging
from typing import Optional
from google.cloud import secretmanager
from contextlib import asynccontextmanager
from decimal import Decimal


class DatabaseConnector:
    def __init__(self, project_id: str, connection_pool_size: int = 10):
        self.project_id = project_id
        self.connection_pool_size = connection_pool_size
        self.pool: Optional[asyncpg.Pool] = None
        self.logger = logging.getLogger("DatabaseConnector")
        self.secret_client = secretmanager.SecretManagerServiceClient()

    async def initialize(self):
        """Initialize database connections and cache."""
        try:
            # Get database credentials from Secret Manager
            creds = await self._get_database_credentials()
            
            # Initialize PostgreSQL connection pool
            self.pool = await asyncpg.create_pool(
                **creds,
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
    async def get_connection(self):
        if not self.pool:
            raise RuntimeError("DatabaseConnector not initialized.")
        async with self.pool.acquire() as conn:
            yield conn