import asyncio
import logging
import json
import asyncpg
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from google.cloud import secretmanager


class DatabaseConnector:
    """
    Improved database connection manager with better error handling
    and connection lifecycle management.
    """
    _instance: Optional['DatabaseConnector'] = None
    _lock = asyncio.Lock()
    
    def __new__(cls, project_id: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, project_id: str = None):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        if not project_id:
            raise ValueError("project_id is required for first initialization")
            
        self.project_id = project_id
        self.logger = logging.getLogger("DatabaseConnector")
        self.connection_pool: Optional[asyncpg.Pool] = None
        self.secret_client = secretmanager.SecretManagerServiceClient()
        self._db_config: Optional[Dict[str, Any]] = None
        self._initialized = True
        self._initialization_in_progress = False
        self._pool_closed = False
        
        # Connection retry settings
        self.max_retries = 3
        self.retry_delay = 1.0
    
    async def initialize(self) -> None:
        """Initialize the database connection pool with improved error handling."""
        async with self._lock:
            # Check if already initialized and pool is healthy
            if self.connection_pool is not None and not self._pool_closed:
                try:
                    # Test the pool with a quick query
                    async with self.connection_pool.acquire() as conn:
                        await conn.fetchval("SELECT 1")
                    self.logger.info("Connection pool already initialized and healthy")
                    return
                except Exception as e:
                    self.logger.warning(f"Existing pool unhealthy: {e}. Reinitializing...")
                    await self._close_pool()
            
            # Check if initialization is in progress
            if self._initialization_in_progress:
                self.logger.info("Initialization already in progress, waiting...")
                while self._initialization_in_progress:
                    await asyncio.sleep(0.1)
                return
                
            try:
                self._initialization_in_progress = True
                self._pool_closed = False
                
                # Load database configuration from Secret Manager
                if self._db_config is None:
                    await self._load_db_config()
                
                # Validate configuration
                if not self._db_config:
                    raise RuntimeError("Database configuration is empty")
                
                # Create connection pool with improved settings
                self.connection_pool = await asyncpg.create_pool(
                    host=self._db_config['host'],
                    port=self._db_config['port'],
                    database=self._db_config['database'],
                    user=self._db_config['user'],
                    password=self._db_config['password'],
                    min_size=self._db_config.get('min_size', 5),
                    max_size=self._db_config.get('max_size', 20),
                    command_timeout=self._db_config.get('command_timeout', 60),
                    max_queries=50000,  # Reset connections after this many queries
                    max_inactive_connection_lifetime=300.0,  # 5 minutes
                    setup=self._setup_connection,  # Setup function for each connection
                    server_settings={
                        'application_name': 'vector_search_tool',
                        'tcp_keepalives_idle': '600',
                        'tcp_keepalives_interval': '30',
                        'tcp_keepalives_count': '3'
                    }
                )
                
                # Test the connection with retry logic
                await self._test_connection_with_retry()
                
                self.logger.info("PostgreSQL connection pool initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize connection pool: {e}")
                # Clean up on failure
                await self._close_pool()
                raise
            finally:
                self._initialization_in_progress = False
    
    async def _setup_connection(self, conn: asyncpg.Connection) -> None:
        """Setup function called for each new connection."""
        try:
            # Enable vector extension if needed
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            # Set connection-specific settings
            await conn.execute("SET statement_timeout = '60s'")
            await conn.execute("SET lock_timeout = '30s'")
        except Exception as e:
            self.logger.warning(f"Connection setup warning: {e}")
    
    async def _test_connection_with_retry(self) -> None:
        """Test connection with retry logic."""
        for attempt in range(self.max_retries):
            try:
                async with self.connection_pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                return
            except Exception as e:
                self.logger.warning(f"Connection test attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay * (attempt + 1))
    
    async def _load_db_config(self) -> None:
        """Load database configuration from Google Secret Manager."""
        try:
            secret_name = f"projects/{self.project_id}/secrets/postgres-config/versions/latest"
            response = self.secret_client.access_secret_version(request={"name": secret_name})
            secret_data = json.loads(response.payload.data.decode("UTF-8"))
            
            # Validate required fields
            required_fields = ['host', 'database', 'user', 'password']
            for field in required_fields:
                if field not in secret_data:
                    raise ValueError(f"Missing required field in secret: {field}")
            
            self._db_config = {
                'host': secret_data["host"],
                'port': secret_data.get("port", 5432),
                'database': secret_data["database"],
                'user': secret_data["user"],
                'password': secret_data["password"],
                'min_size': secret_data.get('min_size', 3),  # Reduced min size
                'max_size': secret_data.get('max_size', 15),  # Reduced max size
                'command_timeout': secret_data.get('command_timeout', 60)
            }
            
            self.logger.info("Database configuration loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load database configuration: {e}")
            raise
    
    async def get_pool(self) -> asyncpg.Pool:
        """Get the connection pool, initializing if necessary."""
        if self.connection_pool is None or self._pool_closed:
            await self.initialize()
        
        if self.connection_pool is None:
            raise RuntimeError("Failed to initialize database connection pool")
            
        return self.connection_pool
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool with improved error handling."""
        connection = None
        pool = None
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                pool = await self.get_pool()
                
                if pool is None or pool.is_closing():
                    raise RuntimeError("Connection pool is not available or closing")
                
                connection = await asyncio.wait_for(
                    pool.acquire(), 
                    timeout=30.0  # 30 second timeout for acquiring connection
                )
                
                if connection is None or connection.is_closed():
                    raise RuntimeError("Failed to acquire valid connection from pool")
                
                # Test the connection before yielding
                await connection.fetchval("SELECT 1")
                
                yield connection
                break  # Success, exit retry loop
                
            except (asyncpg.ConnectionDoesNotExistError, 
                    asyncpg.InterfaceError, 
                    OSError, 
                    ConnectionResetError,
                    asyncio.TimeoutError) as e:
                self.logger.warning(f"Connection error (attempt {retry_count + 1}): {e}")
                
                # Release the problematic connection
                if connection is not None:
                    try:
                        await pool.release(connection, discard=True)
                    except Exception as release_error:
                        self.logger.warning(f"Error discarding connection: {release_error}")
                    connection = None
                
                retry_count += 1
                if retry_count >= self.max_retries:
                    self.logger.error(f"Max retries ({self.max_retries}) exceeded for connection")
                    # Try to reinitialize the pool as last resort
                    try:
                        await self._close_pool()
                        await self.initialize()
                    except Exception as reinit_error:
                        self.logger.error(f"Failed to reinitialize pool: {reinit_error}")
                    raise e
                
                # Wait before retry
                await asyncio.sleep(self.retry_delay * retry_count)
                
            except Exception as e:
                self.logger.error(f"Unexpected error with database connection: {e}")
                if connection is not None:
                    try:
                        await pool.release(connection, discard=True)
                    except Exception as release_error:
                        self.logger.warning(f"Error discarding connection: {release_error}")
                raise
            finally:
                # Only release if we didn't hit an error that requires discarding
                if connection is not None and not connection.is_closed() and pool is not None:
                    try:
                        await pool.release(connection)
                    except Exception as e:
                        self.logger.warning(f"Error releasing connection: {e}")
    
    async def execute_query(self, query: str, *args) -> list:
        """Execute a query and return results with retry logic."""
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                async with self.get_connection() as conn:
                    return await conn.fetch(query, *args)
            except (asyncpg.ConnectionDoesNotExistError, 
                    asyncpg.InterfaceError, 
                    OSError, 
                    ConnectionResetError) as e:
                self.logger.warning(f"Query retry {retry_count + 1} due to: {e}")
                retry_count += 1
                if retry_count >= self.max_retries:
                    self.logger.error(f"Query failed after {self.max_retries} retries: {e}")
                    raise
                await asyncio.sleep(self.retry_delay * retry_count)
            except Exception as e:
                self.logger.error(f"Query execution failed: {e}")
                raise
    
    async def execute_query_one(self, query: str, *args):
        """Execute a query and return single result with retry logic."""
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                async with self.get_connection() as conn:
                    return await conn.fetchrow(query, *args)
            except (asyncpg.ConnectionDoesNotExistError, 
                    asyncpg.InterfaceError, 
                    OSError, 
                    ConnectionResetError) as e:
                self.logger.warning(f"Query retry {retry_count + 1} due to: {e}")
                retry_count += 1
                if retry_count >= self.max_retries:
                    self.logger.error(f"Query failed after {self.max_retries} retries: {e}")
                    raise
                await asyncio.sleep(self.retry_delay * retry_count)
            except Exception as e:
                self.logger.error(f"Query execution failed: {e}")
                raise
    
    async def execute_command(self, command: str, *args) -> str:
        """Execute a command (INSERT, UPDATE, DELETE) with retry logic."""
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                async with self.get_connection() as conn:
                    return await conn.execute(command, *args)
            except (asyncpg.ConnectionDoesNotExistError, 
                    asyncpg.InterfaceError, 
                    OSError, 
                    ConnectionResetError) as e:
                self.logger.warning(f"Command retry {retry_count + 1} due to: {e}")
                retry_count += 1
                if retry_count >= self.max_retries:
                    self.logger.error(f"Command failed after {self.max_retries} retries: {e}")
                    raise
                await asyncio.sleep(self.retry_delay * retry_count)
            except Exception as e:
                self.logger.error(f"Command execution failed: {e}")
                raise
    
    async def _close_pool(self) -> None:
        """Internal method to close the connection pool."""
        if self.connection_pool and not self.connection_pool.is_closing():
            try:
                await asyncio.wait_for(self.connection_pool.close(), timeout=30.0)
                self.logger.info("PostgreSQL connection pool closed")
            except Exception as e:
                self.logger.error(f"Error closing connection pool: {e}")
            finally:
                self.connection_pool = None
                self._pool_closed = True
    
    async def close(self) -> None:
        """Close the connection pool."""
        async with self._lock:
            await self._close_pool()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check."""
        try:
            start_time = asyncio.get_event_loop().time()
            
            async with self.get_connection() as conn:
                result = await conn.fetchval("SELECT 1")
                
                response_time = (asyncio.get_event_loop().time() - start_time) * 1000
                
                # Get pool statistics if available
                pool_stats = {}
                if self.connection_pool:
                    pool_stats = {
                        "pool_size": self.connection_pool.get_size(),
                        "pool_min_size": self.connection_pool.get_min_size(),
                        "pool_max_size": self.connection_pool.get_max_size(),
                        "pool_idle_size": self.connection_pool.get_idle_size(),
                        "pool_is_closing": self.connection_pool.is_closing()
                    }
                
                return {
                    "status": "healthy",
                    "test_query_result": result,
                    "response_time_ms": round(response_time, 2),
                    "pool_statistics": pool_stats,
                    "pool_closed": self._pool_closed
                }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "pool_closed": self._pool_closed
            }
    
    @classmethod
    async def get_instance(cls, project_id: str = None) -> 'DatabaseConnector':
        """Get the singleton instance, creating if necessary."""
        if cls._instance is None:
            if not project_id:
                raise ValueError("project_id required for first instance creation")
            cls._instance = cls(project_id)
        
        # Ensure it's initialized
        await cls._instance.initialize()
        return cls._instance
    
    @classmethod
    async def reset_instance(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        async with cls._lock:
            if cls._instance and cls._instance.connection_pool:
                await cls._instance.close()
            cls._instance = None
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'connection_pool') and self.connection_pool and not self.connection_pool.is_closing():
            self.logger.warning("DatabaseConnector destroyed without proper cleanup. Call close() explicitly.")