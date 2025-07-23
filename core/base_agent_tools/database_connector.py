import asyncpg
import json
import logging
from typing import Optional
from google.cloud import secretmanager
from contextlib import asynccontextmanager


class DatabaseConnector:
    def __init__(self, project_id: str, connection_pool_size: int = 10):
        self.project_id = project_id
        self.connection_pool_size = connection_pool_size
        self.pool: Optional[asyncpg.Pool] = None
        self.logger = logging.getLogger("DatabaseConnector")
        self.secret_client = secretmanager.SecretManagerServiceClient()

    async def initialize(self):
        creds = await self._get_database_credentials()
        self.pool = await asyncpg.create_pool(
            **creds,
            min_size=5,
            max_size=self.connection_pool_size,
            command_timeout=30
        )
        self.logger.info("Database pool initialized.")

    async def _get_database_credentials(self) -> dict:
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

    async def get_pool(self) -> asyncpg.Pool:
        if not self.pool:
            raise RuntimeError("DatabaseConnector not initialized.")
        return self.pool

    @asynccontextmanager
    async def get_connection(self):
        if not self.pool:
            raise RuntimeError("DatabaseConnector not initialized.")
        async with self.pool.acquire() as conn:
            yield conn