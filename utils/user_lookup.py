import logging
from typing import Optional
from utils.database_connector import DatabaseConnector
import asyncio

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:\Google Agentic AI\Code Lab\Raseed\server\serviceAccountKey.json"

async def get_user_id_by_firebase_uid(
    firebase_uid: str,
    project_id: str
) -> Optional[str]:
    """
    Fetch the user_id from the database using the provided firebase_uid.

    Args:
        firebase_uid: Firebase UID of the user
        project_id: Google Cloud project ID for database config

    Returns:
        user_id if found, else None
    """
    try:
        db_manager = await DatabaseConnector.get_instance(project_id)
        query = "SELECT user_id FROM users WHERE firebase_uid = $1"

        async with db_manager.get_connection() as conn:
            row = await conn.fetchrow(query, firebase_uid)
            if row:
                return row["user_id"]
            else:
                logger.warning(f"No user found for firebase_uid: {firebase_uid}")
                return None

    except Exception as e:
        logger.error(f"Failed to fetch user_id for firebase_uid {firebase_uid}: {e}")
        return None
