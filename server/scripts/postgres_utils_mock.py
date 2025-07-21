import os
from sqlalchemy import create_engine
from google.cloud.sql.connector import Connector, IPTypes
import logging

def init_connection_pool() -> create_engine:
    """
    Initializes a connection pool for a Cloud SQL instance of Postgres.
    This function securely connects using the Cloud SQL Python Connector.
    It reads connection details from environment variables.
    """
    try:
        # --- Get connection details from environment variables ---
        # e.g., "your-project-id:your-region:your-instance-id"
        instance_connection_name = os.environ["INSTANCE_CONNECTION_NAME"]
        db_user = os.environ["DB_USER"]  # e.g., "postgres"
        db_pass = os.environ["DB_PASS"]  # The password you set for the user
        db_name = os.environ["DB_NAME"]  # e.g., "postgres"
    except KeyError as e:
        raise RuntimeError(f"Missing required database environment variable: {e}")

    # Initialize the Cloud SQL connector
    connector = Connector()

    def getconn():
        """Helper function called by SQLAlchemy to create a new database connection."""
        conn = connector.connect(
            instance_connection_name,
            "pg8000",
            user=db_user,
            password=db_pass,
            db=db_name,
            ip_type=IPTypes.PUBLIC, # Use public IP for local testing and authorized networks
        )
        return conn

    # Create the connection pool using SQLAlchemy
    pool = create_engine(
        "postgresql+pg8000://",
        creator=getconn,
        pool_size=5,          # Max number of connections to keep open
        max_overflow=2,       # Extra connections to allow during high traffic
        pool_timeout=30,      # Time to wait for a connection before raising an error
        pool_recycle=1800,    # Recycle connections every 30 minutes
    )
    return pool
