from google.cloud import firestore, storage
import os
import json
import tempfile
import logging
import atexit

logger = logging.getLogger(__name__)

# Global variable to keep track of temporary credentials file
_temp_credentials_path = None

def _cleanup_temp_credentials():
    """Clean up temporary credentials file on exit"""
    global _temp_credentials_path
    if _temp_credentials_path and os.path.exists(_temp_credentials_path):
        try:
            os.unlink(_temp_credentials_path)
            logger.info("Cleaned up temporary credentials file")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary credentials file: {e}")

def _setup_credentials():
    """Set up credentials from environment variable and return the temp file path"""
    global _temp_credentials_path
    
    # If credentials are already set up, return the existing path
    if _temp_credentials_path and os.path.exists(_temp_credentials_path):
        return _temp_credentials_path
    
    try:
        service_account_json = os.getenv("FIREBASE_CREDENTIALS")
        if not service_account_json:
            raise ValueError("Service account credentials not found in environment.")

        # Parse JSON to verify it's valid
        credentials_dict = json.loads(service_account_json)

        # Create a temporary file for credentials
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
            temp_file.write(json.dumps(credentials_dict).encode("utf-8"))
            _temp_credentials_path = temp_file.name

        # Set the environment variable for GOOGLE_APPLICATION_CREDENTIALS
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _temp_credentials_path
        
        # Register cleanup function to run on exit
        atexit.register(_cleanup_temp_credentials)
        
        logger.info(f"Temporary credentials file created at: {_temp_credentials_path}")
        return _temp_credentials_path

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse service account JSON: {e}")
        raise
    except Exception as e:
        logger.error(f"Credentials setup error: {e}")
        raise

def initialize_firestore(database=None):
    """
    Dynamically create a temporary credentials file from environment variables
    and initialize Firestore with the specified database.
    """
    try:
        # Set up credentials (this will reuse existing temp file if already created)
        _setup_credentials()

        # Initialize Firestore client
        if database:
            client = firestore.Client(database=database)
        else:
            client = firestore.Client()

        logger.info("Firestore client initialized successfully")
        return client

    except Exception as e:
        logger.error(f"Firestore initialization error: {e}")
        raise

def initialize_gcs_client():
    """
    Dynamically create a temporary credentials file from environment variables
    and initialize the Google Cloud Storage client.
    """
    try:
        # Set up credentials (this will reuse existing temp file if already created)
        _setup_credentials()

        # Initialize the Google Cloud Storage client
        storage_client = storage.Client()
        
        logger.info("GCS client initialized successfully")
        return storage_client

    except Exception as e:
        logger.error(f"GCS client initialization error: {e}")
        raise