import os
from firebase_admin import credentials
import firebase_admin
from google.cloud import firestore, storage
import tempfile
import logging
import atexit
import json
import re

logger = logging.getLogger(__name__)

def get_credentials():
    """Load credentials from environment variables."""

    try:
        PROJECT_ID = os.getenv("PROJECT_ID")
        LOCATION = os.getenv("GCP_LOCATION")
        GOOGLE_WALLET_ISSUER_ID = os.getenv("GOOGLE_WALLET_ISSUER_ID")
    except KeyError:
        raise RuntimeError("GCP_PROJECT_ID not found in .env file.")

    #Initialize cloud storage and firestore clients

    try:
        BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
    except KeyError:
        raise RuntimeError("GCS_BUCKET_NAME not found in .env file. Please set it.")

    return PROJECT_ID, LOCATION, BUCKET_NAME, GOOGLE_WALLET_ISSUER_ID

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

def parse_json(raw_text):
    """
    Extracts and parses a JSON object from a Markdown code block (e.g., ```json ... ```).

    Args:
        raw_text (str): Raw text that includes a Markdown-style JSON block.

    Returns:
        dict: Parsed JSON data as a Python dictionary.

    Raises:
        ValueError: If no valid JSON block is found or JSON parsing fails.
    """
    if not isinstance(raw_text, str):
        raise ValueError("Input must be a string")

    # Match ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON block found in the input text")

    json_text = match.group(1).strip()

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")

