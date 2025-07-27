import os
from firebase_admin import credentials
import firebase_admin
from google.cloud import firestore, storage
import tempfile
import logging
import atexit
import json
import re
import uuid
from datetime import datetime, timezone
from fastapi import UploadFile
from typing import Dict, Any

logger = logging.getLogger(__name__)

from typing import Dict, Any
from google.cloud import firestore, storage
from fastapi import UploadFile
import logging, uuid, random
from datetime import datetime, timedelta

def save_receipt_to_cloud(
    db: firestore.Client, 
    bucket: storage.Bucket, 
    parsed_data: Dict[str, Any], 
    image_bytes: bytes, 
    file: UploadFile,
    uploadedAt: datetime,
    processedAt: datetime,
    receipt_id: str,
    user_id: str,
    email: str,
) -> Dict[str, Any]:
    """
    Uploads a receipt image to GCS and saves its data to Firestore.
    Returns a JSON-serializable dictionary.
    """
    try:
        # 1. Upload image to GCS
        unique_filename = f"receipts/{user_id or 'anonymous'}/{receipt_id}-{file.filename}"
        blob = bucket.blob(unique_filename)
        blob.upload_from_string(image_bytes, content_type=file.content_type)
        logging.info(f"Image uploaded to {unique_filename}.")

        # 2. Check if user exists
        users_ref = db.collection("users")
        query = users_ref.where("userId", "==", user_id)
        results = query.get()
        user_doc = results[0] if results else None

        current_time = datetime.utcnow().isoformat()

        if not user_doc:
            display_name = email.split("@")[0].capitalize()
            user_data = {
                "userId": user_id,
                "email": email,
                "displayName": display_name,
                "photoURL": f"https://api.dicebear.com/8.x/initials/svg?seed={display_name}",
                "createdAt": current_time,
                "lastLoginAt": current_time,
                "preferences": {
                    "currency": "USD",
                    "language": "en",
                    "timezone": "America/New_York",
                    "notifications": {
                        "pushEnabled": True,
                        "emailEnabled": False,
                        "budgetAlerts": True,
                        "spendingInsights": True,
                        "weeklyReports": False,
                        "proactiveInsights": True,
                        "frequency": "daily"
                    },
                    "privacySettings": {
                        "shareData": False,
                        "anonymousAnalytics": True
                    }
                },
                "financialProfile": {
                    "monthlyIncome": random.choice([50000, 75000, 100000]),
                    "budgetLimits": {
                        "total": random.randint(35000, 50000),
                        "groceries": random.randint(12000, 18000),
                        "dining": random.randint(6000, 12000),
                        "entertainment": random.randint(3000, 8000),
                        "transportation": random.randint(5000, 10000),
                        "shopping": random.randint(5000, 15000),
                        "utilities": random.randint(3000, 8000),
                        "healthcare": random.randint(2000, 6000),
                        "other": random.randint(3000, 8000)
                    },
                    "financialGoals": [
                        {
                            "id": str(uuid.uuid4()),
                            "type": "saving",
                            "targetAmount": random.choice([100000, 250000, 500000]),
                            "currentAmount": random.randint(10000, 200000),
                            "deadline": (datetime.utcnow() + timedelta(days=random.randint(30, 365))).isoformat(),
                            "category": random.choice(["Travel", "Emergency Fund", "Car"]),
                            "priority": "high"
                        }
                    ],
                    "riskTolerance": random.choice(["Low", "Moderate", "High"])
                }
            }
            db.collection("users").document().set(user_data)
            logging.info(f"User created with userId: {user_id}")
        else:
            db.collection("users").document(user_doc.id).update({
                "lastLoginAt": current_time
            })
            logging.info(f"Existing user {user_id} updated.")

        # 3. Save receipt
        receipt_data = {
            'receiptId': receipt_id,
            'userId': user_id,
            'uploadedAt': uploadedAt.isoformat(),
            'processedAt': processedAt.isoformat(),
            'status': "completed",
            'ocrData': parsed_data.copy(),
            'gcsUri': f"gs://{bucket.name}/{unique_filename}",
            'processingErrors': [],
            'retryCount': 0,
            'walletPass': {}
        }

        db.collection("receiptQueue").document().set(receipt_data)
        logging.info(f"Receipt saved to Firestore with ID: {receipt_data['receiptId']}")

        return receipt_data

    except Exception as e:
        logging.error(f"Error in save_receipt_to_cloud: {e}")
        raise

def get_credentials():
    """Load credentials from environment variables."""

    try:
        PROJECT_ID = os.getenv("GCP_PROJECT_ID")
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

import json
import re # Ensure re is imported

def parse_json(raw_text) -> dict:
    """
    Extracts and parses a JSON object from a Markdown code block (e.g., ```json ... ```)
    or directly from a JSON string/byte string.

    Args:
        raw_text (str or bytes): Raw text or byte data that includes a Markdown-style JSON block
                                 or is a direct JSON string.

    Returns:
        dict: Parsed JSON data as a Python dictionary.

    Raises:
        ValueError: If no valid JSON block is found or JSON parsing fails.
    """
    # If input is bytes, decode it first
    if isinstance(raw_text, bytes):
        try:
            raw_text = raw_text.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode byte string to UTF-8: {e}")

    if not isinstance(raw_text, str):
        raise ValueError("Input must be a string or bytes convertible to string.")

    # Attempt to match ```json ... ``` markdown block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    
    if match:
        json_text = match.group(1).strip()
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from markdown block: {e}")
    else:
        # If no markdown block, assume the entire raw_text is a direct JSON string
        try:
            return json.loads(raw_text.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"No valid JSON block found or direct JSON parsing failed: {e}")

