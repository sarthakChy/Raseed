import os
from firebase_admin import credentials
import firebase_admin

def get_credentials():
    """Load credentials from environment variables."""

    try:
        PROJECT_ID = os.getenv("PROJECT_ID")
        LOCATION = os.getenv("GCP_LOCATION")
    except KeyError:
        raise RuntimeError("GCP_PROJECT_ID not found in .env file.")

    #Initialize cloud storage and firestore clients

    try:
        BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
    except KeyError:
        raise RuntimeError("GCS_BUCKET_NAME not found in .env file. Please set it.")

    return PROJECT_ID, LOCATION, BUCKET_NAME
