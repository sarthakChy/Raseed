import uuid
import logging
from datetime import datetime, timezone
from google.cloud import firestore, storage
from fastapi import UploadFile
from typing import Dict, Any

def save_receipt_to_cloud(
    db: firestore.Client, 
    bucket: storage.Bucket, 
    parsed_data: Dict[str, Any], 
    image_bytes: bytes, 
    file: UploadFile,
    user_id: str = None
) -> Dict[str, Any]:
    """
    Uploads a receipt image to GCS and saves its data to Firestore.
    Returns a JSON-serializable dictionary.
    """
    try:
        # --- Step 1: Upload Image to Cloud Storage (No Change) ---
        unique_filename = f"receipts/{user_id or 'anonymous'}/{uuid.uuid4()}-{file.filename}"
        blob = bucket.blob(unique_filename)
        blob.upload_from_string(image_bytes, content_type=file.content_type)
        logging.info(f"Image uploaded to {unique_filename}.")

        # --- Step 2: Prepare and Save Data to Firestore ---
        doc_ref = db.collection("receipts").document()
        
        # This dictionary is for Firestore, which understands datetime objects
        firestore_data = parsed_data.copy()
        firestore_data['id'] = doc_ref.id
        firestore_data['userId'] = 'anonymous' if user_id is None else user_id
        firestore_data['gcs_uri'] = f"gs://{bucket.name}/{unique_filename}"
        
        doc_ref.set(firestore_data)
        logging.info(f"Data saved to Firestore with ID: {doc_ref.id}")
        # Return the JSON-serializable dictionary
        return firestore_data

    except Exception as e:
        logging.error(f"Error in save_receipt_to_cloud: {e}")
        raise e