import logging
from typing import List, Dict, Any
from google.cloud import firestore
from datetime import datetime


def fetch_user_data_by_email(db: firestore.Client, email: str) -> List[Dict[str, Any]]:
    """
    Fetches all receipt data for a user by email from Firestore, sorted by uploaded_at descending.

    Args:
        db (firestore.Client): The Firestore client instance.
        email (str): The authenticated user's email.

    Returns:
        List[Dict[str, Any]]: A list of receipts (may be empty), sorted by upload time.
    """
    try:
        logging.info(f"Fetching receipts for email: {email}")

        user_data_ref = db.collection("receipts").where("userEmail", "==", email)
        user_data_docs = user_data_ref.stream()
        items = [doc.to_dict() for doc in user_data_docs]

        if not items:
            logging.warning(f"No receipts found for email: {email}")
            return []

        # ✅ Sort using actual datetime objects from Firestore
        items.sort(
            key=lambda r: r.get("uploaded_at") or datetime.min,
            reverse=True
        )

        logging.info(f"Fetched {len(items)} receipt(s) for email: {email}")
        return items

    except Exception as e:
        logging.error(f"Error fetching user data for {email}: {e}")
        return []
