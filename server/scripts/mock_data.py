import os
import random
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from google.cloud import firestore
import uuid

# This script connects to your Firestore and populates it with mock data.

# --- Configuration ---
NUM_RECEIPTS_TO_CREATE = 100
NUM_DAYS_HISTORY = 90 # Generate receipts over the last 90 days

# --- Mock Data Pools ---
MOCK_USERS = [
    {"userId": "user_alice_123", "email": "alice@example.com"},
    {"userId": "aryannkaushikk", "email": "aryan.kaushik2015@yahoo.com"},
]

# --- UPDATED: Defined staple items that users will buy repeatedly ---
STAPLE_GROCERIES = [
    ("AMUL TAAZA MILK 1L", 70.00),
    ("BRITANNIA BREAD", 45.00),
    ("TATA SALT 1KG", 28.00),
]

MOCK_MERCHANTS = {
    "BigBasket": {
        "category": "Groceries",
        "items": STAPLE_GROCERIES + [ # Include staples plus other items
            ("AASHIRVAAD ATTA 5KG", 250.00),
            ("MAGGI NOODLES 4PK", 56.00),
            ("DAAWAT BASMATI RICE 1KG", 120.00),
            ("SURF EXCEL 1KG", 215.00),
            ("ONIONS 1KG", 40.00),
        ]
    },
    "Starbucks": {
        "category": "Dining",
        "items": [
            ("CAFFE LATTE", 280.00), ("CAPPUCCINO", 280.00),
            ("CHOCOLATE CROISSANT", 250.00), ("ICED AMERICANO", 260.00),
        ]
    },
    "Croma": {
        "category": "Electronics",
        "items": [
            ("BOAT ROCKERZ 450", 1499.00), ("SAMSUNG 25W ADAPTER", 1299.00),
            ("JBL GO 3 SPEAKER", 2999.00), ("SANDISK 128GB PENDRIVE", 899.00),
        ]
    },
    "WALL-MART-SUPERSTORE": {
        "category": "Shopping",
        "items": [
            ("HAND TOWEL", 250.00), ("GATORADE", 50.00),
            ("GRAPHIC T-SHIRT", 899.00), ("PUSH PINS", 100.00),
        ]
    }
}


def generate_mock_receipts():
    """Generates and saves a specified number of mock receipts to Firestore."""
    
    load_dotenv()
    try:
        PROJECT_ID = os.getenv("GCP_PROJECT_ID")
        db = firestore.Client(project=PROJECT_ID)
        print(f"Successfully connected to Firestore for project: {PROJECT_ID}")
    except Exception as e:
        print(f"Error connecting to Firestore: {e}")
        return

    batch = db.batch()
    receipts_collection = db.collection("receipts")
    
    print(f"Generating {NUM_RECEIPTS_TO_CREATE} mock receipts...")

    # --- UPDATED: More intelligent generation logic ---
    for i in range(NUM_RECEIPTS_TO_CREATE):
        user = random.choice(MOCK_USERS)
        
        # Make grocery shopping more frequent
        is_grocery_trip = random.random() < 0.7 
        
        if is_grocery_trip:
            merchant_name = "BigBasket"
            merchant_data = MOCK_MERCHANTS[merchant_name]
            # On a grocery trip, user always buys some staples plus some random items
            num_staples = random.randint(1, len(STAPLE_GROCERIES))
            num_random_items = random.randint(0, 2)
            
            items_to_choose_from = merchant_data["items"][len(STAPLE_GROCERIES):]
            random.shuffle(items_to_choose_from)
            
            receipt_items_tuples = random.sample(STAPLE_GROCERIES, num_staples) + items_to_choose_from[:num_random_items]
        else:
            # For other trips, pick a random non-grocery store
            other_merchants = {k: v for k, v in MOCK_MERCHANTS.items() if k != "BigBasket"}
            merchant_name, merchant_data = random.choice(list(other_merchants.items()))
            num_items_on_receipt = random.randint(1, 2)
            receipt_items_tuples = random.sample(merchant_data["items"], num_items_on_receipt)

        receipt_items = []
        receipt_total = 0
        for item_name, base_price in receipt_items_tuples:
            price = round(base_price * random.uniform(0.95, 1.05), 2)
            receipt_items.append({"name": item_name, "price": price})
            receipt_total += price
            
        days_ago = random.randint(0, NUM_DAYS_HISTORY)
        random_date = datetime.now(timezone.utc) - timedelta(days=days_ago)
        
        doc_ref = receipts_collection.document()
        receipt_doc = {
            "id": doc_ref.id,
            "userId": user["userId"],
            "userEmail": user["email"],
            "merchantName": merchant_name,
            "category": merchant_data["category"],
            "transactionDate": random_date.strftime("%Y-%m-%d"),
            "items": receipt_items,
            "total": round(receipt_total, 2),
            "tax": round(receipt_total * 0.05, 2),
            "gcs_uri": f"gs://{os.getenv('GCS_BUCKET_NAME')}/receipts/mock/{uuid.uuid4()}.jpg",
            "uploaded_at": random_date,
            "original_filename": "mock_receipt.jpg"
        }
        
        batch.set(doc_ref, receipt_doc)

    try:
        batch.commit()
        print(f"\nSuccessfully saved {NUM_RECEIPTS_TO_CREATE} mock receipts to your Firestore database.")
        print("You can now run your analytics pipeline!")
    except Exception as e:
        print(f"\nError committing batch to Firestore: {e}")


if __name__ == "__main__":
    generate_mock_receipts()