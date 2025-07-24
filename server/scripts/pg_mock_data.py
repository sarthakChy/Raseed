import os
import random
import uuid
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import vertexai
from vertexai.language_models import TextEmbeddingModel
import time
import json

# Assuming your postgres_utils.py is in server/utils/
from postgres_utils_mock import init_connection_pool

# --- Configuration ---
NUM_RECEIPTS_TO_CREATE = 100
NUM_DAYS_HISTORY = 90

# --- Mock Data Pools (Expanded) ---
MOCK_USERS = [
    {"userId": "user_alice_123", "email": "alice@example.com"},
    {"userId": "user_bob_456", "email": "bob@example.com"},
    {"userId": "user_charlie_789", "email": "charlie@example.com"},
]

MOCK_MERCHANTS = {
    "SuperMart": {
        "category": "Groceries",
        "items": [
            {"name": "Amul Gold Milk 1L", "price": 72.00, "item_category": "Dairy"},
            {"name": "Britannia Bread", "price": 45.00, "item_category": "Bakery"},
            {"name": "Tata Salt 1kg", "price": 28.00, "item_category": "Pantry"},
            {"name": "Aashirvaad Atta 5kg", "price": 250.00, "item_category": "Pantry"},
            {"name": "Maggi Noodles 4pk", "price": 56.00, "item_category": "Snacks"},
        ]
    },
    "Daily Needs": {
        "category": "Groceries",
        "items": [
            {"name": "Store Brand Milk 1L", "price": 65.00, "item_category": "Dairy"},
            {"name": "Local Bakery Bread", "price": 40.00, "item_category": "Bakery"},
        ]
    },
    "Croma": {
        "category": "Electronics",
        "items": [
            {"name": "Boat Rockerz 450", "price": 1499.00, "item_category": "Audio"},
            {"name": "Samsung 25W Adapter", "price": 1299.00, "item_category": "Accessories"},
        ]
    },
    "Binge Cafe": {
        "category": "Dining",
        "items": [
            {"name": "Espresso", "price": 180.00, "item_category": "Beverages"},
            {"name": "Veg Sandwich", "price": 250.00, "item_category": "Food"},
        ]
    }
}

def seed_database():
    """Generates and saves mock receipts with embeddings directly to PostgreSQL."""
    
    load_dotenv()
    print("Initializing clients...")
    try:
        PROJECT_ID = os.getenv("GCP_PROJECT_ID")
        LOCATION = os.getenv("GCP_LOCATION", "us-central1")
        # --- NEW: Get bucket name for mock URI ---
        BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
        if not BUCKET_NAME:
            raise RuntimeError("GCS_BUCKET_NAME not found in .env file.")

        vertexai.init(project=PROJECT_ID, location=LOCATION)
        
        embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        db_pool = init_connection_pool()
        print("Clients initialized successfully.")
    except Exception as e:
        print(f"Error during initialization: {e}")
        return

    print(f"Generating {NUM_RECEIPTS_TO_CREATE} mock receipts in memory...")
    
    # Step 1: Generate all receipt data and collect all text to be embedded
    all_receipts_data = []
    texts_to_embed = []
    for _ in range(NUM_RECEIPTS_TO_CREATE):
        user = random.choice(MOCK_USERS)
        merchant_name, merchant_data = random.choice(list(MOCK_MERCHANTS.items()))
        
        num_items = random.randint(1, len(merchant_data["items"]))
        receipt_items_data = random.sample(merchant_data["items"], num_items)
        
        for item in receipt_items_data:
            item['quantity'] = random.randint(1, 3)

        item_names = [item['name'] for item in receipt_items_data]
        category = merchant_data["category"]
        context_text = f"A receipt from {merchant_name} for {category}, including: {', '.join(item_names)}"
        
        texts_to_embed.append(context_text)
        texts_to_embed.extend(item_names)
        
        total = sum(item['price'] * item['quantity'] for item in receipt_items_data)
        random_date = datetime.now(timezone.utc) - timedelta(days=random.randint(0, NUM_DAYS_HISTORY))

        all_receipts_data.append({
            "user": user, "merchant_name": merchant_name, "category": category,
            "items": receipt_items_data, "total": total, "date": random_date
        })

    # Step 2: Get all embeddings in smaller chunks to avoid quota errors
    print(f"Generating {len(texts_to_embed)} embeddings in batches...")
    all_embeddings = []
    batch_size = 250

    for i in range(0, len(texts_to_embed), batch_size):
        batch = texts_to_embed[i:i + batch_size]
        try:
            print(f"  -> Processing batch {i//batch_size + 1} of {len(texts_to_embed)//batch_size + 1}...")
            embeddings_batch = embedding_model.get_embeddings(batch)
            all_embeddings.extend([emb.values for emb in embeddings_batch])
            time.sleep(1) 
        except Exception as e:
            print(f"FATAL: An error occurred during embedding generation on batch {i//batch_size + 1}: {e}")
            return

    if len(all_embeddings) != len(texts_to_embed):
        print(f"FATAL: Embedding count mismatch. Expected {len(texts_to_embed)}, but got {len(all_embeddings)}.")
        return

    print("Embeddings generated successfully.")

    # Step 3: Insert data into PostgreSQL
    print("Inserting data into PostgreSQL...")
    embedding_counter = 0
    with db_pool.connect() as db_conn:
        for i, receipt_data in enumerate(all_receipts_data):
            tx = db_conn.begin()
            try:
                receipt_embedding = all_embeddings[embedding_counter]
                embedding_counter += 1
                
                item_embeddings = all_embeddings[embedding_counter : embedding_counter + len(receipt_data["items"])]
                embedding_counter += len(receipt_data["items"])

                # --- UPDATED: Added gcs_uri to the INSERT statement ---
                receipt_insert_stmt = text(
                    """
                    INSERT INTO receipts (user_id, merchant_name, transaction_date, total_amount, category, embedding, gcs_uri)
                    VALUES (:user_id, :merchant_name, :transaction_date, :total_amount, :category, :embedding, :gcs_uri)
                    RETURNING id;
                    """
                )
                result = db_conn.execute(
                    receipt_insert_stmt,
                    parameters={
                        "user_id": receipt_data["user"]["userId"], "merchant_name": receipt_data["merchant_name"],
                        "transaction_date": receipt_data["date"].date(), "total_amount": receipt_data["total"],
                        "category": receipt_data["category"], 
                        "embedding": json.dumps(receipt_embedding),
                        # --- NEW: Added a mock GCS URI ---
                        "gcs_uri": f"gs://{BUCKET_NAME}/receipts/mock/{uuid.uuid4()}.jpg"
                    }
                )
                receipt_id = result.scalar_one()

                item_insert_stmt = text(
                    "INSERT INTO items (receipt_id, name, price, quantity, embedding) VALUES (:r_id, :name, :price, :quantity, :emb);"
                )
                for j, item_data in enumerate(receipt_items_data):
                    db_conn.execute(
                        item_insert_stmt,
                        parameters={
                            "r_id": receipt_id, "name": item_data['name'],
                            "price": item_data['price'], "quantity": item_data['quantity'],
                            "emb": json.dumps(item_embeddings[j])
                        }
                    )
                
                tx.commit()
                print(f"  -> Inserted receipt {i+1}/{NUM_RECEIPTS_TO_CREATE}")

            except Exception as e:
                print(f"An error occurred while inserting receipt {i+1}. Rolling back. Error: {e}")
                tx.rollback()

    print("\nDatabase seeding complete!")


if __name__ == "__main__":
    seed_database()