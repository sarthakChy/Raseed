import os
import random
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from google.cloud import firestore
import uuid

def generate_random_date_in_range():
    """Generate a random date within the last 6 months."""
    current_date = datetime.now(timezone.utc)
    start_date = current_date - timedelta(days=NUM_DAYS_HISTORY)
    
    # Generate random number of days between start_date and current_date
    random_days = random.randint(0, NUM_DAYS_HISTORY)
    random_date = start_date + timedelta(days=random_days)
    
    return random_date

# --- Configuration ---
NUM_RECEIPTS_TO_QUEUE = 50
NUM_DAYS_HISTORY = 90
NUM_CHAT_SESSIONS_PER_USER = 3
NUM_MESSAGES_PER_SESSION = 6
NUM_NOTIFICATIONS_PER_USER = 5

# --- Mock Data Pools ---
MOCK_USERS = [
    {"userId": "user_jigisha_2314", "email": "jigisha.6@gmail.com", "displayName": "jigisha"}
]

# Enhanced merchants with better alternatives finding data
MOCK_MERCHANTS = {
    # Grocery Stores - Great for price comparison
    "BigBasket Store #123": {
        "normalized": "BigBasket",
        "category": "groceries",
        "items": [
            {"name": "Amul Gold Milk 1L", "price": 72.00, "quantity": 1},
            {"name": "Britannia Bread Whole Wheat", "price": 45.00, "quantity": 1},
            {"name": "Organic Tomatoes 1kg", "price": 80.00, "quantity": 1},
            {"name": "Basmati Rice Kohinoor 5kg", "price": 450.00, "quantity": 1},
            {"name": "Tata Salt 1kg", "price": 28.00, "quantity": 1},
            {"name": "Fortune Sunflower Oil 1L", "price": 165.00, "quantity": 1},
        ],
        "address": "MG Road, Bangalore, KA 560001"
    },
    
    "Reliance Fresh #456": {
        "normalized": "Reliance Fresh",
        "category": "groceries", 
        "items": [
            {"name": "Amul Gold Milk 1L", "price": 70.00, "quantity": 1},
            {"name": "Britannia Bread Whole Wheat", "price": 40.00, "quantity": 1},
            {"name": "Fresh Tomatoes 1kg", "price": 60.00, "quantity": 1},
            {"name": "Basmati Rice India Gate 5kg", "price": 500.00, "quantity": 1},
            {"name": "Refined Oil Saffola 1L", "price": 185.00, "quantity": 1},
        ],
        "address": "Forum Mall, Bangalore, KA 560029"
    },

    "Spencer's Retail": {
        "normalized": "Spencer's",
        "category": "groceries",
        "items": [
            {"name": "Mother Dairy Milk 1L", "price": 68.00, "quantity": 1},
            {"name": "Harvest Gold Bread", "price": 42.00, "quantity": 1},
            {"name": "Organic Carrots 500g", "price": 45.00, "quantity": 1},
            {"name": "Daawat Basmati Rice 5kg", "price": 475.00, "quantity": 1},
        ],
        "address": "UB City Mall, Bangalore, KA 560001"
    },
    
    # Electronics - Good for feature and price comparison
    "Croma Electronics": {
        "normalized": "Croma",
        "category": "electronics",
        "items": [
            {"name": "boAt Rockerz 450 Bluetooth Headphones", "price": 1499.00, "quantity": 1},
            {"name": "Samsung 25W Fast Charger", "price": 1299.00, "quantity": 1},
            {"name": "Realme Buds Air 3", "price": 3999.00, "quantity": 1},
            {"name": "Mi Power Bank 10000mAh", "price": 1199.00, "quantity": 1},
        ],
        "address": "Forum Mall, Bangalore, KA 560029"
    },

    "Vijay Sales": {
        "normalized": "Vijay Sales", 
        "category": "electronics",
        "items": [
            {"name": "JBL C100SI Wired Earphones", "price": 799.00, "quantity": 1},
            {"name": "OnePlus Warp Charge 65W", "price": 1599.00, "quantity": 1},
            {"name": "Sony WH-CH520 Headphones", "price": 2990.00, "quantity": 1},
            {"name": "Anker PowerCore 20000mAh", "price": 2299.00, "quantity": 1},
        ],
        "address": "Commercial Street, Bangalore, KA 560001"
    },
    
    # Restaurants/Cafes - Good for cuisine and price alternatives
    "Binge Cafe": {
        "normalized": "Binge Cafe",
        "category": "food",
        "items": [
            {"name": "Cappuccino", "price": 180.00, "quantity": 1},
            {"name": "Grilled Chicken Sandwich", "price": 320.00, "quantity": 1},
            {"name": "Pasta Arrabbiata", "price": 380.00, "quantity": 1},
            {"name": "Chocolate Brownie", "price": 220.00, "quantity": 1},
        ],
        "address": "Indiranagar, Bangalore, KA 560038"
    },

    "Cafe Coffee Day": {
        "normalized": "CCD",
        "category": "food", 
        "items": [
            {"name": "Regular Coffee", "price": 120.00, "quantity": 1},
            {"name": "Veg Club Sandwich", "price": 250.00, "quantity": 1},
            {"name": "Chicken Tikka Wrap", "price": 280.00, "quantity": 1},
            {"name": "Blueberry Muffin", "price": 150.00, "quantity": 1},
        ],
        "address": "Brigade Road, Bangalore, KA 560001"
    },

    "Domino's Pizza": {
        "normalized": "Dominos",
        "category": "food",
        "items": [
            {"name": "Margherita Pizza Medium", "price": 299.00, "quantity": 1},
            {"name": "Chicken Pepperoni Pizza Medium", "price": 449.00, "quantity": 1},
            {"name": "Garlic Breadsticks", "price": 149.00, "quantity": 1},
            {"name": "Coke 500ml", "price": 57.00, "quantity": 1},
        ],
        "address": "Koramangala, Bangalore, KA 560034"
    },

    # Fashion - Good for style and price alternatives  
    "Westside": {
        "normalized": "Westside",
        "category": "shopping",
        "items": [
            {"name": "Cotton Casual Shirt", "price": 1299.00, "quantity": 1},
            {"name": "Denim Jeans", "price": 1899.00, "quantity": 1},
            {"name": "Leather Wallet", "price": 699.00, "quantity": 1},
        ],
        "address": "UB City Mall, Bangalore, KA 560001"
    },

    "Max Fashion": {
        "normalized": "Max",
        "category": "shopping",
        "items": [
            {"name": "Polo T-Shirt", "price": 799.00, "quantity": 1},
            {"name": "Chino Pants", "price": 1199.00, "quantity": 1},
            {"name": "Canvas Sneakers", "price": 1599.00, "quantity": 1},
        ],
        "address": "Phoenix MarketCity, Bangalore, KA 560048"
    },

    # Health & Beauty - Good for ingredient and brand alternatives
    "Apollo Pharmacy": {
        "normalized": "Apollo Pharmacy",
        "category": "shopping",
        "items": [
            {"name": "Paracetamol 500mg Strip", "price": 15.00, "quantity": 2},
            {"name": "Vitamin D3 Tablets", "price": 180.00, "quantity": 1},
            {"name": "Face Wash Himalaya", "price": 165.00, "quantity": 1},
        ],
        "address": "Jayanagar, Bangalore, KA 560011"
    },

    "Nykaa Store": {
        "normalized": "Nykaa",
        "category": "shopping", 
        "items": [
            {"name": "Lakme Foundation", "price": 825.00, "quantity": 1},
            {"name": "The Ordinary Niacinamide Serum", "price": 590.00, "quantity": 1},
            {"name": "Biotique Shampoo", "price": 245.00, "quantity": 1},
        ],
        "address": "Brigade Road, Bangalore, KA 560001"
    }
}

PAYMENT_METHODS = ["Credit Card", "Debit Card", "UPI", "Cash", "Wallet", "Net Banking"]
CURRENCIES = ["INR"]

MOCK_CHAT_QUERIES = [
    {"text": "How much did I spend on groceries last month?", "intent": "ANALYTICAL", "entities": { "timeframe": "last_month"}},
    {"text": "Show me my top 5 expenses this week", "intent": "EXPLORATORY", "entities": {"timeframe": "this_week"}},
    {"text": "Can you help me create a budget for dining out?", "intent": "ACTIONABLE", "entities": {"timeframe": "this_month"}},
    {"text": "Compare my spending this month vs last month", "intent": "COMPARATIVE", "entities": {"timeframe": "this_month"}},
    {"text": "Will I exceed my monthly budget if I continue spending like this?", "intent": "PREDICTIVE", "entities": {}},
]

MOCK_AGENT_RESPONSES = [
    "You spent ₹15,247 on groceries last month, which is 8% higher than your average.",
    "Here are your top 5 expenses this week based on your recent transactions.",
    "Based on your current dining spending, I recommend setting a monthly budget of ₹8,000 for dining out.",
    "This month you've spent ₹32,450 compared to ₹28,900 last month - an increase of 12.3%.",
    "At your current spending rate, you're likely to exceed your monthly budget by ₹4,200.",
]

def generate_random_timestamp(days_back=None):
    """Generate a random timestamp within the specified range."""
    if days_back is None:
        days_back = random.randint(0, NUM_DAYS_HISTORY)
    return datetime.now(timezone.utc) - timedelta(days=days_back)

def generate_mock_data():
    """Generates and saves comprehensive mock data to Firestore."""
    
    # --- Setup ---
    load_dotenv()
    try:
        PROJECT_ID = os.getenv("GCP_PROJECT_ID")
        if not PROJECT_ID:
            print("Error: GCP_PROJECT_ID not found in environment variables")
            return
        db = firestore.Client(project=PROJECT_ID)
        print(f"Successfully connected to Firestore for project: {PROJECT_ID}")
    except Exception as e:
        print(f"Error connecting to Firestore: {e}")
        return

    # Get current timestamp for consistency
    current_time = datetime.now(timezone.utc)

    # --- Create User Documents ---
    print("Generating user documents...")
    users_batch = db.batch()
    users_collection = db.collection("users")
    
    for user_data in MOCK_USERS:
        user_doc_ref = users_collection.document(user_data["userId"])
        user_doc = {
            "userId": user_data["userId"],
            "email": user_data["email"],
            "displayName": user_data["displayName"],
            "photoURL": f"https://api.dicebear.com/8.x/initials/svg?seed={user_data['displayName']}",
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
                        "deadline": current_time + timedelta(days=random.randint(30, 365)),
                        "category":random.choice(["Travel", "Emergency Fund", "Car"]),
                        "priority": "high"
                    }
                ],
                "riskTolerance": "moderate"
            }
        }
        users_batch.set(user_doc_ref, user_doc)
    
    users_batch.commit()
    print(f"Successfully created {len(MOCK_USERS)} user documents.")

    # --- Create Receipt Queue Documents ---
    print(f"Generating {NUM_RECEIPTS_TO_QUEUE} receipt queue documents...")
    receipt_batch = db.batch()
    receipt_queue_collection = db.collection("receiptQueue")
    
    for i in range(NUM_RECEIPTS_TO_QUEUE):
        user = random.choice(MOCK_USERS)
        merchant_name, merchant_data = random.choice(list(MOCK_MERCHANTS.items()))
        
        # Select random items from the merchant
        num_items = random.randint(1, min(3, len(merchant_data["items"])))
        receipt_items = random.sample(merchant_data["items"], num_items)
        
        # Calculate totals
        subtotal = sum(item['price'] * item['quantity'] for item in receipt_items)
        tax = round(subtotal * 0.18, 2)  # 18% tax
        total = round(subtotal + tax, 2)
        
        is_processed = random.choice([True, True, False])  # 66% processed
        upload_time = generate_random_timestamp()
        purchase_date = generate_random_date_in_range()

        doc_ref = receipt_queue_collection.document()
        receipt_doc = {
            "receiptId": str(uuid.uuid4()),
            "userId": user["userId"],
            "uploadedAt": upload_time,
            "processedAt": upload_time + timedelta(minutes=random.randint(1, 30)) if is_processed else None,
            "status": "completed",
            "ocrData": {
                "rawText": f"Mock OCR text for {merchant_name}",
                "confidence": round(random.uniform(0.80, 0.99), 2),
                "extractedData": {
                    "merchantName": merchant_name,
                    "normalizedMerchant": merchant_data["normalized"],
                    "date": purchase_date.strftime("%Y-%m-%d"),
                    "totalAmount": total,
                    "subtotal": subtotal,
                    "tax": tax,
                    "items": receipt_items,
                    "paymentMethod": random.choice(["Credit Card", "Debit Card", "UPI","CASH"]),
                    "address": merchant_data["address"],
                    "currency": random.choice(CURRENCIES),
                    "category": merchant_data["category"]

                }
            },
            "processingErrors": [],
            "retryCount": 0,
            "walletPass": {
                "objectId": str(uuid.uuid4()) if random.choice([True, False]) else "string",
                "walletLink": f"https://wallet.example.com/{doc_ref.id}" if random.choice([True, False]) else "string",
                "created": False
            }
        }
        receipt_batch.set(doc_ref, receipt_doc)
    
    receipt_batch.commit()
    print(f"Successfully created {NUM_RECEIPTS_TO_QUEUE} receipt queue documents.")

    # --- Create Chat Sessions and Messages ---
    print("Generating chat sessions and messages...")
    chat_batch = db.batch()
    sessions_collection = db.collection("chatSessions")
    
    for user in MOCK_USERS:
        for session_num in range(NUM_CHAT_SESSIONS_PER_USER):
            session_id = str(uuid.uuid4())
            session_start_time = generate_random_timestamp()
            
            session_doc_ref = sessions_collection.document(session_id)
            session_doc = {
                "sessionId": session_id,
                "userId": user["userId"],
                "startedAt": session_start_time,
                "lastActiveAt": session_start_time + timedelta(minutes=random.randint(5, 60)),
                "status": "active",
                "context": {
                    "currentTopic": "spending_analysis",
                    "lastIntent": "ANALYTICAL",
                    "conversationFlow": ["translate_query", "analyze_data"],
                    "activeWorkflowId": str(uuid.uuid4())
                },
                "metadata": {
                    "deviceType": "web",
                    "userAgent": "string",
                    "ipAddress": "string"
                }
            }
            chat_batch.set(session_doc_ref, session_doc)

            # Create messages for this session
            messages_collection = session_doc_ref.collection("messages")
            message_time = session_start_time
            
            for msg_num in range(NUM_MESSAGES_PER_SESSION):
                message_time += timedelta(minutes=random.randint(1, 5))
                
                # User message
                if msg_num % 2 == 0:
                    query = random.choice(MOCK_CHAT_QUERIES)
                    user_message_ref = messages_collection.document()
                    user_message = {
                        "messageId": user_message_ref.id,
                        "sessionId": session_id,
                        "userId": user["userId"],
                        "timestamp": message_time,
                        "type": "user",
                        "content": {
                            "text": query["text"],
                            "intent": query["intent"],
                            "entities": {
                                "amount": 450.75,
                                
                                "timeframe": "last_month",
                                "merchants": ["Whole Foods"]
                            },
                            "attachments": []
                        },
                        "isRead": True,
                        "reactions": {
                            "helpful": False,
                            "notHelpful": False
                        }
                    }
                    chat_batch.set(user_message_ref, user_message)
                
                # Agent response
                else:
                    agent_message_ref = messages_collection.document()
                    response_text = random.choice(MOCK_AGENT_RESPONSES)
                    agent_message = {
                        "messageId": agent_message_ref.id,
                        "sessionId": session_id,
                        "userId": user["userId"],
                        "timestamp": message_time,
                        "type": "agent",
                        "content": {
                            "text": response_text,
                            "attachments": [
                                {
                                    "type": "chart",
                                    "url": f"https://charts.example.com/chart_{uuid.uuid4()}.png",
                                    "metadata": {}
                                }
                            ] if random.choice([True, False]) else []
                        },
                        "agentContext": {
                            "agentId": "financial_analysis_agent",
                            "workflowId": str(uuid.uuid4()),
                            "stepId": "analyze_data",
                            "processingTime": 1250,
                            "confidence": 0.95
                        },
                        "isRead": True,
                        "reactions": {
                            "helpful": False,
                            "notHelpful": False
                        }
                    }
                    chat_batch.set(agent_message_ref, agent_message)
    
    chat_batch.commit()
    print(f"Successfully created {NUM_CHAT_SESSIONS_PER_USER * len(MOCK_USERS)} chat sessions with messages.")

    # --- Create Notifications ---
    print("Generating notifications...")
    notifications_batch = db.batch()
    notifications_collection = db.collection("notifications")
    
    for user in MOCK_USERS:
        for i in range(NUM_NOTIFICATIONS_PER_USER):
            notification_ref = notifications_collection.document()
            notification_time = generate_random_timestamp()
            
            notification_doc = {
                "notificationId": notification_ref.id,
                "userId": user["userId"],
                "type": "proactive_insight",
                "title": "Unusual Spending Pattern Detected",
                "message": "You've spent 30% more on dining out this week compared to your average.",
                "data": {
                                      "amount": 285.50,
                    "comparison": "weekly_average",
                    "variance": 0.30,
                    "actionable": True,
                    "recommendations": ["dining_alternatives", "budget_adjustment"]
                },
                "priority": "medium",
                "status": "pending",
                "scheduledFor": notification_time,
                "sentAt": notification_time + timedelta(minutes=1),
                "readAt": notification_time + timedelta(minutes=random.randint(5, 120)),
                "channels": ["push", "in_app"],
                "metadata": {
                    "source": "proactive_insights_agent",
                    "triggerEvent": "spending_pattern_analysis",
                    "relevanceScore": 0.87
                }
            }
            notifications_batch.set(notification_ref, notification_doc)
    
    notifications_batch.commit()
    print(f"Successfully created {NUM_NOTIFICATIONS_PER_USER * len(MOCK_USERS)} notifications.")

    print("\n=== Mock Data Generation Complete ===")
    print(f"✅ Users: {len(MOCK_USERS)}")
    print(f"✅ Receipts: {NUM_RECEIPTS_TO_QUEUE}")
    print(f"✅ Chat Sessions: {NUM_CHAT_SESSIONS_PER_USER * len(MOCK_USERS)}")
    print(f"✅ Messages: {NUM_MESSAGES_PER_SESSION * NUM_CHAT_SESSIONS_PER_USER * len(MOCK_USERS)}")
    print(f"✅ Notifications: {NUM_NOTIFICATIONS_PER_USER * len(MOCK_USERS)}")


if __name__ == "__main__":
    generate_mock_data()