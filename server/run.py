# ========================== Imports ==========================
import os
import json
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any, Literal
from datetime import datetime, timedelta, timezone
from google.oauth2 import id_token
from google.auth.transport import requests
from datetime import datetime, timezone
from google.cloud import firestore, storage
import requests as req
import uuid
import asyncio
import logging
import time
import sys
import os
import json
import psutil
import jwt
import random
import re
import traceback
import base64
from dotenv import load_dotenv

from fastapi import (
    FastAPI, HTTPException, Query, Body, Request,
    File, UploadFile, Form, Depends, Header
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.background import BackgroundTasks

from dotenv import load_dotenv
import requests as req

from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import vertexai
from vertexai.generative_models import GenerativeModel

import firebase_admin
from firebase_admin import credentials, auth

from utils.utils import get_credentials,parse_json,initialize_firestore, initialize_gcs_client,save_receipt_to_cloud

from agents.receipts_agent import ReceiptAgent
from wallet.receipt_manager import ReceiptWalletManager
from wallet.shopping_list_manager import ShoppingListWalletManager
from agents.agent_orchestrator import MasterOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================== Environment Setup ==========================
load_dotenv()

if not firebase_admin._apps:
    FIREBASE_CRED_PATH = os.environ.get("FIREBASE_CRED_PATH", "firebase-sdk.json")
    try:
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        raise RuntimeError(f"Could not initialize Firebase Admin SDK: {str(e)}")

PROJECT_ID,LOCATION,BUCKET_NAME,GOOGLE_WALLET_ISSUER_ID = get_credentials()

try:
    vertexai.init(project=PROJECT_ID, location=LOCATION) #global init for Vertex AI SDK
    logging.info(f"Vertex AI SDK initialized for project '{PROJECT_ID}' in '{LOCATION}'.")
except Exception as e:
    logging.error(f"Critical: Failed to initialize Vertex AI SDK: {e}")
    sys.exit(1) 

storage_client = initialize_gcs_client()
db = initialize_firestore()
bucket = storage_client.bucket(BUCKET_NAME)

# ========================== App Initialization ==========================

app = FastAPI(
    title="Raseed API",
    version="1.0.0",
    root_path="/api",
    docs_url="/docs",
)


# ========================== Middleware ==========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.google.com",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========================== Firebase Auth Dependency ==========================
async def firebase_auth_required(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization token missing or invalid.")
    token = auth_header.split("Bearer ")[1]
    try:
        decoded_token = auth.verify_id_token(token)
        request.state.user = decoded_token
        return decoded_token
    except Exception as e:
        logging.error(f"Token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


# ========================== Routes ==========================
@app.get("/")
async def root(request: Request, auth=Depends(firebase_auth_required)):
    return {"message": "Welcome to the API. Visit /docs for documentation."}


@app.post("/receipts/analyze")
async def analyze_receipt(
    request: Request,
    auth=Depends(firebase_auth_required),
    file: UploadFile = File(...),
):
    try:    
        #Reads fils bytes   
        user_image_bytes = await file.read()

        uploadedAt = datetime.now()

        #Initialize the ReceiptAgent
        agent = ReceiptAgent(
                        file_bytes=user_image_bytes,
                        file_content_type=file.content_type,
                    )
        
        # Analyze the receipt
        response = agent.analyze()
        processedAt = datetime.now()
        parsed_data = parse_json(response.text) 

        #Get user data
        user_id = auth.get("uid")
        email = auth.get("email")

        #Save the receipt to cloud storage and Firestore
        receipt_data = save_receipt_to_cloud(
                    db=db,
                    bucket=bucket,
                    parsed_data=parsed_data,
                    image_bytes=user_image_bytes, 
                    file=file,
                    uploadedAt = uploadedAt,
                    processedAt = processedAt,
                    receipt_id = str(uuid.uuid4()),
                    user_id = user_id,
                    email = email
                )

        return JSONResponse(content=receipt_data, status_code=200)
    except Exception as e:
        logging.error(f"Receipt analysis error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ========================== Google Wallet Integration ==========================

# ========================== Receipt Pass ===============================

wallet_manager = ReceiptWalletManager()

@app.post("/receipts/create-wallet-pass")
async def create_wallet_pass(
    request: Request,
    auth=Depends(firebase_auth_required),
    body: dict = Body(...),
):
    """
    Create a Google Wallet pass for a receipt.
    
    This endpoint takes receipt data and creates a digital wallet pass
    that users can add to their Google Wallet app.
    """
    try:
        print(body)
        uuid = body.get('uuid')
        query_result = db.collection("receiptQueue").where("receiptId", "==", uuid).get()

        if not query_result:
            logger.error(f"No receipt found for UUID: {uuid}")
            return {"success": False, "message": "Receipt not found"}

        doc = query_result[0]
        receipt_data = doc.to_dict()

        logger.info(f"Retrieved receipt data for UUID {uuid}: {receipt_data}")

        # Create wallet pass
        result = await wallet_manager.create_receipt_pass(receipt_data, uuid)

        # Update the Firestore document with the new pass info
        doc.reference.update({'walletPass': result})

        if result['success']:
            return result
        else:
            logger.error(f"Failed to create wallet pass: {result.get('error', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Internal error creating wallet pass: {e}")

@app.put("/receipts/update-wallet-pass")
async def update_wallet_pass(
    request: Request,
    auth=Depends(firebase_auth_required),
    body:dict = Body(...)
):
    """
    Update an existing Google Wallet pass with new information.
    
    This can be used to add AI-generated insights, spending tips, or other
    dynamic content to existing wallet passes.
    """
    try:
        
        uuid = body.get('uuid')
        query_result = db.collection("receiptQueue").where("receiptId", "==", uuid).get()

        if not query_result:
            logger.error(f"No receipt found for UUID: {uuid}")
            return {"success": False, "message": "Receipt not found"}

        doc = query_result[0]
        receipt_data = doc.to_dict()

        logger.info(f"Retrieved receipt data for UUID {uuid}: {receipt_data}")

        object_id = f"{GOOGLE_WALLET_ISSUER_ID}.{uuid}"

        result = await wallet_manager.update_wallet_pass(object_id, body)
        
        doc.reference.update({
            'walletPass': {
                'pass_data':result['pass_data'],
                'lastUpdatedAt':datetime.now().isoformat()
            }

            })

        if result['success']:
            return result
        else:
            logging.error(f"Failed to update pass")

    except Exception as e:
        logger.error(f"Unexpected error updating wallet pass: {e}")

@app.patch("/receipts/expire-wallet-pass")
async def expire_wallet_pass(
    request: Request,
    auth=Depends(firebase_auth_required),
    body:dict = Body(...)
):
    """
    Expire a Google Wallet pass.
    
    This marks the wallet pass as expired, which will update its status
    in the user's Google Wallet app.
    """
    try:            
        

        uuid = body.get('uuid')
        query_result = db.collection("receiptQueue").where("receiptId", "==", uuid).get()

        if not query_result:
            logger.error(f"No receipt found for UUID: {uuid}")
            return {"success": False, "message": "Receipt not found"}

        doc = query_result[0]
        receipt_data = doc.to_dict()

        logger.info(f"Retrieved receipt data for UUID {uuid}: {receipt_data}")

        object_id = f"{GOOGLE_WALLET_ISSUER_ID}.{uuid}"
        
        result = await wallet_manager.expire_wallet_pass(object_id)
        
        doc.reference.update({
            'walletPass': {
                'pass_data':result['pass_data'],
                'lastUpdatedAt':datetime.now().isoformat()
            }

            })

        if result['success']:
            return result
        else:
            logging.error("Failed to expire pass")
            
    
    except Exception as e:
        logger.error(f"Unexpected error expiring wallet pass: {e}")


# ============================= ShoppingList Pass ===============================================

shopping_list_manager = ShoppingListWalletManager()

@app.post("/shopping-list/create-shopping-pass")
async def create_wallet_pass(
    request: Request,
    auth=Depends(firebase_auth_required),
    #body: dict = Body(...),
):
    """
    Create a Google Wallet pass for a Shopping List.
    
    This endpoint takes shopping data and creates a digital wallet pass
    that users can add to their Google Wallet app.
    """
    try:

        shopping_data = {
            'listName': 'Weekly Groceries',
            'items': [
                {'name': 'Milk', 'quantity': 2, 'estimatedPrice': 5.99, 'completed': False},
                {'name': 'Bread', 'quantity': 1, 'estimatedPrice': 2.50, 'completed': True},
                {'name': 'Eggs', 'quantity': 1, 'estimatedPrice': 3.99, 'completed': False}
            ],
            'estimatedTotal': 12.48,
            'category': 'Groceries',
            'storePreference': 'Walmart',
            'priority': 'High'
        }

        result = await shopping_list_manager.create_shopping_list_pass(shopping_data,str(uuid.uuid4()))
        logger.info(result)
        if result['success']:
            return result
        else:
            logger.error(f"Failed to create wallet pass: {result.get('error', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Internal error creating wallet pass: {e}")


@app.put("/shopping-list/update-shopping-pass")
async def update_wallet_pass(
    request: Request,
    auth=Depends(firebase_auth_required),
    body: dict = Body(...)
):
    """
    Update a Google Wallet pass. Pass `object_id` in the body along with updates.
    """
    try:
        object_id = body.get("object_id")
        if not object_id:
            return JSONResponse(status_code=400, content={"error": "Missing object_id in body"})

        update_data = {k: v for k, v in body.items() if k != "object_id"}
        result = await shopping_list_manager.update_shopping_list_pass(object_id, update_data)

        if result['success']:
            return result
        else:
            logger.error(f"Failed to update wallet pass: {result.get('error', 'Unknown error')}")
            return JSONResponse(status_code=500, content={"error": result.get("error", "Unknown error")})
    except Exception as e:
        logger.error(f"Internal error updating wallet pass: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

# ============================= Chat ===============================================

@app.post("/chat")
async def chat_handler(
    request: Request,
    auth=Depends(firebase_auth_required),
    body: dict = Body(...)
):
    try:
        user_query = body['query']

        # # Uncomment this for actual processing
        # orchestrator = MasterOrchestrator(
        #     project_id=os.getenv("PROJECT_ID"),
        #     config_path="config/agent_config.yaml",
        #     location="us-central1",
        #     model_name="gemini-2.0-flash-001"
        # )
        # result = await orchestrator.process_query(
        #     query=user_query,
        #     user_id="a73ff731-9018-45ed-86ff-214e91baf702",
        #     additional_context={
        #         "currency": "USD",
        #         "timezone": "America/New_York",
        #         "preferred_categories": ["groceries", "food", "transport"]
        #     }
        # )

        # Dummy response for testing with frontend
        mock_result = {
            "insights": "In July 2025, you spent a total of $5326.11 on food. A significant portion of your food spending went to Domino's Pizza ($3695.72), Binge Cafe ($3386.1), and Cafe Coffee Day ($2979). It appears you frequently visit these merchants. Consider exploring other dining options for more variety!",
            "visualization": {
                "type": "bar_chart",
                "fields": {
                    "x_axis": [
                        "2025-07-01",
                        "2025-06-01",
                        "2025-04-01",
                        "2025-03-01",
                        "2025-02-01"
                    ],
                    "y_axis": [
                        5668.46,
                        1905.75,
                        1125.72,
                        7942.6,
                        1575.5
                    ]
                },
                "caption": "The bar chart shows the amount spent on groceries each month."
            },
            "explanation": "A bar chart is suitable because it effectively compares the total spending on groceries across different months, visually highlighting the changes in expenditure over time. It clearly presents the monthly grocery expenses for easy comparison."
        }

#         mock_result = {
#     "insights": "In July 2025, your food spending was distributed among several key merchants.",
#     "visualization": {
#         "type": "pie_chart",
#         "fields": {
#             "x_axis": [
#                 "Domino's Pizza",
#                 "Binge Cafe",
#                 "Cafe Coffee Day",
#                 "Others"
#             ],
#             "y_axis": [
#                 3695.72,
#                 3386.1,
#                 2979,
#                 1123.56
#             ]
#         },
#         "caption": "Pie chart showing the share of food spending by merchant in July 2025."
#     },
#     "explanation": "A pie chart is appropriate here as it highlights the proportional distribution of spending among different merchants, making it easy to identify dominant expenses."
# }

<<<<<<< HEAD
<<<<<<< HEAD
        # response = chat.send_message(user_message)

        return JSONResponse(content=json.loads(result.model_dump_json()))
=======
        response = chat.send_message(user_query)
        return JSONResponse(content={"reply": response.text} )
>>>>>>> 52dac17 (Frontend Integration - History, Scan, Chat)
=======
#         mock_result = {
#             "insights": "Your monthly spending on transportation shows a fluctuating pattern across recent months.",
#             "visualization": {
#                 "type": "line_chart",
#                 "fields": {
#                     "x_axis": [
#                         "2025-02-01",
#                         "2025-03-01",
#                         "2025-04-01",
#                         "2025-05-01",
#                         "2025-06-01",
#                         "2025-07-01"
#                     ],
#                     "y_axis": [
#                         320.5,
#                         540.0,
#                         410.2,
#                         660.75,
#                         500.3,
#                         720.4
#                     ]
#                 },
#                 "caption": "Line chart showing monthly transportation expenses over time."
#             },
#             "explanation": "A line chart is used here to effectively show the trend of your transportation spending. It helps highlight increases or decreases across months, making patterns easily recognizable."
# }



        return JSONResponse(content={"reply": mock_result})
>>>>>>> 6752978 (Frontend Integration - Chat Charts Setup)

    except Exception as e:
        logging.error(f"Chat endpoint error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
# ============================= Receipt History ===============================================

@app.get("/receipts/user/{user_id}")
async def get_user_receipts(
    user_id: str,
    request: Request,
    auth=Depends(firebase_auth_required)
):
    try:
        # Fetch receipts for the given user ID
        receipts_ref = db.collection("receiptQueue").where("userId", "==", user_id)
        docs = receipts_ref.stream()
        
        receipts = []
        for doc in docs:
            receipt_data = doc.to_dict()
            receipt_data["receiptId"] = doc.id
            receipts.append(receipt_data)

        return JSONResponse(content={"status": "success", "receipts": receipts}, status_code=200)

    except Exception as e:
        logging.error(f"Error fetching receipts for user {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to fetch receipts")

# ============================= Receipt Delete ===============================================

@app.delete("/receipts/{receipt_id}")
async def delete_receipt(
    receipt_id: str,
    request: Request,
    auth=Depends(firebase_auth_required)
):
    try:
        # Reference the document in Firestore
        doc_ref = db.collection("receiptQueue").document(receipt_id)

        # Check if the document exists before attempting deletion
        doc = doc_ref.get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Receipt not found")

        # Optional: verify that the user owns this receipt
        user_id = request.state.user["uid"]
        if doc.to_dict().get("userId") != user_id:
            raise HTTPException(status_code=403, detail="You are not authorized to delete this receipt")

        # Delete the document
        doc_ref.delete()

        return JSONResponse(
            content={"status": "success", "message": f"Receipt {receipt_id} deleted successfully"},
            status_code=200
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error deleting receipt {receipt_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to delete receipt")


@app.post("/orchestrator/query")
async def orchestrator_query(
    request: Request,
    body: dict = Body(...) 
):
    """
    Process a user query through the Master Orchestrator
    Uses hardcoded values for testing purposes.
    
    Expected body:
    {
        "query": "How much did I spend on groceries last month?"
    }
    """
    try:
        # Extract query from request body
        user_query = body.get("query")
        if not user_query:
            raise HTTPException(status_code=400, detail="Missing 'query' in request body")
        
        # HARDCODED VALUES FOR TESTING
        hardcoded_user_id = "user_alice_123"
        hardcoded_config_path = "config/agent_config.yaml"
        hardcoded_additional_context = {
            "currency": "USD",
            "timezone": "America/New_York",
            "preferred_categories": ["groceries", "food", "transport"]
        }
        
        # Initialize the orchestrator with hardcoded values
        orchestrator = MasterOrchestrator(
            project_id=PROJECT_ID,
            config_path=hardcoded_config_path,
            location=LOCATION,
            model_name="gemini-2.0-flash-001"
        )
        
        # Process the query with hardcoded values
        result = await orchestrator.process_query(
            query=user_query,
            user_id=hardcoded_user_id,
            additional_context=hardcoded_additional_context
        )
        
        # Convert Pydantic model to dict before JSONResponse
        return JSONResponse(content={"result": result.dict()}, status_code=200)
        
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}")
        raise HTTPException(status_code=500, detail="Agent configuration not found")
    except Exception as e:
        logging.error(f"Orchestrator endpoint error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")