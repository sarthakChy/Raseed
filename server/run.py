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
import uuid
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

import firebase_admin
from firebase_admin import credentials, auth

from utils.utils import get_credentials,parse_json,initialize_firestore, initialize_gcs_client

from agents.insights_agent import PurchaseInsightsAgent
from agents.receipt.receipts_agent import ReceiptAgent
from agents.wallet_agent import ReceiptWalletManager, WalletPassResponse, UpdatePassData, HealthResponse
from server.utils.storage_utils import save_receipt_to_cloud

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
        #Initialize the ReceiptAgent
        agent = ReceiptAgent(
                        file_bytes=user_image_bytes,
                        file_content_type=file.content_type,
                    )
        # Analyze the receipt
        response = agent.analyze()
        parsed_data = parse_json(response.text) 
            
        
        #Save the receipt to cloud storage and Firestore
        final_data = save_receipt_to_cloud(
                    db=db,
                    bucket=bucket,
                    parsed_data=parsed_data,
                    image_bytes=user_image_bytes, 
                    file=file,
                    user_id=None 
                )

        return JSONResponse(content=final_data, status_code=200)
    except Exception as e:
        logging.error(f"Receipt analysis error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
        

@app.post("/chat")
async def chat_handler(
    request: Request,
    auth=Depends(firebase_auth_required),
    body: dict = Body(...)
):
    try:
        user_message = body.get("message")
        if not user_message:
            raise HTTPException(status_code=400, detail="Missing 'message' in body.")

        model = GenerativeModel("gemini-2.0-flash-001")
        chat = model.start_chat()

        response = chat.send_message(user_message)

        return JSONResponse(content={"reply": response.text})

    except Exception as e:
        logging.error(f"Chat endpoint error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    
@app.get("/analyze-insights")
async def analyze_insights(
    request: Request,
    auth=Depends(firebase_auth_required)
    ):
    try:
        
        # Load local items.json each time the endpoint is hit
        with open("items.json", "r", encoding="utf-8") as f:
            sample_data = json.load(f)
        
        # Initialize agent
        agent = PurchaseInsightsAgent(project_id=PROJECT_ID)

        # Run analysis
        results = agent.analyze_purchases(sample_data)

        # Return as JSON
        return JSONResponse(content=results.model_dump(), status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ========================== Google Wallet Integration ==========================

wallet_manager = ReceiptWalletManager()

@app.post("/receipts/create-wallet-pass", response_model=WalletPassResponse)
async def create_wallet_pass(
    request: Request,
    auth=Depends(firebase_auth_required)
):
    """
    Create a Google Wallet pass for a receipt.
    
    This endpoint takes receipt data and creates a digital wallet pass
    that users can add to their Google Wallet app.
    """
    try:
        
        # Create wallet pass
        receipt_data = await request.body()
        logger.info(f"Creating wallet pass with data: {parse_json(receipt_data)}")
        result = await wallet_manager.create_receipt_pass(parse_json(receipt_data))
        
        if result['success']:
            return WalletPassResponse(
                success=True,
                message="Wallet pass created successfully",
                object_id=result['object_id'],
                wallet_link=result['wallet_link']
            )
        else:
            logger.error(f"Failed to create wallet pass: {result.get('error', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Internal error creating wallet pass: {e}")

# @app.patch("/receipts/update-wallet-pass/{object_id}", response_model=WalletPassResponse)
# async def update_wallet_pass(object_id: str, update_data: UpdatePassData):
#     """
#     Update an existing Google Wallet pass with new information.
    
#     This can be used to add AI-generated insights, spending tips, or other
#     dynamic content to existing wallet passes.
#     """
#     try:
#         if not object_id.strip():
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Object ID is required"
#             )
        
#         result = await wallet_manager.update_wallet_pass(object_id, update_data)
        
#         if result['success']:
#             return WalletPassResponse(
#                 success=True,
#                 message="Wallet pass updated successfully",
#                 object_id=object_id
#             )
#         else:
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail=result.get('error', 'Failed to update wallet pass')
#             )
            
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error updating wallet pass: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=str(e)
#         )

# @app.post("/receipts/expire-wallet-pass/{object_id}", response_model=WalletPassResponse)
# async def expire_wallet_pass(object_id: str):
#     """
#     Expire a Google Wallet pass.
    
#     This marks the wallet pass as expired, which will update its status
#     in the user's Google Wallet app.
#     """
#     try:
#         if not object_id.strip():
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Object ID is required"
#             )
        
#         result = await wallet_manager.expire_wallet_pass(object_id)
        
#         if result['success']:
#             return WalletPassResponse(
#                 success=True,
#                 message="Wallet pass expired successfully",
#                 object_id=object_id
#             )
#         else:
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail=result.get('error', 'Failed to expire wallet pass')
#             )
            
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error expiring wallet pass: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=str(e)
#         )


