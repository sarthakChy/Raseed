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
from utils.google_services_utils import initialize_firestore, initialize_gcs_client
from server.utils.storage_utils import save_receipt_to_cloud
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

import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig

from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from utils.credentials import get_credentials

from agents.insights_agent import PurchaseInsightsAgent
from agents.receipt.receipts_agent import ReceiptAgent
from agents.receipt.prompt import SYSTEM_INSTRUCTION, FEW_SHOT_EXAMPLES
from utils.parse_json import parse_json

import firebase_admin
from firebase_admin import credentials, auth

# ========================== Environment Setup ==========================
load_dotenv()

if not firebase_admin._apps:
    FIREBASE_CRED_PATH = os.environ.get("FIREBASE_CRED_PATH", "firebase-sdk.json")
    try:
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        raise RuntimeError(f"Could not initialize Firebase Admin SDK: {str(e)}")

PROJECT_ID,LOCATION, BUCKET_NAME = get_credentials()

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
                    project_id=PROJECT_ID,
                    location=LOCATION,
                    file_bytes=user_image_bytes,
                    file_content_type=file.content_type,
                    system_instruction=SYSTEM_INSTRUCTION
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
        return JSONResponse(content={"error": str(e)}, status_code=501)
        

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