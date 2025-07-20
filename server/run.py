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
from agents.prompts import SYSTEM_INSTRUCTION, FEW_SHOT_EXAMPLES
from datetime import datetime, timezone
from google.cloud import firestore, storage
from utils.google_services_utils import initialize_firestore, initialize_gcs_client
from server.utils.storage_utils import save_receipt_to_cloud
from server.utils.fetch_data_utils import fetch_user_data_by_email
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

import firebase_admin
from firebase_admin import credentials, auth

from agents.prompts import SYSTEM_INSTRUCTION, FEW_SHOT_EXAMPLES
from agents.insights_agent import PurchaseInsightsAgent


# ========================== Environment Setup ==========================
load_dotenv()

try:
    PROJECT_ID = os.getenv("PROJECT_ID")
    LOCATION = os.getenv("GCP_LOCATION")
    vertexai.init(project=PROJECT_ID, location=LOCATION)
except KeyError:
    raise RuntimeError("GCP_PROJECT_ID not found in .env file.")

if not firebase_admin._apps:
    FIREBASE_CRED_PATH = os.environ.get("FIREBASE_CRED_PATH", "firebase-sdk.json")
    try:
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        raise RuntimeError(f"Could not initialize Firebase Admin SDK: {str(e)}")

#Initialize cloud storage and firestore clients

try:
    BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
except KeyError:
    raise RuntimeError("GCS_BUCKET_NAME not found in .env file. Please set it.")

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
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        model = GenerativeModel(
            "gemini-2.0-flash-001",
            system_instruction=SYSTEM_INSTRUCTION
        )

        conversation_history = []
        for example in FEW_SHOT_EXAMPLES:
            json_part = Part.from_text(json.dumps(example["expected_json"]))
            conversation_history.append(json_part)

        user_image_bytes = await file.read()
        user_image_part = Part.from_data(data=user_image_bytes, mime_type=file.content_type)
        conversation_history.append(user_image_part)

        generation_config = GenerationConfig(
            response_mime_type="application/json",
            temperature=1.5,
            max_output_tokens=3072,
        )

        response = await model.generate_content_async(
            conversation_history,
            generation_config=generation_config,
        )

        parsed_data = json.loads(response.text)

        final_data = save_receipt_to_cloud(
            db=db,
            bucket=bucket,
            parsed_data=parsed_data,
            image_bytes=user_image_bytes,
            file=file,
            user_id=None 
        )

        return JSONResponse(content=final_data)

    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from Vertex AI response: {response.text}")
        raise HTTPException(status_code=500, detail="Could not parse the AI model's response.")
    except Exception as e:
        logging.error(f"An error occurred during receipt analysis: {e}")
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
        
        # --- Extract user info from Firebase token ---
        user = request.state.user
        email = user.get("email")

        if not email:
            return JSONResponse(content={}, status_code=200)

        # --- Fetch all receipts from Firestore ---
        receipts = fetch_user_data_by_email(db, email)
        if not receipts:
            return JSONResponse(content={}, status_code=200)
        
        # Initialize agent
        agent = PurchaseInsightsAgent(project_id=PROJECT_ID)

        # Run analysis
        results = agent.analyze_purchases(receipts)

        # Return as JSON
        return JSONResponse(content=results.model_dump(), status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
