import vertexai
from fastapi import ( FastAPI, HTTPException, Query, Body, Request, File, UploadFile, Form, Depends, Header)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.background import BackgroundTasks
from fastapi.responses import JSONResponse
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
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

load_dotenv()

# Initialize Vertex AI
try:
    PROJECT_ID = os.getenv("PROJECT_ID")
    LOCATION = os.getenv("GCP_LOCATION")
    vertexai.init(project=PROJECT_ID, location=LOCATION)
except KeyError:
    raise RuntimeError("GCP_PROJECT_ID not found in .env file. Please set it.")

#Initialize cloud storage and firestore clients

try:
    BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
except KeyError:
    raise RuntimeError("GCS_BUCKET_NAME not found in .env file. Please set it.")

storage_client = initialize_gcs_client()
db = initialize_firestore()
bucket = storage_client.bucket(BUCKET_NAME)

app = FastAPI(
    title="Raseed API",
    version="1.0.0",
    root_path ="/api",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.google.com",
        "http://localhost:3000",
        "http://localhost:8000"# Allow local development
        "http://localhost:5173", # Added for Vite's default port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Welcome to the API. Visit /docs for documentation."}

@app.post("/receipts/analyze")
async def analyze_receipt(file: UploadFile = File(...)):
    """
    Analyzes a receipt image using a few-shot conversational prompt with a system instruction.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # 1. Initialize the model with our detailed system instruction
        model = GenerativeModel(
            "gemini-2.0-flash-001",
            system_instruction=SYSTEM_INSTRUCTION

        )

        # 2. Construct the conversational prompt with few-shot examples
        conversation_history = []
        for example in FEW_SHOT_EXAMPLES:
            # # Add the example image
            # image_part = Part.from_data(
            #     data=base64.b64decode(example["image_base64"]), 
            #     mime_type=example["mime_type"]
            # )
            # Add the expected perfect JSON output
            json_part = Part.from_text(json.dumps(example["expected_json"]))
            
            # Add to history as a user/model turn
            #conversation_history.append(image_part)
            conversation_history.append(json_part)

        # 3. Add the user's actual uploaded image to the end of the conversation
        user_image_bytes = await file.read()
        user_image_part = Part.from_data(data=user_image_bytes, mime_type=file.content_type)
        conversation_history.append(user_image_part)
        
        generation_config = GenerationConfig(
            response_mime_type="application/json",
            temperature=1.5,
            max_output_tokens=3072,
        )

        # 4. Send the full conversation to the model for analysis
        response = await model.generate_content_async(
            conversation_history,
            generation_config=generation_config,
        )

        # 5. Parse the clean JSON response and return it to the frontend
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
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")