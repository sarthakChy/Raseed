from fastapi import ( FastAPI, HTTPException, Query, Body, Request, File, UploadFile, Form, Depends, Header)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.background import BackgroundTasks
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import List, Dict, Any, Literal
from datetime import datetime, timedelta, timezone
from google.oauth2 import id_token
from google.auth.transport import requests
import requests as req
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
from dotenv import load_dotenv
load_dotenv()

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
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Welcome to the API. Visit /docs for documentation."}