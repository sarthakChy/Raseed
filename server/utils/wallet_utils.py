from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import os
import uuid
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.service_account import Credentials
from google.auth import jwt, crypt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RASEED Wallet API",
    description="Google Wallet integration for receipt management",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ReceiptItem(BaseModel):
    name: str
    price: str

class ReceiptData(BaseModel):
    store_name: str
    total_amount: str
    date: Optional[str] = None
    category: Optional[str] = "Groceries"
    items: Optional[List[ReceiptItem]] = []

class UpdatePassData(BaseModel):
    insights: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None

class WalletPassResponse(BaseModel):
    success: bool
    message: str
    object_id: Optional[str] = None
    wallet_link: Optional[str] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str

class ReceiptWalletManager:
    """Manages Google Wallet passes for receipts in RASEED app."""
    
    def __init__(self):
        self.key_file_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '/home/sarthak/Desktop/Raseed/server/serviceAccountKey.json')
        self.issuer_id = os.environ.get('GOOGLE_WALLET_ISSUER_ID','3388000000022970942')
        self.base_domain = os.environ.get('RASEED_DOMAIN', 'Raseed.com')
        self.auth()
    
    def auth(self):
        """Create authenticated HTTP client using a service account file."""
        try:
            self.credentials = Credentials.from_service_account_file(
                self.key_file_path,
                scopes=['https://www.googleapis.com/auth/wallet_object.issuer']
            )
            self.client = build('walletobjects', 'v1', credentials=self.credentials)
            logger.info("Google Wallet API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google Wallet API client: {e}")
            raise
    
    def create_receipt_class(self, class_suffix: str = "receipt_class") -> str:
        """Create a receipt pass class if it doesn't exist."""
        class_id = f'{self.issuer_id}.{class_suffix}'
        
        try:
            self.client.genericclass().get(resourceId=class_id).execute()
            logger.info(f'Class {class_id} already exists!')
            return class_id
        except HttpError as e:
            if e.status_code != 404:
                logger.error(f'Error checking class: {e}')
                return class_id
        
        # Create new receipt class with enhanced template
        new_class = {
            'id': class_id,
            'classTemplateInfo': {
                'cardTemplateOverride': {
                    'cardRowTemplateInfos': [
                        {
                            'twoItems': {
                                'startItem': {
                                    'firstValue': {
                                        'fields': [{
                                            'fieldPath': 'object.textModulesData[\'store_name\']'
                                        }]
                                    }
                                },
                                'endItem': {
                                    'firstValue': {
                                        'fields': [{
                                            'fieldPath': 'object.textModulesData[\'total_amount\']'
                                        }]
                                    }
                                }
                            }
                        },
                        {
                            'oneItem': {
                                'item': {
                                    'firstValue': {
                                        'fields': [{
                                            'fieldPath': 'object.textModulesData[\'category\']'
                                        }]
                                    }
                                }
                            }
                        }
                    ]
                }
            },
            'enableSmartTap': True,
            'redemptionIssuers': [self.issuer_id]
        }
        
        try:
            response = self.client.genericclass().insert(body=new_class).execute()
            logger.info('Receipt class created successfully')
            return class_id
        except HttpError as e:
            logger.error(f'Error creating class: {e}')
            return class_id
    
    async def create_receipt_pass(self, receipt_data: ReceiptData) -> Dict[str, Any]:
        """Create a Google Wallet pass for a receipt."""
        
        # Generate unique identifiers
        object_suffix = f"receipt_{uuid.uuid4().hex[:8]}"
        class_suffix = "receipt_class"
        
        # Ensure class exists
        class_id = self.create_receipt_class(class_suffix)
        object_id = f'{self.issuer_id}.{object_suffix}'
        
        # Parse receipt data
        store_name = receipt_data.store_name
        date = receipt_data.date or datetime.now().strftime('%B %d, %Y')
        total_amount = receipt_data.total_amount
        items = receipt_data.items or []
        category = receipt_data.category
        
        # Create items text for display
        items_text = ""
        for item in items[:5]:  # Show only first 5 items
            items_text += f"• {item.name}: {item.price}\n"
        
        if len(items) > 5:
            items_text += f"... and {len(items) - 5} more items"
        
        # Create the wallet pass object
        receipt_object = {
            'id': object_id,
            'classId': class_id,
            'state': 'ACTIVE',
            'heroImage': {
                'sourceUri': {
                    'uri': 'https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?w=400&h=200&fit=crop'
                },
                'contentDescription': {
                    'defaultValue': {
                        'language': 'en-US',
                        'value': 'Receipt hero image'
                    }
                }
            },
            'logo': {
                'sourceUri': {
                    'uri': 'https://storage.googleapis.com/wallet-lab-tools-codelab-artifacts-public/pass_google_logo.jpg'
                },
                'contentDescription': {
                    'defaultValue': {
                        'language': 'en-US',
                        'value': 'RASEED Receipt Manager'
                    }
                }
            },
            'cardTitle': {
                'defaultValue': {
                    'language': 'en-US',
                    'value': store_name
                }
            },
            'header': {
                'defaultValue': {
                    'language': 'en-US',
                    'value': f'Receipt - {date}'
                }
            },
            'textModulesData': [
                {
                    'id': 'store_name',
                    'header': 'Store',
                    'body': store_name
                },
                {
                    'id': 'total_amount',
                    'header': 'Total',
                    'body': total_amount
                },
                {
                    'id': 'category',
                    'header': 'Category',
                    'body': category
                },
                {
                    'id': 'date',
                    'header': 'Date',
                    'body': date
                },
                {
                    'id': 'items_list',
                    'header': 'Items',
                    'body': items_text if items_text else 'No items listed'
                }
            ],
            'linksModuleData': {
                'uris': [
                    {
                        'uri': f'https://{self.base_domain}/receipts',
                        'description': 'View in RASEED App',
                        'id': 'raseed_app_link'
                    },
                    {
                        'uri': 'tel:+919876543210',
                        'description': 'Contact Support',
                        'id': 'support_link'
                    }
                ]
            },
            'barcode': {
                'type': 'QR_CODE',
                'value': f'RASEED_RECEIPT_{object_suffix}',
                'alternateText': object_suffix
            },
            'hexBackgroundColor': '#4285f4',
            'validTimeInterval': {
                'start': {
                    'date': datetime.now().isoformat() + 'Z'
                },
                'end': {
                    'date': (datetime.now() + timedelta(days=365)).isoformat() + 'Z'
                }
            },
            'smartTapRedemptionValue': f'RASEED_{object_suffix}',
            'hasUsers': True
        }
        
        try:
            # Create the object
            response = self.client.genericobject().insert(body=receipt_object).execute()
            logger.info(f'Receipt pass created successfully: {object_id}')
            
            # Generate "Add to Wallet" link
            wallet_link = self.create_wallet_link(receipt_object)
            
            return {
                'success': True,
                'object_id': object_id,
                'wallet_link': wallet_link,
                'pass_data': response
            }
            
        except HttpError as e:
            logger.error(f'Error creating receipt pass: {e}')
            return {
                'success': False,
                'error': str(e),
                'object_id': object_id
            }
    
    def create_wallet_link(self, receipt_object: dict) -> str:
        """Generate a signed JWT for 'Add to Google Wallet' link."""
        
        # Create the JWT claims
        claims = {
            'iss': self.credentials.service_account_email,
            'aud': 'google',
            'origins': [self.base_domain],
            'typ': 'savetowallet',
            'payload': {
                'genericObjects': [receipt_object]
            }
        }
        
        # Sign the JWT
        signer = crypt.RSASigner.from_service_account_file(self.key_file_path)
        token = jwt.encode(signer, claims).decode('utf-8')
        
        return f'https://pay.google.com/gp/v/save/{token}'
    
    async def update_wallet_pass(self, object_id: str, update_data: UpdatePassData) -> Dict[str, Any]:
        """Update an existing wallet pass with new information."""
        try:
            patch_body = {}
            
            if update_data.insights:
                # Add AI insights to the pass
                patch_body['textModulesData'] = [
                    {
                        'id': 'ai_insights',
                        'header': 'AI Insights & Tips',
                        'body': update_data.insights
                    }
                ]
            
            if update_data.additional_info:
                # Add any additional information
                for key, value in update_data.additional_info.items():
                    if 'textModulesData' not in patch_body:
                        patch_body['textModulesData'] = []
                    patch_body['textModulesData'].append({
                        'id': f'additional_{key}',
                        'header': key.replace('_', ' ').title(),
                        'body': str(value)
                    })
            
            response = self.client.genericobject().patch(
                resourceId=object_id,
                body=patch_body
            ).execute()
            
            logger.info(f'Wallet pass updated successfully: {object_id}')
            return {
                'success': True,
                'object_id': object_id,
                'response': response
            }
            
        except HttpError as e:
            logger.error(f'Error updating wallet pass: {e}')
            return {
                'success': False,
                'error': str(e)
            }
    
    async def expire_wallet_pass(self, object_id: str) -> Dict[str, Any]:
        """Expire a wallet pass."""
        try:
            patch_body = {'state': 'EXPIRED'}
            
            response = self.client.genericobject().patch(
                resourceId=object_id,
                body=patch_body
            ).execute()
            
            logger.info(f'Wallet pass expired successfully: {object_id}')
            return {
                'success': True,
                'object_id': object_id
            }
            
        except HttpError as e:
            logger.error(f'Error expiring wallet pass: {e}')
            return {
                'success': False,
                'error': str(e)
            }

# Initialize wallet manager
wallet_manager = ReceiptWalletManager()

@app.post("/api/receipts/create-wallet-pass", response_model=WalletPassResponse)
async def create_wallet_pass(receipt_data: ReceiptData):
    """
    Create a Google Wallet pass for a receipt.
    
    This endpoint takes receipt data and creates a digital wallet pass
    that users can add to their Google Wallet app.
    """
    try:
        # Validate receipt data
        if not receipt_data.store_name.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Store name is required"
            )
        
        if not receipt_data.total_amount.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Total amount is required"
            )
        
        # Create wallet pass
        result = await wallet_manager.create_receipt_pass(receipt_data)
        
        if result['success']:
            return WalletPassResponse(
                success=True,
                message="Wallet pass created successfully",
                object_id=result['object_id'],
                wallet_link=result['wallet_link']
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get('error', 'Failed to create wallet pass')
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating wallet pass: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.patch("/api/receipts/update-wallet-pass/{object_id}", response_model=WalletPassResponse)
async def update_wallet_pass(object_id: str, update_data: UpdatePassData):
    """
    Update an existing Google Wallet pass with new information.
    
    This can be used to add AI-generated insights, spending tips, or other
    dynamic content to existing wallet passes.
    """
    try:
        if not object_id.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Object ID is required"
            )
        
        result = await wallet_manager.update_wallet_pass(object_id, update_data)
        
        if result['success']:
            return WalletPassResponse(
                success=True,
                message="Wallet pass updated successfully",
                object_id=object_id
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get('error', 'Failed to update wallet pass')
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error updating wallet pass: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/api/receipts/expire-wallet-pass/{object_id}", response_model=WalletPassResponse)
async def expire_wallet_pass(object_id: str):
    """
    Expire a Google Wallet pass.
    
    This marks the wallet pass as expired, which will update its status
    in the user's Google Wallet app.
    """
    try:
        if not object_id.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Object ID is required"
            )
        
        result = await wallet_manager.expire_wallet_pass(object_id)
        
        if result['success']:
            return WalletPassResponse(
                success=True,
                message="Wallet pass expired successfully",
                object_id=object_id
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get('error', 'Failed to expire wallet pass')
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error expiring wallet pass: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify service status."""
    return HealthResponse(
        status="healthy",
        service="RASEED Wallet API",
        timestamp=datetime.now().isoformat()
    )

@app.get("/api/docs-info")
async def get_api_info():
    """Get API documentation information."""
    return {
        "title": "RASEED Google Wallet API",
        "description": "API for creating and managing Google Wallet passes for receipts",
        "version": "1.0.0",
        "endpoints": {
            "create_pass": "POST /api/receipts/create-wallet-pass",
            "update_pass": "PATCH /api/receipts/update-wallet-pass/{object_id}",
            "expire_pass": "POST /api/receipts/expire-wallet-pass/{object_id}",
            "health": "GET /api/health"
        }
    }