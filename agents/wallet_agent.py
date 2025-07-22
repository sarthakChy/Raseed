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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReceiptWalletManager:
    """Manages Google Wallet passes for receipts in RASEED app."""
    
    def __init__(self):
        self.key_file_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        self.issuer_id = os.environ.get('GOOGLE_WALLET_ISSUER_ID')
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
    
    async def create_receipt_pass(self, receipt_data: dict,uuid:str) -> Dict[str, Any]:
        """Create a Google Wallet pass for a receipt."""
        
        # Generate unique identifiers
        class_suffix = "receipt_class"
        
        # Ensure class exists
        class_id = self.create_receipt_class(class_suffix)
        object_id = f'{self.issuer_id}.{uuid}'
        
        # Parse receipt data
        store_name = receipt_data['merchantName']
        date = receipt_data['transactionDate'] or datetime.now().strftime('%B %d, %Y')
        total_amount = receipt_data['total']
        items = receipt_data['items'] or []
        category = receipt_data['category'] if 'category' in receipt_data else 'General'
        
        # Create items text for display
        items_text = ""
        for item in items[:5]:  # Show only first 5 items
            logger.info(f"{item['name']} - Creating wallet pass with ID: {object_id}")
            items_text += f"• {item['name']}: {item['price']}\n"
        
        if len(items) > 5:
            items_text += f"... and {len(items) - 5} more items"
        
        # Create the wallet pass object
        receipt_object = {
            'id': object_id,
            'classId': class_id,
            'state': 'ACTIVE',
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
            'heroImage': {
                'sourceUri': {
                    'uri': 'https://storage.googleapis.com/wallet-lab-tools-codelab-artifacts-public/google-io-hero-demo-only.png'
                },
                'contentDescription': {
                    'defaultValue': {
                        'language': 'en-US',
                        'value': 'Receipt hero image'
                    }
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
                'value': " ",
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
            'smartTapRedemptionValue': f'RASEED_{uuid}',
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
    
    async def update_wallet_pass(self, object_id: str, update_data) -> Dict[str, Any]:
        """Update an existing wallet pass by preserving all existing fields and adding insights."""
        try:
            
            current_pass = self.client.genericobject().get(resourceId=object_id).execute()

            patch_body = current_pass.copy()

            text_modules = patch_body.get("textModulesData", [])

            insights_module = {
                'id': 'ai_insights',
                'header': 'AI Insights & Tips',
                'body': update_data['insights']
            }

            text_modules = [m for m in text_modules if m.get('id') != 'ai_insights']
            text_modules.append(insights_module)

            patch_body['textModulesData'] = text_modules

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
