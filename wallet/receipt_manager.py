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
                            'oneItem': {
                                'item': {
                                    'firstValue': {
                                        'fields': [{
                                            'fieldPath': 'object.textModulesData[\'date\']'
                                        }]
                                    }
                                }
                            }
                        },
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
    
    async def create_receipt_pass(self, receipt_data: dict, uuid: str) -> Dict[str, Any]:
        """Create a Google Wallet pass for a receipt."""
        
        # Generate unique identifiers
        class_suffix = "receipt_class"
        
        # Ensure class exists
        class_id = self.create_receipt_class(class_suffix)
        object_id = f'{self.issuer_id}.{uuid}'
        
        # Parse receipt data from your actual structure
        ocr_data = receipt_data.get('ocrData', {})
        extracted_data = ocr_data.get('extractedData',{})
        
        # Get the correct field names from your structure
        merchant_name = extracted_data.get('merchantName') or extracted_data.get('normalizedMerchant') or 'Unknown Store'
        date = extracted_data.get('date') or datetime.now().strftime('%Y-%m-%d')
        total_amount = extracted_data.get('totalAmount')
        subtotal = extracted_data.get('subtotal')
        tax = extracted_data.get('tax')
        items = extracted_data.get('items') or []
        payment_method = extracted_data.get('paymentMethod') or 'Not specified'
        currency = extracted_data.get('currency', 'USD')
        address = extracted_data.get('address') or 'Not available'
        
        # Format date for display
        try:
            if date:
                date_obj = datetime.strptime(date, '%Y-%m-%d')
                formatted_date = date_obj.strftime('%B %d, %Y')
            else:
                formatted_date = datetime.now().strftime('%B %d, %Y')
        except:
            formatted_date = date or datetime.now().strftime('%B %d, %Y')
        
        # Format amounts with currency
        def format_amount(amount):
            if amount is not None:
                return f"{amount:.2f}{currency}" if isinstance(amount, (int, float)) else f"{amount}{currency}"
            return "Not available"
        
        total_display = format_amount(total_amount)
        subtotal_display = format_amount(subtotal)
        tax_display = format_amount(tax)
        
        # Create items text for display
        items_text = ""
        if items and isinstance(items, list):
            for item in items[:10]:  # Show only first 10 items
                if isinstance(item, dict):
                    item_name = item.get('name', 'Unknown Item')
                    item_price = item.get('price')
                    logger.info(f"Processing item: {item_name} - Creating wallet pass with ID: {object_id}")
                    
                    items_text += f"• {item_name}"
                    if item_price is not None:
                        items_text += f": {format_amount(item_price)}"
                    items_text += "\n"
            
            if len(items) > 10:
                items_text += f"... and {len(items) - 10} more items"
        else:
            items_text = "No items listed"
        
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
                    'value': merchant_name
                }
            },
            'header': {
                'defaultValue': {
                    'language': 'en-US',
                    'value': f'Receipt - {formatted_date}'
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
                    'body': merchant_name
                },
                {
                    'id': 'total_amount',
                    'header': 'Total',
                    'body': total_display
                },
                {
                    'id': 'date',
                    'header': 'Date',
                    'body': formatted_date
                },
                {
                    'id': 'subtotal',
                    'header': 'Subtotal',
                    'body': subtotal_display
                },
                {
                    'id': 'tax',
                    'header': 'Tax',
                    'body': tax_display
                },
                {
                    'id': 'payment_method',
                    'header': 'Payment Method',
                    'body': payment_method
                },
                {
                    'id': 'currency',
                    'header': 'Currency',
                    'body': currency
                },
                {
                    'id': 'address',
                    'header': 'Store Address',
                    'body': address
                },
                {
                    'id': 'items_list',
                    'header': 'Items Purchased',
                    'body': items_text
                }
            ],
            'linksModuleData': {
                'uris': [
                    {
                        'uri': f'https://{self.base_domain}/receipts/{uuid}',
                        'description': 'View in RASEED App',
                        'id': 'raseed_app_link'
                    },
                    {
                        'uri': f'https://{self.base_domain}/receipts/{uuid}/details',
                        'description': 'Receipt Details',
                        'id': 'receipt_details_link'
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
                'value': f'RASEED_RECEIPT_{uuid}',
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
            'smartTapRedemptionValue': f'RASEED_RECEIPT_{uuid}',
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
                'id': 'raseed_insights',
                'header': 'RASEED Insights & Tips',
                'body': update_data['insights']
            }
            
            # Remove existing insights and add new one
            text_modules = [m for m in text_modules if m.get('id') != 'raseed_insights']
            text_modules.append(insights_module)
            
            patch_body['textModulesData'] = text_modules
            
            response = self.client.genericobject().update(
                resourceId=object_id,
                body=patch_body
            ).execute()
            
            logger.info(f'Wallet pass updated successfully: {object_id}')
            return {
                'success': True,
                'object_id': object_id,
                'pass_data': response
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
                'object_id': object_id,
                'pass_data':response
            }
            
        except HttpError as e:
            logger.error(f'Error expiring wallet pass: {e}')
            return {
                'success': False,
                'error': str(e)
            }