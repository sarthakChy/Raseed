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

class ShoppingListWalletManager:
    """Manages Google Wallet passes for shopping lists in RASEED app."""
    
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
    
    def create_shopping_list_class(self, class_suffix: str = "shopping_list_class") -> str:
        """Create a shopping list pass class if it doesn't exist."""
        class_id = f'{self.issuer_id}.{class_suffix}'
        
        try:
            self.client.genericclass().get(resourceId=class_id).execute()
            logger.info(f'Shopping list class {class_id} already exists!')
            return class_id
        except HttpError as e:
            if e.status_code != 404:
                logger.error(f'Error checking shopping list class: {e}')
                return class_id
        
        # Create new shopping list class with enhanced template
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
                                            'fieldPath': 'object.textModulesData[\'list_name\']'
                                        }]
                                    }
                                },
                                'endItem': {
                                    'firstValue': {
                                        'fields': [{
                                            'fieldPath': 'object.textModulesData[\'item_count\']'
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
                                            'fieldPath': 'object.textModulesData[\'status\']'
                                        }]
                                    }
                                },
                                'endItem': {
                                    'firstValue': {
                                        'fields': [{
                                            'fieldPath': 'object.textModulesData[\'estimated_total\']'
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
            logger.info('Shopping list class created successfully')
            return class_id
        except HttpError as e:
            logger.error(f'Error creating shopping list class: {e}')
            return class_id
    
    async def create_shopping_list_pass(self, shopping_list_data: dict, uuid: str) -> Dict[str, Any]:
        """Create a Google Wallet pass for a shopping list."""
        
        # Generate unique identifiers
        class_suffix = "shopping_list_class"
        
        # Ensure class exists
        class_id = self.create_shopping_list_class(class_suffix)
        object_id = f'{self.issuer_id}.{uuid}'
        
        # Parse shopping list data
        list_name = shopping_list_data.get('listName', 'My Shopping List')
        created_date = shopping_list_data.get('createdDate', datetime.now().strftime('%B %d, %Y'))
        items = shopping_list_data.get('items', [])
        estimated_total = shopping_list_data.get('estimatedTotal', 'Not calculated')
        category = shopping_list_data.get('category', 'General Shopping')
        store_preference = shopping_list_data.get('storePreference', 'Any Store')
        priority = shopping_list_data.get('priority', 'Normal')
        
        # Calculate statistics
        total_items = len(items)
        completed_items = len([item for item in items if item.get('completed', False)])
        pending_items = total_items - completed_items
        
        # Create items text for display
        items_text = ""
        pending_items_text = ""
        completed_items_text = ""
        
        for item in items[:10]:  # Show only first 10 items
            item_name = item.get('name', 'Unknown Item')
            item_quantity = item.get('quantity', 1)
            item_price = item.get('estimatedPrice', '')
            completed = item.get('completed', False)
            
            item_line = f"• {item_name}"
            if item_quantity > 1:
                item_line += f" (x{item_quantity})"
            if item_price:
                item_line += f" - ${item_price}"
            item_line += "\n"
            
            if completed:
                completed_items_text += f"✓ {item_line}"
            else:
                pending_items_text += item_line
        
        if len(items) > 10:
            items_text = pending_items_text + f"... and {len(items) - 10} more items"
        else:
            items_text = pending_items_text
        
        # Create the wallet pass object
        shopping_list_object = {
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
                        'value': 'RASEED Shopping List'
                    }
                }
            },
            'cardTitle': {
                'defaultValue': {
                    'language': 'en-US',
                    'value': list_name
                }
            },
            'header': {
                'defaultValue': {
                    'language': 'en-US',
                    'value': f'Shopping List - {created_date}'
                }
            },
            'heroImage': {
                'sourceUri': {
                    'uri': 'https://storage.googleapis.com/wallet-lab-tools-codelab-artifacts-public/google-io-hero-demo-only.png'
                },
                'contentDescription': {
                    'defaultValue': {
                        'language': 'en-US',
                        'value': 'Shopping list hero image'
                    }
                }
            },
            'textModulesData': [
                {
                    'id': 'list_name',
                    'header': 'List Name',
                    'body': list_name
                },
                {
                    'id': 'item_count',
                    'header': 'Items',
                    'body': f'{pending_items} pending, {completed_items} completed'
                },
                {
                    'id': 'status',
                    'header': 'Status',
                    'body': 'Completed' if pending_items == 0 else f'{pending_items} items remaining'
                },
                {
                    'id': 'estimated_total',
                    'header': 'Est. Total',
                    'body': f'${estimated_total}' if isinstance(estimated_total, (int, float)) else str(estimated_total)
                },
                {
                    'id': 'category',
                    'header': 'Category',
                    'body': category
                },
                {
                    'id': 'store_preference',
                    'header': 'Preferred Store',
                    'body': store_preference
                },
                {
                    'id': 'priority',
                    'header': 'Priority',
                    'body': priority
                },
                {
                    'id': 'pending_items',
                    'header': 'Pending Items',
                    'body': items_text if items_text.strip() else 'No pending items'
                }
            ],
            'linksModuleData': {
                'uris': [
                    {
                        'uri': f'https://{self.base_domain}/shopping-lists/{uuid}',
                        'description': 'View in RASEED App',
                        'id': 'raseed_app_link'
                    },
                    {
                        'uri': f'https://{self.base_domain}/shopping-lists/{uuid}/edit',
                        'description': 'Edit List',
                        'id': 'edit_list_link'
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
                'value': f'RASEED_SHOPPING_LIST_{uuid}',
            },
            'hexBackgroundColor': '#34a853',  # Green color for shopping lists
            'validTimeInterval': {
                'start': {
                    'date': datetime.now().isoformat() + 'Z'
                },
                'end': {
                    'date': (datetime.now() + timedelta(days=90)).isoformat() + 'Z'  # 90 days validity
                }
            },
            'smartTapRedemptionValue': f'RASEED_SHOPPING_LIST_{uuid}',
            'hasUsers': True
        }
        
        # Add completed items section if there are any
        if completed_items_text.strip():
            shopping_list_object['textModulesData'].append({
                'id': 'completed_items',
                'header': 'Completed Items',
                'body': completed_items_text.strip()
            })
        
        try:
            # Create the object
            response = self.client.genericobject().insert(body=shopping_list_object).execute()
            logger.info(f'Shopping list pass created successfully: {object_id}')
            
            # Generate "Add to Wallet" link
            wallet_link = self.create_wallet_link(shopping_list_object)
            
            return {
                'success': True,
                'object_id': object_id,
                'wallet_link': wallet_link,
                'pass_data': response
            }
            
        except HttpError as e:
            logger.error(f'Error creating shopping list pass: {e}')
            return {
                'success': False,
                'error': str(e),
                'object_id': object_id
            }
    
    def create_wallet_link(self, shopping_list_object: dict) -> str:
        """Generate a signed JWT for 'Add to Google Wallet' link."""
        
        # Create the JWT claims
        claims = {
            'iss': self.credentials.service_account_email,
            'aud': 'google',
            'origins': [self.base_domain],
            'typ': 'savetowallet',
            'payload': {
                'genericObjects': [shopping_list_object]
            }
        }
        
        # Sign the JWT
        signer = crypt.RSASigner.from_service_account_file(self.key_file_path)
        token = jwt.encode(signer, claims).decode('utf-8')
        
        return f'https://pay.google.com/gp/v/save/{token}'
    
    async def update_shopping_list_pass(self, object_id: str, update_data: dict) -> Dict[str, Any]:
        """Update an existing shopping list pass with new data."""
        try:
            # Get current pass
            current_pass = self.client.genericobject().get(resourceId=object_id).execute()
            patch_body = current_pass.copy()
            
            # Parse updated shopping list data
            items = update_data.get('items', [])
            estimated_total = update_data.get('estimatedTotal', 'Not calculated')
            
            # Calculate new statistics
            total_items = len(items)
            completed_items = len([item for item in items if item.get('completed', False)])
            pending_items = total_items - completed_items
            
            
            # Update text modules
            text_modules = patch_body.get("textModulesData", [])
            
            # Update specific modules
            for module in text_modules:
                if module['id'] == 'item_count':
                    module['body'] = f'{pending_items} pending, {completed_items} completed'
                elif module['id'] == 'status':
                    module['body'] = 'Completed' if pending_items == 0 else f'{pending_items} items remaining'
                elif module['id'] == 'estimated_total':
                    module['body'] = f'${estimated_total}' if isinstance(estimated_total, (int, float)) else str(estimated_total)
                elif module['id'] == 'pending_items':
                    # Rebuild pending items text
                    pending_items_text = ""
                    for item in [i for i in items if not i.get('completed', False)][:10]:
                        item_name = item.get('name', 'Unknown Item')
                        item_quantity = item.get('quantity', 1)
                        item_price = item.get('estimatedPrice', '')
                        
                        item_line = f"• {item_name}"
                        if item_quantity > 1:
                            item_line += f" (x{item_quantity})"
                        if item_price:
                            item_line += f" - ${item_price}"
                        pending_items_text += item_line + "\n"
                    
                    module['body'] = pending_items_text if pending_items_text.strip() else 'No pending items'
            
            # Update completed items or add if doesn't exist
            completed_items_text = ""
            for item in [i for i in items if i.get('completed', False)][:10]:
                item_name = item.get('name', 'Unknown Item')
                item_quantity = item.get('quantity', 1)
                item_price = item.get('estimatedPrice', '')
                
                item_line = f"✓ {item_name}"
                if item_quantity > 1:
                    item_line += f" (x{item_quantity})"
                if item_price:
                    item_line += f" - ${item_price}"
                completed_items_text += item_line + "\n"
            
            # Remove existing completed items module
            text_modules = [m for m in text_modules if m.get('id') != 'completed_items']
            
            # Add completed items module if there are completed items
            if completed_items_text.strip():
                text_modules.append({
                    'id': 'completed_items',
                    'header': 'Completed Items',
                    'body': completed_items_text.strip()
                })
            
            # Add shopping insights if provided
            if 'insights' in update_data:
                # Remove existing insights
                text_modules = [m for m in text_modules if m.get('id') != 'shopping_insights']
                text_modules.append({
                    'id': 'shopping_insights',
                    'header': 'Shopping Insights & Tips',
                    'body': update_data['insights']
                })
            
            patch_body['textModulesData'] = text_modules
            
            # Update pass state if shopping is completed
            if pending_items == 0 and update_data.get('markCompleted', False):
                patch_body['state'] = 'COMPLETED'
            
            response = self.client.genericobject().update(
                resourceId=object_id,
                body=patch_body
            ).execute()
            
            logger.info(f'Shopping list pass updated successfully: {object_id}')
            return {
                'success': True,
                'object_id': object_id,
                'response': response
            }
            
        except HttpError as e:
            logger.error(f'Error updating shopping list pass: {e}')
            return {
                'success': False,
                'error': str(e)
            }
    
    async def expire_shopping_list_pass(self, object_id: str) -> Dict[str, Any]:
        """Expire a shopping list pass."""
        try:
            patch_body = {'state': 'EXPIRED'}
            
            response = self.client.genericobject().patch(
                resourceId=object_id,
                body=patch_body
            ).execute()
            
            logger.info(f'Shopping list pass expired successfully: {object_id}')
            return {
                'success': True,
                'object_id': object_id
            }
            
        except HttpError as e:
            logger.error(f'Error expiring shopping list pass: {e}')
            return {
                'success': False,
                'error': str(e)
            }
    
    async def mark_shopping_completed(self, object_id: str) -> Dict[str, Any]:
        """Mark a shopping list as completed."""
        try:
            current_pass = self.client.genericobject().get(resourceId=object_id).execute()
            patch_body = current_pass.copy()
            
            # Update status
            text_modules = patch_body.get("textModulesData", [])
            for module in text_modules:
                if module['id'] == 'status':
                    module['body'] = 'Shopping Completed! ✓'
            
            patch_body['textModulesData'] = text_modules
            patch_body['state'] = 'COMPLETED'
            patch_body['hexBackgroundColor'] = '#0f9d58'  # Darker green for completed
            
            response = self.client.genericobject().patch(
                resourceId=object_id,
                body=patch_body
            ).execute()
            
            logger.info(f'Shopping list marked as completed: {object_id}')
            return {
                'success': True,
                'object_id': object_id,
                'response': response
            }
            
        except HttpError as e:
            logger.error(f'Error marking shopping list as completed: {e}')
            return {
                'success': False,
                'error': str(e)
            }