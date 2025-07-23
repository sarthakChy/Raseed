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

class SmartInsightsWalletManager:
    """Manages Google Wallet passes for smart insights based on receipts in RASEED app."""
    
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
    
    def create_insights_class(self, class_suffix: str = "smart_insights_class") -> str:
        """Create a smart insights pass class if it doesn't exist."""
        class_id = f'{self.issuer_id}.{class_suffix}'
        
        try:
            self.client.genericclass().get(resourceId=class_id).execute()
            logger.info(f'Smart insights class {class_id} already exists!')
            return class_id
        except HttpError as e:
            if e.status_code != 404:
                logger.error(f'Error checking smart insights class: {e}')
                return class_id
        
        # Create new insights class with enhanced template
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
                                            'fieldPath': 'object.textModulesData[\'merchant_name\']'
                                        }]
                                    }
                                },
                                'endItem': {
                                    'firstValue': {
                                        'fields': [{
                                            'fieldPath': 'object.textModulesData[\'category\']'
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
                                            'fieldPath': 'object.textModulesData[\'insight_type\']'
                                        }]
                                    }
                                },
                                'endItem': {
                                    'firstValue': {
                                        'fields': [{
                                            'fieldPath': 'object.textModulesData[\'confidence_score\']'
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
            logger.info('Smart insights class created successfully')
            return class_id
        except HttpError as e:
            logger.error(f'Error creating smart insights class: {e}')
            return class_id
    
    async def create_insights_pass(self, insights_data: dict, uuid: str) -> Dict[str, Any]:
        """Create a Google Wallet pass for smart insights."""
        
        # Generate unique identifiers
        class_suffix = "smart_insights_class"
        
        # Ensure class exists
        class_id = self.create_insights_class(class_suffix)
        object_id = f'{self.issuer_id}.insights_{uuid}'
        
        # Parse insights data
        merchant_name = insights_data.get('merchantName', 'Unknown Merchant')
        category = insights_data.get('category', 'General')
        receipt_date = insights_data.get('receiptDate', datetime.now().strftime('%Y-%m-%d'))
        insight_type = insights_data.get('insightType', 'General Analysis')
        
        # Format date for display
        try:
            if receipt_date:
                date_obj = datetime.strptime(receipt_date, '%Y-%m-%d')
                formatted_date = date_obj.strftime('%B %d, %Y')
            else:
                formatted_date = datetime.now().strftime('%B %d, %Y')
        except:
            formatted_date = receipt_date or datetime.now().strftime('%B %d, %Y')
        
        # Get insights sections
        spending_insights = insights_data.get('spendingInsights', 'No spending insights available')
        savings_tips = insights_data.get('savingsTips', 'No savings tips available')
        merchant_insights = insights_data.get('merchantInsights', 'No merchant insights available')
        category_trends = insights_data.get('categoryTrends', 'No category trends available')
        budget_impact = insights_data.get('budgetImpact', 'No budget impact analysis available')
        seasonal_tips = insights_data.get('seasonalTips', 'No seasonal tips available')
        
        # AI confidence score
        confidence_score = insights_data.get('confidenceScore', 0.85)
        confidence_display = f'{confidence_score:.1%}' if isinstance(confidence_score, (int, float)) else str(confidence_score)
        
        # Related receipts
        related_receipts = insights_data.get('relatedReceipts', [])
        related_count = len(related_receipts) if related_receipts else 0
        
        # Create the wallet pass object
        insights_object = {
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
                        'value': 'RASEED Smart Insights'
                    }
                }
            },
            'cardTitle': {
                'defaultValue': {
                    'language': 'en-US',
                    'value': f'💡 Smart Insights - {merchant_name}'
                }
            },
            'header': {
                'defaultValue': {
                    'language': 'en-US',
                    'value': f'Insights for {formatted_date}'
                }
            },
            'heroImage': {
                'sourceUri': {
                    'uri': 'https://storage.googleapis.com/wallet-lab-tools-codelab-artifacts-public/google-io-hero-demo-only.png'
                },
                'contentDescription': {
                    'defaultValue': {
                        'language': 'en-US',
                        'value': 'Smart insights hero image'
                    }
                }
            },
            'textModulesData': [
                {
                    'id': 'merchant_name',
                    'header': 'Merchant',
                    'body': merchant_name
                },
                {
                    'id': 'category',
                    'header': 'Category',
                    'body': category
                },
                {
                    'id': 'insight_type',
                    'header': 'Analysis Type',
                    'body': insight_type
                },
                {
                    'id': 'confidence_score',
                    'header': 'AI Confidence',
                    'body': confidence_display
                },
                {
                    'id': 'receipt_date',
                    'header': 'Receipt Date',
                    'body': formatted_date
                },
                {
                    'id': 'related_receipts',
                    'header': 'Related Receipts',
                    'body': f'{related_count} similar transactions analyzed'
                },
                {
                    'id': 'spending_insights',
                    'header': '💰 Spending Analysis',
                    'body': spending_insights
                },
                {
                    'id': 'savings_tips',
                    'header': '💡 Money-Saving Tips',
                    'body': savings_tips
                },
                {
                    'id': 'merchant_insights',
                    'header': '🏪 Merchant Insights',
                    'body': merchant_insights
                },
                {
                    'id': 'category_trends',
                    'header': '📊 Category Trends',
                    'body': category_trends
                },
                {
                    'id': 'budget_impact',
                    'header': '📈 Budget Impact',
                    'body': budget_impact
                },
                {
                    'id': 'seasonal_tips',
                    'header': '🗓️ Seasonal Tips',
                    'body': seasonal_tips
                }
            ],
            'linksModuleData': {
                'uris': [
                    {
                        'uri': f'https://{self.base_domain}/insights/{uuid}',
                        'description': 'View Full Analysis',
                        'id': 'full_analysis_link'
                    },
                    {
                        'uri': f'https://{self.base_domain}/insights/{uuid}/budget',
                        'description': 'Budget Tracker',
                        'id': 'budget_link'
                    },
                    {
                        'uri': f'https://{self.base_domain}/insights/merchant/{merchant_name.replace(" ", "-").lower()}',
                        'description': f'More {merchant_name} Insights',
                        'id': 'merchant_insights_link'
                    },
                    {
                        'uri': f'https://{self.base_domain}/insights/category/{category.lower()}',
                        'description': f'{category} Spending Trends',
                        'id': 'category_insights_link'
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
                'value': f'RASEED_INSIGHTS_{uuid}',
            },
            'hexBackgroundColor': '#9c27b0',  # Purple color for insights
            'validTimeInterval': {
                'start': {
                    'date': datetime.now().isoformat() + 'Z'
                },
                'end': {
                    'date': (datetime.now() + timedelta(days=30)).isoformat() + 'Z'  # 30 days validity for insights
                }
            },
            'smartTapRedemptionValue': f'RASEED_INSIGHTS_{uuid}',
            'hasUsers': True
        }
        
        # Add personalized recommendations if available
        if 'personalizedRecommendations' in insights_data:
            insights_object['textModulesData'].append({
                'id': 'personalized_recommendations',
                'header': '🎯 Personalized for You',
                'body': insights_data['personalizedRecommendations']
            })
        
        # Add alternative stores suggestions
        if 'alternativeStores' in insights_data:
            insights_object['textModulesData'].append({
                'id': 'alternative_stores',
                'header': '🏪 Try These Alternatives',
                'body': insights_data['alternativeStores']
            })
        
        # Add deal alerts
        if 'dealAlerts' in insights_data:
            insights_object['textModulesData'].append({
                'id': 'deal_alerts',
                'header': '🔔 Deal Alerts',
                'body': insights_data['dealAlerts']
            })
        
        try:
            # Create the object
            response = self.client.genericobject().insert(body=insights_object).execute()
            logger.info(f'Smart insights pass created successfully: {object_id}')
            
            # Generate "Add to Wallet" link
            wallet_link = self.create_wallet_link(insights_object)
            
            return {
                'success': True,
                'object_id': object_id,
                'wallet_link': wallet_link,
                'pass_data': response
            }
            
        except HttpError as e:
            logger.error(f'Error creating smart insights pass: {e}')
            return {
                'success': False,
                'error': str(e),
                'object_id': object_id
            }
    
    def create_wallet_link(self, insights_object: dict) -> str:
        """Generate a signed JWT for 'Add to Google Wallet' link."""
        
        # Create the JWT claims
        claims = {
            'iss': self.credentials.service_account_email,
            'aud': 'google',
            'origins': [self.base_domain],
            'typ': 'savetowallet',
            'payload': {
                'genericObjects': [insights_object]
            }
        }
        
        # Sign the JWT
        signer = crypt.RSASigner.from_service_account_file(self.key_file_path)
        token = jwt.encode(signer, claims).decode('utf-8')
        
        return f'https://pay.google.com/gp/v/save/{token}'
    
    async def update_insights_pass(self, object_id: str, update_data: dict) -> Dict[str, Any]:
        """Update an existing insights pass with new analysis."""
        try:
            current_pass = self.client.genericobject().get(resourceId=object_id).execute()
            patch_body = current_pass.copy()
            
            text_modules = patch_body.get("textModulesData", [])
            
            # Update specific insight sections
            for module in text_modules:
                module_id = module.get('id')
                
                if module_id == 'spending_insights' and 'spendingInsights' in update_data:
                    module['body'] = update_data['spendingInsights']
                elif module_id == 'savings_tips' and 'savingsTips' in update_data:
                    module['body'] = update_data['savingsTips']
                elif module_id == 'merchant_insights' and 'merchantInsights' in update_data:
                    module['body'] = update_data['merchantInsights']
                elif module_id == 'category_trends' and 'categoryTrends' in update_data:
                    module['body'] = update_data['categoryTrends']
                elif module_id == 'budget_impact' and 'budgetImpact' in update_data:
                    module['body'] = update_data['budgetImpact']
                elif module_id == 'seasonal_tips' and 'seasonalTips' in update_data:
                    module['body'] = update_data['seasonalTips']
                elif module_id == 'confidence_score' and 'confidenceScore' in update_data:
                    confidence = update_data['confidenceScore']
                    module['body'] = f'{confidence:.1%}' if isinstance(confidence, (int, float)) else str(confidence)
            
            # Add new insights sections if provided
            if 'newInsights' in update_data:
                for insight_id, insight_data in update_data['newInsights'].items():
                    # Remove existing module with same ID
                    text_modules = [m for m in text_modules if m.get('id') != insight_id]
                    # Add new module
                    text_modules.append({
                        'id': insight_id,
                        'header': insight_data.get('header', 'New Insight'),
                        'body': insight_data.get('body', 'No content available')
                    })
            
            patch_body['textModulesData'] = text_modules
            
            # Update timestamp
            patch_body['textModulesData'].append({
                'id': 'last_updated',
                'header': 'Last Updated',
                'body': datetime.now().strftime('%B %d, %Y at %I:%M %p')
            })
            
            response = self.client.genericobject().patch(
                resourceId=object_id,
                body=patch_body
            ).execute()
            
            logger.info(f'Smart insights pass updated successfully: {object_id}')
            return {
                'success': True,
                'object_id': object_id,
                'response': response
            }
            
        except HttpError as e:
            logger.error(f'Error updating smart insights pass: {e}')
            return {
                'success': False,
                'error': str(e)
            }
    
    async def expire_insights_pass(self, object_id: str) -> Dict[str, Any]:
        """Expire a smart insights pass."""
        try:
            patch_body = {'state': 'EXPIRED'}
            
            response = self.client.genericobject().patch(
                resourceId=object_id,
                body=patch_body
            ).execute()
            
            logger.info(f'Smart insights pass expired successfully: {object_id}')
            return {
                'success': True,
                'object_id': object_id
            }
            
        except HttpError as e:
            logger.error(f'Error expiring smart insights pass: {e}')
            return {
                'success': False,
                'error': str(e)
            }
    
    async def refresh_insights(self, object_id: str, fresh_insights_data: dict) -> Dict[str, Any]:
        """Completely refresh insights with new AI analysis."""
        try:
            # This essentially recreates the insights with fresh data
            return await self.update_insights_pass(object_id, fresh_insights_data)
            
        except Exception as e:
            logger.error(f'Error refreshing insights: {e}')
            return {
                'success': False,
                'error': str(e)
            }