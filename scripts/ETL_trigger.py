import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any
import uuid
import numpy as np
from dotenv import load_dotenv
import random
import psycopg2
from psycopg2.extras import RealDictCursor
from google.cloud import firestore, secretmanager

# --- Configuration & Initialization ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

class FirestorePostgresETL:
    def __init__(self):
        try:
            self.project_id = os.getenv("GCP_PROJECT_ID")
            if not self.project_id:
                raise ValueError("GCP_PROJECT_ID environment variable not set.")
            
            self.db_firestore = firestore.Client(project=self.project_id)
            self.secret_client = secretmanager.SecretManagerServiceClient()
            self.db_postgres = None
            self.db_config = None
        except Exception as e:
            logger.critical(f"Failed to initialize clients: {e}")
            raise

    def _get_db_config_from_secret_manager(self) -> Dict:
        """Fetches the database configuration from Google Secret Manager."""
        if self.db_config:
            return self.db_config

        secret_name = f"projects/{self.project_id}/secrets/postgres-config/versions/latest"
        try:
            logger.info("Fetching database configuration from Secret Manager...")
            response = self.secret_client.access_secret_version(request={"name": secret_name})
            payload = response.payload.data.decode("UTF-8")
            config = json.loads(payload)
            config['sslmode'] = 'require' # Ensure SSL is required
            self.db_config = config
            logger.info("Successfully fetched database configuration.")
            return self.db_config
        except Exception as e:
            logger.error(f"Failed to retrieve database credentials from Secret Manager: {e}")
            raise

    def get_postgres_connection(self):
        """Establishes or reuses a connection to the PostgreSQL database."""
        try:
            if self.db_postgres is None or self.db_postgres.closed:
                db_params = self._get_db_config_from_secret_manager()
                self.db_postgres = psycopg2.connect(**db_params)
                self.db_postgres.autocommit = False
            return self.db_postgres
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def generate_embedding(self, text: str, dimension: int = 768) -> List[float]:
        """Generate mock embedding vector for the given text."""
        # In production, use actual embedding service like Vertex AI
        np.random.seed(hash(text) % (2**32))
        return np.random.normal(0, 1, dimension).tolist()

    def safe_uuid(self, value: str) -> str:
        """Convert string to valid UUID or generate new one."""
        try:
            return str(uuid.UUID(value))
        except (ValueError, TypeError):
            return str(uuid.uuid4())

    def calculate_progress_percentage(self, current_amount, target_amount):
        """Calculate progress percentage, capped at 100%"""
        if target_amount <= 0:
            return 0
        progress = (current_amount / target_amount) * 100
        return min(progress, 100.0)  # Cap at 100%

    def process_users(self) -> None:
        """Process users collection from Firestore to PostgreSQL."""
        logger.info("Processing users collection...")

        conn = self.get_postgres_connection()
        cursor = conn.cursor()

        try:
            users_ref = self.db_firestore.collection('users')
            users = users_ref.stream()

            for user_doc in users:
                user_data = user_doc.to_dict()

                # Generate user embedding based on preferences and financial profile
                user_text = f"{user_data.get('displayName', '')} {json.dumps(user_data.get('preferences', {}))}"
                preference_embedding = self.generate_embedding(user_text)

                # Extract financial profile
                financial_profile = user_data.get('financialProfile', {})

                # Generate a UUID for the user_id if it's missing or invalid from Firestore
                # This 'candidate' user_id will be used for insertion
                candidate_user_id = self.safe_uuid(user_data.get('userId', ''))

                # Prepare user record for upsert
                user_record = {
                    'user_id': candidate_user_id, # Use the generated/validated UUID here
                    'firebase_uid': user_data.get('userId', ''),
                    'email': user_data.get('email', ''),
                    'display_name': user_data.get('displayName', ''),
                    'created_at': user_data.get('createdAt', datetime.now(timezone.utc)),
                    'last_login_at': user_data.get('lastLoginAt'),
                    'monthly_income': financial_profile.get('monthlyIncome', 0),
                    'risk_tolerance': financial_profile.get('riskTolerance', 'moderate'),
                    'preferences': json.dumps(user_data.get('preferences', {})),
                    'spending_patterns': json.dumps({}),  # Will be populated later
                    'preference_embedding': preference_embedding
                }

                # Upsert user and RETURNING the actual user_id from the database
                cursor.execute("""
                    INSERT INTO users (user_id, firebase_uid, email, display_name, created_at,
                                    last_login_at, monthly_income, risk_tolerance, preferences,
                                    spending_patterns, preference_embedding)
                    VALUES (%(user_id)s, %(firebase_uid)s, %(email)s, %(display_name)s, %(created_at)s,
                            %(last_login_at)s, %(monthly_income)s, %(risk_tolerance)s, %(preferences)s,
                            %(spending_patterns)s, %(preference_embedding)s)
                    ON CONFLICT (firebase_uid)
                    DO UPDATE SET
                        email = EXCLUDED.email,
                        display_name = EXCLUDED.display_name,
                        last_login_at = EXCLUDED.last_login_at,
                        monthly_income = EXCLUDED.monthly_income,
                        risk_tolerance = EXCLUDED.risk_tolerance,
                        preferences = EXCLUDED.preferences,
                        preference_embedding = EXCLUDED.preference_embedding,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING user_id; -- Crucially, return the user_id that was used/updated
                """, user_record)

                # Fetch the user_id that was actually used/returned by the database
                # This handles both new insertions and updates correctly
                actual_user_id = cursor.fetchone()[0]

                # Process budget limits - USE actual_user_id
                budget_limits = financial_profile.get('budgetLimits', {})
                for category, limit_amount in budget_limits.items():
                    if category != 'total' and limit_amount > 0:
                        budget_record = {
                            'budget_id': str(uuid.uuid4()),
                            'user_id': actual_user_id, # Use the actual user_id here!
                            'category': category,
                            'period_type': 'monthly',
                            'limit_amount': limit_amount,
                            'effective_from': datetime.now(timezone.utc).date(),
                            'current_spent': random.randint(1, limit_amount),
                        }

                        cursor.execute("""
                            INSERT INTO budget_limits (budget_id, user_id, category, period_type,
                                                    limit_amount, effective_from, current_spent)
                            VALUES (%(budget_id)s, %(user_id)s, %(category)s, %(period_type)s,
                                    %(limit_amount)s, %(effective_from)s, %(current_spent)s)
                            ON CONFLICT (user_id, category, period_type, effective_from)
                            DO UPDATE SET
                                limit_amount = EXCLUDED.limit_amount,
                                updated_at = CURRENT_TIMESTAMP
                        """, budget_record)

                # Process financial goals - USE actual_user_id
                financial_goals = financial_profile.get('financialGoals', [])
                for goal in financial_goals:
                    target_amount = goal.get('targetAmount', 0)
                    current_amount = goal.get('currentAmount', 0)
                    
                    # Calculate progress percentage, capped at 100%
                    if target_amount <= 0:
                        progress_percentage = 0.0
                    else:
                        raw_progress = (current_amount / target_amount) * 100
                        progress_percentage = min(raw_progress, 100.0)  # Cap at 100%
                    
                    # Debug logging
                    logger.info(f"Goal: {goal.get('category', 'Unknown')} - Current: {current_amount}, Target: {target_amount}, Progress: {progress_percentage}")
                    
                    # Determine status based on progress
                    if progress_percentage >= 100:
                        status = 'completed'
                        completed_at = datetime.now(timezone.utc)
                    else:
                        status = 'active'
                        completed_at = None
                    
                    goal_record = {
                        'goal_id': self.safe_uuid(goal.get('id', '')),
                        'user_id': actual_user_id, # Use the actual user_id here!
                        'description':"None",
                        'title': goal.get('category', 'Financial Goal'),
                        'goal_type': goal.get('type', 'saving'),
                        'target_amount': target_amount,
                        'current_amount': current_amount,
                        'target_date': goal.get('deadline'),
                        'category': goal.get('category'),
                        'priority': goal.get('priority', 'medium'),
                        'status': status,
                        'progress_percentage': round(progress_percentage, 2),  # Round to 2 decimal places
                        'completed_at': completed_at
                    }

                    cursor.execute("""
                        INSERT INTO financial_goals (goal_id, user_id, title, goal_type, target_amount,
                                                current_amount, target_date, category, priority, status,
                                                progress_percentage, completed_at)
                        VALUES (%(goal_id)s, %(user_id)s, %(title)s, %(goal_type)s, %(target_amount)s,
                                %(current_amount)s, %(target_date)s, %(category)s, %(priority)s, %(status)s,
                                %(progress_percentage)s, %(completed_at)s)
                        ON CONFLICT (goal_id)
                        DO UPDATE SET
                            title = EXCLUDED.title,
                            target_amount = EXCLUDED.target_amount,
                            current_amount = EXCLUDED.current_amount,
                            target_date = EXCLUDED.target_date,
                            status = EXCLUDED.status,
                            progress_percentage = EXCLUDED.progress_percentage,
                            completed_at = EXCLUDED.completed_at,
                            updated_at = CURRENT_TIMESTAMP
                    """, goal_record)

            conn.commit()
            logger.info("Successfully processed users collection")

        except Exception as e:
            conn.rollback()
            logger.error(f"Error processing users: {e}")
            raise
        finally:
            cursor.close()

    def process_receipts(self) -> None:
        """Process receiptQueue collection from Firestore to PostgreSQL."""
        logger.info("Processing receipt queue...")
        
        conn = self.get_postgres_connection()
        cursor = conn.cursor()
        
        try:
            receipts_ref = self.db_firestore.collection('receiptQueue')
            receipts = receipts_ref.stream()
            
            for receipt_doc in receipts:
                receipt_data = receipt_doc.to_dict()
                
                # Skip if not processed
                if receipt_data.get('status') != 'completed':
                    continue
                    
                ocr_data = receipt_data.get('ocrData', {})
                extracted_data = ocr_data.get('extractedData', {})
                
                # Process merchant
                merchant_name = extracted_data.get('merchantName', 'Unknown')
                normalized_name = extracted_data.get('normalizedMerchant', merchant_name)
                address = extracted_data.get('address', {})
                
                merchant_embedding = self.generate_embedding(f"{merchant_name} {normalized_name}")
                
                merchant_record = {
                    'merchant_id': str(uuid.uuid4()),
                    'name': merchant_name,
                    'normalized_name': normalized_name,
                    'address': json.dumps(address) if address else None,
                    'merchant_embedding': merchant_embedding
                }
                
                # Upsert merchant
                cursor.execute("""
                    INSERT INTO merchants (merchant_id, name, normalized_name, address, merchant_embedding)
                    VALUES (%(merchant_id)s, %(name)s, %(normalized_name)s, %(address)s, %(merchant_embedding)s)
                    ON CONFLICT (normalized_name) 
                    DO UPDATE SET
                        name = EXCLUDED.name,
                        address = EXCLUDED.address,
                        merchant_embedding = EXCLUDED.merchant_embedding,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING merchant_id
                """, merchant_record)
                
                result = cursor.fetchone()
                if result:
                    merchant_id = result[0]
                else:
                    # Get existing merchant_id
                    cursor.execute("SELECT merchant_id FROM merchants WHERE normalized_name = %s", 
                                 (normalized_name,))
                    merchant_id = cursor.fetchone()[0]
                
                # Get user_id
                cursor.execute("SELECT user_id FROM users WHERE firebase_uid = %s", 
                             (receipt_data.get('userId'),))
                user_result = cursor.fetchone()
                if not user_result:
                    logger.warning(f"User not found: {receipt_data.get('userId')}")
                    continue
                    
                user_id = user_result[0]
                
                # Process transaction
                total_amount = extracted_data.get('totalAmount', 0)
                transaction_date = datetime.strptime(extracted_data.get('date', '2024-01-01'), '%Y-%m-%d').date()
                
                # Determine category based on items
                items = extracted_data.get('items', [])
                main_category = self.determine_transaction_category(items)
                
                transaction_embedding = self.generate_embedding(
                    f"{merchant_name} {main_category} {total_amount} {' '.join([item.get('name', '') for item in items])}"
                )
                
                transaction_record = {
                    'transaction_id': str(uuid.uuid4()),
                    'user_id': user_id,
                    'merchant_id': merchant_id,
                    'amount': total_amount,
                    'transaction_date': transaction_date,
                    'category': main_category,
                    'payment_method': extracted_data.get('paymentMethod'),
                    'receipt_id': receipt_data.get('receiptId'),
                    'confidence_score': ocr_data.get('confidence', 0.8),
                    'is_verified': True,
                    'verification_method': 'auto',
                    'transaction_embedding': transaction_embedding
                }
                
                cursor.execute("""
                    INSERT INTO transactions (transaction_id, user_id, merchant_id, amount, 
                                            transaction_date, category, payment_method, receipt_id,
                                            confidence_score, is_verified, verification_method,
                                            transaction_embedding)
                    VALUES (%(transaction_id)s, %(user_id)s, %(merchant_id)s, %(amount)s,
                            %(transaction_date)s, %(category)s, %(payment_method)s, %(receipt_id)s,
                            %(confidence_score)s, %(is_verified)s, %(verification_method)s,
                            %(transaction_embedding)s)
                    ON CONFLICT (receipt_id) 
                    DO UPDATE SET
                        amount = EXCLUDED.amount,
                        category = EXCLUDED.category,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING transaction_id
                """, transaction_record)
                
                transaction_id = cursor.fetchone()[0]
                
                # Process transaction items
                for item in items:
                    item_embedding = self.generate_embedding(f"{item.get('name', '')} {item.get('category', '')}")
                    
                    item_record = {
                        'item_id': str(uuid.uuid4()),
                        'transaction_id': transaction_id,
                        'name': item.get('name', ''),
                        'quantity': item.get('quantity', 1),
                        'unit_price': item.get('price', 0),
                        'total_price': item.get('price', 0) * item.get('quantity', 1),
                        'category': item.get('category'),
                        'is_organic': item.get('isOrganic', False),
                        'item_embedding': item_embedding
                    }
                    
                    cursor.execute("""
                        INSERT INTO transaction_items (item_id, transaction_id, name, quantity,
                                                     unit_price, total_price, category, is_organic,
                                                     item_embedding)
                        VALUES (%(item_id)s, %(transaction_id)s, %(name)s, %(quantity)s,
                                %(unit_price)s, %(total_price)s, %(category)s, %(is_organic)s,
                                %(item_embedding)s)
                    """, item_record)
                
                # Process wallet pass if exists
                wallet_pass = receipt_data.get('walletPass', {})
                if wallet_pass.get('objectId') and wallet_pass.get('objectId') != 'string':
                    pass_record = {
                        'pass_id': str(uuid.uuid4()),
                        'user_id': user_id,
                        'transaction_id': transaction_id,
                        'object_id': wallet_pass.get('objectId'),
                        'class_id': 'receipt_class',
                        'wallet_link': wallet_pass.get('walletLink', ''),
                        'pass_status': 'active',
                        'pass_data': json.dumps(wallet_pass)
                    }
                    
                    cursor.execute("""
                        INSERT INTO wallet_passes (pass_id, user_id, transaction_id, object_id,
                                                 class_id, wallet_link, pass_status, pass_data)
                        VALUES (%(pass_id)s, %(user_id)s, %(transaction_id)s, %(object_id)s,
                                %(class_id)s, %(wallet_link)s, %(pass_status)s, %(pass_data)s)
                        ON CONFLICT (object_id) DO NOTHING
                    """, pass_record)
            
            conn.commit()
            logger.info("Successfully processed receipt queue")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error processing receipts: {e}")
            raise
        finally:
            cursor.close()

    def process_chat_sessions(self) -> None:
        """Process chat sessions and messages from Firestore to PostgreSQL."""
        logger.info("Processing chat sessions...")
        
        conn = self.get_postgres_connection()
        cursor = conn.cursor()
        
        try:
            sessions_ref = self.db_firestore.collection('chatSessions')
            sessions = sessions_ref.stream()
            
            for session_doc in sessions:
                session_data = session_doc.to_dict()
                
                # Get user_id
                cursor.execute("SELECT user_id FROM users WHERE firebase_uid = %s", 
                             (session_data.get('userId'),))
                user_result = cursor.fetchone()
                if not user_result:
                    continue
                    
                user_id = user_result[0]
                
                # Process messages
                messages_ref = session_doc.reference.collection('messages')
                messages = messages_ref.stream()
                
                for message_doc in messages:
                    message_data = message_doc.to_dict()
                    
                    # Only process user messages for query embeddings
                    if message_data.get('type') == 'user':
                        content = message_data.get('content', {})
                        query_text = content.get('text', '')
                        
                        query_embedding = self.generate_embedding(query_text)
                        
                        query_record = {
                            'query_id': str(uuid.uuid4()),
                            'user_id': user_id,
                            'session_id': session_data.get('sessionId'),
                            'original_query': query_text,
                            'intent_type': content.get('intent', 'ANALYTICAL'),
                            'entities': json.dumps(content.get('entities', {})),
                            'confidence_score': 0.9,  # Mock confidence
                            'query_embedding': query_embedding,
                            'conversation_context': json.dumps({}),
                            'user_context': json.dumps({}),
                            'response_provided': True,
                            'created_at': message_data.get('timestamp', datetime.now(timezone.utc))
                        }
                        
                        cursor.execute("""
                            INSERT INTO chat_queries (query_id, user_id, session_id, original_query,
                                                     intent_type, entities, confidence_score, query_embedding,
                                                     conversation_context, user_context, response_provided, created_at)
                            VALUES (%(query_id)s, %(user_id)s, %(session_id)s, %(original_query)s,
                                    %(intent_type)s, %(entities)s, %(confidence_score)s, %(query_embedding)s,
                                    %(conversation_context)s, %(user_context)s, %(response_provided)s, %(created_at)s)
                            ON CONFLICT DO NOTHING
                        """, query_record)
            
            conn.commit()
            logger.info("Successfully processed chat sessions")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error processing chat sessions: {e}")
            raise
        finally:
            cursor.close()

    def process_notifications(self) -> None:
        """Process notifications from Firestore and create insights in PostgreSQL."""
        logger.info("Processing notifications as insights...")
        
        conn = self.get_postgres_connection()
        cursor = conn.cursor()
        
        try:
            notifications_ref = self.db_firestore.collection('notifications')
            notifications = notifications_ref.stream()
            
            for notification_doc in notifications:
                notification_data = notification_doc.to_dict()
                
                # Get user_id
                cursor.execute("SELECT user_id FROM users WHERE firebase_uid = %s", 
                             (notification_data.get('userId'),))
                user_result = cursor.fetchone()
                if not user_result:
                    continue
                    
                user_id = user_result[0]
                
                # Convert notification to insight
                data_payload = notification_data.get('data', {})
                
                insight_record = {
                    'insight_id': str(uuid.uuid4()),
                    'user_id': user_id,
                    'insight_type': notification_data.get('type', 'spending_spike'),
                    'category': data_payload.get('category'),
                    'title': notification_data.get('title', ''),
                    'description': notification_data.get('message', ''),
                    'severity': notification_data.get('priority', 'medium'),
                    'confidence_score': 0.87,  # From mock data
                    'relevance_score': data_payload.get('relevanceScore', 0.8),
                    'data_points': json.dumps(data_payload),
                    'recommendations': json.dumps(data_payload.get('recommendations', [])),
                    'status': 'active' if notification_data.get('readAt') is None else 'dismissed',
                    'viewed_at': notification_data.get('readAt'),
                    'generated_by': 'proactive_insights_agent',
                    'generation_context': json.dumps(notification_data.get('metadata', {})),
                    'created_at': notification_data.get('scheduledFor', datetime.now(timezone.utc))
                }
                
                cursor.execute("""
                    INSERT INTO insights (insight_id, user_id, insight_type, category, title,
                                        description, severity, confidence_score, relevance_score,
                                        data_points, recommendations, status, viewed_at,
                                        generated_by, generation_context, created_at)
                    VALUES (%(insight_id)s, %(user_id)s, %(insight_type)s, %(category)s, %(title)s,
                            %(description)s, %(severity)s, %(confidence_score)s, %(relevance_score)s,
                            %(data_points)s, %(recommendations)s, %(status)s, %(viewed_at)s,
                            %(generated_by)s, %(generation_context)s, %(created_at)s)
                    ON CONFLICT DO NOTHING
                """, insight_record)
            
            conn.commit()
            logger.info("Successfully processed notifications as insights")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error processing notifications: {e}")
            raise
        finally:
            cursor.close()

    def determine_transaction_category(self, items: List[Dict]) -> str:
        """Determine transaction category based on items."""
        if not items:
            return 'other'
            
        # Count categories
        category_counts = {}
        for item in items:
            category = item.get('category', 'other')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Map item categories to transaction categories
        category_mapping = {
            'dairy': 'groceries',
            'bakery': 'groceries',
            'produce': 'groceries',
            'grains': 'groceries',
            'audio': 'electronics',
            'accessories': 'electronics',
            'beverages': 'dining',
            'food': 'dining'
        }
        
        # Find most common category and map it
        most_common = max(category_counts.keys(), key=lambda k: category_counts[k])
        return category_mapping.get(most_common, most_common)

    def run_full_etl(self) -> Dict[str, str]:
        """Run complete ETL process."""
        results = {}
        
        try:
            logger.info("Starting Firestore to PostgreSQL ETL process...")
            
            # Process in order to maintain referential integrity
            self.process_users()
            results['users'] = 'success'
            
            # self.process_receipts()
            # results['receipts'] = 'success'
            
            # self.process_chat_sessions()
            # results['chat_sessions'] = 'success'
            
            # self.process_notifications()
            # results['notifications'] = 'success'
            
            logger.info("ETL process completed successfully")
            
        except Exception as e:
            logger.error(f"ETL process failed: {e}")
            results['error'] = str(e)
            raise
        
        finally:
            if self.db_postgres:
                self.db_postgres.close()
        
        return results


def firestore_to_postgres_etl():
    
    try:
        # Initialize and run ETL
        etl = FirestorePostgresETL()
        results = etl.run_full_etl()
        
        return {
            'status': 'success',
            'message': 'ETL process completed',
            'results': results,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }, 200
        
    except Exception as e:
        logger.error(f"ETL function failed: {e}")
        return {
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }, 500


# For testing locally
if __name__ == "__main__":
    etl = FirestorePostgresETL()
    results = etl.run_full_etl()
    print(f"ETL Results: {results}")