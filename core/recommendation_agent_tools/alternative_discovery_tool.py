import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from collections import defaultdict, Counter
import statistics
import numpy as np

from utils.database_connector import DatabaseConnector


class AlternativeDiscoveryTool:
    """
    Tool for finding alternative merchants, products, and services based on user's 
    transaction history and spending patterns. Provides cost-effective alternatives
    and better options for financial optimization.
    """
    
    def __init__(
        self,
        project_id: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Alternative Finding Tool.
        
        Args:
            project_id: Google Cloud project ID
            logger: Logger instance
        """
        self.project_id = project_id
        self.logger = logger or logging.getLogger(__name__)
        self._db_manager: Optional[DatabaseConnector] = None
        
        # Alternative finding settings
        self.similarity_threshold = 0.3  # Lower threshold for broader alternatives
        self.price_improvement_threshold = 0.1  # 10% minimum savings to recommend
        self.min_transaction_count = 2  # Minimum transactions to consider patterns
        self.location_radius_km = 10  # Distance radius for nearby alternatives
        
        self.logger.info("Alternative Finding Tool initialized")

    async def _get_db_manager(self) -> DatabaseConnector:
        """Get the database manager instance."""
        if self._db_manager is None:
            self._db_manager = await DatabaseConnector.get_instance(self.project_id)
        return self._db_manager

    async def find_merchant_alternatives(
        self,
        transaction_data: List[Dict[str, Any]],
        user_id: str,
        criteria: str = "cost_savings",
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Find alternative merchants based on user's transaction history.
        
        Args:
            transaction_data: List of user transactions from recent analysis
            user_id: User identifier
            criteria: Criteria for alternatives ("cost_savings", "value", "convenience", "quality")
            limit: Number of alternatives per merchant
            
        Returns:
            Dictionary containing merchant alternatives and analysis
        """
        try:
            start_time = datetime.now()
            
            # Extract merchant patterns from transaction data
            merchant_patterns = await self._analyze_merchant_patterns(transaction_data)
            
            # Find alternatives for each frequent merchant
            alternatives = {}
            
            for merchant_name, pattern_data in merchant_patterns.items():
                if pattern_data['transaction_count'] >= self.min_transaction_count:
                    merchant_alternatives = await self._find_alternatives_for_merchant(
                        merchant_name=merchant_name,
                        pattern_data=pattern_data,
                        user_id=user_id,
                        criteria=criteria,
                        limit=limit
                    )
                    
                    if merchant_alternatives:
                        alternatives[merchant_name] = merchant_alternatives
            
            # Generate recommendations and insights
            recommendations = await self._generate_alternative_recommendations(
                alternatives, 
                merchant_patterns,
                criteria
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "user_id": user_id,
                "criteria": criteria,
                "merchant_patterns": merchant_patterns,
                "alternatives": alternatives,
                "recommendations": recommendations,
                "total_potential_savings": self._calculate_total_savings(alternatives),
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to find merchant alternatives: {e}")
            return {
                "success": False,
                "error": str(e),
                "user_id": user_id
            }

    async def find_category_alternatives(
        self,
        transaction_data: List[Dict[str, Any]],  
        user_id: str,
        target_category: str,
        optimization_type: str = "cost_reduction"
    ) -> Dict[str, Any]:
        """
        Find alternatives within a specific spending category.
        
        Args:
            transaction_data: List of user transactions
            user_id: User identifier
            target_category: Category to find alternatives for (e.g., 'food', 'transportation')
            optimization_type: Type of optimization ("cost_reduction", "quality_improvement", "convenience")
            
        Returns:
            Dictionary containing category-specific alternatives
        """
        try:
            # Filter transactions for target category
            category_transactions = [
                t for t in transaction_data 
                if t.get('category', '').lower() == target_category.lower()
            ]
            
            if not category_transactions:
                return {
                    "success": True,
                    "message": f"No transactions found for category: {target_category}",
                    "alternatives": []
                }
            
            # Analyze spending patterns in this category
            category_analysis = await self._analyze_category_spending(
                category_transactions, 
                target_category,
                user_id
            )
            
            # Find alternative merchants in the same category
            category_alternatives = await self._find_category_specific_alternatives(
                category_analysis,
                target_category,
                optimization_type,
                user_id
            )
            
            # Generate category-specific recommendations
            category_recommendations = await self._generate_category_recommendations(
                category_alternatives,
                category_analysis,
                optimization_type
            )
            
            return {
                "success": True,
                "user_id": user_id,
                "target_category": target_category,
                "optimization_type": optimization_type,
                "category_analysis": category_analysis,
                "alternatives": category_alternatives,
                "recommendations": category_recommendations,
            }
            
        except Exception as e:
            self.logger.error(f"Failed to find category alternatives: {e}")
            return {
                "success": False,
                "error": str(e),
                "user_id": user_id,
                "target_category": target_category
            }

    async def find_product_alternatives(
        self,
        transaction_data: List[Dict[str, Any]],
        user_id: str,
        focus_items: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Find alternative products based on transaction items and user preferences.
        
        Args:
            transaction_data: List of user transactions
            user_id: User identifier
            focus_items: Specific items to find alternatives for (optional)
            
        Returns:
            Dictionary containing product alternatives
        """
        try:
            # Get detailed item-level data from transactions
            item_analysis = await self._analyze_item_spending(
                transaction_data,
                user_id,
                focus_items
            )
            
            # Find alternative products using embeddings
            product_alternatives = await self._find_similar_products(
                item_analysis,
                user_id
            )
            
            # Calculate potential savings from product switches
            savings_analysis = await self._calculate_product_savings(
                product_alternatives,
                item_analysis
            )
            
            return {
                "success": True,
                "user_id": user_id,
                "item_analysis": item_analysis,
                "product_alternatives": product_alternatives,
                "savings_analysis": savings_analysis,
                "focus_items": focus_items
            }
            
        except Exception as e:
            self.logger.error(f"Failed to find product alternatives: {e}")
            return {
                "success": False,
                "error": str(e),
                "user_id": user_id
            }

    # Private helper methods

    async def _analyze_merchant_patterns(
        self, 
        transaction_data: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze merchant spending patterns from transaction data."""
        merchant_patterns = defaultdict(lambda: {
            'transaction_count': 0,
            'total_spent': 0.0,
            'avg_transaction': 0.0,
            'transactions': [],
            'categories': set(),
            'payment_methods': set(),
            'date_range': {'start': None, 'end': None}
        })
        
        for transaction in transaction_data:
            merchant_name = transaction.get('merchant_name', 'Unknown')
            amount = float(transaction.get('amount', 0))
            
            pattern = merchant_patterns[merchant_name]
            pattern['transaction_count'] += 1
            pattern['total_spent'] += amount
            pattern['transactions'].append(transaction)
            pattern['categories'].add(transaction.get('category', 'other'))
            pattern['payment_methods'].add(transaction.get('payment_method', 'unknown'))
            
            # Update date range
            trans_date = transaction.get('transaction_date')
            if trans_date:
                if isinstance(trans_date, str):
                    trans_date = datetime.fromisoformat(trans_date).date()
                elif isinstance(trans_date, datetime):
                    trans_date = trans_date.date()
                
                if pattern['date_range']['start'] is None or trans_date < pattern['date_range']['start']:
                    pattern['date_range']['start'] = trans_date
                if pattern['date_range']['end'] is None or trans_date > pattern['date_range']['end']:
                    pattern['date_range']['end'] = trans_date
        
        # Calculate averages and convert sets to lists for JSON serialization
        for merchant_name, pattern in merchant_patterns.items():
            if pattern['transaction_count'] > 0:
                pattern['avg_transaction'] = pattern['total_spent'] / pattern['transaction_count']
            pattern['categories'] = list(pattern['categories'])
            pattern['payment_methods'] = list(pattern['payment_methods'])
            
            # Convert dates to strings for JSON serialization
            if pattern['date_range']['start']:
                pattern['date_range']['start'] = pattern['date_range']['start'].isoformat()
            if pattern['date_range']['end']:
                pattern['date_range']['end'] = pattern['date_range']['end'].isoformat()
        
        return dict(merchant_patterns)

    async def _find_alternatives_for_merchant(
        self,
        merchant_name: str,
        pattern_data: Dict[str, Any],
        user_id: str,
        criteria: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Find alternative merchants for a specific merchant."""
        try:
            db_manager = await self._get_db_manager()
            
            # Get the primary category for this merchant from user's transactions
            primary_category = pattern_data['categories'][0] if pattern_data['categories'] else None
            
            # Build query based on whether we have a category filter
            if primary_category:
                query = """
                WITH user_merchant AS (
                    SELECT m.merchant_embedding, m.category, m.subcategory
                    FROM merchants m
                    WHERE m.normalized_name = $1 OR m.name = $2
                    LIMIT 1
                ),
                similar_merchants AS (
                    SELECT 
                        m.merchant_id,
                        m.name,
                        m.normalized_name,
                        m.category,
                        m.subcategory,
                        m.address,
                        m.avg_transaction_amount,
                        m.total_transactions,
                        m.price_range,
                        1 - (m.merchant_embedding <=> um.merchant_embedding) as similarity_score
                    FROM merchants m, user_merchant um
                    WHERE m.normalized_name != $3 
                        AND m.name != $4
                        AND (m.merchant_embedding <=> um.merchant_embedding) < $5
                        AND m.category = $6
                    ORDER BY m.merchant_embedding <=> um.merchant_embedding
                    LIMIT $7
                ),
                user_spending_at_alternatives AS (
                    SELECT 
                        sm.merchant_id,
                        sm.name,
                        sm.normalized_name,
                        sm.category,
                        sm.subcategory,
                        sm.address,
                        sm.avg_transaction_amount,
                        sm.total_transactions,
                        sm.price_range,
                        sm.similarity_score,
                        COALESCE(AVG(t.amount), 0) as user_avg_spent,
                        COUNT(t.transaction_id) as user_transaction_count,
                        MAX(t.transaction_date) as last_transaction_date
                    FROM similar_merchants sm
                    LEFT JOIN transactions t ON sm.merchant_id = t.merchant_id 
                        AND t.user_id = $8 
                        AND t.deleted_at IS NULL
                    GROUP BY sm.merchant_id, sm.name, sm.normalized_name, sm.category, 
                             sm.subcategory, sm.address, sm.avg_transaction_amount, 
                             sm.total_transactions, sm.price_range, sm.similarity_score
                )
                SELECT * FROM user_spending_at_alternatives
                ORDER BY 
                    CASE WHEN $9 = 'cost_savings' THEN avg_transaction_amount END ASC,
                    CASE WHEN $10 = 'quality' THEN total_transactions END DESC,
                    CASE WHEN $11 = 'convenience' THEN user_transaction_count END DESC,
                    similarity_score DESC
                """
                
                params = [
                    merchant_name, merchant_name,  # user_merchant lookup
                    merchant_name, merchant_name,  # exclusion filters
                    1 - self.similarity_threshold,  # similarity threshold
                    primary_category,  # category filter
                    limit * 2,  # get more results for filtering
                    user_id,  # user spending analysis
                    criteria, criteria, criteria  # ordering criteria
                ]
            else:
                # Query without category filter
                query = """
                WITH user_merchant AS (
                    SELECT m.merchant_embedding, m.category, m.subcategory
                    FROM merchants m
                    WHERE m.normalized_name = $1 OR m.name = $2
                    LIMIT 1
                ),
                similar_merchants AS (
                    SELECT 
                        m.merchant_id,
                        m.name,
                        m.normalized_name,
                        m.category,
                        m.subcategory,
                        m.address,
                        m.avg_transaction_amount,
                        m.total_transactions,
                        m.price_range,
                        1 - (m.merchant_embedding <=> um.merchant_embedding) as similarity_score
                    FROM merchants m, user_merchant um
                    WHERE m.normalized_name != $3 
                        AND m.name != $4
                        AND (m.merchant_embedding <=> um.merchant_embedding) < $5
                    ORDER BY m.merchant_embedding <=> um.merchant_embedding
                    LIMIT $6
                ),
                user_spending_at_alternatives AS (
                    SELECT 
                        sm.merchant_id,
                        sm.name,
                        sm.normalized_name,
                        sm.category,
                        sm.subcategory,
                        sm.address,
                        sm.avg_transaction_amount,
                        sm.total_transactions,
                        sm.price_range,
                        sm.similarity_score,
                        COALESCE(AVG(t.amount), 0) as user_avg_spent,
                        COUNT(t.transaction_id) as user_transaction_count,
                        MAX(t.transaction_date) as last_transaction_date
                    FROM similar_merchants sm
                    LEFT JOIN transactions t ON sm.merchant_id = t.merchant_id 
                        AND t.user_id = $7 
                        AND t.deleted_at IS NULL
                    GROUP BY sm.merchant_id, sm.name, sm.normalized_name, sm.category, 
                             sm.subcategory, sm.address, sm.avg_transaction_amount, 
                             sm.total_transactions, sm.price_range, sm.similarity_score
                )
                SELECT * FROM user_spending_at_alternatives
                ORDER BY 
                    CASE WHEN $8 = 'cost_savings' THEN avg_transaction_amount END ASC,
                    CASE WHEN $9 = 'quality' THEN total_transactions END DESC,
                    CASE WHEN $10 = 'convenience' THEN user_transaction_count END DESC,
                    similarity_score DESC
                """
                
                params = [
                    merchant_name, merchant_name,  # user_merchant lookup
                    merchant_name, merchant_name,  # exclusion filters
                    1 - self.similarity_threshold,  # similarity threshold
                    limit * 2,  # get more results for filtering
                    user_id,  # user spending analysis
                    criteria, criteria, criteria  # ordering criteria
                ]
            
            async with db_manager.get_connection() as conn:
                results = await conn.fetch(query, *params)
            
            # Process and rank alternatives
            alternatives = []
            user_avg_amount = pattern_data['avg_transaction']
            
            for row in results:
                alternative = dict(row)
                
                # Calculate potential savings/benefits
                merchant_avg = float(alternative['avg_transaction_amount']) if alternative['avg_transaction_amount'] else user_avg_amount
                user_spent_avg = float(alternative['user_avg_spent']) if alternative['user_avg_spent'] else merchant_avg
                
                # Use user's actual spending at this merchant if available, otherwise use merchant average
                comparison_amount = user_spent_avg if alternative['user_transaction_count'] > 0 else merchant_avg
                
                savings_potential = user_avg_amount - comparison_amount
                savings_percentage = (savings_potential / user_avg_amount * 100) if user_avg_amount > 0 else 0
                
                # Parse address if it exists
                address_data = None
                if alternative['address']:
                    try:
                        address_data = json.loads(alternative['address'])
                    except (json.JSONDecodeError, TypeError):
                        address_data = {"raw": str(alternative['address'])}
                
                # Calculate recommendation score based on criteria
                recommendation_score = self._calculate_recommendation_score(
                    alternative, 
                    pattern_data, 
                    criteria,
                    savings_percentage
                )
                
                alternative_info = {
                    'merchant_id': alternative['merchant_id'],
                    'name': alternative['name'],
                    'normalized_name': alternative['normalized_name'],
                    'category': alternative['category'],
                    'subcategory': alternative['subcategory'],
                    'address': address_data,
                    'similarity_score': float(alternative['similarity_score']),
                    'avg_transaction_amount': float(merchant_avg),
                    'total_transactions': alternative['total_transactions'],
                    'price_range': alternative['price_range'],
                    'user_experience': {
                        'has_visited': alternative['user_transaction_count'] > 0,
                        'visit_count': alternative['user_transaction_count'],
                        'avg_spent': float(user_spent_avg),
                        'last_visit': alternative['last_transaction_date'].isoformat() if alternative['last_transaction_date'] else None
                    },
                    'financial_impact': {
                        'potential_savings': round(savings_potential, 2),
                        'savings_percentage': round(savings_percentage, 1),
                        'comparison_base': 'user_average' if alternative['user_transaction_count'] > 0 else 'merchant_average'
                    },
                    'recommendation_score': recommendation_score,
                    'recommendation_reason': self._generate_recommendation_reason(
                        alternative, criteria, savings_percentage
                    )
                }
                
                alternatives.append(alternative_info)
            
            # Sort by recommendation score and return top results
            alternatives.sort(key=lambda x: x['recommendation_score'], reverse=True)
            return alternatives[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to find alternatives for merchant {merchant_name}: {e}")
            return []

    async def _analyze_category_spending(
        self,
        category_transactions: List[Dict[str, Any]],
        target_category: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Analyze spending patterns within a specific category."""
        if not category_transactions:
            return {}
        
        # Calculate basic statistics
        amounts = [float(t.get('amount', 0)) for t in category_transactions]
        merchants = [t.get('merchant_name', 'Unknown') for t in category_transactions]
        
        # Merchant frequency analysis
        merchant_counts = Counter(merchants)
        merchant_spending = defaultdict(float)
        for t in category_transactions:
            merchant_spending[t.get('merchant_name', 'Unknown')] += float(t.get('amount', 0))
        
        # Calculate date patterns
        dates = []
        for t in category_transactions:
            trans_date = t.get('transaction_date')
            if trans_date:
                if isinstance(trans_date, str):
                    dates.append(datetime.fromisoformat(trans_date).date())
                elif isinstance(trans_date, datetime):
                    dates.append(trans_date.date())
        
        analysis = {
            'category': target_category,
            'total_transactions': len(category_transactions),
            'total_spent': sum(amounts),
            'avg_transaction': sum(amounts) / len(amounts) if amounts else 0,
            'min_transaction': min(amounts) if amounts else 0,
            'max_transaction': max(amounts) if amounts else 0,
            'median_transaction': statistics.median(amounts) if amounts else 0,
            'std_deviation': statistics.stdev(amounts) if len(amounts) > 1 else 0,
            'unique_merchants': len(merchant_counts),
            'top_merchants': dict(merchant_counts.most_common(5)),
            'merchant_spending': dict(merchant_spending),
            'date_range': {
                'start': min(dates).isoformat() if dates else None,
                'end': max(dates).isoformat() if dates else None,
                'days_span': (max(dates) - min(dates)).days if len(dates) > 1 else 0
            },
            'frequency_analysis': {
                'transactions_per_week': len(category_transactions) / max(1, ((max(dates) - min(dates)).days / 7)) if len(dates) > 1 else 0
            }
        }
        
        return analysis

    async def _find_category_specific_alternatives(
        self,
        category_analysis: Dict[str, Any],
        target_category: str,
        optimization_type: str,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Find alternative merchants within the same category."""
        try:
            db_manager = await self._get_db_manager()
            
            # Get merchants in the same category that user hasn't used much
            query = """
            WITH user_merchants AS (
                SELECT DISTINCT t.merchant_id
                FROM transactions t
                WHERE t.user_id = $1 
                    AND t.category = $2 
                    AND t.deleted_at IS NULL
            ),
            category_merchants AS (
                SELECT 
                    m.merchant_id,
                    m.name,
                    m.normalized_name,
                    m.category,
                    m.subcategory,
                    m.address,
                    m.avg_transaction_amount,
                    m.total_transactions,
                    m.price_range,
                    COALESCE(user_stats.transaction_count, 0) as user_transaction_count,
                    COALESCE(user_stats.avg_amount, 0) as user_avg_amount,
                    COALESCE(user_stats.total_spent, 0) as user_total_spent
                FROM merchants m
                LEFT JOIN (
                    SELECT 
                        t.merchant_id,
                        COUNT(*) as transaction_count,
                        AVG(t.amount) as avg_amount,
                        SUM(t.amount) as total_spent
                    FROM transactions t
                    WHERE t.user_id = $3 AND t.deleted_at IS NULL
                    GROUP BY t.merchant_id
                ) user_stats ON m.merchant_id = user_stats.merchant_id
                WHERE m.category = $4 
                    AND m.total_transactions > 5  -- Only consider established merchants
            )
            SELECT * FROM category_merchants
            ORDER BY 
                CASE WHEN $5 = 'cost_reduction' THEN avg_transaction_amount END ASC,
                CASE WHEN $6 = 'quality_improvement' THEN total_transactions END DESC,
                CASE WHEN $7 = 'convenience' THEN user_transaction_count END DESC,
                total_transactions DESC
            LIMIT 20
            """
            
            async with db_manager.get_connection() as conn:
                results = await conn.fetch(
                    query,
                    user_id, target_category,  # user_merchants CTE
                    user_id,  # user_stats in main query
                    target_category,  # category filter
                    optimization_type, optimization_type, optimization_type  # ordering
                )
            
            alternatives = []
            user_category_avg = category_analysis.get('avg_transaction', 0)
            
            for row in results:
                alternative = dict(row)
                
                # Skip if this is already a top merchant for the user
                if alternative['name'] in category_analysis.get('top_merchants', {}):
                    continue
                
                # Calculate potential impact
                merchant_avg = float(alternative['avg_transaction_amount']) if alternative['avg_transaction_amount'] else user_category_avg
                savings_potential = user_category_avg - merchant_avg
                savings_percentage = (savings_potential / user_category_avg * 100) if user_category_avg > 0 else 0
                
                # Parse address
                address_data = None
                if alternative['address']:
                    try:
                        address_data = json.loads(alternative['address'])
                    except (json.JSONDecodeError, TypeError):
                        address_data = {"raw": str(alternative['address'])}
                
                alternative_info = {
                    'merchant_id': alternative['merchant_id'],
                    'name': alternative['name'],
                    'normalized_name': alternative['normalized_name'],
                    'category': alternative['category'],
                    'subcategory': alternative['subcategory'],
                    'address': address_data,
                    'avg_transaction_amount': float(merchant_avg),
                    'total_transactions': alternative['total_transactions'],
                    'price_range': alternative['price_range'],
                    'user_experience': {
                        'has_visited': alternative['user_transaction_count'] > 0,
                        'visit_count': alternative['user_transaction_count'],
                        'avg_spent': float(alternative['user_avg_amount']),
                        'total_spent': float(alternative['user_total_spent'])
                    },
                    'financial_impact': {
                        'potential_savings': round(savings_potential, 2),
                        'savings_percentage': round(savings_percentage, 1),
                        'optimization_type': optimization_type
                    },
                    'recommendation_reason': self._generate_category_recommendation_reason(
                        alternative, optimization_type, savings_percentage
                    )
                }
                
                alternatives.append(alternative_info)
            
            return alternatives
            
        except Exception as e:
            self.logger.error(f"Failed to find category alternatives: {e}")
            return []

    async def _analyze_item_spending(
        self,
        transaction_data: List[Dict[str, Any]],
        user_id: str,
        focus_items: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze item-level spending patterns."""
        try:
            db_manager = await self._get_db_manager()
            
            # Get transaction IDs from the provided data
            transaction_ids = [t.get('transaction_id') for t in transaction_data if t.get('transaction_id')]
            
            if not transaction_ids:
                return {"message": "No transaction IDs found in data"}
            
            # Query for item-level data using ANY() for array parameter
            query = """
            SELECT 
                ti.item_id,
                ti.transaction_id,
                ti.name,
                ti.quantity,
                ti.unit_price,
                ti.total_price,
                ti.category,
                ti.brand,
                ti.is_organic,
                t.merchant_name,
                t.transaction_date
            FROM transaction_items ti
            JOIN transactions t ON ti.transaction_id = t.transaction_id
            WHERE ti.transaction_id = ANY($1)
            ORDER BY ti.total_price DESC
            """
            
            async with db_manager.get_connection() as conn:
                results = await conn.fetch(query, transaction_ids)
            
            # Analyze item patterns
            item_analysis = defaultdict(lambda: {
                'total_spent': 0.0,
                'total_quantity': 0,
                'purchase_count': 0,
                'avg_unit_price': 0.0,
                'merchants': set(),
                'categories': set(),
                'is_organic_purchases': 0,
                'transactions': []
            })
            
            for row in results:
                item_name = row['name']
                if focus_items and item_name not in focus_items:
                    continue
                    
                analysis = item_analysis[item_name]
                analysis['total_spent'] += float(row['total_price'])
                analysis['total_quantity'] += row['quantity']
                analysis['purchase_count'] += 1
                analysis['merchants'].add(row['merchant_name'])
                analysis['categories'].add(row['category'])
                if row['is_organic']:
                    analysis['is_organic_purchases'] += 1
                analysis['transactions'].append(dict(row))
            
            # Calculate averages and convert sets to lists
            processed_analysis = {}
            for item_name, analysis in item_analysis.items():
                if analysis['purchase_count'] > 0:
                    analysis['avg_unit_price'] = analysis['total_spent'] / analysis['total_quantity'] if analysis['total_quantity'] > 0 else 0
                    analysis['avg_purchase_amount'] = analysis['total_spent'] / analysis['purchase_count']
                    analysis['organic_percentage'] = (analysis['is_organic_purchases'] / analysis['purchase_count']) * 100
                
                # Convert sets to lists for JSON serialization
                analysis['merchants'] = list(analysis['merchants'])
                analysis['categories'] = list(analysis['categories'])
                
                processed_analysis[item_name] = analysis
            
            return {
                'total_items_analyzed': len(processed_analysis),
                'focus_items': focus_items,
                'item_analysis': processed_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze item spending: {e}")
            return {"error": str(e)}

    async def _find_similar_products(
        self,
        item_analysis: Dict[str, Any],
        user_id: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Find similar products using item embeddings."""
        try:
            db_manager = await self._get_db_manager()
            
            product_alternatives = {}
            
            for item_name, analysis in item_analysis.get('item_analysis', {}).items():
                # Find similar items using embeddings
                query = """
                WITH target_item AS (
                    SELECT item_embedding
                    FROM transaction_items
                    WHERE name = $1
                    LIMIT 1
                ),
                similar_items AS (
                    SELECT DISTINCT
                        ti.name,
                        ti.category,
                        ti.brand,
                        AVG(ti.unit_price) as avg_unit_price,
                        COUNT(*) as purchase_frequency,
                        1 - (ti.item_embedding <=> ti_target.item_embedding) as similarity_score
                    FROM transaction_items ti, target_item ti_target
                    WHERE ti.name != $2
                        AND ti.item_embedding IS NOT NULL
                        AND (ti.item_embedding <=> ti_target.item_embedding) < 0.7
                    GROUP BY ti.name, ti.category, ti.brand, ti.item_embedding, ti_target.item_embedding
                    HAVING COUNT(*) >= 2
                    ORDER BY similarity_score DESC
                    LIMIT 10
                )
                SELECT * FROM similar_items
                """
                
                async with db_manager.get_connection() as conn:
                    results = await conn.fetch(query, item_name, item_name)
                
                alternatives = []
                user_avg_price = analysis.get('avg_unit_price', 0)
                
                for row in results:
                    alternative = dict(row)
                    alt_price = float(alternative['avg_unit_price'])
                    
                    savings_potential = user_avg_price - alt_price
                    savings_percentage = (savings_potential / user_avg_price * 100) if user_avg_price > 0 else 0
                    
                    alternative_info = {
                        'name': alternative['name'],
                        'category': alternative['category'],
                        'brand': alternative['brand'],
                        'avg_unit_price': alt_price,
                        'purchase_frequency': alternative['purchase_frequency'],
                        'similarity_score': float(alternative['similarity_score']),
                        'financial_impact': {
                            'potential_savings_per_unit': round(savings_potential, 2),
                            'savings_percentage': round(savings_percentage, 1)
                        }
                    }
                    
                    alternatives.append(alternative_info)
                
                if alternatives:
                    product_alternatives[item_name] = alternatives
            
            return product_alternatives
            
        except Exception as e:
            self.logger.error(f"Failed to find similar products: {e}")
            return {}

    def _calculate_recommendation_score(
        self,
        alternative: Dict[str, Any],
        pattern_data: Dict[str, Any],
        criteria: str,
        savings_percentage: float
    ) -> float:
        """Calculate recommendation score based on criteria."""
        base_score = float(alternative.get('similarity_score', 0)) * 0.3
        
        if criteria == 'cost_savings':
            # Higher score for better savings
            savings_score = min(savings_percentage / 20, 1) * 0.4  # Normalize to 20% max savings
            popularity_score = min(alternative.get('total_transactions', 0) / 1000, 1) * 0.3
            return base_score + savings_score + popularity_score
            
        elif criteria == 'quality':
            # Higher score for more popular merchants
            popularity_score = min(alternative.get('total_transactions', 0) / 1000, 1) * 0.5
            price_reasonableness = 1 - min(abs(savings_percentage) / 50, 1) * 0.2  # Prefer reasonable prices
            return base_score + popularity_score + price_reasonableness
            
        elif criteria == 'convenience':
            # Higher score for merchants user has visited
            user_familiarity = 1.0 if alternative.get('user_transaction_count', 0) > 0 else 0.0
            familiarity_score = user_familiarity * 0.4
            popularity_score = min(alternative.get('total_transactions', 0) / 500, 1) * 0.3
            return base_score + familiarity_score + popularity_score
            
        else:  # default/value
            # Balanced approach
            savings_score = min(max(savings_percentage, 0) / 15, 1) * 0.25
            popularity_score = min(alternative.get('total_transactions', 0) / 750, 1) * 0.25
            familiarity_bonus = 0.2 if alternative.get('user_transaction_count', 0) > 0 else 0
            return base_score + savings_score + popularity_score + familiarity_bonus

    def _generate_recommendation_reason(
        self,
        alternative: Dict[str, Any],
        criteria: str,
        savings_percentage: float
    ) -> str:
        """Generate explanation for why this alternative is recommended."""
        reasons = []
        
        if criteria == 'cost_savings' and savings_percentage > 5:
            reasons.append(f"Could save {abs(savings_percentage):.1f}% per transaction")
        elif criteria == 'cost_savings' and savings_percentage < -5:
            reasons.append(f"Higher quality option (${abs(savings_percentage):.1f}% premium)")
            
        if alternative.get('total_transactions', 0) > 100:
            reasons.append("Popular choice with many customers")
            
        if alternative.get('user_transaction_count', 0) > 0:
            reasons.append("You've shopped here before")
            
        similarity_score = alternative.get('similarity_score', 0)
        if similarity_score > 0.8:
            reasons.append("Very similar to your current choice")
        elif similarity_score > 0.6:
            reasons.append("Similar merchant category and type")
            
        return " • ".join(reasons) if reasons else "Alternative option in same category"

    def _generate_category_recommendation_reason(
        self,
        alternative: Dict[str, Any],
        optimization_type: str,
        savings_percentage: float
    ) -> str:
        """Generate recommendation reason for category alternatives."""
        reasons = []
        
        if optimization_type == 'cost_reduction' and savings_percentage > 0:
            reasons.append(f"Average savings of {savings_percentage:.1f}% per transaction")
            
        if alternative.get('total_transactions', 0) > 50:
            reasons.append("Well-established merchant")
            
        if alternative.get('user_transaction_count', 0) == 0:
            reasons.append("New option to explore")
        else:
            reasons.append("Previous positive experience")
            
        return " • ".join(reasons) if reasons else "Alternative in your category"

    async def _generate_alternative_recommendations(
        self,
        alternatives: Dict[str, List[Dict[str, Any]]],
        merchant_patterns: Dict[str, Dict[str, Any]],
        criteria: str
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on alternatives found."""
        recommendations = []
        
        # Overall savings potential
        total_potential_savings = 0
        high_impact_switches = []
        
        for merchant_name, merchant_alternatives in alternatives.items():
            if not merchant_alternatives:
                continue
                
            pattern_data = merchant_patterns[merchant_name]
            monthly_spending = pattern_data['total_spent']
            
            # Find best alternative
            best_alternative = max(merchant_alternatives, key=lambda x: x['recommendation_score'])
            potential_monthly_savings = (best_alternative['financial_impact']['potential_savings'] * 
                                       pattern_data['transaction_count'])
            
            total_potential_savings += potential_monthly_savings
            
            if potential_monthly_savings > 50:  # High impact threshold
                high_impact_switches.append({
                    'current_merchant': merchant_name,
                    'recommended_alternative': best_alternative['name'],
                    'monthly_savings': round(potential_monthly_savings, 2),
                    'savings_percentage': best_alternative['financial_impact']['savings_percentage'],
                    'confidence': best_alternative['recommendation_score']
                })
        
        # Generate recommendations
        if total_potential_savings > 20:
            recommendations.append({
                'type': 'high_savings_potential',
                'title': 'Significant Savings Opportunity',
                'description': f'You could save approximately ₹{total_potential_savings:.0f} monthly by switching merchants',
                'priority': 'high',
                'action_items': [
                    f"Try {switch['recommended_alternative']} instead of {switch['current_merchant']}"
                    for switch in high_impact_switches[:3]
                ]
            })
        
        if len(high_impact_switches) > 2:
            recommendations.append({
                'type': 'diversification',
                'title': 'Diversify Your Merchant Portfolio',
                'description': 'Consider trying different merchants to optimize your spending',
                'priority': 'medium',
                'action_items': [
                    'Start with one new merchant per category',
                    'Compare quality and service along with price',
                    'Track your experience for future decisions'
                ]
            })
        
        # Category-specific recommendations
        category_analysis = self._analyze_categories_for_recommendations(alternatives, merchant_patterns)
        recommendations.extend(category_analysis)
        
        return recommendations

    async def _generate_category_recommendations(
        self,
        category_alternatives: List[Dict[str, Any]],
        category_analysis: Dict[str, Any],
        optimization_type: str
    ) -> List[Dict[str, Any]]:
        """Generate category-specific recommendations."""
        recommendations = []
        
        if not category_alternatives:
            return recommendations
        
        category = category_analysis.get('category', 'Unknown')
        total_spent = category_analysis.get('total_spent', 0)
        
        # Find top alternatives by different criteria
        cost_effective = sorted(category_alternatives, 
                              key=lambda x: x['financial_impact']['savings_percentage'], 
                              reverse=True)[:3]
        
        if cost_effective and cost_effective[0]['financial_impact']['savings_percentage'] > 10:
            potential_savings = (cost_effective[0]['financial_impact']['savings_percentage'] / 100) * total_spent
            recommendations.append({
                'type': 'cost_optimization',
                'title': f'Optimize {category.title()} Spending',
                'description': f'Could save ₹{potential_savings:.0f} by switching to more cost-effective options',
                'priority': 'high',
                'top_alternatives': [alt['name'] for alt in cost_effective],
                'action': f"Try {cost_effective[0]['name']} for your next {category} purchase"
            })
        
        # Quality improvement suggestions
        quality_options = [alt for alt in category_alternatives 
                         if alt.get('total_transactions', 0) > 100]
        if quality_options and optimization_type in ['quality_improvement', 'value']:
            recommendations.append({
                'type': 'quality_improvement',
                'title': f'Quality Options in {category.title()}',
                'description': 'Well-established merchants with good customer base',
                'priority': 'medium',
                'alternatives': [
                    {
                        'name': alt['name'],
                        'customer_base': alt.get('total_transactions', 0),
                        'your_experience': 'Previous customer' if alt['user_experience']['has_visited'] else 'New option'
                    }
                    for alt in quality_options[:3]
                ]
            })
        
        return recommendations

    async def _calculate_product_savings(
        self,
        product_alternatives: Dict[str, List[Dict[str, Any]]],
        item_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate potential savings from product alternatives."""
        total_potential_savings = 0
        item_savings_breakdown = {}
        
        for item_name, alternatives in product_alternatives.items():
            if not alternatives:
                continue
                
            item_data = item_analysis.get('item_analysis', {}).get(item_name, {})
            current_spending = item_data.get('total_spent', 0)
            purchase_frequency = item_data.get('purchase_count', 0)
            
            if purchase_frequency == 0:
                continue
                
            # Find best savings alternative
            best_savings_alt = max(alternatives, 
                                 key=lambda x: x['financial_impact']['savings_percentage'])
            
            if best_savings_alt['financial_impact']['savings_percentage'] > 0:
                annual_savings = (best_savings_alt['financial_impact']['potential_savings_per_unit'] * 
                                purchase_frequency * 12)  # Annualized
                
                total_potential_savings += annual_savings
                item_savings_breakdown[item_name] = {
                    'current_annual_spending': current_spending * 12,
                    'potential_annual_savings': annual_savings,
                    'best_alternative': best_savings_alt['name'],
                    'savings_percentage': best_savings_alt['financial_impact']['savings_percentage']
                }
        
        return {
            'total_annual_savings_potential': round(total_potential_savings, 2),
            'item_breakdown': item_savings_breakdown,
            'high_impact_items': [
                item for item, data in item_savings_breakdown.items()
                if data['potential_annual_savings'] > 100
            ]
        }

    def _analyze_categories_for_recommendations(
        self,
        alternatives: Dict[str, List[Dict[str, Any]]],
        merchant_patterns: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze alternatives by category for targeted recommendations."""
        category_groups = defaultdict(list)
        
        # Group alternatives by category
        for merchant_name, merchant_alternatives in alternatives.items():
            pattern = merchant_patterns[merchant_name]
            primary_category = pattern['categories'][0] if pattern['categories'] else 'other'
            
            for alt in merchant_alternatives:
                category_groups[primary_category].append({
                    'current_merchant': merchant_name,
                    'alternative': alt,
                    'monthly_spending': pattern['total_spent']
                })
        
        recommendations = []
        
        # Generate category-specific insights
        for category, category_alts in category_groups.items():
            if len(category_alts) < 2:
                continue
                
            total_category_spending = sum(alt['monthly_spending'] for alt in category_alts)
            avg_savings_potential = sum(alt['alternative']['financial_impact']['savings_percentage'] 
                                      for alt in category_alts) / len(category_alts)
            
            if avg_savings_potential > 8:
                recommendations.append({
                    'type': 'category_optimization',
                    'title': f'Optimize {category.title()} Category',
                    'description': f'Multiple alternatives available with average {avg_savings_potential:.1f}% savings',
                    'priority': 'medium',
                    'category': category,
                    'monthly_spending': round(total_category_spending, 2),
                    'alternatives_count': len(category_alts),
                    'action': f'Review your {category} merchants for better options'
                })
        
        return recommendations

    def _calculate_total_savings(self, alternatives: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """Calculate total potential savings across all alternatives."""
        monthly_savings = 0
        annual_savings = 0
        
        for merchant_alternatives in alternatives.values():
            if merchant_alternatives:
                # Use the best alternative for each merchant
                best_alternative = max(merchant_alternatives, 
                                     key=lambda x: x['financial_impact']['potential_savings'])
                monthly_savings += max(0, best_alternative['financial_impact']['potential_savings'])
        
        annual_savings = monthly_savings * 12
        
        return {
            'monthly_potential': round(monthly_savings, 2),
            'annual_potential': round(annual_savings, 2)
        }