import logging
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import json

import vertexai
from vertexai.language_models import TextEmbeddingModel
from google.cloud import aiplatform
from google.cloud import secretmanager
from utils.database_connector import DatabaseConnector

import asyncpg

class VectorSearchTool:
    """
    Advanced vector search tool for financial analysis using PostgreSQL with pgvector
    and Vertex AI embeddings. Supports semantic similarity searches on spending patterns,
    merchants, products, and transaction contexts.
    """
    
    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Vector Search Tool.
        
        Args:
            project_id: Google Cloud project ID
            location: Vertex AI location
            logger: Logger instance
        """
        self.project_id = project_id
        self.secret_client = secretmanager.SecretManagerServiceClient()
        self.location = location
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        self.text_embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        self.embedding_dimension = 768

        # Vector search settings
        self.default_similarity_threshold = 0.7
        self.max_search_results = 50

        self._db_manager: Optional[DatabaseConnector] = None

    async def _get_db_manager(self) -> DatabaseConnector:
        """Get the database manager instance."""
        if self._db_manager is None:
            self._db_manager = await DatabaseConnector.get_instance(self.project_id)
        return self._db_manager
    
    async def search_similar_transactions(
        self,
        query_text: str,
        search_type: str = "description",
        limit: int = 10,
        threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find similar transactions using vector search.
        
        Args:
            query_text: Transaction description or pattern to search for
            search_type: Type of similarity search (description, merchant, category, amount_pattern)
            limit: Maximum number of results to return
            threshold: Similarity threshold (0-1)
            filters: Additional filters to apply
            user_id: User ID to scope search to specific user
            
        Returns:
            Dictionary containing similar transactions and metadata
        """
        try:
            start_time = datetime.now()
            
            # Generate embedding for the query
            query_embedding = await self._generate_text_embedding(query_text)
            
            # Build search query based on search type
            search_results = await self._execute_vector_search(
                embedding=query_embedding,
                search_type=search_type,
                limit=limit,
                threshold=threshold,
                filters=filters,
                user_id=user_id
            )
            
            # Enhance results with additional context
            enhanced_results = await self._enhance_search_results(search_results, search_type)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "query": query_text,
                "search_type": search_type,
                "results": enhanced_results,
                "total_results": len(enhanced_results),
                "execution_time": execution_time,
                "threshold_used": threshold,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query_text,
                "search_type": search_type
            }
    
    async def find_similar_merchants(
        self,
        merchant_name: str,
        limit: int = 10,
        threshold: float = 0.8,
        include_alternatives: bool = True,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find similar merchants based on name and characteristics.
        
        Args:
            merchant_name: Name of merchant to find similarities for
            limit: Maximum number of results
            threshold: Similarity threshold
            include_alternatives: Include alternative merchant suggestions
            user_id: User ID for personalized results
            
        Returns:
            Dictionary containing similar merchants and alternatives
        """
        try:
            db_manager = await self._get_db_manager()
            # Generate embedding for merchant name
            merchant_embedding = await self._generate_text_embedding(merchant_name)
            
            # Search for similar merchants
            query = """
            SELECT 
                m.merchant_id,
                m.name,
                m.normalized_name,
                m.category,
                m.subcategory,
                m.price_range,
                m.merchant_type,
                m.address,
                m.avg_transaction_amount,
                m.total_transactions,
                1 - (m.merchant_embedding <=> %s::vector) as similarity_score
            FROM merchants m
            WHERE (m.merchant_embedding <=> %s::vector) < %s
            ORDER BY m.merchant_embedding <=> %s::vector
            LIMIT %s;
            """
            
            async with db_manager.get_connection() as conn:
                results = await conn.fetch(
                    query, 
                    merchant_embedding, merchant_embedding, 
                    1 - threshold, merchant_embedding, limit
                )
            
            # Format results
            similar_merchants = []
            for row in results:
                merchant_data = dict(row)
                merchant_data['similarity_score'] = float(merchant_data['similarity_score'])
                merchant_data['address'] = json.loads(merchant_data['address']) if merchant_data['address'] else None
                similar_merchants.append(merchant_data)
            
            # Get alternatives if requested
            alternatives = []
            if include_alternatives and similar_merchants:
                alternatives = await self._get_merchant_alternatives(
                    similar_merchants[0]['category'],
                    merchant_name,
                    user_id
                )
            
            return {
                "success": True,
                "query_merchant": merchant_name,
                "similar_merchants": similar_merchants,
                "alternatives": alternatives,
                "total_found": len(similar_merchants)
            }
            
        except Exception as e:
            self.logger.error(f"Merchant similarity search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query_merchant": merchant_name
            }
    
    async def identify_spending_clusters(
        self,
        user_id: str,
        time_period: str = "6_months",
        min_cluster_size: int = 3,
        similarity_threshold: float = 0.75
    ) -> Dict[str, Any]:
        """
        Identify spending behavior clusters using vector similarity.
        
        Args:
            user_id: User ID to analyze
            time_period: Time period for analysis
            min_cluster_size: Minimum transactions per cluster
            similarity_threshold: Threshold for clustering
            
        Returns:
            Dictionary containing identified spending clusters
        """
        try:
            db_manager = await self._get_db_manager()
            # Get user transactions with embeddings
            date_range = self._get_date_range_filter(time_period)
            
            query = """
            SELECT 
                t.transaction_id,
                t.amount,
                t.category,
                t.subcategory,
                t.transaction_date,
                t.transaction_embedding,
                m.name as merchant_name,
                m.category as merchant_category
            FROM transactions t
            LEFT JOIN merchants m ON t.merchant_id = m.merchant_id
            WHERE t.user_id = %s 
                AND t.transaction_date >= %s 
                AND t.transaction_date <= %s
                AND t.deleted_at IS NULL
                AND t.transaction_embedding IS NOT NULL
            ORDER BY t.transaction_date DESC;
            """
            
            async with db_manager.get_connection() as conn:
                transactions = await conn.fetch(query, user_id, date_range['start'], date_range['end'])
            
            if len(transactions) < min_cluster_size:
                return {
                    "success": True,
                    "clusters": [],
                    "message": "Insufficient transactions for clustering analysis"
                }
            
            # Perform clustering analysis
            clusters = await self._perform_vector_clustering(
                transactions, 
                similarity_threshold,
                min_cluster_size
            )
            
            # Analyze clusters for insights
            cluster_insights = await self._analyze_spending_clusters(clusters, user_id)
            
            return {
                "success": True,
                "user_id": user_id,
                "time_period": time_period,
                "total_transactions": len(transactions),
                "clusters": clusters,
                "insights": cluster_insights,
                "clustering_parameters": {
                    "similarity_threshold": similarity_threshold,
                    "min_cluster_size": min_cluster_size
                }
            }
            
        except Exception as e:
            self.logger.error(f"Spending cluster analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "user_id": user_id
            }
    
    async def detect_spending_anomalies(
        self,
        user_id: str,
        time_period: str = "3_months",
        anomaly_threshold: float = 0.3,
        min_pattern_length: int = 5
    ) -> Dict[str, Any]:
        """
        Detect spending anomalies using vector similarity to established patterns.
        
        Args:
            user_id: User ID to analyze
            time_period: Time period for analysis
            anomaly_threshold: Threshold for anomaly detection (lower = more sensitive)
            min_pattern_length: Minimum transactions to establish a pattern
            
        Returns:
            Dictionary containing detected anomalies
        """
        try:
            db_manager = await self._get_db_manager()
            # Get user's spending patterns
            patterns = await self._get_user_spending_patterns(user_id, time_period)
            
            if len(patterns) < min_pattern_length:
                return {
                    "success": True,
                    "anomalies": [],
                    "message": "Insufficient transaction history to detect anomalies"
                }
            
            # Get recent transactions for anomaly detection
            recent_date_range = self._get_date_range_filter("1_month")
            
            query = """
            SELECT 
                t.transaction_id,
                t.amount,
                t.category,
                t.subcategory,
                t.transaction_date,
                t.transaction_embedding,
                m.name as merchant_name,
                m.normalized_name
            FROM transactions t
            LEFT JOIN merchants m ON t.merchant_id = m.merchant_id
            WHERE t.user_id = %s 
                AND t.transaction_date >= %s
                AND t.deleted_at IS NULL
                AND t.transaction_embedding IS NOT NULL
            ORDER BY t.transaction_date DESC;
            """
            
            async with db_manager.get_connection() as conn:
                recent_transactions = await conn.fetch(
                    query, user_id, recent_date_range['start']
                )
            
            # Detect anomalies by comparing to established patterns
            anomalies = await self._detect_transaction_anomalies(
                recent_transactions,
                patterns,
                anomaly_threshold
            )
            
            # Classify and prioritize anomalies
            classified_anomalies = await self._classify_anomalies(anomalies, user_id)
            
            return {
                "success": True,
                "user_id": user_id,
                "analysis_period": time_period,
                "recent_period": "1_month",
                "total_recent_transactions": len(recent_transactions),
                "total_anomalies": len(classified_anomalies),
                "anomalies": classified_anomalies,
                "detection_parameters": {
                    "anomaly_threshold": anomaly_threshold,
                    "min_pattern_length": min_pattern_length
                }
            }
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "user_id": user_id
            }
    
    async def match_historical_patterns(
        self,
        user_id: str,
        current_transaction: Dict[str, Any],
        pattern_window: str = "1_year",
        similarity_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Match current spending to historical patterns.
        
        Args:
            user_id: User ID
            current_transaction: Current transaction to match
            pattern_window: Time window for historical patterns
            similarity_threshold: Similarity threshold for matching
            
        Returns:
            Dictionary containing pattern matches and insights
        """
        try:
            db_manager = await self._get_db_manager()
            # Generate embedding for current transaction
            transaction_text = self._format_transaction_for_embedding(current_transaction)
            current_embedding = await self._generate_text_embedding(transaction_text)
            
            # Search for similar historical transactions
            date_range = self._get_date_range_filter(pattern_window)
            
            query = """
            SELECT 
                t.transaction_id,
                t.amount,
                t.category,
                t.subcategory,
                t.transaction_date,
                t.transaction_embedding,
                m.name as merchant_name,
                1 - (t.transaction_embedding <=> %s::vector) as similarity_score
            FROM transactions t
            LEFT JOIN merchants m ON t.merchant_id = m.merchant_id
            WHERE t.user_id = %s 
                AND t.transaction_date >= %s 
                AND t.transaction_date <= %s
                AND t.deleted_at IS NULL
                AND t.transaction_embedding IS NOT NULL
                AND (t.transaction_embedding <=> %s::vector) < %s
            ORDER BY t.transaction_embedding <=> %s::vector
            LIMIT 20;
            """
            
            async with db_manager.get_connection() as conn:
                matches = await conn.fetch(
                    query, 
                    current_embedding, user_id, 
                    date_range['start'], date_range['end'],
                    current_embedding, 1 - similarity_threshold,
                    current_embedding
                )
            
            # Analyze patterns in matches
            pattern_analysis = await self._analyze_pattern_matches(matches, current_transaction)
            
            # Generate insights and predictions
            insights = await self._generate_pattern_insights(pattern_analysis, user_id)
            
            return {
                "success": True,
                "current_transaction": current_transaction,
                "historical_matches": [dict(match) for match in matches],
                "pattern_analysis": pattern_analysis,
                "insights": insights,
                "match_parameters": {
                    "pattern_window": pattern_window,
                    "similarity_threshold": similarity_threshold,
                    "total_matches": len(matches)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Pattern matching failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "current_transaction": current_transaction
            }
    
    # Private helper methods
    
    async def _generate_text_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text using Vertex AI."""
        try:
            embeddings = self.text_embedding_model.get_embeddings([text])
            return embeddings[0].values
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def _execute_vector_search(
        self,
        embedding: List[float],
        search_type: str,
        limit: int,
        threshold: float,
        filters: Optional[Dict[str, Any]],
        user_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Execute vector search query based on search type."""
        db_manager = await self._get_db_manager()
        base_conditions = []
        params = [embedding, embedding, 1 - threshold, embedding, limit]
        
        # Add user filter if specified
        if user_id:
            base_conditions.append("t.user_id = %s")
            params.insert(-1, user_id)  # Insert before limit
        
        # Add date range filter if specified
        if filters and 'date_range' in filters:
            date_range = filters['date_range']
            if 'start_date' in date_range:
                base_conditions.append("t.transaction_date >= %s")
                params.insert(-1, date_range['start_date'])
            if 'end_date' in date_range:
                base_conditions.append("t.transaction_date <= %s")
                params.insert(-1, date_range['end_date'])
        
        # Add category filter if specified
        if filters and 'categories' in filters:
            category_placeholders = ','.join(['%s'] * len(filters['categories']))
            base_conditions.append(f"t.category = ANY(ARRAY[{category_placeholders}])")
            for category in filters['categories']:
                params.insert(-1, category)
        
        # Add amount range filter if specified
        if filters and 'amount_range' in filters:
            amount_range = filters['amount_range']
            if 'min_amount' in amount_range:
                base_conditions.append("t.amount >= %s")
                params.insert(-1, amount_range['min_amount'])
            if 'max_amount' in amount_range:
                base_conditions.append("t.amount <= %s")
                params.insert(-1, amount_range['max_amount'])
        
        # Build WHERE clause
        where_clause = "WHERE " + " AND ".join([
            "t.deleted_at IS NULL",
            "t.transaction_embedding IS NOT NULL",
            "(t.transaction_embedding <=> %s::vector) < %s"
        ] + base_conditions)
        
        query = f"""
        SELECT 
            t.transaction_id,
            t.amount,
            t.category,
            t.subcategory,
            t.transaction_date,
            t.payment_method,
            m.name as merchant_name,
            m.normalized_name,
            m.category as merchant_category,
            1 - (t.transaction_embedding <=> %s::vector) as similarity_score
        FROM transactions t
        LEFT JOIN merchants m ON t.merchant_id = m.merchant_id
        {where_clause}
        ORDER BY t.transaction_embedding <=> %s::vector
        LIMIT %s;
        """
        
        async with db_manager.get_connection() as conn:
            results = await conn.fetch(query, *params)
        
        return [dict(row) for row in results]
    
    async def _enhance_search_results(
        self, 
        results: List[Dict[str, Any]], 
        search_type: str
    ) -> List[Dict[str, Any]]:
        """Enhance search results with additional context and metadata."""
        enhanced_results = []
        
        for result in results:
            enhanced_result = result.copy()
            
            # Convert Decimal to float for JSON serialization
            if 'amount' in enhanced_result and enhanced_result['amount'] is not None:
                enhanced_result['amount'] = float(enhanced_result['amount'])
            if 'similarity_score' in enhanced_result and enhanced_result['similarity_score'] is not None:
                enhanced_result['similarity_score'] = float(enhanced_result['similarity_score'])
            
            # Convert date to string
            if 'transaction_date' in enhanced_result and enhanced_result['transaction_date'] is not None:
                enhanced_result['transaction_date'] = enhanced_result['transaction_date'].isoformat()
            
            # Add search context
            enhanced_result['search_context'] = {
                'search_type': search_type,
                'relevance_level': self._categorize_similarity_score(enhanced_result.get('similarity_score', 0))
            }
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _categorize_similarity_score(self, score: float) -> str:
        """Categorize similarity score into relevance levels."""
        if score >= 0.9:
            return "very_high"
        elif score >= 0.8:
            return "high"
        elif score >= 0.7:
            return "medium"
        elif score >= 0.6:
            return "low"
        else:
            return "very_low"
    
    def _get_date_range_filter(self, period: str) -> Dict[str, str]:
        """Convert period string to date range."""
        end_date = datetime.now().date()
        
        period_map = {
            "1_week": timedelta(days=7),
            "1_month": timedelta(days=30),
            "3_months": timedelta(days=90),
            "6_months": timedelta(days=180),
            "1_year": timedelta(days=365),
            "2_years": timedelta(days=730)
        }
        
        start_date = end_date - period_map.get(period, timedelta(days=90))
        
        return {
            "start": start_date.isoformat(),
            "end": end_date.isoformat()
        }
    
    def _format_transaction_for_embedding(self, transaction: Dict[str, Any]) -> str:
        """Format transaction data for embedding generation."""
        parts = []
        
        if 'merchant_name' in transaction and transaction['merchant_name']:
            parts.append(f"Merchant: {transaction['merchant_name']}")
        if 'category' in transaction and transaction['category']:
            parts.append(f"Category: {transaction['category']}")
        if 'amount' in transaction and transaction['amount']:
            parts.append(f"Amount: ${transaction['amount']}")
        if 'description' in transaction and transaction['description']:
            parts.append(f"Description: {transaction['description']}")
        
        return " | ".join(parts) if parts else "Unknown transaction"
    
    async def _get_merchant_alternatives(
        self,
        category: str,
        original_merchant: str,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get alternative merchants in the same category."""
        try:
            db_manager = await self._get_db_manager()
            query = """
            SELECT 
                m.merchant_id,
                m.name,
                m.normalized_name,
                m.price_range,
                m.avg_transaction_amount,
                m.address,
                COALESCE(ut.transaction_count, 0) as user_transaction_count,
                COALESCE(ut.avg_amount, 0) as user_avg_amount
            FROM merchants m
            LEFT JOIN (
                SELECT 
                    merchant_id,
                    COUNT(*) as transaction_count,
                    AVG(amount) as avg_amount
                FROM transactions 
                WHERE user_id = %s AND deleted_at IS NULL
                GROUP BY merchant_id
            ) ut ON m.merchant_id = ut.merchant_id
            WHERE m.category = %s 
                AND m.normalized_name != %s
            ORDER BY ut.transaction_count DESC NULLS LAST, m.total_transactions DESC
            LIMIT 5;
            """
            
            normalized_original = original_merchant.lower().strip() if original_merchant else ""
            
            async with db_manager.get_connection() as conn:
                results = await conn.fetch(query, user_id or '', category or '', normalized_original)
            
            alternatives = []
            for row in results:
                alt = dict(row)
                alt['avg_transaction_amount'] = float(alt['avg_transaction_amount']) if alt['avg_transaction_amount'] else 0
                alt['user_avg_amount'] = float(alt['user_avg_amount']) if alt['user_avg_amount'] else 0
                alt['address'] = json.loads(alt['address']) if alt['address'] else None
                alternatives.append(alt)
            
            return alternatives
            
        except Exception as e:
            self.logger.error(f"Failed to get merchant alternatives: {e}")
            return []
    
    async def _perform_vector_clustering(
        self,
        transactions: List[Dict[str, Any]],
        similarity_threshold: float,
        min_cluster_size: int
    ) -> List[Dict[str, Any]]:
        """Perform clustering analysis on transaction vectors."""
        # This is a simplified clustering approach
        # For production, you might want to use more sophisticated clustering algorithms
        
        clusters = []
        processed_transactions = set()
        
        for i, transaction in enumerate(transactions):
            if transaction['transaction_id'] in processed_transactions:
                continue
            
            # Start a new cluster with this transaction
            cluster = {
                'cluster_id': len(clusters),
                'transactions': [transaction],
                'center_transaction': transaction,
                'characteristics': {
                    'dominant_category': transaction['category'],
                    'avg_amount': float(transaction['amount']),
                    'date_range': {
                        'start': transaction['transaction_date'],
                        'end': transaction['transaction_date']
                    }
                }
            }
            
            processed_transactions.add(transaction['transaction_id'])
            
            # Find similar transactions for this cluster
            for j, other_transaction in enumerate(transactions[i+1:], i+1):
                if other_transaction['transaction_id'] in processed_transactions:
                    continue
                
                # Calculate similarity between embeddings
                if (transaction['transaction_embedding'] and 
                    other_transaction['transaction_embedding']):
                    
                    similarity = await self._calculate_vector_similarity(
                        transaction['transaction_embedding'],
                        other_transaction['transaction_embedding']
                    )
                    
                    if similarity >= similarity_threshold:
                        cluster['transactions'].append(other_transaction)
                        processed_transactions.add(other_transaction['transaction_id'])
            
            # Only keep clusters that meet minimum size requirement
            if len(cluster['transactions']) >= min_cluster_size:
                # Update cluster characteristics
                cluster['characteristics'] = await self._calculate_cluster_characteristics(
                    cluster['transactions']
                )
                clusters.append(cluster)
        
        return clusters
    
    async def _calculate_vector_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            # Convert to numpy arrays for calculation
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            # Calculate cosine similarity
            dot_product = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            if norms == 0:
                return 0.0
            
            return dot_product / norms
            
        except Exception as e:
            self.logger.error(f"Failed to calculate vector similarity: {e}")
            return 0.0
    
    async def _calculate_cluster_characteristics(
        self, 
        transactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate characteristics of a transaction cluster."""
        if not transactions:
            return {}
        
        # Calculate basic statistics
        amounts = [float(t['amount']) for t in transactions if t['amount'] is not None]
        categories = [t['category'] for t in transactions if t['category']]
        dates = [t['transaction_date'] for t in transactions if t['transaction_date']]
        
        if not amounts or not categories or not dates:
            return {"error": "Insufficient data for cluster analysis"}
        
        # Find dominant category
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        dominant_category = max(category_counts, key=category_counts.get)
        
        return {
            'transaction_count': len(transactions),
            'dominant_category': dominant_category,
            'category_distribution': category_counts,
            'avg_amount': sum(amounts) / len(amounts),
            'min_amount': min(amounts),
            'max_amount': max(amounts),
            'date_range': {
                'start': min(dates),
                'end': max(dates)
            },
            'frequency': len(transactions) / max(1, (max(dates) - min(dates)).days or 1)
        }
    
    async def _analyze_spending_clusters(
        self, 
        clusters: List[Dict[str, Any]], 
        user_id: str
    ) -> Dict[str, Any]:
        """Analyze spending clusters for insights."""
        if not clusters:
            return {"message": "No clusters found"}
        
        insights = {
            "total_clusters": len(clusters),
            "cluster_summary": [],
            "patterns": [],
            "recommendations": []
        }
        
        for cluster in clusters:
            chars = cluster['characteristics']
            if 'error' in chars:
                continue
                
            summary = {
                "cluster_id": cluster['cluster_id'],
                "size": chars['transaction_count'],
                "dominant_category": chars['dominant_category'],
                "avg_amount": round(chars['avg_amount'], 2),
                "frequency": round(chars['frequency'], 2)
            }
            insights["cluster_summary"].append(summary)
            
            # Identify patterns
            if chars['frequency'] > 0.5:  # More than once every 2 days
                insights["patterns"].append(f"High frequency spending in {chars['dominant_category']}")
            
            if chars['avg_amount'] > 100:
                insights["patterns"].append(f"High-value transactions in {chars['dominant_category']}")
        
        return insights
    
    async def _get_user_spending_patterns(
        self, 
        user_id: str, 
        time_period: str
    ) -> List[Dict[str, Any]]:
        """Get established spending patterns for a user."""
        date_range = self._get_date_range_filter(time_period)
        db_manager = await self._get_db_manager()
        query = """
        SELECT 
            t.category,
            t.subcategory,
            AVG(t.amount) as avg_amount,
            STDDEV(t.amount) as amount_stddev,
            COUNT(*) as transaction_count
        FROM transactions t
        WHERE t.user_id = %s 
            AND t.transaction_date >= %s 
            AND t.transaction_date <= %s
            AND t.deleted_at IS NULL
            AND t.transaction_embedding IS NOT NULL
        GROUP BY t.category, t.subcategory
        HAVING COUNT(*) >= 3
        ORDER BY transaction_count DESC;
        """
        
        async with db_manager.get_connection() as conn:
            results = await conn.fetch(query, user_id, date_range['start'], date_range['end'])
        
        return [dict(row) for row in results]
    
    async def _detect_transaction_anomalies(
        self,
        recent_transactions: List[Dict[str, Any]],
        patterns: List[Dict[str, Any]],
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Detect anomalies by comparing recent transactions to established patterns."""
        anomalies = []
        
        # Create pattern lookup by category
        pattern_lookup = {}
        for pattern in patterns:
            key = f"{pattern['category']}_{pattern['subcategory'] or ''}"
            pattern_lookup[key] = pattern
        
        for transaction in recent_transactions:
            key = f"{transaction['category']}_{transaction['subcategory'] or ''}"
            
            if key in pattern_lookup:
                pattern = pattern_lookup[key]
                
                # Check amount anomaly
                expected_amount = float(pattern['avg_amount']) if pattern['avg_amount'] else 0
                actual_amount = float(transaction['amount']) if transaction['amount'] else 0
                amount_stddev = float(pattern['amount_stddev']) if pattern['amount_stddev'] else 0
                
                # Calculate z-score for amount deviation
                if amount_stddev > 0:
                    z_score = abs(actual_amount - expected_amount) / amount_stddev
                    if z_score > 2:  # More than 2 standard deviations
                        anomalies.append({
                            'transaction': transaction,
                            'anomaly_type': 'amount_deviation',
                            'expected_amount': expected_amount,
                            'actual_amount': actual_amount,
                            'z_score': z_score,
                            'severity': 'high' if z_score > 3 else 'medium'
                        })
            else:
                # Transaction in category with no established pattern
                anomalies.append({
                    'transaction': transaction,
                    'anomaly_type': 'new_category',
                    'severity': 'low'
                })
        
        return anomalies
    
    async def _classify_anomalies(
        self,
        anomalies: List[Dict[str, Any]],
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Classify and prioritize detected anomalies."""
        classified_anomalies = []
        
        for anomaly in anomalies:
            transaction = anomaly.get('transaction', {})
            classification = {
                'anomaly_id': f"anomaly_{len(classified_anomalies)}",
                'user_id': user_id,
                'transaction_id': transaction.get('transaction_id'),
                'type': anomaly['anomaly_type'],
                'severity': anomaly['severity'],
                'detected_at': datetime.now().isoformat(),
                'transaction_details': {
                    'amount': float(transaction['amount']) if transaction.get('amount') else 0,
                    'merchant': transaction.get('merchant_name', 'Unknown'),
                    'category': transaction.get('category', 'Unknown'),
                    'date': transaction['transaction_date'].isoformat() if transaction.get('transaction_date') else None
                }
            }
            
            # Add type-specific details
            if anomaly['anomaly_type'] == 'amount_deviation':
                classification['details'] = {
                    'expected_amount': anomaly.get('expected_amount', 0),
                    'actual_amount': anomaly.get('actual_amount', 0),
                    'z_score': anomaly.get('z_score', 0),
                    'description': f"Amount ${anomaly.get('actual_amount', 0):.2f} is significantly different from expected ${anomaly.get('expected_amount', 0):.2f}"
                }
            elif anomaly['anomaly_type'] == 'behavioral_deviation':
                classification['details'] = {
                    'similarity_score': anomaly.get('similarity_score', 0),
                    'description': f"Transaction pattern unusual for this category (similarity: {anomaly.get('similarity_score', 0):.2f})"
                }
            elif anomaly['anomaly_type'] == 'new_category':
                classification['details'] = {
                    'description': f"First transaction in category '{transaction.get('category', 'Unknown')}'"
                }
            
            # Generate recommendations based on anomaly type
            classification['recommendations'] = await self._generate_anomaly_recommendations(anomaly)
            
            classified_anomalies.append(classification)
        
        # Sort by severity and amount impact
        severity_order = {'high': 3, 'medium': 2, 'low': 1}
        classified_anomalies.sort(
            key=lambda x: (severity_order.get(x['severity'], 0), x['transaction_details']['amount']),
            reverse=True
        )
        
        return classified_anomalies
    
    async def _analyze_pattern_matches(
        self,
        matches: List[Dict[str, Any]],
        current_transaction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze patterns in historical matches."""
        if not matches:
            return {"message": "No historical patterns found"}
        
        # Group matches by time periods
        monthly_patterns = {}
        seasonal_patterns = {}
        frequency_analysis = {}
        
        for match in matches:
            match_date = match.get('transaction_date')
            if not match_date:
                continue
                
            if isinstance(match_date, str):
                try:
                    match_date = datetime.fromisoformat(match_date).date()
                except ValueError:
                    continue
            
            # Monthly pattern analysis
            month_key = f"{match_date.year}-{match_date.month:02d}"
            if month_key not in monthly_patterns:
                monthly_patterns[month_key] = []
            monthly_patterns[month_key].append(match)
            
            # Seasonal pattern analysis
            season = self._get_season(match_date)
            if season not in seasonal_patterns:
                seasonal_patterns[season] = []
            seasonal_patterns[season].append(match)
            
            # Frequency analysis by merchant
            merchant = match.get('merchant_name', 'Unknown')
            if merchant not in frequency_analysis:
                frequency_analysis[merchant] = []
            frequency_analysis[merchant].append(match)
        
        # Calculate pattern statistics
        amounts = [float(m['amount']) for m in matches if m.get('amount') is not None]
        similarities = [float(m['similarity_score']) for m in matches if m.get('similarity_score') is not None]
        
        if not amounts or not similarities:
            return {"message": "Insufficient data for pattern analysis"}
        
        analysis = {
            'total_matches': len(matches),
            'avg_similarity': sum(similarities) / len(similarities),
            'amount_statistics': {
                'avg': sum(amounts) / len(amounts),
                'min': min(amounts),
                'max': max(amounts),
                'std_dev': np.std(amounts) if len(amounts) > 1 else 0
            },
            'temporal_patterns': {
                'monthly_distribution': {k: len(v) for k, v in monthly_patterns.items()},
                'seasonal_distribution': {k: len(v) for k, v in seasonal_patterns.items()},
                'most_frequent_month': max(monthly_patterns.keys(), key=lambda k: len(monthly_patterns[k])) if monthly_patterns else None
            },
            'merchant_patterns': {
                'merchant_frequency': {k: len(v) for k, v in frequency_analysis.items()},
                'most_frequent_merchant': max(frequency_analysis.keys(), key=lambda k: len(frequency_analysis[k])) if frequency_analysis else None
            }
        }
        
        return analysis
    
    async def _generate_pattern_insights(
        self,
        pattern_analysis: Dict[str, Any],
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Generate actionable insights from pattern analysis."""
        insights = []
        
        if 'amount_statistics' in pattern_analysis:
            stats = pattern_analysis['amount_statistics']
            
            # Amount trend insight
            if stats.get('std_dev', 0) > stats.get('avg', 0) * 0.3:  # High variability
                insights.append({
                    'type': 'amount_variability',
                    'title': 'Variable Spending Pattern',
                    'description': f"Your spending in this category varies significantly (${stats.get('min', 0):.2f} - ${stats.get('max', 0):.2f})",
                    'recommendation': 'Consider setting a more flexible budget or reviewing the necessity of higher amounts'
                })
        
        if 'temporal_patterns' in pattern_analysis:
            temporal = pattern_analysis['temporal_patterns']
            
            # Seasonal insight
            if temporal.get('seasonal_distribution'):
                peak_season = max(temporal['seasonal_distribution'].keys(), 
                                key=lambda k: temporal['seasonal_distribution'][k])
                insights.append({
                    'type': 'seasonal_pattern',
                    'title': f'Peak Spending in {peak_season}',
                    'description': f"You tend to spend more in this category during {peak_season}",
                    'recommendation': f'Plan ahead for increased {peak_season} spending'
                })
        
        if 'merchant_patterns' in pattern_analysis:
            merchant = pattern_analysis['merchant_patterns']
            
            if merchant.get('most_frequent_merchant'):
                insights.append({
                    'type': 'merchant_loyalty',
                    'title': 'Preferred Merchant Identified',
                    'description': f"You frequently shop at {merchant['most_frequent_merchant']}",
                    'recommendation': 'Check if loyalty programs or bulk discounts are available'
                })
        
        return insights
    
    async def _generate_anomaly_recommendations(
        self,
        anomaly: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on anomaly type."""
        recommendations = []
        
        anomaly_type = anomaly.get('anomaly_type')
        transaction = anomaly.get('transaction', {})
        
        if anomaly_type == 'amount_deviation':
            if anomaly.get('z_score', 0) > 2:
                recommendations.extend([
                    {
                        'type': 'review_transaction',
                        'title': 'Review Transaction Details',
                        'description': 'This transaction amount is unusual for this category',
                        'action': 'Verify the transaction details and ensure it was intentional'
                    },
                    {
                        'type': 'budget_adjustment',
                        'title': 'Consider Budget Adjustment',
                        'description': 'If this spending level is becoming regular, adjust your budget',
                        'action': f'Update budget for {transaction.get("category", "this")} category'
                    }
                ])
        
        elif anomaly_type == 'behavioral_deviation':
            recommendations.extend([
                {
                    'type': 'pattern_review',
                    'title': 'Spending Pattern Change',
                    'description': 'Your spending pattern for this category has changed',
                    'action': 'Review if this change aligns with your financial goals'
                },
                {
                    'type': 'find_alternatives',
                    'title': 'Explore Alternatives',
                    'description': 'Consider if there are more cost-effective alternatives',
                    'action': f'Look for alternative merchants in {transaction.get("category", "this category")}'
                }
            ])
        
        elif anomaly_type == 'new_category':
            recommendations.extend([
                {
                    'type': 'budget_planning',
                    'title': 'Set Budget for New Category',
                    'description': 'This is a new spending category for you',
                    'action': f'Set a monthly budget limit for {transaction.get("category", "this category")}'
                },
                {
                    'type': 'track_spending',
                    'title': 'Monitor Future Spending',
                    'description': 'Keep track of spending in this new category',
                    'action': 'Review this category in your next monthly budget review'
                }
            ])
        
        return recommendations
    
    def _get_season(self, date) -> str:
        """Determine season from date."""
        month = date.month
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    async def create_transaction_embedding(
        self,
        transaction_data: Dict[str, Any]
    ) -> List[float]:
        """
        Create embedding for a transaction to be stored in the database.
        This is a utility method for other agents to use.
        """
        try:
            # Format transaction for embedding
            transaction_text = self._format_transaction_for_embedding(transaction_data)
            
            # Generate embedding
            embedding = await self._generate_text_embedding(transaction_text)
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to create transaction embedding: {e}")
            raise
    
    async def create_merchant_embedding(
        self,
        merchant_data: Dict[str, Any]
    ) -> List[float]:
        """
        Create embedding for a merchant to be stored in the database.
        """
        try:
            # Format merchant data for embedding
            parts = []
            if 'name' in merchant_data and merchant_data['name']:
                parts.append(f"Name: {merchant_data['name']}")
            if 'category' in merchant_data and merchant_data['category']:
                parts.append(f"Category: {merchant_data['category']}")
            if 'subcategory' in merchant_data and merchant_data['subcategory']:
                parts.append(f"Subcategory: {merchant_data['subcategory']}")
            if 'merchant_type' in merchant_data and merchant_data['merchant_type']:
                parts.append(f"Type: {merchant_data['merchant_type']}")
            if 'address' in merchant_data and merchant_data['address']:
                if isinstance(merchant_data['address'], dict):
                    city = merchant_data['address'].get('city', '')
                    state = merchant_data['address'].get('state', '')
                    if city or state:
                        parts.append(f"Location: {city} {state}".strip())
            
            merchant_text = " | ".join(parts) if parts else "Unknown merchant"
            
            # Generate embedding
            embedding = await self._generate_text_embedding(merchant_text)
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to create merchant embedding: {e}")
            raise
    
    async def create_item_embedding(
        self,
        item_data: Dict[str, Any]
    ) -> List[float]:
        """
        Create embedding for a transaction item.
        """
        try:
            # Format item data for embedding
            parts = []
            if 'name' in item_data and item_data['name']:
                parts.append(f"Item: {item_data['name']}")
            if 'category' in item_data and item_data['category']:
                parts.append(f"Category: {item_data['category']}")
            if 'brand' in item_data and item_data['brand']:
                parts.append(f"Brand: {item_data['brand']}")
            if 'description' in item_data and item_data['description']:
                parts.append(f"Description: {item_data['description']}")
            if 'size_info' in item_data and item_data['size_info']:
                parts.append(f"Size: {item_data['size_info']}")
            if item_data.get('is_organic'):
                parts.append("Organic")
            
            item_text = " | ".join(parts) if parts else "Unknown item"
            
            # Generate embedding
            embedding = await self._generate_text_embedding(item_text)
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to create item embedding: {e}")
            raise
    
    async def batch_update_embeddings(
        self,
        table_name: str,
        id_column: str,
        embedding_column: str,
        data_formatter_func,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Utility method to batch update embeddings for existing records.
        Useful for migrating data or updating embeddings with new models.
        """
        updated_count = 0
        error_count = 0
        
        try:
            db_manager = await self._get_db_manager()
            
            # Get records without embeddings
            query = f"""
            SELECT {id_column}, *
            FROM {table_name}
            WHERE {embedding_column} IS NULL
            ORDER BY created_at DESC
            LIMIT %s;
            """
            
            async with db_manager.get_connection() as conn:
                while True:
                    records = await conn.fetch(query, batch_size)
                    
                    if not records:
                        break
                    
                    # Process batch
                    for record in records:
                        try:
                            record_dict = dict(record)
                            formatted_text = data_formatter_func(record_dict)
                            embedding = await self._generate_text_embedding(formatted_text)
                            
                            # Update record with embedding
                            update_query = f"""
                            UPDATE {table_name}
                            SET {embedding_column} = %s::vector
                            WHERE {id_column} = %s;
                            """
                            
                            await conn.execute(update_query, embedding, record_dict[id_column])
                            updated_count += 1
                            
                        except Exception as e:
                            self.logger.error(f"Failed to update embedding for {record_dict.get(id_column)}: {e}")
                            error_count += 1
                    
                    # Small delay between batches to avoid overwhelming the API
                    await asyncio.sleep(0.1)
            
            return {
                "success": True,
                "updated_count": updated_count,
                "error_count": error_count,
                "table": table_name
            }
            
        except Exception as e:
            self.logger.error(f"Batch embedding update failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "updated_count": updated_count,
                "error_count": error_count
            }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the vector search tool."""
        try:
            db_manager = await self._get_db_manager()
            # Test database connection
            async with db_manager.get_connection() as conn:
                await conn.fetchval("SELECT 1")
            db_status = "healthy"
            
            # Test embedding generation
            test_embedding = await self._generate_text_embedding("test query")
            embedding_status = "healthy" if len(test_embedding) == self.embedding_dimension else "unhealthy"
            
            # Get some statistics
            async with db_manager.get_connection() as conn:
                stats = await conn.fetchrow("""
                SELECT 
                    (SELECT COUNT(*) FROM transactions WHERE transaction_embedding IS NOT NULL) as transactions_with_embeddings,
                    (SELECT COUNT(*) FROM merchants WHERE merchant_embedding IS NOT NULL) as merchants_with_embeddings,
                    (SELECT COUNT(*) FROM transaction_items WHERE item_embedding IS NOT NULL) as items_with_embeddings
                """)
            
            return {
                "status": "healthy" if db_status == "healthy" and embedding_status == "healthy" else "unhealthy",
                "database_connection": db_status,
                "embedding_service": embedding_status,
                "embedding_dimension": self.embedding_dimension,
                "statistics": dict(stats) if stats else {},
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }