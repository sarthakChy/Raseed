import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import asyncio
import logging
from datetime import datetime, timedelta

# Assuming you're using pgvector with PostgreSQL
import asyncpg
from pgvector.asyncpg import register_vector

@dataclass
class SearchResult:
    """Structured result from vector search"""
    item_id: str
    item_name: str
    category: str
    price: float
    store: str
    similarity_score: float
    last_seen: datetime
    potential_saving: float
    metadata: Dict[str, Any] = None

@dataclass
class SearchFilter:
    """Search filters for vector queries"""
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    categories: Optional[List[str]] = None
    stores: Optional[List[str]] = None
    max_distance_km: Optional[float] = None
    exclude_user_purchases: bool = True
    min_similarity: float = 0.7
    max_results: int = 10

class VectorSearchTool:
    """
    Vector search tool specialized for finding cheaper alternatives to products.
    Integrates with PostgreSQL + pgvector for similarity search.
    """
    
    def __init__(self, db_connector, logger=None):
        self.db_connector = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        # Vector search configuration
        self.default_similarity_threshold = 0.7
        self.max_results_limit = 50
        
        # Price comparison thresholds
        self.min_savings_threshold = 0.05  # Minimum 5% savings to recommend
        self.max_price_ratio = 0.95  # Only recommend items up to 95% of original price
        
    async def _get_connection(self):
        """Get database connection with vector support"""
        conn = await self.db_connector.get_connection()
        await register_vector(conn)
        return conn

    async def find_cheaper_alternatives(
        self, 
        item_embedding: List[float], 
        original_price: float,
        category: str,
        user_id: str,
        filters: Optional[SearchFilter] = None
    ) -> List[SearchResult]:
        """
        Find cheaper alternatives to a given item using vector similarity search.
        
        Args:
            item_embedding: Vector embedding of the original item
            original_price: Price of the original item
            category: Category of the item (for filtering)
            user_id: User ID (to exclude their own purchases)
            filters: Additional search filters
            
        Returns:
            List of SearchResult objects with cheaper alternatives
        """
        if not filters:
            filters = SearchFilter()
        
        # Set price filters based on original price
        filters.max_price = min(
            filters.max_price or float('inf'),
            original_price * self.max_price_ratio
        )
        
        try:
            conn = await self._get_connection()
            
            # Build the vector similarity query
            query = self._build_similarity_query(filters, user_id)
            
            # Execute the search
            embedding_array = np.array(item_embedding, dtype=np.float32)
            
            rows = await conn.fetch(
                query,
                embedding_array,
                category,
                user_id,
                filters.max_price,
                filters.min_similarity,
                filters.max_results or 10
            )
            
            # Process results
            results = []
            for row in rows:
                potential_saving = original_price - row['price']
                saving_percentage = potential_saving / original_price
                
                # Only include if savings meet threshold
                if saving_percentage >= self.min_savings_threshold:
                    result = SearchResult(
                        item_id=row['item_id'],
                        item_name=row['item_name'],
                        category=row['category'],
                        price=row['price'],
                        store=row['store'],
                        similarity_score=1 - row['distance'],  # Convert distance to similarity
                        last_seen=row['last_seen'],
                        potential_saving=potential_saving,
                        metadata={
                            'saving_percentage': saving_percentage,
                            'brand': row.get('brand'),
                            'location': row.get('location'),
                            'availability': row.get('availability', 'unknown')
                        }
                    )
                    results.append(result)
            
            # Sort by best combination of savings and similarity
            results.sort(key=lambda x: (x.potential_saving * x.similarity_score), reverse=True)
            
            self.logger.info(f"Found {len(results)} alternatives for category {category}")
            return results[:filters.max_results]
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []
        finally:
            if 'conn' in locals():
                await conn.close()

    def _build_similarity_query(self, filters: SearchFilter, user_id: str) -> str:
        """Build PostgreSQL query with vector similarity search"""
        
        base_query = """
        SELECT 
            pi.item_id,
            pi.item_name,
            pi.category,
            pi.price,
            pi.store,
            pi.brand,
            pi.last_seen,
            pi.location,
            pi.availability,
            (pi.embedding <=> $1) as distance
        FROM product_index pi
        WHERE 1=1
        """
        
        conditions = []
        
        # Category filter
        conditions.append("AND pi.category = $2")
        
        # Exclude user's own purchases
        if filters.exclude_user_purchases:
            conditions.append("""
                AND pi.item_id NOT IN (
                    SELECT DISTINCT product_id 
                    FROM user_purchases 
                    WHERE user_id = $3
                    AND purchase_date > NOW() - INTERVAL '30 days'
                )
            """)
        
        # Price filter
        conditions.append("AND pi.price <= $4")
        
        # Similarity threshold
        conditions.append("AND (pi.embedding <=> $1) <= (1 - $5)")  # Convert similarity to distance
        
        # Only include recently seen items (within last 30 days)
        conditions.append("AND pi.last_seen > NOW() - INTERVAL '30 days'")
        
        # Only include available items
        conditions.append("AND (pi.availability IS NULL OR pi.availability != 'out_of_stock')")
        
        query = base_query + " ".join(conditions)
        query += " ORDER BY (pi.embedding <=> $1) LIMIT $6"
        
        return query

    async def find_similar_products_by_name(
        self, 
        product_name: str, 
        category: str,
        max_price: Optional[float] = None,
        user_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Find similar products by name using text search + vector similarity.
        Useful when you don't have an embedding but have a product name.
        """
        try:
            conn = await self._get_connection()
            
            # First, get embedding for the product name using text similarity
            text_search_query = """
            SELECT embedding, price 
            FROM product_index 
            WHERE similarity(item_name, $1) > 0.3
            AND category = $2
            ORDER BY similarity(item_name, $1) DESC
            LIMIT 1
            """
            
            row = await conn.fetchrow(text_search_query, product_name, category)
            
            if not row:
                self.logger.warning(f"No similar product found for: {product_name}")
                return []
            
            # Use the found embedding to search for alternatives
            filters = SearchFilter(max_price=max_price or row['price'])
            
            return await self.find_cheaper_alternatives(
                item_embedding=row['embedding'],
                original_price=row['price'],
                category=category,
                user_id=user_id or 'anonymous',
                filters=filters
            )
            
        except Exception as e:
            self.logger.error(f"Text-based search failed: {e}")
            return []
        finally:
            if 'conn' in locals():
                await conn.close()

    async def find_store_alternatives(
        self, 
        product_name: str,
        current_store: str,
        category: str,
        user_location: Optional[Dict[str, float]] = None
    ) -> List[SearchResult]:
        """
        Find the same or similar product at different stores.
        """
        try:
            conn = await self._get_connection()
            
            # Search for same product at different stores
            query = """
            SELECT 
                item_id, item_name, category, price, store, brand, last_seen,
                location, availability
            FROM product_index
            WHERE similarity(item_name, $1) > 0.8
            AND category = $2  
            AND store != $3
            AND (availability IS NULL OR availability != 'out_of_stock')
            ORDER BY price ASC
            LIMIT 10
            """
            
            rows = await conn.fetch(query, product_name, category, current_store)
            
            results = []
            for row in rows:
                result = SearchResult(
                    item_id=row['item_id'],
                    item_name=row['item_name'],
                    category=row['category'],
                    price=row['price'],
                    store=row['store'],
                    similarity_score=0.9,  # High similarity for same product
                    last_seen=row['last_seen'],
                    potential_saving=0,  # Will be calculated by caller
                    metadata={
                        'brand': row.get('brand'),
                        'location': row.get('location'),
                        'search_type': 'store_alternative'
                    }
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Store alternatives search failed: {e}")
            return []
        finally:
            if 'conn' in locals():
                await conn.close()

    async def find_category_alternatives(
        self,
        spending_category: str,
        user_id: str,
        time_period: str = "3months"
    ) -> List[Dict[str, Any]]:
        """
        Find alternative categories that could satisfy similar needs.
        E.g., "fast_food" alternatives might include "groceries" + "meal_prep"
        """
        try:
            conn = await self._get_connection()
            
            # Get user's spending in the target category
            period_map = {
                "1month": "1 month",
                "3months": "3 months", 
                "6months": "6 months"
            }
            
            query = """
            WITH user_category_spending AS (
                SELECT 
                    category,
                    AVG(amount) as avg_amount,
                    COUNT(*) as frequency,
                    SUM(amount) as total_spent
                FROM user_purchases 
                WHERE user_id = $1 
                AND purchase_date > NOW() - INTERVAL %s
                GROUP BY category
            ),
            target_spending AS (
                SELECT * FROM user_category_spending WHERE category = $2
            ),
            alternative_categories AS (
                SELECT 
                    ucs.category,
                    ucs.avg_amount,
                    ucs.frequency,
                    ucs.total_spent,
                    -- Calculate potential substitution score
                    CASE 
                        WHEN ucs.avg_amount < ts.avg_amount THEN 
                            (ts.avg_amount - ucs.avg_amount) / ts.avg_amount
                        ELSE 0 
                    END as potential_savings_ratio
                FROM user_category_spending ucs, target_spending ts
                WHERE ucs.category != ts.category
                AND ucs.category IN (
                    -- Categories that could be alternatives based on business logic
                    SELECT alternative_category 
                    FROM category_alternatives 
                    WHERE primary_category = $2
                )
            )
            SELECT * FROM alternative_categories 
            WHERE potential_savings_ratio > 0
            ORDER BY potential_savings_ratio DESC
            LIMIT 5
            """ % period_map[time_period]
            
            rows = await conn.fetch(query, user_id, spending_category)
            
            alternatives = []
            for row in rows:
                alternatives.append({
                    'alternative_category': row['category'],
                    'avg_amount': float(row['avg_amount']),
                    'frequency': row['frequency'],
                    'potential_monthly_savings': float(row['potential_savings_ratio']) * float(row['total_spent']) / 3,
                    'substitution_feasibility': self._calculate_substitution_feasibility(
                        spending_category, row['category']
                    )
                })
            
            return alternatives
            
        except Exception as e:
            self.logger.error(f"Category alternatives search failed: {e}")
            return []
        finally:
            if 'conn' in locals():
                await conn.close()

    def _calculate_substitution_feasibility(self, original_category: str, alternative_category: str) -> float:
        """
        Calculate how feasible it is to substitute one category for another.
        This would be enhanced with ML models in production.
        """
        # Simple rule-based feasibility scoring
        substitution_rules = {
            ('restaurant', 'groceries'): 0.8,
            ('fast_food', 'groceries'): 0.9,
            ('coffee_shop', 'groceries'): 0.7,
            ('delivery', 'groceries'): 0.8,
            ('brand_name', 'generic'): 0.9,
            ('premium_gas', 'regular_gas'): 0.8,
        }
        
        return substitution_rules.get((original_category, alternative_category), 0.5)

    async def update_product_embeddings(
        self, 
        products: List[Dict[str, Any]]
    ) -> bool:
        """
        Update or insert product embeddings in the vector database.
        This would typically be called by a data pipeline.
        """
        try:
            conn = await self._get_connection()
            
            for product in products:
                await conn.execute("""
                    INSERT INTO product_index (
                        item_id, item_name, category, price, store, brand, 
                        embedding, location, availability, last_seen
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                    ON CONFLICT (item_id) DO UPDATE SET
                        price = EXCLUDED.price,
                        embedding = EXCLUDED.embedding,
                        availability = EXCLUDED.availability,
                        last_seen = NOW()
                """,
                    product['item_id'],
                    product['item_name'], 
                    product['category'],
                    product['price'],
                    product['store'],
                    product.get('brand'),
                    np.array(product['embedding'], dtype=np.float32),
                    product.get('location'),
                    product.get('availability', 'available')
                )
            
            self.logger.info(f"Updated {len(products)} product embeddings")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update embeddings: {e}")
            return False
        finally:
            if 'conn' in locals():
                await conn.close()

    async def get_search_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about search effectiveness for a user"""
        try:
            conn = await self._get_connection()
            
            # Get stats on recommendations and their effectiveness
            stats_query = """
            SELECT 
                COUNT(*) as total_searches,
                AVG(similarity_score) as avg_similarity,
                AVG(potential_saving) as avg_savings,
                COUNT(DISTINCT category) as categories_searched
            FROM search_history 
            WHERE user_id = $1 
            AND search_date > NOW() - INTERVAL '30 days'
            """
            
            row = await conn.fetchrow(stats_query, user_id)
            
            return {
                'total_searches': row['total_searches'] or 0,
                'avg_similarity': float(row['avg_similarity'] or 0),
                'avg_potential_savings': float(row['avg_savings'] or 0),
                'categories_searched': row['categories_searched'] or 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get search stats: {e}")
            return {}
        finally:
            if 'conn' in locals():
                await conn.close()


# ===== DATABASE SCHEMA (for reference) =====
"""
-- Product index table for vector search
CREATE TABLE product_index (
    item_id VARCHAR PRIMARY KEY,
    item_name TEXT NOT NULL,
    category VARCHAR NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    store VARCHAR NOT NULL,
    brand VARCHAR,
    embedding vector(384),  -- Assuming 384-dim embeddings
    location JSONB,  -- Store location info
    availability VARCHAR DEFAULT 'available',
    last_seen TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Vector similarity index
CREATE INDEX ON product_index USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Regular indexes
CREATE INDEX idx_product_category ON product_index(category);
CREATE INDEX idx_product_price ON product_index(price);
CREATE INDEX idx_product_store ON product_index(store);
CREATE INDEX idx_product_availability ON product_index(availability);

-- User purchases table
CREATE TABLE user_purchases (
    purchase_id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    product_id VARCHAR,
    amount DECIMAL(10,2) NOT NULL,
    category VARCHAR NOT NULL,
    purchase_date TIMESTAMP NOT NULL,
    store VARCHAR,
    FOREIGN KEY (product_id) REFERENCES product_index(item_id)
);

-- Category alternatives mapping
CREATE TABLE category_alternatives (
    primary_category VARCHAR,
    alternative_category VARCHAR,
    substitution_score FLOAT DEFAULT 0.5,
    PRIMARY KEY (primary_category, alternative_category)
);

-- Search history for analytics
CREATE TABLE search_history (
    search_id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    original_item JSONB,
    search_results JSONB,
    search_date TIMESTAMP DEFAULT NOW()
);
"""