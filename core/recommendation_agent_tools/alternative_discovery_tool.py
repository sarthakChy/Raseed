import json
import logging
from typing import Dict, Any, Optional
from vertexai.generative_models import GenerativeModel
from core.recommendation_agent_tools.tools_instructions import alternatives_synthesis_instruction


class AlternativeDiscoveryTool:
    """
    Tool for discovering cheaper alternatives using vector similarity search.
    Finds substitute products/services with better prices.
    """
    
    def __init__(self, project_id: str, location: str, logger: logging.Logger):
        """
        Initialize the Alternative Discovery Tool.
        
        Args:
            project_id: Google Cloud project ID
            location: Vertex AI location
            logger: Logger instance for this tool
        """
        self.project_id = project_id
        self.location = location
        self.logger = logger
        self.model_name = "gemini-2.0-flash-001"
        
        # Initialize synthesis model with specific instructions
        self.synthesis_model = GenerativeModel(
            self.model_name,
            system_instruction=alternatives_synthesis_instruction
        )
        
        self.logger.info("AlternativeDiscoveryTool initialized")
    
    async def discover_alternatives(
        self,
        user_id: str,
        item_description: str,
        item_category: str,
        price: float,
        item_embedding: str,
        location_preference: Optional[str] = None,
        db_connector = None
    ) -> Dict[str, Any]:
        """
        Discover cheaper alternatives for a given item using vector similarity search.
        
        Args:
            user_id: User identifier
            item_description: Description of the item to find alternatives for
            item_category: Category of the item
            price: Current price of the item
            item_embedding: Vector embedding of the item for similarity search
            location_preference: Optional location preference
            db_connector: Database connection (injected by agent)
            
        Returns:
            Dictionary containing alternative recommendations
        """
        self.logger.info(f"Discovering alternatives for user: {user_id}, item: {item_description}")
        
        try:
            # Ensure embedding is in correct format
            if isinstance(item_embedding, str):
                embedding_str = item_embedding
            else:
                embedding_str = '[' + ','.join(map(str, item_embedding)) + ']'
            
            # Vector similarity search query
            query = """
                SELECT DISTINCT ON (ti.name, ti.total_price, m.name)
                    ti.name AS item_name,
                    ti.total_price AS alternative_price,
                    m.name AS merchant_name,
                    (1 - (ti.item_embedding <=> $1::vector(768))) AS similarity_score -- Cosine similarity
                FROM
                    transaction_items ti
                JOIN
                    transactions t ON ti.transaction_id = t.transaction_id
                JOIN
                    merchants m ON t.merchant_id = m.merchant_id
                WHERE
                    t.category ILIKE $2          -- Filter by category (case-insensitive)
                    AND ti.unit_price < $3   
                    AND t.user_id != $4
                    AND ti.item_embedding IS NOT NULL -- Ensure the item has an embedding
                ORDER BY
                    ti.name, ti.total_price, m.name,
                    similarity_score DESC,        -- Most similar items first
                    alternative_price ASC         -- Then by lowest price
                LIMIT 10; 
            """
            
            # Set parameters
            params = [embedding_str, item_category, price, user_id]
            
            # Execute query
            if db_connector:
                alternatives_result = await db_connector.execute_query(query, params)
                alternatives_found = getattr(alternatives_result, "data", [])
            else:
                # Mock data for testing
                alternatives_found = []
            
            if not alternatives_found:
                return {
                    "success": True,
                    "alternatives_found": [],
                    "raw_data_summary": f"No cheaper alternatives found for user {user_id}",
                    "potential_areas_for_recommendation": ["No similar but cheaper items found in category"],
                    "search_metadata": {
                        "item_description": item_description,
                        "category": item_category,
                        "original_price": price,
                        "alternatives_count": 0
                    }
                }

            self.logger.info(f"Found {len(alternatives_found)} alternatives for {item_description}")

            # Prepare data for synthesis
            analysis_data = {
                "original_item": {
                    "description": item_description,
                    "category": item_category,
                    "price": price
                },
                "alternatives": alternatives_found,
                "location_preference": location_preference
            }

            # Create synthesis prompt
            prompt = f"""
            High-cost item:
            ```json
            {json.dumps(analysis_data["original_item"], indent=2)}
            ```
            
            Alternatives found:
            ```json
            {json.dumps(alternatives_found, indent=2, default=str)}
            ```
            
            Location preference: {location_preference or 'Not specified'}
            
            Please analyze these alternatives and provide structured recommendations following the system instructions.
            """

            # Generate synthesis
            response = await self.synthesis_model.generate_content_async(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )

            result = json.loads(response.text)
            result["success"] = True
            result["search_metadata"] = {
                "item_description": item_description,
                "category": item_category,
                "original_price": price,
                "alternatives_count": len(alternatives_found)
            }
            return result

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error in alternative discovery: {e}")
            return {
                "success": False,
                "error": str(e),
                "alternatives_found": [],
                "raw_data_summary": "Failed to parse LLM response",
                "potential_areas_for_recommendation": ["Analysis unavailable due to processing error"]
            }

        except Exception as e:
            self.logger.error(f"Unexpected error during alternative discovery: {e}")
            return {
                "success": False,
                "error": str(e),
                "alternatives_found": [],
                "raw_data_summary": "Error occurred during recommendation discovery",
                "potential_areas_for_recommendation": ["System error"]
            }
    
    def _calculate_savings_potential(self, original_price: float, alternatives: list) -> Dict[str, Any]:
        """
        Calculate potential savings from alternatives.
        
        Args:
            original_price: Original item price
            alternatives: List of alternative items with prices
            
        Returns:
            Savings analysis dictionary
        """
        if not alternatives:
            return {"max_savings": 0, "avg_savings": 0, "savings_percentage": 0}
        
        try:
            alternative_prices = [float(alt.get('alternative_price', original_price)) for alt in alternatives]
            
            min_price = min(alternative_prices)
            avg_price = sum(alternative_prices) / len(alternative_prices)
            
            max_savings = original_price - min_price
            avg_savings = original_price - avg_price
            max_savings_percentage = (max_savings / original_price) * 100 if original_price > 0 else 0
            
            return {
                "max_savings": max_savings,
                "avg_savings": avg_savings,
                "savings_percentage": max_savings_percentage,
                "best_alternative_price": min_price,
                "alternatives_count": len(alternatives)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating savings potential: {e}")
            return {"error": str(e)}