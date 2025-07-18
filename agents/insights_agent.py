import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
import pandas as pd
import numpy as np
from models.pydantic_models import *


class PurchaseInsightsAgent:
    def __init__(self, project_id: str, location: str = "us-central1"):
        """
        Initialize the Vertex AI agent for purchase insights
        
        Args:
            project_id: Google Cloud project ID
            location: Vertex AI location
        """
        self.project_id = project_id
        self.location = location
        vertexai.init(project=project_id, location=location)
        
        # Initialize Gemini model
        self.model = GenerativeModel("gemini-2.0-flash")
        
    def preprocess_purchase_data(self, purchase_data: List[Dict]) -> pd.DataFrame:
        """
        Convert JSON purchase data to structured DataFrame for analysis
        """
        processed_data = []
        
        for receipt in purchase_data:
            receipt_date = datetime.fromisoformat(receipt['uploaded_at'].replace('Z', '+00:00'))
            
            for item in receipt['items']:
                processed_data.append({
                    'receipt_id': receipt['id'],
                    'date': receipt_date.date(),
                    'datetime': receipt_date,
                    'merchant': receipt['merchantName'],
                    'item_name': item['name'],
                    'price': item['price'],
                    'total_receipt': receipt['total'],
                    'user_id': receipt['userId']
                })
        
        return pd.DataFrame(processed_data)
    
    def generate_spending_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate basic spending insights from the data
        """
        insights = {}
        
        # Daily spending patterns
        daily_spending = df.groupby('date')['price'].sum().reset_index()
        insights['daily_avg_spending'] = daily_spending['price'].mean()
        insights['daily_max_spending'] = daily_spending['price'].max()
        insights['daily_min_spending'] = daily_spending['price'].min()
        
        # Most frequent items
        item_frequency = df['item_name'].value_counts()
        insights['most_frequent_items'] = item_frequency.head(5).to_dict()
        
        # Category analysis (basic categorization)
        category_mapping = {
            'MILK': 'Dairy',
            'BREAD': 'Bakery',
            'EGGS': 'Dairy',
            'RICE': 'Grains',
            'TOMATOES': 'Vegetables',
            'POTATOES': 'Vegetables',
            'PASTA': 'Grains',
            'CHEESE': 'Dairy',
            'YOGURT': 'Dairy',
            'TOOTHPASTE': 'Personal Care',
            'TOOTHBRUSH': 'Personal Care',
            'FACE WASH': 'Personal Care',
            'ICE CREAM': 'Desserts',
            'COOKIES': 'Snacks',
            'CHIPS': 'Snacks'
        }
        
        df['category'] = df['item_name'].apply(
            lambda x: next((cat for key, cat in category_mapping.items() if key in x), 'Other')
        )
        
        category_spending = df.groupby('category')['price'].sum().to_dict()
        insights['category_spending'] = category_spending
        
        # Shopping frequency
        shopping_days = df['date'].nunique()
        total_days = (df['date'].max() - df['date'].min()).days + 1
        insights['shopping_frequency'] = shopping_days / total_days
        
        # Price trend analysis
        price_trends = {}
        for item in df['item_name'].unique():
            item_data = df[df['item_name'] == item]
            if len(item_data) > 1:
                price_trends[item] = {
                    'avg_price': item_data['price'].mean(),
                    'price_volatility': item_data['price'].std(),
                    'purchase_frequency': len(item_data)
                }
        insights['price_trends'] = price_trends
        
        return insights
    
    def create_gemini_prompt(self, insights: Dict[str, Any], raw_data: List[Dict]) -> str:
        """
        Create a comprehensive prompt for Gemini to analyze purchase patterns
        """
        prompt = f"""
        You are an AI agent specialized in analyzing consumer purchase behavior to provide actionable insights about daily usage patterns and spending habits.

        PURCHASE DATA SUMMARY:
        - Daily Average Spending: ₹{insights['daily_avg_spending']:.2f}
        - Shopping Frequency: {insights['shopping_frequency']:.2%} of days
        - Most Frequent Items: {list(insights['most_frequent_items'].keys())[:3]}
        - Category Spending Distribution: {insights['category_spending']}

        RAW PURCHASE DATA:
        {json.dumps(raw_data, indent=2)}

        ANALYSIS REQUIREMENTS:
        Please provide a comprehensive analysis in the following JSON format:

        {{
            "daily_usage_patterns": ["pattern1", "pattern2", "pattern3"],
            "spending_efficiency": [
                {{
                    "insight_type": "bulk_buying",
                    "description": "detailed description",
                    "potential_savings": 50.0,
                    "action_items": ["action1", "action2"]
                }}
            ],
            "consumption_trends": [
                {{
                    "trend_name": "increasing_dairy_consumption",
                    "description": "detailed description",
                    "time_period": "weekly",
                    "trend_strength": 0.8
                }}
            ],
            "budget_recommendations": [
                {{
                    "category": "Dairy",
                    "current_spending": 45.0,
                    "recommended_budget": 40.0,
                    "potential_savings": 5.0,
                    "reasoning": "detailed reasoning"
                }}
            ],
            "health_insights": [
                {{
                    "category": "Dairy",
                    "health_score": 8,
                    "recommendation": "recommendation text",
                    "items_in_category": ["MILK 1L", "EGGS 12 PACK"]
                }}
            ],
            "predictive_recommendations": [
                {{
                    "item_name": "MILK 1L",
                    "predicted_next_purchase": "2025-07-22",
                    "confidence": 0.9,
                    "reasoning": "detailed reasoning"
                }}
            ],
            "key_findings": ["finding1", "finding2", "finding3"],
            "cost_savings_opportunities": ["opportunity1", "opportunity2"]
        }}

        Provide ONLY the JSON response with no additional text or formatting.
        """
        return prompt
    
    def get_gemini_insights(self, prompt: str) -> Dict[str, Any]:
        """
        Get insights from Gemini model and parse JSON response
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.3,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=2048,
                )
            )
            
            # Parse JSON response
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            return json.loads(response_text)
        except Exception as e:
            # Return default structure if parsing fails
            return {
                "daily_usage_patterns": [f"Error generating insights: {str(e)}"],
                "spending_efficiency": [],
                "consumption_trends": [],
                "budget_recommendations": [],
                "health_insights": [],
                "predictive_recommendations": [],
                "key_findings": ["Analysis failed - please try again"],
                "cost_savings_opportunities": []
            }
    
    def convert_to_pydantic_models(self, insights: Dict[str, Any], df: pd.DataFrame, raw_data: List[Dict], ai_insights: Dict[str, Any]) -> PurchaseInsightsOutput:
        """
        Convert raw insights to Pydantic models
        """
        # Convert basic insights
        most_frequent_items = [
            ItemFrequency(
                item_name=item,
                frequency=freq,
                avg_price=df[df['item_name'] == item]['price'].mean()
            ) for item, freq in insights['most_frequent_items'].items()
        ]
        
        total_spending = sum(insights['category_spending'].values())
        category_spending = [
            CategorySpending(
                category=cat,
                total_spent=amount,
                percentage=(amount / total_spending) * 100 if total_spending > 0 else 0
            ) for cat, amount in insights['category_spending'].items()
        ]
        
        price_trends = []
        for item, trend_data in insights['price_trends'].items():
            # Simple trend analysis based on price volatility
            if trend_data['price_volatility'] < 0.5:
                trend_direction = "stable"
            elif df[df['item_name'] == item]['price'].iloc[-1] > df[df['item_name'] == item]['price'].iloc[0]:
                trend_direction = "increasing"
            else:
                trend_direction = "decreasing"
            
            price_trends.append(PriceTrend(
                item_name=item,
                avg_price=trend_data['avg_price'],
                price_volatility=trend_data['price_volatility'],
                purchase_frequency=trend_data['purchase_frequency'],
                trend_direction=trend_direction
            ))
        
        basic_insights = BasicInsights(
            daily_avg_spending=insights['daily_avg_spending'],
            daily_max_spending=insights['daily_max_spending'],
            daily_min_spending=insights['daily_min_spending'],
            most_frequent_items=most_frequent_items,
            category_spending=category_spending,
            shopping_frequency=insights['shopping_frequency'],
            price_trends=price_trends
        )
        
        # Convert AI insights
        ai_analysis = AIAnalysis(
            daily_usage_patterns=ai_insights.get('daily_usage_patterns', []),
            spending_efficiency=[
                SpendingEfficiencyInsight(**item) for item in ai_insights.get('spending_efficiency', [])
            ],
            consumption_trends=[
                ConsumptionTrend(**item) for item in ai_insights.get('consumption_trends', [])
            ],
            budget_recommendations=[
                BudgetRecommendation(**item) for item in ai_insights.get('budget_recommendations', [])
            ],
            health_insights=[
                HealthInsight(**item) for item in ai_insights.get('health_insights', [])
            ],
            predictive_recommendations=[
                PredictiveRecommendation(**item) for item in ai_insights.get('predictive_recommendations', [])
            ],
            key_findings=ai_insights.get('key_findings', []),
            cost_savings_opportunities=ai_insights.get('cost_savings_opportunities', [])
        )
        
        # Generate daily patterns
        daily_patterns = []
        daily_data = df.groupby('date').agg({
            'price': 'sum',
            'item_name': 'count',
            'receipt_id': 'nunique'
        }).reset_index()
        
        for _, row in daily_data.iterrows():
            daily_patterns.append(DailySpendingPattern(
                date=row['date'].isoformat(),
                total_spent=row['price'],
                items_bought=row['item_name'],
                receipts_count=row['receipt_id']
            ))
        
        # Data summary
        data_summary = DataSummary(
            total_receipts=len(raw_data),
            date_range=f"{df['date'].min()} to {df['date'].max()}",
            total_spending=df['price'].sum(),
            unique_items=df['item_name'].nunique(),
            shopping_frequency=insights['shopping_frequency'],
            avg_daily_spending=insights['daily_avg_spending']
        )
        
        return PurchaseInsightsOutput(
            basic_insights=basic_insights,
            ai_analysis=ai_analysis,
            data_summary=data_summary,
            daily_patterns=daily_patterns
        )
    
    def analyze_purchases(self, purchase_data: List[Dict]) -> PurchaseInsightsOutput:
        """
        Main method to analyze purchase data and generate structured insights
        """
        # Preprocess data
        df = self.preprocess_purchase_data(purchase_data)
        
        # Generate basic insights
        insights = self.generate_spending_insights(df)
        
        # Create Gemini prompt
        prompt = self.create_gemini_prompt(insights, purchase_data)
        
        # Get AI insights
        ai_insights = self.get_gemini_insights(prompt)
        
        # Convert to Pydantic models
        structured_output = self.convert_to_pydantic_models(insights, df, purchase_data, ai_insights)
        
        return structured_output