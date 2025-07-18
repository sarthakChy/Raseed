from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ItemFrequency(BaseModel):
    item_name: str
    frequency: int
    avg_price: float

class CategorySpending(BaseModel):
    category: str
    total_spent: float
    percentage: float

class PriceTrend(BaseModel):
    item_name: str
    avg_price: float
    price_volatility: float
    purchase_frequency: int
    trend_direction: str  # "increasing", "decreasing", "stable"

class DailySpendingPattern(BaseModel):
    date: str
    total_spent: float
    items_bought: int
    receipts_count: int

class BudgetRecommendation(BaseModel):
    category: str
    current_spending: float
    recommended_budget: float
    potential_savings: float
    reasoning: str

class HealthInsight(BaseModel):
    category: str
    health_score: int = Field(ge=1, le=10)  # 1-10 scale
    recommendation: str
    items_in_category: List[str]

class PredictiveRecommendation(BaseModel):
    item_name: str
    predicted_next_purchase: str  # ISO date
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

class SpendingEfficiencyInsight(BaseModel):
    insight_type: str
    description: str
    potential_savings: float
    action_items: List[str]

class ConsumptionTrend(BaseModel):
    trend_name: str
    description: str
    time_period: str
    trend_strength: float = Field(ge=0.0, le=1.0)

class DataSummary(BaseModel):
    total_receipts: int
    date_range: str
    total_spending: float
    unique_items: int
    shopping_frequency: float
    avg_daily_spending: float

class BasicInsights(BaseModel):
    daily_avg_spending: float
    daily_max_spending: float
    daily_min_spending: float
    most_frequent_items: List[ItemFrequency]
    category_spending: List[CategorySpending]
    shopping_frequency: float
    price_trends: List[PriceTrend]

class AIAnalysis(BaseModel):
    daily_usage_patterns: List[str]
    spending_efficiency: List[SpendingEfficiencyInsight]
    consumption_trends: List[ConsumptionTrend]
    budget_recommendations: List[BudgetRecommendation]
    health_insights: List[HealthInsight]
    predictive_recommendations: List[PredictiveRecommendation]
    key_findings: List[str]
    cost_savings_opportunities: List[str]

class PurchaseInsightsOutput(BaseModel):
    basic_insights: BasicInsights
    ai_analysis: AIAnalysis
    data_summary: DataSummary
    daily_patterns: List[DailySpendingPattern]