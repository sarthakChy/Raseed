import logging
import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import uuid
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from core.base_agent_tools.database_connector import DatabaseConnector
from core.base_agent_tools.config_manager import AgentConfig
from core.base_agent_tools.error_handler import ErrorHandler


class TrendType(Enum):
    """Types of trends that can be predicted."""
    SPENDING_INCREASE = "spending_increase"
    SPENDING_DECREASE = "spending_decrease"
    SEASONAL_PATTERN = "seasonal_pattern"
    BUDGET_CHALLENGE = "budget_challenge"
    EMERGING_CATEGORY = "emerging_category"
    BEHAVIOR_SHIFT = "behavior_shift"
    MERCHANT_LOYALTY = "merchant_loyalty"
    PAYMENT_METHOD_SHIFT = "payment_method_shift"


class TrendSeverity(Enum):
    """Severity levels for predicted trends."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PredictionConfidence(Enum):
    """Confidence levels for predictions."""
    LOW = "low"        # < 60%
    MEDIUM = "medium"  # 60-80%
    HIGH = "high"      # 80-95%
    VERY_HIGH = "very_high"  # > 95%


@dataclass
class TrendPrediction:
    """Structure for a trend prediction."""
    prediction_id: str
    user_id: str
    trend_type: TrendType
    category: str
    severity: TrendSeverity
    confidence: PredictionConfidence
    confidence_score: float
    
    # Prediction details
    title: str
    description: str
    predicted_impact: Dict[str, Any]
    timeline: Dict[str, Any]  # when_starts, when_peaks, duration
    
    # Supporting data
    historical_data: Dict[str, Any]
    statistical_evidence: Dict[str, Any]
    model_metrics: Dict[str, Any]
    
    # Recommendations
    preparation_actions: List[Dict[str, Any]]
    mitigation_strategies: List[Dict[str, Any]]
    opportunities: List[Dict[str, Any]]
    
    # Metadata
    prediction_date: datetime
    valid_until: datetime
    model_used: str
    data_quality_score: float


@dataclass
class SeasonalPattern:
    """Structure for seasonal spending patterns."""
    pattern_id: str
    category: str
    seasonal_factors: Dict[int, float]  # month -> multiplier
    peak_months: List[int]
    low_months: List[int]
    year_over_year_growth: float
    confidence_score: float
    last_updated: datetime


class TrendPredictionTool:
    """
    Advanced trend prediction tool for financial behavior forecasting.
    
    This tool analyzes historical spending patterns to predict future trends,
    seasonal changes, budget challenges, and emerging behavioral shifts.
    """
    
    def __init__(
        self,
        db_connector: Optional[DatabaseConnector] = None,
        config: Optional[AgentConfig] = None
    ):
        # Initialize configuration
        self.config = config or AgentConfig.from_env()
        self.db_connector = db_connector or DatabaseConnector()
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Error handling
        self.error_handler = ErrorHandler(self.logger)
        
        # Model parameters
        self.min_data_points = 30  # Minimum transactions needed for predictions
        self.seasonal_lookback_months = 24  # Look back 2 years for seasonal patterns
        self.trend_lookback_months = 6     # Look back 6 months for trend analysis
        self.prediction_horizon_days = 90  # Predict 3 months ahead
        
        # Statistical thresholds
        self.significance_threshold = 0.05
        self.trend_strength_threshold = 0.3
        self.seasonal_strength_threshold = 0.4
        
        # Model cache
        self.model_cache = {}
        self.pattern_cache = {}
        
        self.logger.info("Trend Prediction Tool initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up structured logging."""
        logger = logging.getLogger("financial_agent.trend_prediction_tool")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def predict_trends(
        self,
        user_id: str,
        categories: Optional[List[str]] = None,
        prediction_types: Optional[List[TrendType]] = None,
        horizon_days: Optional[int] = None
    ) -> List[TrendPrediction]:
        """
        Generate comprehensive trend predictions for a user.
        
        Args:
            user_id: Target user ID
            categories: Specific categories to analyze (None for all)
            prediction_types: Types of predictions to generate (None for all)
            horizon_days: Prediction horizon (default: 90 days)
            
        Returns:
            List of trend predictions
        """
        try:
            self.logger.info(f"Generating trend predictions for user {user_id}")
            
            horizon_days = horizon_days or self.prediction_horizon_days
            
            # Get historical transaction data
            historical_data = await self._get_historical_data(user_id, categories)
            
            if not self._validate_data_quality(historical_data):
                self.logger.warning(f"Insufficient data quality for user {user_id}")
                return []
            
            predictions = []
            
            # Generate different types of predictions
            prediction_types = prediction_types or list(TrendType)
            
            for trend_type in prediction_types:
                if trend_type == TrendType.SPENDING_INCREASE:
                    predictions.extend(await self._predict_spending_increases(
                        user_id, historical_data, horizon_days
                    ))
                elif trend_type == TrendType.SPENDING_DECREASE:
                    predictions.extend(await self._predict_spending_decreases(
                        user_id, historical_data, horizon_days
                    ))
                elif trend_type == TrendType.SEASONAL_PATTERN:
                    predictions.extend(await self._predict_seasonal_changes(
                        user_id, historical_data, horizon_days
                    ))
                elif trend_type == TrendType.BUDGET_CHALLENGE:
                    predictions.extend(await self._predict_budget_challenges(
                        user_id, historical_data, horizon_days
                    ))
                elif trend_type == TrendType.EMERGING_CATEGORY:
                    predictions.extend(await self._predict_emerging_categories(
                        user_id, historical_data, horizon_days
                    ))
                elif trend_type == TrendType.BEHAVIOR_SHIFT:
                    predictions.extend(await self._predict_behavior_shifts(
                        user_id, historical_data, horizon_days
                    ))
                elif trend_type == TrendType.MERCHANT_LOYALTY:
                    predictions.extend(await self._predict_merchant_changes(
                        user_id, historical_data, horizon_days
                    ))
                elif trend_type == TrendType.PAYMENT_METHOD_SHIFT:
                    predictions.extend(await self._predict_payment_method_shifts(
                        user_id, historical_data, horizon_days
                    ))
            
            # Filter and prioritize predictions
            filtered_predictions = await self._filter_and_rank_predictions(predictions)
            
            # Store predictions for future reference
            await self._store_predictions(filtered_predictions)
            
            self.logger.info(f"Generated {len(filtered_predictions)} trend predictions for user {user_id}")
            return filtered_predictions
            
        except Exception as e:
            self.logger.error(f"Error generating trend predictions: {str(e)}")
            return await self.error_handler.handle_error(e, {
                "operation": "predict_trends",
                "user_id": user_id,
                "categories": categories
            })
    
    async def _get_historical_data(self, user_id: str, categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get historical transaction data for analysis."""
        try:
            # Get lookback period
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=self.seasonal_lookback_months * 30)
            
            # Build query
            query = """
                SELECT 
                    t.transaction_id,
                    t.amount,
                    t.transaction_date,
                    t.category,
                    t.subcategory,
                    t.merchant_id,
                    t.payment_method,
                    m.name as merchant_name,
                    m.normalized_name as merchant_normalized,
                    EXTRACT(YEAR FROM t.transaction_date) as year,
                    EXTRACT(MONTH FROM t.transaction_date) as month,
                    EXTRACT(DOW FROM t.transaction_date) as day_of_week,
                    EXTRACT(DAY FROM t.transaction_date) as day_of_month
                FROM transactions t
                LEFT JOIN merchants m ON t.merchant_id = m.merchant_id
                WHERE t.user_id = %s 
                    AND t.transaction_date >= %s 
                    AND t.transaction_date <= %s
                    AND t.deleted_at IS NULL
                    AND t.amount > 0
            """
            
            params = [user_id, start_date, end_date]
            
            if categories:
                query += " AND t.category = ANY(%s)"
                params.append(categories)
            
            query += " ORDER BY t.transaction_date"
            
            # Execute query
            result = await self.db_connector.execute_query(query, params)
            transactions = result.get('data', [])
            
            if not transactions:
                return {}
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(transactions)
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])
            df['amount'] = df['amount'].astype(float)
            
            # Get user budget limits
            budget_query = """
                SELECT category, limit_amount, period_type
                FROM budget_limits 
                WHERE user_id = %s 
                    AND (effective_to IS NULL OR effective_to >= CURRENT_DATE)
                    AND effective_from <= CURRENT_DATE
            """
            budget_result = await self.db_connector.execute_query(budget_query, [user_id])
            budget_limits = {row['category']: row['limit_amount'] for row in budget_result.get('data', [])}
            
            # Get user financial goals
            goals_query = """
                SELECT goal_id, goal_type, category, target_amount, current_amount, 
                       target_date, status, progress_percentage
                FROM financial_goals 
                WHERE user_id = %s AND status = 'active'
            """
            goals_result = await self.db_connector.execute_query(goals_query, [user_id])
            financial_goals = goals_result.get('data', [])
            
            return {
                'transactions': df,
                'budget_limits': budget_limits,
                'financial_goals': financial_goals,
                'data_quality': {
                    'total_transactions': len(df),
                    'date_range': {
                        'start': df['transaction_date'].min(),
                        'end': df['transaction_date'].max()
                    },
                    'categories': df['category'].nunique(),
                    'merchants': df['merchant_id'].nunique()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting historical data: {str(e)}")
            return {}
    
    def _validate_data_quality(self, historical_data: Dict[str, Any]) -> bool:
        """Validate if historical data is sufficient for predictions."""
        if not historical_data or 'transactions' not in historical_data:
            return False
        
        df = historical_data['transactions']
        data_quality = historical_data.get('data_quality', {})
        
        # Check minimum data requirements
        if len(df) < self.min_data_points:
            self.logger.warning(f"Insufficient transactions: {len(df)} < {self.min_data_points}")
            return False
        
        # Check date range coverage
        date_range = (df['transaction_date'].max() - df['transaction_date'].min()).days
        if date_range < 60:  # Need at least 2 months of data
            self.logger.warning(f"Insufficient date range: {date_range} days")
            return False
        
        # Check data distribution
        if data_quality.get('categories', 0) < 2:
            self.logger.warning("Insufficient category diversity")
            return False
        
        return True
    
    async def _predict_spending_increases(
        self, 
        user_id: str, 
        historical_data: Dict[str, Any], 
        horizon_days: int
    ) -> List[TrendPrediction]:
        """Predict categories with increasing spending trends."""
        predictions = []
        
        try:
            df = historical_data['transactions']
            budget_limits = historical_data.get('budget_limits', {})
            
            # Group by category and month
            monthly_spending = df.groupby(['category', df['transaction_date'].dt.to_period('M')])['amount'].sum().reset_index()
            monthly_spending['month_num'] = monthly_spending['transaction_date'].dt.month
            monthly_spending['year_month'] = monthly_spending['transaction_date'].astype(str)
            
            for category in df['category'].unique():
                category_data = monthly_spending[monthly_spending['category'] == category].copy()
                
                if len(category_data) < 3:  # Need at least 3 months of data
                    continue
                
                # Prepare data for trend analysis
                category_data = category_data.sort_values('transaction_date')
                category_data['period_index'] = range(len(category_data))
                
                # Fit linear regression to detect trend
                X = category_data['period_index'].values.reshape(-1, 1)
                y = category_data['amount'].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Calculate trend strength and significance
                slope = model.coef_[0]
                r_squared = model.score(X, y)
                
                # Statistical significance test
                n = len(category_data)
                if n > 2:
                    y_pred = model.predict(X)
                    residuals = y - y_pred
                    mse = np.mean(residuals ** 2)
                    se_slope = np.sqrt(mse / np.sum((X.flatten() - np.mean(X)) ** 2))
                    t_stat = slope / se_slope if se_slope > 0 else 0
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2)) if n > 2 else 1
                else:
                    p_value = 1
                
                # Only consider significant increasing trends
                if slope > 0 and p_value < self.significance_threshold and r_squared > self.trend_strength_threshold:
                    # Project future spending
                    future_periods = int(horizon_days / 30)  # Convert to months
                    future_X = np.array([[len(category_data) + i] for i in range(1, future_periods + 1)])
                    projected_amounts = model.predict(future_X)
                    
                    # Calculate impact metrics
                    current_avg = category_data['amount'].tail(3).mean()
                    projected_avg = projected_amounts.mean()
                    increase_percentage = ((projected_avg - current_avg) / current_avg) * 100
                    
                    # Determine severity
                    budget_limit = budget_limits.get(category, 0)
                    severity = self._calculate_trend_severity(
                        increase_percentage, projected_avg, budget_limit, slope
                    )
                    
                    # Determine confidence
                    confidence_score = min(r_squared * (1 - p_value), 1.0)
                    confidence = self._get_confidence_level(confidence_score)
                    
                    # Generate prediction
                    prediction = TrendPrediction(
                        prediction_id=str(uuid.uuid4()),
                        user_id=user_id,
                        trend_type=TrendType.SPENDING_INCREASE,
                        category=category,
                        severity=severity,
                        confidence=confidence,
                        confidence_score=confidence_score,
                        title=f"Increasing {category.title()} Spending Trend",
                        description=f"Your {category} spending is trending upward with a {increase_percentage:.1f}% projected increase over the next {horizon_days} days.",
                        predicted_impact={
                            "current_monthly_avg": float(current_avg),
                            "projected_monthly_avg": float(projected_avg),
                            "increase_percentage": float(increase_percentage),
                            "projected_total_increase": float(sum(projected_amounts) - current_avg * future_periods),
                            "budget_impact": float(projected_avg / budget_limit) if budget_limit > 0 else None
                        },
                        timeline={
                            "trend_start": category_data['transaction_date'].iloc[0].isoformat(),
                            "projection_start": datetime.now().isoformat(),
                            "projection_end": (datetime.now() + timedelta(days=horizon_days)).isoformat(),
                            "peak_expected": None  # Could be calculated based on seasonal patterns
                        },
                        historical_data={
                            "monthly_amounts": category_data['amount'].tolist(),
                            "months": category_data['year_month'].tolist(),
                            "trend_slope": float(slope),
                            "data_points": len(category_data)
                        },
                        statistical_evidence={
                            "r_squared": float(r_squared),
                            "p_value": float(p_value),
                            "slope": float(slope),
                            "trend_strength": "strong" if r_squared > 0.7 else "moderate"
                        },
                        model_metrics={
                            "model_type": "linear_regression",
                            "mae": float(mean_absolute_error(y, model.predict(X))),
                            "mse": float(mean_squared_error(y, model.predict(X))),
                            "data_quality_score": min(len(category_data) / 12, 1.0)  # Normalize by ideal 12 months
                        },
                        preparation_actions=self._generate_increase_preparations(
                            category, increase_percentage, projected_avg, budget_limit
                        ),
                        mitigation_strategies=self._generate_increase_mitigations(
                            category, increase_percentage, projected_avg
                        ),
                        opportunities=self._generate_increase_opportunities(
                            category, increase_percentage
                        ),
                        prediction_date=datetime.now(),
                        valid_until=datetime.now() + timedelta(days=horizon_days),
                        model_used="linear_regression_trend",
                        data_quality_score=min(len(category_data) / 12, 1.0)
                    )
                    
                    predictions.append(prediction)
            
        except Exception as e:
            self.logger.error(f"Error predicting spending increases: {str(e)}")
        
        return predictions
    
    async def _predict_spending_decreases(
        self, 
        user_id: str, 
        historical_data: Dict[str, Any], 
        horizon_days: int
    ) -> List[TrendPrediction]:
        """Predict categories with decreasing spending trends (opportunities)."""
        predictions = []
        
        try:
            df = historical_data['transactions']
            
            # Similar to increase prediction but for decreasing trends
            monthly_spending = df.groupby(['category', df['transaction_date'].dt.to_period('M')])['amount'].sum().reset_index()
            monthly_spending['year_month'] = monthly_spending['transaction_date'].astype(str)
            
            for category in df['category'].unique():
                category_data = monthly_spending[monthly_spending['category'] == category].copy()
                
                if len(category_data) < 3:
                    continue
                
                category_data = category_data.sort_values('transaction_date')
                category_data['period_index'] = range(len(category_data))
                
                X = category_data['period_index'].values.reshape(-1, 1)
                y = category_data['amount'].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                slope = model.coef_[0]
                r_squared = model.score(X, y)
                
                # Calculate p-value for significance
                n = len(category_data)
                if n > 2:
                    y_pred = model.predict(X)
                    residuals = y - y_pred
                    mse = np.mean(residuals ** 2)
                    se_slope = np.sqrt(mse / np.sum((X.flatten() - np.mean(X)) ** 2))
                    t_stat = slope / se_slope if se_slope > 0 else 0
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                else:
                    p_value = 1
                
                # Only consider significant decreasing trends
                if slope < 0 and p_value < self.significance_threshold and r_squared > self.trend_strength_threshold:
                    current_avg = category_data['amount'].tail(3).mean()
                    future_periods = int(horizon_days / 30)
                    future_X = np.array([[len(category_data) + i] for i in range(1, future_periods + 1)])
                    projected_amounts = model.predict(future_X)
                    projected_avg = projected_amounts.mean()
                    
                    decrease_percentage = abs(((projected_avg - current_avg) / current_avg) * 100)
                    potential_savings = (current_avg - projected_avg) * future_periods
                    
                    confidence_score = min(r_squared * (1 - p_value), 1.0)
                    confidence = self._get_confidence_level(confidence_score)
                    
                    prediction = TrendPrediction(
                        prediction_id=str(uuid.uuid4()),
                        user_id=user_id,
                        trend_type=TrendType.SPENDING_DECREASE,
                        category=category,
                        severity=TrendSeverity.LOW,  # Decreases are generally positive
                        confidence=confidence,
                        confidence_score=confidence_score,
                        title=f"Decreasing {category.title()} Spending Trend",
                        description=f"Great news! Your {category} spending is trending downward with a {decrease_percentage:.1f}% projected decrease, potentially saving ${potential_savings:.2f}.",
                        predicted_impact={
                            "current_monthly_avg": float(current_avg),
                            "projected_monthly_avg": float(projected_avg),
                            "decrease_percentage": float(decrease_percentage),
                            "potential_savings": float(potential_savings)
                        },
                        timeline={
                            "trend_start": category_data['transaction_date'].iloc[0].isoformat(),
                            "projection_start": datetime.now().isoformat(),
                            "projection_end": (datetime.now() + timedelta(days=horizon_days)).isoformat(),
                        },
                        historical_data={
                            "monthly_amounts": category_data['amount'].tolist(),
                            "months": category_data['year_month'].tolist(),
                            "trend_slope": float(slope),
                            "data_points": len(category_data)
                        },
                        statistical_evidence={
                            "r_squared": float(r_squared),
                            "p_value": float(p_value),
                            "slope": float(slope),
                            "trend_strength": "strong" if r_squared > 0.7 else "moderate"
                        },
                        model_metrics={
                            "model_type": "linear_regression_trend",
                            "mae": float(mean_absolute_error(y, model.predict(X))),
                            "mse": float(mean_squared_error(y, model.predict(X))),
                            "data_quality_score": min(len(category_data) / 12, 1.0)
                        },
                        preparation_actions=[
                            {
                                "action": "reinforce_positive_behavior",
                                "description": f"Continue your current approach to {category} spending",
                                "priority": "medium"
                            }
                        ],
                        mitigation_strategies=[],  # No mitigation needed for positive trends
                        opportunities=self._generate_decrease_opportunities(
                            category, decrease_percentage, potential_savings
                        ),
                        prediction_date=datetime.now(),
                        valid_until=datetime.now() + timedelta(days=horizon_days),
                        model_used="linear_regression_trend",
                        data_quality_score=min(len(category_data) / 12, 1.0)
                    )
                    
                    predictions.append(prediction)
        
        except Exception as e:
            self.logger.error(f"Error predicting spending decreases: {str(e)}")
        
        return predictions
    
    async def _predict_seasonal_changes(
        self, 
        user_id: str, 
        historical_data: Dict[str, Any], 
        horizon_days: int
    ) -> List[TrendPrediction]:
        """Predict seasonal spending pattern changes."""
        predictions = []
        
        try:
            df = historical_data['transactions']
            
            # Get seasonal patterns for each category
            seasonal_patterns = await self._analyze_seasonal_patterns(df)
            
            current_month = datetime.now().month
            target_months = []
            
            # Determine which months to predict for based on horizon
            for i in range(1, (horizon_days // 30) + 2):
                future_month = ((current_month + i - 1) % 12) + 1
                target_months.append(future_month)
            
            for category, pattern in seasonal_patterns.items():
                if pattern.confidence_score < 0.6:  # Skip low-confidence patterns
                    continue
                
                # Check if any target months are peak or low months
                upcoming_peaks = [m for m in target_months if m in pattern.peak_months]
                upcoming_lows = [m for m in target_months if m in pattern.low_months]
                
                if upcoming_peaks or upcoming_lows:
                    # Calculate expected impact
                    current_factor = pattern.seasonal_factors.get(current_month, 1.0)
                    
                    if upcoming_peaks:
                        # Predict seasonal increase
                        peak_month = upcoming_peaks[0]
                        peak_factor = pattern.seasonal_factors.get(peak_month, 1.0)
                        seasonal_increase = ((peak_factor / current_factor) - 1) * 100
                        
                        if seasonal_increase > 20:  # Only alert for significant increases
                            severity = TrendSeverity.HIGH if seasonal_increase > 50 else TrendSeverity.MEDIUM
                            
                            prediction = TrendPrediction(
                                prediction_id=str(uuid.uuid4()),
                                user_id=user_id,
                                trend_type=TrendType.SEASONAL_PATTERN,
                                category=category,
                                severity=severity,
                                confidence=self._get_confidence_level(pattern.confidence_score),
                                confidence_score=pattern.confidence_score,
                                title=f"Seasonal {category.title()} Spending Increase Expected",
                                description=f"Based on historical patterns, expect a {seasonal_increase:.1f}% increase in {category} spending during {self._month_name(peak_month)}.",
                                predicted_impact={
                                    "seasonal_increase_percentage": float(seasonal_increase),
                                    "peak_month": peak_month,
                                    "current_factor": float(current_factor),
                                    "peak_factor": float(peak_factor),
                                    "year_over_year_growth": float(pattern.year_over_year_growth)
                                },
                                timeline={
                                    "seasonal_start": f"{datetime.now().year}-{peak_month:02d}-01",
                                    "pattern_based_on": f"{len(df[df['category'] == category])} historical transactions",
                                    "next_low_period": f"{datetime.now().year}-{pattern.low_months[0] if pattern.low_months else 'N/A'}-01"
                                },
                                historical_data={
                                    "seasonal_factors": pattern.seasonal_factors,
                                    "peak_months": pattern.peak_months,
                                    "low_months": pattern.low_months,
                                    "pattern_strength": float(pattern.confidence_score)
                                },
                                statistical_evidence={
                                    "pattern_confidence": float(pattern.confidence_score),
                                    "year_over_year_growth": float(pattern.year_over_year_growth),
                                    "seasonal_variance": float(np.var(list(pattern.seasonal_factors.values())))
                                },
                                model_metrics={
                                    "model_type": "seasonal_decomposition",
                                    "data_quality_score": min(len(df[df['category'] == category]) / 50, 1.0)
                                },
                                preparation_actions=self._generate_seasonal_preparations(
                                    category, peak_month, seasonal_increase
                                ),
                                mitigation_strategies=self._generate_seasonal_mitigations(
                                    category, seasonal_increase
                                ),
                                opportunities=self._generate_seasonal_opportunities(
                                    category, pattern
                                ),
                                prediction_date=datetime.now(),
                                valid_until=datetime.now() + timedelta(days=horizon_days),
                                model_used="seasonal_pattern_analysis",
                                data_quality_score=min(len(df[df['category'] == category]) / 50, 1.0)
                            )
                            
                            predictions.append(prediction)
        
        except Exception as e:
            self.logger.error(f"Error predicting seasonal changes: {str(e)}")
        
        return predictions
    
    async def _predict_budget_challenges(
        self, 
        user_id: str, 
        historical_data: Dict[str, Any], 
        horizon_days: int
    ) -> List[TrendPrediction]:
        """Predict potential budget challenges based on spending trends and limits."""
        predictions = []
        
        try:
            df = historical_data['transactions']
            budget_limits = historical_data.get('budget_limits', {})
            
            if not budget_limits:
                return predictions  # No budget limits to challenge
            
            # Analyze spending velocity for each category with budget limits
            for category, limit in budget_limits.items():
                category_transactions = df[df['category'] == category]
                
                if len(category_transactions) < 5:  # Need minimum data
                    continue
                
                # Calculate current spending rate (last 30 days)
                recent_date = df['transaction_date'].max()
                last_30_days = recent_date - timedelta(days=30)
                recent_spending = category_transactions[
                    category_transactions['transaction_date'] >= last_30_days
                ]['amount'].sum()
                
                # Calculate average monthly spending
                monthly_spending = category_transactions.groupby(
                    category_transactions['transaction_date'].dt.to_period('M')
                )['amount'].sum()
                
                if len(monthly_spending) < 2:
                    continue
                
                # Predict spending for next period
                current_monthly_avg = monthly_spending.mean()
                recent_trend = monthly_spending.tail(3).mean() / monthly_spending.head(3).mean()
                projected_spending = current_monthly_avg * recent_trend
                
                # Check for budget challenges
                utilization_rate = projected_spending / limit
                days_until_limit = None
                
                if recent_spending > 0:
                    daily_rate = recent_spending / 30
                    remaining_budget = limit - recent_spending
                    if daily_rate > 0 and remaining_budget > 0:
                        days_until_limit = remaining_budget / daily_rate
                
                # Determine if this is a challenge
                is_challenge = False
                severity = TrendSeverity.LOW
                
                if utilization_rate > 1.2:  # Projected to exceed by 20%+
                    is_challenge = True
                    severity = TrendSeverity.HIGH
                elif utilization_rate > 1.0:  # Projected to exceed
                    is_challenge = True
                    severity = TrendSeverity.MEDIUM
                elif days_until_limit and days_until_limit < 20:  # Will hit limit soon
                    is_challenge = True
                    severity = TrendSeverity.MEDIUM
                
                if is_challenge:
                    confidence_score = min(
                        len(monthly_spending) / 6,  # More data = higher confidence
                        0.9 if len(category_transactions) > 20 else 0.7
                    )
                    
                    prediction = TrendPrediction(
                        prediction_id=str(uuid.uuid4()),
                        user_id=user_id,
                        trend_type=TrendType.BUDGET_CHALLENGE,
                        category=category,
                        severity=severity,
                        confidence=self._get_confidence_level(confidence_score),
                        confidence_score=confidence_score,
                        title=f"Budget Challenge Alert: {category.title()}",
                        description=f"You're projected to exceed your {category} budget by {((utilization_rate - 1) * 100):.1f}% this month.",
                        predicted_impact={
                            "budget_limit": float(limit),
                            "projected_spending": float(projected_spending),
                            "current_utilization": float(recent_spending / limit),
                            "projected_utilization": float(utilization_rate),
                            "overage_amount": float(max(0, projected_spending - limit)),
                            "days_until_limit": days_until_limit
                        },
                        timeline={
                            "current_period_start": (recent_date - timedelta(days=30)).isoformat(),
                            "budget_limit_date": days_until_limit and (datetime.now() + timedelta(days=days_until_limit)).isoformat(),
                            "month_end": (recent_date.replace(day=1) + timedelta(days=32)).replace(day=1).isoformat()
                        },
                        historical_data={
                            "monthly_spending": monthly_spending.tolist(),
                            "recent_spending": float(recent_spending),
                            "average_monthly": float(current_monthly_avg),
                            "trend_factor": float(recent_trend)
                        },
                        statistical_evidence={
                            "utilization_trend": float(recent_trend),
                            "spending_volatility": float(monthly_spending.std()),
                            "budget_history": len(monthly_spending)
                        },
                        model_metrics={
                            "model_type": "budget_projection",
                            "data_quality_score": min(len(category_transactions) / 30, 1.0)
                        },
                        preparation_actions=self._generate_budget_challenge_preparations(
                            category, utilization_rate, days_until_limit
                        ),
                        mitigation_strategies=self._generate_budget_challenge_mitigations(
                            category, utilization_rate, projected_spending - limit
                        ),
                        opportunities=self._generate_budget_challenge_opportunities(
                            category, limit, projected_spending
                        ),
                        prediction_date=datetime.now(),
                        valid_until=datetime.now() + timedelta(days=min(horizon_days, 30)),
                        model_used="budget_utilization_trend",
                        data_quality_score=min(len(category_transactions) / 30, 1.0)
                    )
                    
                    predictions.append(prediction)
        
        except Exception as e:
            self.logger.error(f"Error predicting budget challenges: {str(e)}")
        
        return predictions
    
    async def _predict_emerging_categories(
        self, 
        user_id: str, 
        historical_data: Dict[str, Any], 
        horizon_days: int
    ) -> List[TrendPrediction]:
        """Predict emerging spending categories based on recent activity."""
        predictions = []
        
        try:
            df = historical_data['transactions']
            
            # Compare recent vs historical category spending
            cutoff_date = datetime.now().date() - timedelta(days=60)
            recent_df = df[df['transaction_date'] >= cutoff_date]
            historical_df = df[df['transaction_date'] < cutoff_date]
            
            if len(historical_df) < 20:  # Need sufficient historical data
                return predictions
            
            # Calculate category spending by period
            recent_category_spending = recent_df.groupby('category')['amount'].agg(['sum', 'count', 'mean'])
            historical_category_spending = historical_df.groupby('category')['amount'].agg(['sum', 'count', 'mean'])
            
            # Find new or significantly increased categories
            for category in recent_category_spending.index:
                recent_stats = recent_category_spending.loc[category]
                
                # Check if this is a new category
                is_new_category = category not in historical_category_spending.index
                
                if is_new_category and recent_stats['count'] >= 3:  # New category with multiple transactions
                    confidence_score = min(recent_stats['count'] / 10, 0.8)
                    
                    prediction = TrendPrediction(
                        prediction_id=str(uuid.uuid4()),
                        user_id=user_id,
                        trend_type=TrendType.EMERGING_CATEGORY,
                        category=category,
                        severity=TrendSeverity.MEDIUM,
                        confidence=self._get_confidence_level(confidence_score),
                        confidence_score=confidence_score,
                        title=f"New Spending Category: {category.title()}",
                        description=f"You've started spending in {category} - ${recent_stats['sum']:.2f} over {recent_stats['count']} transactions in the last 60 days.",
                        predicted_impact={
                            "recent_total": float(recent_stats['sum']),
                            "transaction_count": int(recent_stats['count']),
                            "average_transaction": float(recent_stats['mean']),
                            "projected_monthly": float(recent_stats['sum'] / 2),  # 60 days -> monthly
                            "is_new_category": True
                        },
                        timeline={
                            "first_transaction": recent_df[recent_df['category'] == category]['transaction_date'].min().isoformat(),
                            "analysis_period": "last_60_days",
                            "projection_horizon": f"{horizon_days}_days"
                        },
                        historical_data={
                            "recent_transactions": int(recent_stats['count']),
                            "recent_spending": float(recent_stats['sum']),
                            "historical_presence": False
                        },
                        statistical_evidence={
                            "emergence_confidence": float(confidence_score),
                            "spending_consistency": float(recent_stats['sum'] / recent_stats['count']) if recent_stats['count'] > 0 else 0
                        },
                        model_metrics={
                            "model_type": "category_emergence_detection",
                            "data_quality_score": min(recent_stats['count'] / 5, 1.0)
                        },
                        preparation_actions=self._generate_emerging_category_preparations(
                            category, recent_stats['sum'], recent_stats['count']
                        ),
                        mitigation_strategies=[],  # Emerging categories aren't necessarily problems
                        opportunities=self._generate_emerging_category_opportunities(
                            category, recent_stats['sum']
                        ),
                        prediction_date=datetime.now(),
                        valid_until=datetime.now() + timedelta(days=horizon_days),
                        model_used="category_emergence_analysis",
                        data_quality_score=min(recent_stats['count'] / 5, 1.0)
                    )
                    predictions.append(prediction)
                
                elif not is_new_category:
                    # Check for significant increase in existing category
                    historical_stats = historical_category_spending.loc[category]
                    
                    # Calculate growth rate
                    historical_monthly_avg = historical_stats['sum'] / (len(historical_df['transaction_date'].dt.to_period('M').unique()))
                    recent_monthly_avg = recent_stats['sum'] / 2  # 60 days
                    
                    if historical_monthly_avg > 0:
                        growth_rate = ((recent_monthly_avg - historical_monthly_avg) / historical_monthly_avg)
                        
                        if growth_rate > 0.5 and recent_stats['count'] >= 5:  # 50% increase with multiple transactions
                            confidence_score = min(
                                (recent_stats['count'] + historical_stats['count']) / 20,
                                0.85
                            )
                            
                            prediction = TrendPrediction(
                                prediction_id=str(uuid.uuid4()),
                                user_id=user_id,
                                trend_type=TrendType.EMERGING_CATEGORY,
                                category=category,
                                severity=TrendSeverity.MEDIUM if growth_rate > 1.0 else TrendSeverity.LOW,
                                confidence=self._get_confidence_level(confidence_score),
                                confidence_score=confidence_score,
                                title=f"Significant Increase in {category.title()} Spending",
                                description=f"Your {category} spending has increased by {growth_rate*100:.1f}% compared to your historical average.",
                                predicted_impact={
                                    "historical_monthly_avg": float(historical_monthly_avg),
                                    "recent_monthly_avg": float(recent_monthly_avg),
                                    "growth_rate": float(growth_rate),
                                    "additional_spending": float(recent_monthly_avg - historical_monthly_avg),
                                    "is_new_category": False
                                },
                                timeline={
                                    "growth_detected": cutoff_date.isoformat(),
                                    "historical_period": f"{len(historical_df['transaction_date'].dt.to_period('M').unique())}_months",
                                    "recent_period": "60_days"
                                },
                                historical_data={
                                    "historical_monthly_avg": float(historical_monthly_avg),
                                    "recent_total": float(recent_stats['sum']),
                                    "historical_transactions": int(historical_stats['count']),
                                    "recent_transactions": int(recent_stats['count'])
                                },
                                statistical_evidence={
                                    "growth_rate": float(growth_rate),
                                    "spending_acceleration": "significant" if growth_rate > 1.0 else "moderate"
                                },
                                model_metrics={
                                    "model_type": "category_growth_detection",
                                    "data_quality_score": min((recent_stats['count'] + historical_stats['count']) / 15, 1.0)
                                },
                                preparation_actions=self._generate_category_growth_preparations(
                                    category, growth_rate, recent_monthly_avg
                                ),
                                mitigation_strategies=self._generate_category_growth_mitigations(
                                    category, growth_rate
                                ),
                                opportunities=self._generate_category_growth_opportunities(
                                    category, growth_rate
                                ),
                                prediction_date=datetime.now(),
                                valid_until=datetime.now() + timedelta(days=horizon_days),
                                model_used="category_growth_analysis",
                                data_quality_score=min((recent_stats['count'] + historical_stats['count']) / 15, 1.0)
                            )
                            predictions.append(prediction)
        
        except Exception as e:
            self.logger.error(f"Error predicting emerging categories: {str(e)}")
        
        return predictions
    
    async def _predict_behavior_shifts(
        self, 
        user_id: str, 
        historical_data: Dict[str, Any], 
        horizon_days: int
    ) -> List[TrendPrediction]:
        """Predict behavioral shifts in spending patterns."""
        predictions = []
        
        try:
            df = historical_data['transactions']
            
            # Analyze different behavioral dimensions
            behavior_predictions = []
            
            # 1. Time-of-week spending shifts
            time_shifts = await self._analyze_temporal_behavior_shifts(df)
            behavior_predictions.extend(time_shifts)
            
            # 2. Transaction size behavior changes
            size_shifts = await self._analyze_transaction_size_shifts(df)
            behavior_predictions.extend(size_shifts)
            
            # 3. Frequency behavior changes
            frequency_shifts = await self._analyze_frequency_shifts(df)
            behavior_predictions.extend(frequency_shifts)
            
            # Convert behavior shifts to trend predictions
            for shift in behavior_predictions:
                if shift['confidence'] > 0.6:  # Only high-confidence shifts
                    prediction = TrendPrediction(
                        prediction_id=str(uuid.uuid4()),
                        user_id=user_id,
                        trend_type=TrendType.BEHAVIOR_SHIFT,
                        category=shift.get('category', 'general'),
                        severity=TrendSeverity.MEDIUM if shift['magnitude'] > 0.3 else TrendSeverity.LOW,
                        confidence=self._get_confidence_level(shift['confidence']),
                        confidence_score=shift['confidence'],
                        title=f"Behavior Shift Detected: {shift['shift_type']}",
                        description=shift['description'],
                        predicted_impact=shift['impact'],
                        timeline=shift['timeline'],
                        historical_data=shift['historical_data'],
                        statistical_evidence=shift['statistical_evidence'],
                        model_metrics={
                            "model_type": "behavior_shift_detection",
                            "data_quality_score": shift.get('data_quality', 0.7)
                        },
                        preparation_actions=self._generate_behavior_shift_preparations(shift),
                        mitigation_strategies=self._generate_behavior_shift_mitigations(shift),
                        opportunities=self._generate_behavior_shift_opportunities(shift),
                        prediction_date=datetime.now(),
                        valid_until=datetime.now() + timedelta(days=horizon_days),
                        model_used="behavioral_pattern_analysis",
                        data_quality_score=shift.get('data_quality', 0.7)
                    )
                    predictions.append(prediction)
        
        except Exception as e:
            self.logger.error(f"Error predicting behavior shifts: {str(e)}")
        
        return predictions
    
    async def _predict_merchant_changes(
        self, 
        user_id: str, 
        historical_data: Dict[str, Any], 
        horizon_days: int
    ) -> List[TrendPrediction]:
        """Predict changes in merchant loyalty and preferences."""
        predictions = []
        
        try:
            df = historical_data['transactions']
            
            # Analyze merchant loyalty trends
            merchant_spending = df.groupby(['merchant_normalized', df['transaction_date'].dt.to_period('M')]).agg({
                'amount': 'sum',
                'transaction_id': 'count'
            }).reset_index()
            
            # Find merchants with changing patterns
            for merchant in df['merchant_normalized'].value_counts().head(20).index:  # Top 20 merchants
                merchant_data = merchant_spending[merchant_spending['merchant_normalized'] == merchant]
                
                if len(merchant_data) < 4:  # Need at least 4 months of data
                    continue
                
                merchant_data = merchant_data.sort_values('transaction_date')
                
                # Calculate trend in spending
                merchant_data['period_index'] = range(len(merchant_data))
                
                # Spending trend
                X = merchant_data['period_index'].values.reshape(-1, 1)
                y_amount = merchant_data['amount'].values
                y_frequency = merchant_data['transaction_id'].values
                
                amount_model = LinearRegression().fit(X, y_amount)
                frequency_model = LinearRegression().fit(X, y_frequency)
                
                amount_slope = amount_model.coef_[0]
                frequency_slope = frequency_model.coef_[0]
                amount_r2 = amount_model.score(X, y_amount)
                frequency_r2 = frequency_model.score(X, y_frequency)
                
                # Detect significant changes
                significant_amount_change = abs(amount_slope) > merchant_data['amount'].mean() * 0.1 and amount_r2 > 0.4
                significant_frequency_change = abs(frequency_slope) > 0.3 and frequency_r2 > 0.4
                
                if significant_amount_change or significant_frequency_change:
                    # Determine if loyalty is increasing or decreasing
                    is_increasing = (amount_slope > 0 and frequency_slope >= 0) or (amount_slope >= 0 and frequency_slope > 0)
                    
                    change_type = "increasing" if is_increasing else "decreasing"
                    severity = TrendSeverity.LOW if is_increasing else TrendSeverity.MEDIUM
                    
                    confidence_score = min((amount_r2 + frequency_r2) / 2, 0.85)
                    
                    prediction = TrendPrediction(
                        prediction_id=str(uuid.uuid4()),
                        user_id=user_id,
                        trend_type=TrendType.MERCHANT_LOYALTY,
                        category=df[df['merchant_normalized'] == merchant]['category'].mode().iloc[0],
                        severity=severity,
                        confidence=self._get_confidence_level(confidence_score),
                        confidence_score=confidence_score,
                        title=f"Merchant Loyalty Change: {merchant}",
                        description=f"Your loyalty to {merchant} is {change_type} - spending trend: {amount_slope:.2f}/month, frequency trend: {frequency_slope:.2f} visits/month.",
                        predicted_impact={
                            "merchant": merchant,
                            "spending_trend_monthly": float(amount_slope),
                            "frequency_trend_monthly": float(frequency_slope),
                            "current_monthly_spending": float(merchant_data['amount'].tail(3).mean()),
                            "current_monthly_visits": float(merchant_data['transaction_id'].tail(3).mean()),
                            "loyalty_direction": change_type
                        },
                        timeline={
                            "trend_period": f"{len(merchant_data)}_months",
                            "first_transaction": merchant_data['transaction_date'].iloc[0].isoformat(),
                            "latest_transaction": merchant_data['transaction_date'].iloc[-1].isoformat()
                        },
                        historical_data={
                            "monthly_spending": merchant_data['amount'].tolist(),
                            "monthly_visits": merchant_data['transaction_id'].tolist(),
                            "periods": [str(p) for p in merchant_data['transaction_date']],
                            "total_relationship_months": len(merchant_data)
                        },
                        statistical_evidence={
                            "spending_trend_r2": float(amount_r2),
                            "frequency_trend_r2": float(frequency_r2),
                            "spending_slope": float(amount_slope),
                            "frequency_slope": float(frequency_slope)
                        },
                        model_metrics={
                            "model_type": "merchant_loyalty_trend",
                            "data_quality_score": min(len(merchant_data) / 8, 1.0)
                        },
                        preparation_actions=self._generate_merchant_loyalty_preparations(
                            merchant, change_type, amount_slope
                        ),
                        mitigation_strategies=self._generate_merchant_loyalty_mitigations(
                            merchant, change_type
                        ) if not is_increasing else [],
                        opportunities=self._generate_merchant_loyalty_opportunities(
                            merchant, change_type, merchant_data['amount'].mean()
                        ),
                        prediction_date=datetime.now(),
                        valid_until=datetime.now() + timedelta(days=horizon_days),
                        model_used="merchant_loyalty_analysis",
                        data_quality_score=min(len(merchant_data) / 8, 1.0)
                    )
                    predictions.append(prediction)
        
        except Exception as e:
            self.logger.error(f"Error predicting merchant changes: {str(e)}")
        
        return predictions
    
    async def _predict_payment_method_shifts(
        self, 
        user_id: str, 
        historical_data: Dict[str, Any], 
        horizon_days: int
    ) -> List[TrendPrediction]:
        """Predict shifts in payment method preferences."""
        predictions = []
        
        try:
            df = historical_data['transactions']
            
            if 'payment_method' not in df.columns or df['payment_method'].isna().all():
                return predictions  # No payment method data
            
            # Analyze payment method trends over time
            payment_trends = df.groupby([df['transaction_date'].dt.to_period('M'), 'payment_method']).agg({
                'amount': 'sum',
                'transaction_id': 'count'
            }).reset_index()
            
            # Calculate payment method usage by month
            monthly_totals = payment_trends.groupby('transaction_date').agg({
                'amount': 'sum',
                'transaction_id': 'sum'
            }).reset_index()
            monthly_totals.columns = ['transaction_date', 'total_amount', 'total_transactions']
            
            payment_trends = payment_trends.merge(monthly_totals, on='transaction_date')
            payment_trends['amount_percentage'] = payment_trends['amount'] / payment_trends['total_amount']
            payment_trends['transaction_percentage'] = payment_trends['transaction_id'] / payment_trends['total_transactions']
            
            # Analyze trends for each payment method
            for payment_method in df['payment_method'].dropna().unique():
                method_data = payment_trends[payment_trends['payment_method'] == payment_method].copy()
                
                if len(method_data) < 3:  # Need at least 3 months
                    continue
                
                method_data = method_data.sort_values('transaction_date')
                method_data['period_index'] = range(len(method_data))
                
                # Analyze trend in usage percentage
                X = method_data['period_index'].values.reshape(-1, 1)
                y_amount_pct = method_data['amount_percentage'].values
                y_transaction_pct = method_data['transaction_percentage'].values
                
                amount_model = LinearRegression().fit(X, y_amount_pct)
                transaction_model = LinearRegression().fit(X, y_transaction_pct)
                
                amount_slope = amount_model.coef_[0]
                transaction_slope = transaction_model.coef_[0]
                amount_r2 = amount_model.score(X, y_amount_pct)
                transaction_r2 = transaction_model.score(X, y_transaction_pct)
                
                # Detect significant shifts (>5% change per month)
                significant_shift = (abs(amount_slope) > 0.05 or abs(transaction_slope) > 0.05) and max(amount_r2, transaction_r2) > 0.5
                
                if significant_shift:
                    shift_direction = "increasing" if (amount_slope > 0 or transaction_slope > 0) else "decreasing"
                    current_usage = method_data['amount_percentage'].tail(3).mean()
                    
                    confidence_score = min(max(amount_r2, transaction_r2), 0.8)
                    
                    prediction = TrendPrediction(
                        prediction_id=str(uuid.uuid4()),
                        user_id=user_id,
                        trend_type=TrendType.PAYMENT_METHOD_SHIFT,
                        category='payment_behavior',
                        severity=TrendSeverity.LOW,  # Payment method shifts are usually not critical
                        confidence=self._get_confidence_level(confidence_score),
                        confidence_score=confidence_score,
                        title=f"Payment Method Shift: {payment_method}",
                        description=f"Your usage of {payment_method} is {shift_direction} - now {current_usage*100:.1f}% of your spending.",
                        predicted_impact={
                            "payment_method": payment_method,
                            "current_usage_percentage": float(current_usage),
                            "amount_trend_monthly": float(amount_slope),
                            "frequency_trend_monthly": float(transaction_slope),
                            "shift_direction": shift_direction,
                            "projected_usage_3months": float(min(1.0, max(0.0, current_usage + (amount_slope * 3))))
                        },
                        timeline={
                            "trend_period": f"{len(method_data)}_months",
                            "analysis_start": method_data['transaction_date'].iloc[0].isoformat(),
                            "analysis_end": method_data['transaction_date'].iloc[-1].isoformat()
                        },
                        historical_data={
                            "monthly_usage_percentages": method_data['amount_percentage'].tolist(),
                            "monthly_transaction_percentages": method_data['transaction_percentage'].tolist(),
                            "periods": [str(p) for p in method_data['transaction_date']]
                        },
                        statistical_evidence={
                            "amount_trend_r2": float(amount_r2),
                            "transaction_trend_r2": float(transaction_r2),
                            "amount_slope": float(amount_slope),
                            "transaction_slope": float(transaction_slope),
                            "trend_strength": "strong" if max(amount_r2, transaction_r2) > 0.7 else "moderate"
                        },
                        model_metrics={
                            "model_type": "payment_method_trend",
                            "data_quality_score": min(len(method_data) / 6, 1.0)
                        },
                        preparation_actions=self._generate_payment_method_preparations(
                            payment_method, shift_direction, current_usage
                        ),
                        mitigation_strategies=[],  # Payment method shifts usually don't need mitigation
                        opportunities=self._generate_payment_method_opportunities(
                            payment_method, shift_direction
                        ),
                        prediction_date=datetime.now(),
                        valid_until=datetime.now() + timedelta(days=horizon_days),
                        model_used="payment_method_trend_analysis",
                        data_quality_score=min(len(method_data) / 6, 1.0)
                    )
                    predictions.append(prediction)
        
        except Exception as e:
            self.logger.error(f"Error predicting payment method shifts: {str(e)}")
        
        return predictions
    
    # Helper methods for generating recommendations and analyzing patterns
    
    async def _analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, SeasonalPattern]:
        """Analyze seasonal spending patterns for each category."""
        patterns = {}
        
        try:
            for category in df['category'].unique():
                category_data = df[df['category'] == category].copy()
                
                if len(category_data) < 12:  # Need at least a year of data
                    continue
                
                # Group by month and calculate seasonal factors
                monthly_spending = category_data.groupby(category_data['transaction_date'].dt.month)['amount'].agg(['sum', 'count', 'mean'])
                
                if len(monthly_spending) < 6:  # Need data from at least 6 different months
                    continue
                
                # Calculate seasonal factors (compared to average)
                overall_monthly_avg = monthly_spending['sum'].mean()
                seasonal_factors = {}
                
                for month in range(1, 13):
                    if month in monthly_spending.index:
                        seasonal_factors[month] = monthly_spending.loc[month, 'sum'] / overall_monthly_avg
                    else:
                        seasonal_factors[month] = 1.0  # No data, assume average
                
                # Identify peak and low months
                factor_values = list(seasonal_factors.values())
                peak_threshold = np.mean(factor_values) + np.std(factor_values)
                low_threshold = np.mean(factor_values) - np.std(factor_values)
                
                peak_months = [month for month, factor in seasonal_factors.items() if factor > peak_threshold]
                low_months = [month for month, factor in seasonal_factors.items() if factor < low_threshold]
                
                # Calculate year-over-year growth
                # Continuation of _analyze_seasonal_patterns method
                yearly_data = category_data.groupby(category_data['transaction_date'].dt.year)['amount'].sum()
                yoy_growth = 0.0
                if len(yearly_data) > 1:
                    yoy_growth = (yearly_data.iloc[-1] - yearly_data.iloc[0]) / yearly_data.iloc[0]
                
                # Calculate confidence score based on data quality and pattern strength
                seasonal_variance = np.var(list(seasonal_factors.values()))
                confidence_score = min(
                    (len(monthly_spending) / 12) * 0.8 +  # Data coverage
                    (seasonal_variance / 2) * 0.2,        # Pattern strength
                    0.9
                )
                
                if confidence_score >= self.seasonal_strength_threshold:
                    patterns[category] = SeasonalPattern(
                        pattern_id=str(uuid.uuid4()),
                        category=category,
                        seasonal_factors=seasonal_factors,
                        peak_months=peak_months,
                        low_months=low_months,
                        year_over_year_growth=yoy_growth,
                        confidence_score=confidence_score,
                        last_updated=datetime.now()
                    )

        except Exception as e:
            self.logger.error(f"Error analyzing seasonal patterns: {str(e)}")
        
        return patterns

    async def _analyze_temporal_behavior_shifts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze shifts in time-based spending behavior."""
        shifts = []
        
        try:
            # Add time-based features
            df['hour'] = df['transaction_date'].dt.hour
            df['day_of_week'] = df['transaction_date'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6])
            
            # Split data into recent vs historical
            cutoff_date = datetime.now().date() - timedelta(days=90)
            recent_df = df[df['transaction_date'] >= cutoff_date]
            historical_df = df[df['transaction_date'] < cutoff_date]
            
            if len(historical_df) < 30 or len(recent_df) < 30:
                return shifts
            
            # Analyze weekend vs weekday spending shifts
            historical_weekend_ratio = historical_df[historical_df['is_weekend']]['amount'].sum() / historical_df['amount'].sum()
            recent_weekend_ratio = recent_df[recent_df['is_weekend']]['amount'].sum() / recent_df['amount'].sum()
            
            weekend_shift = recent_weekend_ratio - historical_weekend_ratio
            
            if abs(weekend_shift) > 0.1:  # 10% shift threshold
                shifts.append({
                    'shift_type': 'weekend_spending_pattern',
                    'category': 'temporal_behavior',
                    'confidence': min(len(recent_df) / 50, 0.8),
                    'magnitude': abs(weekend_shift),
                    'description': f"Your weekend spending pattern has {'increased' if weekend_shift > 0 else 'decreased'} by {abs(weekend_shift)*100:.1f}%",
                    'impact': {
                        'historical_weekend_ratio': float(historical_weekend_ratio),
                        'recent_weekend_ratio': float(recent_weekend_ratio),
                        'shift_magnitude': float(weekend_shift)
                    },
                    'timeline': {
                        'shift_detected': cutoff_date.isoformat(),
                        'analysis_period': '90_days'
                    },
                    'historical_data': {
                        'historical_weekend_spending': float(historical_df[historical_df['is_weekend']]['amount'].sum()),
                        'recent_weekend_spending': float(recent_df[recent_df['is_weekend']]['amount'].sum())
                    },
                    'statistical_evidence': {
                        'weekend_shift_magnitude': float(abs(weekend_shift)),
                        'confidence_level': 'high' if abs(weekend_shift) > 0.2 else 'medium'
                    },
                    'data_quality': min((len(recent_df) + len(historical_df)) / 100, 1.0)
                })
            
        except Exception as e:
            self.logger.error(f"Error analyzing temporal behavior shifts: {str(e)}")
        
        return shifts

    async def _analyze_transaction_size_shifts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze shifts in transaction size behavior."""
        shifts = []
        
        try:
            # Split data for comparison
            cutoff_date = datetime.now().date() - timedelta(days=90)
            recent_df = df[df['transaction_date'] >= cutoff_date]
            historical_df = df[df['transaction_date'] < cutoff_date]
            
            if len(historical_df) < 20 or len(recent_df) < 20:
                return shifts
            
            # Analyze average transaction size shift
            historical_avg = historical_df['amount'].mean()
            recent_avg = recent_df['amount'].mean()
            size_shift = (recent_avg - historical_avg) / historical_avg
            
            if abs(size_shift) > 0.15:  # 15% change threshold
                # Statistical significance test
                from scipy.stats import ttest_ind
                t_stat, p_value = ttest_ind(recent_df['amount'], historical_df['amount'])
                
                if p_value < 0.05:  # Statistically significant
                    shifts.append({
                        'shift_type': 'transaction_size_pattern',
                        'category': 'spending_behavior',
                        'confidence': min(1 - p_value, 0.9),
                        'magnitude': abs(size_shift),
                        'description': f"Your average transaction size has {'increased' if size_shift > 0 else 'decreased'} by {abs(size_shift)*100:.1f}%",
                        'impact': {
                            'historical_avg_transaction': float(historical_avg),
                            'recent_avg_transaction': float(recent_avg),
                            'size_shift_percentage': float(size_shift * 100)
                        },
                        'timeline': {
                            'shift_detected': cutoff_date.isoformat(),
                            'analysis_period': '90_days'
                        },
                        'historical_data': {
                            'historical_transactions': len(historical_df),
                            'recent_transactions': len(recent_df),
                            'historical_median': float(historical_df['amount'].median()),
                            'recent_median': float(recent_df['amount'].median())
                        },
                        'statistical_evidence': {
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'effect_size': float(size_shift)
                        },
                        'data_quality': min((len(recent_df) + len(historical_df)) / 80, 1.0)
                    })
                    
        except Exception as e:
            self.logger.error(f"Error analyzing transaction size shifts: {str(e)}")
        
        return shifts

    async def _analyze_frequency_shifts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze shifts in spending frequency patterns."""
        shifts = []
        
        try:
            # Group by category and analyze frequency changes
            for category in df['category'].unique():
                category_data = df[df['category'] == category]
                
                if len(category_data) < 20:
                    continue
                
                # Split into periods
                cutoff_date = datetime.now().date() - timedelta(days=90)
                recent_transactions = category_data[category_data['transaction_date'] >= cutoff_date]
                historical_transactions = category_data[category_data['transaction_date'] < cutoff_date]
                
                if len(recent_transactions) < 5 or len(historical_transactions) < 5:
                    continue
                
                # Calculate frequency (transactions per month)
                historical_months = (historical_transactions['transaction_date'].max() - historical_transactions['transaction_date'].min()).days / 30
                recent_months = 3  # 90 days / 30
                
                if historical_months < 1:
                    continue
                
                historical_freq = len(historical_transactions) / historical_months
                recent_freq = len(recent_transactions) / recent_months
                
                freq_shift = (recent_freq - historical_freq) / historical_freq if historical_freq > 0 else 0
                
                if abs(freq_shift) > 0.3:  # 30% frequency change
                    shifts.append({
                        'shift_type': 'spending_frequency_pattern',
                        'category': category,
                        'confidence': min(len(category_data) / 30, 0.85),
                        'magnitude': abs(freq_shift),
                        'description': f"Your {category} spending frequency has {'increased' if freq_shift > 0 else 'decreased'} by {abs(freq_shift)*100:.1f}%",
                        'impact': {
                            'historical_frequency_per_month': float(historical_freq),
                            'recent_frequency_per_month': float(recent_freq),
                            'frequency_shift_percentage': float(freq_shift * 100),
                            'category': category
                        },
                        'timeline': {
                            'shift_detected': cutoff_date.isoformat(),
                            'historical_period_months': float(historical_months),
                            'recent_period_months': recent_months
                        },
                        'historical_data': {
                            'historical_transaction_count': len(historical_transactions),
                            'recent_transaction_count': len(recent_transactions),
                            'total_transactions': len(category_data)
                        },
                        'statistical_evidence': {
                            'frequency_change_ratio': float(recent_freq / historical_freq) if historical_freq > 0 else 0,
                            'trend_strength': 'strong' if abs(freq_shift) > 0.5 else 'moderate'
                        },
                        'data_quality': min(len(category_data) / 25, 1.0)
                    })
                    
        except Exception as e:
            self.logger.error(f"Error analyzing frequency shifts: {str(e)}")
        
        return shifts

    async def _filter_and_rank_predictions(self, predictions: List[TrendPrediction]) -> List[TrendPrediction]:
        """Filter and rank predictions by relevance and confidence."""
        if not predictions:
            return []
        
        # Filter out low-confidence predictions
        filtered = [p for p in predictions if p.confidence_score >= 0.5]
        
        # Calculate composite score for ranking
        for prediction in filtered:
            severity_weight = {
                TrendSeverity.CRITICAL: 1.0,
                TrendSeverity.HIGH: 0.8,
                TrendSeverity.MEDIUM: 0.6,
                TrendSeverity.LOW: 0.4
            }
            
            confidence_weight = {
                PredictionConfidence.VERY_HIGH: 1.0,
                PredictionConfidence.HIGH: 0.8,
                PredictionConfidence.MEDIUM: 0.6,
                PredictionConfidence.LOW: 0.4
            }
            
            # Composite score for ranking
            prediction.composite_score = (
                severity_weight.get(prediction.severity, 0.5) * 0.4 +
                confidence_weight.get(prediction.confidence, 0.5) * 0.4 +
                prediction.data_quality_score * 0.2
            )
        
        # Sort by composite score and limit to top predictions
        filtered.sort(key=lambda x: getattr(x, 'composite_score', 0), reverse=True)
        return filtered[:20]  # Limit to top 20 predictions

    async def _store_predictions(self, predictions: List[TrendPrediction]) -> None:
        """Store predictions in the database for future reference."""
        try:
            if not predictions:
                return
            
            # Prepare batch insert data
            insert_data = []
            for prediction in predictions:
                insert_data.append({
                    'prediction_id': prediction.prediction_id,
                    'user_id': prediction.user_id,
                    'trend_type': prediction.trend_type.value,
                    'category': prediction.category,
                    'severity': prediction.severity.value,
                    'confidence': prediction.confidence.value,
                    'confidence_score': prediction.confidence_score,
                    'title': prediction.title,
                    'description': prediction.description,
                    'predicted_impact': prediction.predicted_impact,
                    'timeline': prediction.timeline,
                    'historical_data': prediction.historical_data,
                    'statistical_evidence': prediction.statistical_evidence,
                    'model_metrics': prediction.model_metrics,
                    'preparation_actions': prediction.preparation_actions,
                    'mitigation_strategies': prediction.mitigation_strategies,
                    'opportunities': prediction.opportunities,
                    'prediction_date': prediction.prediction_date,
                    'valid_until': prediction.valid_until,
                    'model_used': prediction.model_used,
                    'data_quality_score': prediction.data_quality_score
                })
            
            # Batch insert into database
            insert_query = """
                INSERT INTO trend_predictions 
                (prediction_id, user_id, trend_type, category, severity, confidence, 
                 confidence_score, title, description, predicted_impact, timeline, 
                 historical_data, statistical_evidence, model_metrics, preparation_actions, 
                 mitigation_strategies, opportunities, prediction_date, valid_until, 
                 model_used, data_quality_score)
                VALUES %(values)s
                ON CONFLICT (prediction_id) DO UPDATE SET
                    confidence_score = EXCLUDED.confidence_score,
                    predicted_impact = EXCLUDED.predicted_impact,
                    updated_at = CURRENT_TIMESTAMP
            """
            
            await self.db_connector.execute_batch_insert(insert_query, insert_data)
            self.logger.info(f"Stored {len(predictions)} trend predictions")
            
        except Exception as e:
            self.logger.error(f"Error storing predictions: {str(e)}")

    # Helper methods for generating recommendations

    def _calculate_trend_severity(
        self, 
        increase_percentage: float, 
        projected_amount: float, 
        budget_limit: float, 
        slope: float
    ) -> TrendSeverity:
        """Calculate severity based on trend characteristics."""
        if budget_limit > 0 and projected_amount > budget_limit * 1.3:
            return TrendSeverity.CRITICAL
        elif increase_percentage > 50 or (budget_limit > 0 and projected_amount > budget_limit):
            return TrendSeverity.HIGH
        elif increase_percentage > 25 or slope > projected_amount * 0.1:
            return TrendSeverity.MEDIUM
        else:
            return TrendSeverity.LOW

    def _get_confidence_level(self, confidence_score: float) -> PredictionConfidence:
        """Convert confidence score to confidence level."""
        if confidence_score >= 0.95:
            return PredictionConfidence.VERY_HIGH
        elif confidence_score >= 0.80:
            return PredictionConfidence.HIGH
        elif confidence_score >= 0.60:
            return PredictionConfidence.MEDIUM
        else:
            return PredictionConfidence.LOW

    def _month_name(self, month_num: int) -> str:
        """Convert month number to name."""
        months = [
            "", "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        return months[month_num] if 1 <= month_num <= 12 else "Unknown"

    # Recommendation generation methods

    def _generate_increase_preparations(
        self, 
        category: str, 
        increase_percentage: float, 
        projected_avg: float, 
        budget_limit: float
    ) -> List[Dict[str, Any]]:
        """Generate preparation actions for spending increases."""
        actions = []
        
        actions.append({
            "action": "budget_adjustment",
            "description": f"Consider increasing your {category} budget by {increase_percentage:.0f}% to accommodate the trend",
            "priority": "high" if increase_percentage > 30 else "medium",
            "estimated_impact": f"${projected_avg * (increase_percentage/100):.2f} additional budget needed"
        })
        
        if budget_limit > 0 and projected_avg > budget_limit:
            actions.append({
                "action": "spending_review",
                "description": f"Review your {category} spending patterns to identify optimization opportunities",
                "priority": "high",
                "estimated_impact": f"Could prevent ${projected_avg - budget_limit:.2f} budget overage"
            })
        
        actions.append({
            "action": "trend_monitoring",
            "description": f"Set up alerts to monitor {category} spending more closely",
            "priority": "medium",
            "estimated_impact": "Early warning system for budget management"
        })
        
        return actions

    def _generate_increase_mitigations(
        self, 
        category: str, 
        increase_percentage: float, 
        projected_avg: float
    ) -> List[Dict[str, Any]]:
        """Generate mitigation strategies for spending increases."""
        strategies = []
        
        if category.lower() in ['dining', 'food', 'restaurants']:
            strategies.extend([
                {
                    "strategy": "meal_planning",
                    "description": "Implement weekly meal planning to reduce impulsive dining out",
                    "effort": "medium",
                    "potential_savings": f"${projected_avg * 0.2:.2f}/month"
                },
                {
                    "strategy": "cooking_at_home",
                    "description": "Increase home cooking frequency by 2-3 meals per week",
                    "effort": "medium",
                    "potential_savings": f"${projected_avg * 0.3:.2f}/month"
                }
            ])
        
        elif category.lower() in ['shopping', 'retail', 'clothing']:
            strategies.extend([
                {
                    "strategy": "waiting_period",
                    "description": "Implement a 48-hour waiting period before non-essential purchases",
                    "effort": "low",
                    "potential_savings": f"${projected_avg * 0.25:.2f}/month"
                },
                {
                    "strategy": "shopping_list",
                    "description": "Create and stick to a shopping list to avoid impulse purchases",
                    "effort": "low",
                    "potential_savings": f"${projected_avg * 0.15:.2f}/month"
                }
            ])
        
        # Generic strategies
        strategies.append({
            "strategy": "spending_limit",
            "description": f"Set a daily/weekly spending limit for {category}",
            "effort": "low",
            "potential_savings": f"${projected_avg * 0.1:.2f}/month"
        })
        
        return strategies

    def _generate_increase_opportunities(
        self, 
        category: str, 
        increase_percentage: float
    ) -> List[Dict[str, Any]]:
        """Generate opportunities from spending increases."""
        opportunities = []
        
        if increase_percentage > 20:
            opportunities.append({
                "opportunity": "cashback_optimization",
                "description": f"Since your {category} spending is increasing, optimize your credit card rewards for this category",
                "potential_benefit": f"Up to 2-5% cashback on increased spending",
                "action_required": "Review and switch to category-optimized credit cards"
            })
        
        opportunities.append({
            "opportunity": "bulk_purchasing",
            "description": f"Consider bulk purchasing for {category} items to reduce unit costs",
            "potential_benefit": "5-15% savings through bulk discounts",
            "action_required": "Identify frequently purchased items suitable for bulk buying"
        })
        
        return opportunities

    def _generate_decrease_opportunities(
        self, 
        category: str, 
        decrease_percentage: float, 
        potential_savings: float
    ) -> List[Dict[str, Any]]:
        """Generate opportunities from spending decreases."""
        return [
            {
                "opportunity": "savings_reallocation",
                "description": f"Redirect the ${potential_savings:.2f} savings from reduced {category} spending to your financial goals",
                "potential_benefit": f"${potential_savings:.2f} additional monthly savings",
                "action_required": "Set up automatic transfer to savings account"
            },
            {
                "opportunity": "goal_acceleration",
                "description": f"Use the savings to accelerate progress on your financial goals",
                "potential_benefit": "Faster goal achievement",
                "action_required": "Allocate savings to highest priority financial goal"
            }
        ]

    def _generate_seasonal_preparations(
        self, 
        category: str, 
        peak_month: int, 
        seasonal_increase: float
    ) -> List[Dict[str, Any]]:
        """Generate seasonal preparation actions."""
        month_name = self._month_name(peak_month)
        
        return [
            {
                "action": "seasonal_budget_adjustment",
                "description": f"Prepare for {seasonal_increase:.0f}% increase in {category} spending during {month_name}",
                "priority": "high",
                "timing": f"Start preparation 1 month before {month_name}"
            },
            {
                "action": "advance_saving",
                "description": f"Set aside extra funds in the months leading up to {month_name}",
                "priority": "medium",
                "timing": "2-3 months before peak season"
            },
            {
                "action": "early_shopping",
                "description": f"Consider making some {category} purchases early to spread costs",
                "priority": "medium",
                "timing": f"1-2 months before {month_name}"
            }
        ]

    def _generate_seasonal_mitigations(
        self, 
        category: str, 
        seasonal_increase: float
    ) -> List[Dict[str, Any]]:
        """Generate seasonal mitigation strategies."""
        return [
            {
                "strategy": "seasonal_spending_plan",
                "description": f"Create a specific spending plan for the {seasonal_increase:.0f}% seasonal increase",
                "effort": "medium",
                "potential_savings": f"Better control over seasonal spending spikes"
            },
            {
                "strategy": "alternative_timing",
                "description": "Shift some purchases to off-peak times when possible",
                "effort": "low",
                "potential_savings": "5-15% through off-peak pricing"
            }
        ]

    def _generate_seasonal_opportunities(
        self, 
        category: str, 
        pattern: SeasonalPattern
    ) -> List[Dict[str, Any]]:
        """Generate seasonal opportunities."""
        opportunities = []
        
        if pattern.low_months:
            low_month_name = self._month_name(pattern.low_months[0])
            opportunities.append({
                "opportunity": "low_season_stocking",
                "description": f"Stock up on {category} items during {low_month_name} when spending is typically lower",
                "potential_benefit": "10-20% savings through strategic timing",
                "action_required": f"Plan purchases for {low_month_name}"
            })
        
        return opportunities

    def _generate_budget_challenge_preparations(
        self, 
        category: str, 
        utilization_rate: float, 
        days_until_limit: Optional[float]
    ) -> List[Dict[str, Any]]:
        """Generate budget challenge preparation actions."""
        actions = []
        
        if days_until_limit and days_until_limit < 15:
            actions.append({
                "action": "immediate_spending_reduction",
                "description": f"Reduce {category} spending immediately - only {days_until_limit:.0f} days until budget limit",
                "priority": "critical",
                "urgency": "immediate"
            })
        
        actions.extend([
            {
                "action": "spending_review",
                "description": f"Review all {category} transactions this month to identify unnecessary expenses",
                "priority": "high",
                "urgency": "within_24_hours"
            },
            {
                "action": "alternative_planning",
                "description": f"Find lower-cost alternatives for remaining {category} needs this month",
                "priority": "medium",
                "urgency": "within_week"
            }
        ])
        
        return actions

    def _generate_budget_challenge_mitigations(
        self, 
        category: str, 
        utilization_rate: float, 
        overage_amount: float
    ) -> List[Dict[str, Any]]:
        """Generate budget challenge mitigation strategies."""
        return [
            {
                "strategy": "spending_freeze",
                "description": f"Implement a temporary freeze on non-essential {category} spending",
                "effort": "high",
                "potential_savings": f"${overage_amount:.2f} budget overage prevention"
            },
            {
                "strategy": "substitute_alternatives",
                "description": f"Switch to lower-cost alternatives for {category} purchases",
                "effort": "medium",
                "potential_savings": f"${overage_amount * 0.5:.2f} potential reduction"
            },
            {
                "strategy": "defer_purchases",
                "description": f"Defer non-urgent {category} purchases to next month",
                "effort": "low",
                "potential_savings": f"Stay within budget limit"
            }
        ]

    def _generate_budget_challenge_opportunities(
        self, 
        category: str, 
        budget_limit: float, 
        projected_spending: float
    ) -> List[Dict[str, Any]]:
        """Generate budget challenge opportunities."""
        return [
            {
                "opportunity": "budget_rebalancing",
                "description": "Reallocate budget from under-utilized categories to prevent overage",
                "potential_benefit": f"Prevent ${projected_spending - budget_limit:.2f} overage",
                "action_required": "Review and adjust budget allocations"
            },
            {
                "opportunity": "spending_optimization",
                "description": f"Use this challenge to optimize your {category} spending habits long-term",
                "potential_benefit": "Improved budgeting discipline and awareness",
                "action_required": "Implement spending tracking and review processes"
            }
        ]

    def _generate_emerging_category_preparations(
        self, 
        category: str, 
        recent_spending: float, 
        transaction_count: int
    ) -> List[Dict[str, Any]]:
        """Generate emerging category preparation actions."""
        return [
            {
                "action": "budget_allocation",
                "description": f"Consider adding {category} to your budget with ${recent_spending/2:.2f}/month allocation",
                "priority": "medium",
                "rationale": f"Based on {transaction_count} transactions totaling ${recent_spending:.2f}"
            },
            {
                "action": "spending_tracking",
                "description": f"Set up dedicated tracking for {category} spending to monitor growth",
                "priority": "medium",
                "rationale": "Early tracking prevents uncontrolled spending growth"
            },
            {
                "action": "goal_integration",
                "description": f"Evaluate if {category} spending aligns with your financial goals",
                "priority": "low",
                "rationale": "Ensure new spending supports overall financial objectives"
            }
        ]

    def _generate_emerging_category_opportunities(
        self, 
        category: str, 
        recent_spending: float
    ) -> List[Dict[str, Any]]:
        """Generate emerging category opportunities."""
        return [
            {
                "opportunity": "rewards_optimization",
                "description": f"Optimize credit card rewards for your new {category} spending pattern",
                "potential_benefit": f"Up to ${recent_spending * 0.02:.2f}/month in additional rewards",
                "action_required": "Review credit card categories and benefits"
            },
            {
                "opportunity": "bulk_discounts",
                "description": f"Explore bulk purchasing options for {category} items",
                "potential_benefit": "5-15% savings through volume discounts",
                "action_required": "Research bulk purchasing opportunities"
            }
        ]

    def _generate_category_growth_preparations(
        self, 
        category: str, 
        growth_rate: float, 
        recent_monthly_avg: float
    ) -> List[Dict[str, Any]]:
        """Generate category growth preparation actions."""
        return [
            {
                "action": "budget_review",
                "description": f"Review {category} budget - spending has grown {growth_rate*100:.1f}% to ${recent_monthly_avg:.2f}/month",
                "priority": "high" if growth_rate > 1.0 else "medium",
                "urgency": "within_week"
            },
            {
                "action": "spending_analysis",
                "description": f"Analyze what's driving the {growth_rate*100:.1f}% increase in {category} spending",
                "priority": "medium",
                "urgency": "within_two_weeks"
            }
        ]

    def _generate_category_growth_mitigations(
        self, 
        category: str, 
        growth_rate: float
    ) -> List[Dict[str, Any]]:
        """Generate category growth mitigation strategies."""
        if growth_rate < 0.7:  # Only provide mitigations for significant growth
            return []
        
        return [
            {
                "strategy": "spending_cap",
                "description": f"Set a monthly spending cap for {category} to control growth",
                "effort": "low",
                "potential_savings": f"Prevent uncontrolled spending growth"
            },
            {
                "strategy": "value_assessment",
                "description": f"Assess if the increased {category} spending provides proportional value",
                "effort": "medium",
                "potential_savings": "Eliminate low-value spending"
            }
        ]

    def _generate_category_growth_opportunities(
        self, 
        category: str, 
        growth_rate: float
    ) -> List[Dict[str, Any]]:
        """Generate category growth opportunities."""
        return [
            {
                "opportunity": "loyalty_programs",
                "description": f"Join loyalty programs for frequently purchased {category} items",
                "potential_benefit": f"5-10% savings through loyalty rewards",
                "action_required": "Research and join relevant loyalty programs"
            },
            {
                "opportunity": "subscription_services",
                "description": f"Consider subscription services for regular {category} purchases",
                "potential_benefit": "10-20% savings through subscription discounts",
                "action_required": "Evaluate subscription options for frequent purchases"
            }
        ]

    def _generate_behavior_shift_preparations(self, shift: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate behavior shift preparation actions."""
        shift_type = shift['shift_type']
        
        if 'weekend' in shift_type:
            return [
                {
                    "action": "weekend_budget_planning",
                    "description": "Adjust your weekend spending budget to account for the behavioral shift",
                    "priority": "medium",
                    "timing": "before_next_weekend"
                },
                {
                    "action": "weekend_alternatives",
                    "description": "Plan alternative weekend activities that align with your financial goals",
                    "priority": "medium", 
                    "timing": "weekly_planning"
                }
            ]
        
        elif 'transaction_size' in shift_type:
            return [
                {
                    "action": "spending_awareness",
                    "description": "Set up transaction size alerts to maintain awareness of changing spending patterns",
                    "priority": "high",
                    "timing": "immediate"
                },
                {
                    "action": "payment_method_review",
                    "description": "Review if changing transaction sizes warrant different payment methods or cards",
                    "priority": "low",
                    "timing": "monthly_review"
                }
            ]
        
        elif 'frequency' in shift_type:
            category = shift.get('category', 'general')
            return [
                {
                    "action": "frequency_monitoring",
                    "description": f"Monitor your {category} spending frequency to understand the underlying cause",
                    "priority": "medium",
                    "timing": "ongoing"
                },
                {
                    "action": "habit_adjustment",
                    "description": f"Consider if the frequency change in {category} aligns with your financial goals",
                    "priority": "medium",
                    "timing": "weekly_review"
                }
            ]
        
        # Generic behavior shift preparations
        return [
            {
                "action": "pattern_awareness",
                "description": "Increase awareness of the detected behavioral shift",
                "priority": "medium",
                "timing": "immediate"
            },
            {
                "action": "impact_assessment",
                "description": "Assess whether this behavioral shift supports your financial goals",
                "priority": "medium",
                "timing": "within_week"
            }
        ]

    def _generate_behavior_shift_mitigations(self, shift: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate behavior shift mitigation strategies."""
        shift_type = shift['shift_type']
        magnitude = shift.get('magnitude', 0)
        
        # Only provide mitigations for significant negative shifts
        if magnitude < 0.2:
            return []
        
        strategies = []
        
        if 'weekend' in shift_type and 'increase' in shift.get('description', '').lower():
            strategies.extend([
                {
                    "strategy": "weekend_spending_cap",
                    "description": "Set a strict weekend spending limit to control increased weekend spending",
                    "effort": "low",
                    "potential_savings": "10-20% reduction in weekend overspending"
                },
                {
                    "strategy": "weekend_activity_planning",
                    "description": "Plan lower-cost weekend activities in advance",
                    "effort": "medium",
                    "potential_savings": "15-30% weekend spending reduction"
                }
            ])
        
        elif 'transaction_size' in shift_type and 'increase' in shift.get('description', '').lower():
            strategies.extend([
                {
                    "strategy": "purchase_approval_process",
                    "description": "Implement a mental checklist for larger purchases",
                    "effort": "low",
                    "potential_savings": "Prevent unnecessary large purchases"
                },
                {
                    "strategy": "spending_cooling_period",
                    "description": "Wait 24 hours before making purchases above your historical average",
                    "effort": "medium",
                    "potential_savings": "Reduce impulse large purchases by 20-40%"
                }
            ])
        
        elif 'frequency' in shift_type and 'increase' in shift.get('description', '').lower():
            category = shift.get('category', 'general')
            strategies.append({
                "strategy": "frequency_limits",
                "description": f"Set weekly or monthly limits on {category} purchase frequency",
                "effort": "medium",
                "potential_savings": f"Control {category} spending frequency"
            })
        
        return strategies

    def _generate_behavior_shift_opportunities(self, shift: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate behavior shift opportunities."""
        opportunities = []
        shift_type = shift['shift_type']
        
        if 'decrease' in shift.get('description', '').lower():
            # Positive behavioral shifts create opportunities
            opportunities.append({
                "opportunity": "savings_reallocation",
                "description": "Redirect savings from improved spending behavior to financial goals",
                "potential_benefit": "Accelerated progress toward financial objectives",
                "action_required": "Set up automatic savings transfer"
            })
        
        if 'transaction_size' in shift_type:
            opportunities.append({
                "opportunity": "rewards_optimization",
                "description": "Optimize credit card rewards based on new transaction size patterns",
                "potential_benefit": "1-3% additional rewards on changed spending pattern",
                "action_required": "Review and adjust credit card strategy"
            })
        
        if 'frequency' in shift_type:
            opportunities.append({
                "opportunity": "loyalty_program_optimization",
                "description": "Adjust loyalty program participation based on changed purchase frequency",
                "potential_benefit": "Maximized loyalty rewards and benefits",
                "action_required": "Review current loyalty program enrollments"
            })
        
        return opportunities

    def _generate_merchant_loyalty_preparations(
        self, 
        merchant: str, 
        change_type: str, 
        spending_trend: float
    ) -> List[Dict[str, Any]]:
        """Generate merchant loyalty preparation actions."""
        if change_type == "increasing":
            return [
                {
                    "action": "loyalty_program_enrollment",
                    "description": f"Join {merchant}'s loyalty program to maximize benefits from increased spending",
                    "priority": "medium",
                    "potential_benefit": f"5-10% rewards on ${abs(spending_trend) * 12:.2f} annual spending increase"
                },
                {
                    "action": "spending_optimization",
                    "description": f"Look for ways to optimize your {merchant} spending (bulk purchases, sales timing)",
                    "priority": "low",
                    "potential_benefit": "10-15% savings through strategic timing"
                }
            ]
        else:  # decreasing
            return [
                {
                    "action": "alternative_evaluation",
                    "description": f"Evaluate if your reduced {merchant} spending indicates a preference for alternatives",
                    "priority": "low",
                    "potential_benefit": "Ensure spending changes align with preferences"
                },
                {
                    "action": "loyalty_review",
                    "description": f"Review if {merchant} loyalty programs still provide value with reduced spending",
                    "priority": "low",
                    "potential_benefit": "Avoid unused loyalty program commitments"
                }
            ]

    def _generate_merchant_loyalty_mitigations(
        self, 
        merchant: str, 
        change_type: str
    ) -> List[Dict[str, Any]]:
        """Generate merchant loyalty mitigation strategies."""
        if change_type == "decreasing":
            return [
                {
                    "strategy": "merchant_diversity",
                    "description": f"Ensure you're not over-dependent on alternatives to {merchant}",
                    "effort": "low",
                    "potential_savings": "Maintain competitive pricing through choice"
                },
                {
                    "strategy": "loyalty_value_assessment",
                    "description": f"Assess if loyalty benefits at {merchant} justify maintaining minimum spending",
                    "effort": "low",
                    "potential_savings": "Avoid suboptimal loyalty spending"
                }
            ]
        return []

    def _generate_merchant_loyalty_opportunities(
        self, 
        merchant: str, 
        change_type: str, 
        avg_spending: float
    ) -> List[Dict[str, Any]]:
        """Generate merchant loyalty opportunities."""
        opportunities = []
        
        if change_type == "increasing":
            opportunities.extend([
                {
                    "opportunity": "bulk_purchasing",
                    "description": f"Consider bulk purchasing at {merchant} to maximize value from increased loyalty",
                    "potential_benefit": f"5-15% savings on ${avg_spending * 12:.2f} annual spending",
                    "action_required": "Identify bulk-purchase opportunities"
                },
                {
                    "opportunity": "seasonal_timing",
                    "description": f"Time large purchases at {merchant} during sales periods",
                    "potential_benefit": "10-30% savings during promotional periods",
                    "action_required": "Track merchant sales patterns"
                }
            ])
        
        opportunities.append({
            "opportunity": "referral_benefits",
            "description": f"Explore referral benefits or family programs at {merchant}",
            "potential_benefit": "Additional rewards through referrals",
            "action_required": "Research merchant referral programs"
        })
        
        return opportunities

    def _generate_payment_method_preparations(
        self, 
        payment_method: str, 
        shift_direction: str, 
        current_usage: float
    ) -> List[Dict[str, Any]]:
        """Generate payment method preparation actions."""
        if shift_direction == "increasing":
            return [
                {
                    "action": "optimize_rewards",
                    "description": f"Ensure {payment_method} is optimized for your spending categories",
                    "priority": "medium",
                    "potential_benefit": f"Maximize rewards on {current_usage*100:.1f}% of spending"
                },
                {
                    "action": "credit_utilization_monitoring",
                    "description": f"Monitor credit utilization if {payment_method} usage continues increasing",
                    "priority": "high" if "credit" in payment_method.lower() else "low",
                    "potential_benefit": "Maintain healthy credit utilization ratio"
                }
            ]
        else:  # decreasing
            return [
                {
                    "action": "alternative_assessment",
                    "description": f"Ensure your alternatives to {payment_method} are optimally configured",
                    "priority": "medium",
                    "potential_benefit": "Maintain payment method optimization"
                },
                {
                    "action": "account_maintenance_review",
                    "description": f"Review if reduced {payment_method} usage affects account benefits or fees",
                    "priority": "low",
                    "potential_benefit": "Avoid unnecessary fees on unused payment methods"
                }
            ]

    def _generate_payment_method_opportunities(
        self, 
        payment_method: str, 
        shift_direction: str
    ) -> List[Dict[str, Any]]:
        """Generate payment method opportunities."""
        opportunities = []
        
        if shift_direction == "increasing":
            opportunities.append({
                "opportunity": "rewards_maximization",
                "description": f"Maximize rewards categories for increased {payment_method} usage",
                "potential_benefit": "1-3% additional rewards through category optimization",
                "action_required": "Review and adjust rewards categories"
            })
        
        opportunities.extend([
            {
                "opportunity": "payment_method_diversification",
                "description": "Ensure optimal payment method mix for different spending categories",
                "potential_benefit": "2-5% improvement in overall rewards and benefits",
                "action_required": "Analyze payment method effectiveness by category"
            },
            {
                "opportunity": "financial_product_optimization",
                "description": f"Consider if {payment_method} changes indicate need for different financial products",
                "potential_benefit": "Better financial product alignment with spending behavior",
                "action_required": "Review current financial product portfolio"
            }
        ])
        
        return opportunities

    async def get_user_trend_summary(
        self, 
        user_id: str, 
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Get a comprehensive trend summary for a user.
        
        Args:
            user_id: Target user ID
            days_back: How many days back to include in summary
            
        Returns:
            Comprehensive trend summary
        """
        try:
            # Get recent predictions
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            query = """
                SELECT prediction_id, trend_type, category, severity, confidence,
                       title, description, predicted_impact, prediction_date, valid_until
                FROM trend_predictions 
                WHERE user_id = %s 
                    AND prediction_date >= %s 
                    AND valid_until > CURRENT_TIMESTAMP
                ORDER BY prediction_date DESC
            """
            
            result = await self.db_connector.execute_query(query, [user_id, cutoff_date])
            predictions = result.get('data', [])
            
            # Categorize predictions
            summary = {
                'user_id': user_id,
                'summary_period': f'last_{days_back}_days',
                'generated_at': datetime.now().isoformat(),
                'total_predictions': len(predictions),
                'predictions_by_type': {},
                'predictions_by_severity': {},
                'active_challenges': [],
                'opportunities': [],
                'top_concerns': [],
                'trend_categories': set(),
                'recommendations_count': 0
            }
            
            for pred in predictions:
                trend_type = pred['trend_type']
                severity = pred['severity']
                
                # Count by type and severity
                summary['predictions_by_type'][trend_type] = summary['predictions_by_type'].get(trend_type, 0) + 1
                summary['predictions_by_severity'][severity] = summary['predictions_by_severity'].get(severity, 0) + 1
                
                # Track categories
                summary['trend_categories'].add(pred['category'])
                
                # Categorize for user-friendly summary
                if pred['trend_type'] in ['BUDGET_CHALLENGE', 'SPENDING_INCREASE'] and severity in ['HIGH', 'CRITICAL']:
                    summary['active_challenges'].append({
                        'title': pred['title'],
                        'description': pred['description'],
                        'category': pred['category'],
                        'severity': severity,
                        'predicted_impact': pred.get('predicted_impact', {})
                    })
                
                elif pred['trend_type'] in ['SPENDING_DECREASE', 'EMERGING_CATEGORY'] and severity in ['LOW', 'MEDIUM']:
                    summary['opportunities'].append({
                        'title': pred['title'],
                        'description': pred['description'],
                        'category': pred['category'],
                        'predicted_impact': pred.get('predicted_impact', {})
                    })
                
                if severity in ['HIGH', 'CRITICAL']:
                    summary['top_concerns'].append({
                        'title': pred['title'],
                        'category': pred['category'],
                        'severity': severity,
                        'valid_until': pred['valid_until']
                    })
            
            # Convert set to list for JSON serialization
            summary['trend_categories'] = list(summary['trend_categories'])
            
            # Sort by severity/importance
            summary['top_concerns'].sort(key=lambda x: {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}[x['severity']], reverse=True)
            summary['active_challenges'] = summary['active_challenges'][:5]  # Top 5
            summary['opportunities'] = summary['opportunities'][:5]  # Top 5
            summary['top_concerns'] = summary['top_concerns'][:3]  # Top 3
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating trend summary: {str(e)}")
            return {
                'user_id': user_id,
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }

    async def invalidate_expired_predictions(self) -> int:
        """
        Clean up expired predictions from the database.
        
        Returns:
            Number of predictions invalidated
        """
        try:
            query = """
                UPDATE trend_predictions 
                SET updated_at = CURRENT_TIMESTAMP
                WHERE valid_until < CURRENT_TIMESTAMP 
                    AND deleted_at IS NULL
                RETURNING prediction_id
            """
            
            result = await self.db_connector.execute_query(query, [])
            invalidated_count = len(result.get('data', []))
            
            if invalidated_count > 0:
                self.logger.info(f"Invalidated {invalidated_count} expired predictions")
            
            return invalidated_count
            
        except Exception as e:
            self.logger.error(f"Error invalidating expired predictions: {str(e)}")
            return 0

    async def get_prediction_performance_metrics(self, days_back: int = 90) -> Dict[str, Any]:
        """
        Analyze the performance of past predictions for model improvement.
        
        Args:
            days_back: Days back to analyze
            
        Returns:
            Performance metrics for different prediction types
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            query = """
                SELECT 
                    trend_type,
                    confidence,
                    severity,
                    model_used,
                    data_quality_score,
                    predicted_impact,
                    COUNT(*) as prediction_count,
                    AVG(confidence_score) as avg_confidence,
                    AVG(data_quality_score) as avg_data_quality
                FROM trend_predictions 
                WHERE prediction_date >= %s
                GROUP BY trend_type, confidence, severity, model_used, data_quality_score, predicted_impact
                ORDER BY trend_type, avg_confidence DESC
            """
            
            result = await self.db_connector.execute_query(query, [cutoff_date])
            performance_data = result.get('data', [])
            
            metrics = {
                'analysis_period': f'last_{days_back}_days',
                'total_predictions': sum(row['prediction_count'] for row in performance_data),
                'by_trend_type': {},
                'by_confidence_level': {},
                'by_model': {},
                'overall_metrics': {
                    'avg_confidence': 0,
                    'avg_data_quality': 0,
                    'most_common_type': None,
                    'highest_confidence_type': None
                }
            }
            
            # Aggregate metrics
            for row in performance_data:
                trend_type = row['trend_type']
                confidence = row['confidence']
                model = row['model_used']
                
                # By trend type
                if trend_type not in metrics['by_trend_type']:
                    metrics['by_trend_type'][trend_type] = {
                        'count': 0,
                        'avg_confidence': 0,
                        'avg_data_quality': 0
                    }
                
                metrics['by_trend_type'][trend_type]['count'] += row['prediction_count']
                metrics['by_trend_type'][trend_type]['avg_confidence'] = row['avg_confidence']
                metrics['by_trend_type'][trend_type]['avg_data_quality'] = row['avg_data_quality']
                
                # By confidence level
                if confidence not in metrics['by_confidence_level']:
                    metrics['by_confidence_level'][confidence] = 0
                metrics['by_confidence_level'][confidence] += row['prediction_count']
                
                # By model
                if model not in metrics['by_model']:
                    metrics['by_model'][model] = 0
                metrics['by_model'][model] += row['prediction_count']
            
            # Calculate overall metrics
            if performance_data:
                total_predictions = sum(row['prediction_count'] for row in performance_data)
                weighted_confidence = sum(row['avg_confidence'] * row['prediction_count'] for row in performance_data)
                weighted_quality = sum(row['avg_data_quality'] * row['prediction_count'] for row in performance_data)
                
                metrics['overall_metrics']['avg_confidence'] = weighted_confidence / total_predictions if total_predictions > 0 else 0
                metrics['overall_metrics']['avg_data_quality'] = weighted_quality / total_predictions if total_predictions > 0 else 0
                
                # Most common type
                most_common = max(metrics['by_trend_type'].items(), key=lambda x: x[1]['count'])
                metrics['overall_metrics']['most_common_type'] = most_common[0]
                
                # Highest confidence type
                highest_conf = max(metrics['by_trend_type'].items(), key=lambda x: x[1]['avg_confidence'])
                metrics['overall_metrics']['highest_confidence_type'] = highest_conf[0]
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing prediction performance: {str(e)}")
            return {'error': str(e)}

    async def cleanup_resources(self):
        """Clean up resources and close connections."""
        try:
            # Clear caches
            self.model_cache.clear()
            self.pattern_cache.clear()
            
            # Close database connection if needed
            if hasattr(self.db_connector, 'close'):
                await self.db_connector.close()
            
            self.logger.info("Trend Prediction Tool resources cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")