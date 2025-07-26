import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, date
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import asyncpg
from decimal import Decimal
import statistics
import numpy as np

class AnalysisType(Enum):
    """Types of analysis that can be scheduled."""
    BUDGET_CHECK = "budget_check"
    PATTERN_ANALYSIS = "pattern_analysis"
    SEASONAL_REVIEW = "seasonal_review"
    GOAL_TRACKING = "goal_tracking"
    ANOMALY_DETECTION = "anomaly_detection"
    TREND_DETECTION = "trend_detection"

class MonitoringFrequency(Enum):
    """Monitoring frequency options."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

class RiskLevel(Enum):
    """Risk levels for budget and spending analysis."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class MonitoringSchedule:
    """Represents a monitoring schedule for a user and analysis type."""
    user_id: str
    analysis_type: AnalysisType
    frequency: MonitoringFrequency
    next_run_time: datetime
    last_run_time: Optional[datetime] = None
    priority: int = 5  # 1-10, higher = more critical
    conditions: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.conditions is None:
            self.conditions = {}
        if self.metadata is None:
            self.metadata = {}

@dataclass
class BudgetStatus:
    """Represents the current status of a user's budget."""
    user_id: str
    category: str
    current_spent: Decimal
    budget_limit: Decimal
    utilization_percentage: float
    days_remaining: int
    projected_overage: Decimal
    risk_level: RiskLevel
    last_updated: datetime
    spending_velocity: float  # Amount per day
    historical_average: Decimal

@dataclass
class PatternAnalysis:
    """Results from spending pattern analysis."""
    user_id: str
    analysis_period: Tuple[date, date]
    spending_patterns: Dict[str, Any]
    trend_indicators: Dict[str, Any]
    seasonal_factors: Dict[str, Any]
    anomalies_detected: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    recommendations: List[Dict[str, Any]]

@dataclass
class PredictionResult:
    """Results from budget overage prediction."""
    category: str
    projected_overage: Decimal
    confidence: float
    days_until_overage: int
    recommended_daily_limit: Decimal

class ContinuousMonitoringTool:
    """
    Tool for continuous monitoring of user spending data, budget thresholds,
    and pattern detection.
    """

    def __init__(self, db_connection_string: str):
        self.db_connection_string = db_connection_string
        self.connection_pool: Optional[asyncpg.Pool] = None
        self.logger = logging.getLogger(__name__)
        
        # Default monitoring schedules
        self.default_schedules = {
            'budget_critical': {
                'frequency': MonitoringFrequency.HOURLY,
                'trigger_condition': 'budget_usage > 90%',
                'analysis_depth': 'detailed',
                'priority': 10
            },
            'budget_warning': {
                'frequency': MonitoringFrequency.DAILY,
                'trigger_condition': 'budget_usage > 75%',
                'analysis_depth': 'standard',
                'priority': 7
            },
            'pattern_analysis': {
                'frequency': MonitoringFrequency.WEEKLY,
                'trigger_condition': 'always',
                'analysis_depth': 'comprehensive',
                'priority': 5
            },
            'seasonal_review': {
                'frequency': MonitoringFrequency.MONTHLY,
                'trigger_condition': 'month_boundary',
                'analysis_depth': 'historical_comparison',
                'priority': 4
            },
            'goal_tracking': {
                'frequency': MonitoringFrequency.DAILY,
                'trigger_condition': 'active_goals_exist',
                'analysis_depth': 'progress_focused',
                'priority': 6
            }
        }

    async def initialize(self):
        """Initialize the database connection pool."""
        try:
            self.connection_pool = await asyncpg.create_pool(
                self.db_connection_string,
                min_size=5,
                max_size=20,
                command_timeout=30
            )
            self.logger.info("Database connection pool initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize database pool: {str(e)}")
            raise

    async def close(self):
        """Close the database connection pool."""
        if self.connection_pool:
            await self.connection_pool.close()
            self.logger.info("Database connection pool closed")

    # === Analysis Scheduling Methods ===

    async def schedule_user_analysis(self, user_id: str, analysis_type: AnalysisType, priority: int = 5) -> bool:
        """Schedule analysis for a specific user and analysis type."""
        try:
            async with self.connection_pool.acquire() as conn:
                # Get user preferences for monitoring frequency
                user_prefs = await self._get_user_monitoring_preferences(conn, user_id)
                frequency = self._determine_frequency(analysis_type, user_prefs, priority)
                
                # Calculate next run time
                next_run_time = self._calculate_next_run_time(frequency, priority)
                
                schedule = MonitoringSchedule(
                    user_id=user_id,
                    analysis_type=analysis_type,
                    frequency=frequency,
                    next_run_time=next_run_time,
                    priority=priority
                )
                
                # Store schedule in database
                await self._store_monitoring_schedule(conn, schedule)
                
                self.logger.info(f"Scheduled {analysis_type.value} analysis for user {user_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error scheduling analysis: {str(e)}")
            return False

    async def get_next_analysis_time(self, user_id: str, analysis_type: AnalysisType) -> Optional[datetime]:
        """Get the next scheduled analysis time for a user and analysis type."""
        try:
            async with self.connection_pool.acquire() as conn:
                query = """
                SELECT next_run_time FROM monitoring_schedules 
                WHERE user_id = $1 AND analysis_type = $2 AND is_active = true
                """
                result = await conn.fetchval(query, user_id, analysis_type.value)
                return result
                
        except Exception as e:
            self.logger.error(f"Error getting next analysis time: {str(e)}")
            return None

    async def should_run_analysis(self, user_id: str, analysis_type: AnalysisType) -> bool:
        """Check if analysis should be run based on schedule and conditions."""
        try:
            async with self.connection_pool.acquire() as conn:
                # Get schedule info
                query = """
                SELECT next_run_time, conditions, priority 
                FROM monitoring_schedules 
                WHERE user_id = $1 AND analysis_type = $2 AND is_active = true
                """
                result = await conn.fetchrow(query, user_id, analysis_type.value)
                
                if not result:
                    return False
                
                # Check if it's time to run
                if datetime.now() < result['next_run_time']:
                    return False
                
                # Check additional conditions
                conditions = result['conditions'] or {}
                if not await self._check_analysis_conditions(conn, user_id, analysis_type, conditions):
                    return False
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error checking if analysis should run: {str(e)}")
            return False

    async def update_analysis_schedule(self, user_id: str, new_schedule: Dict[str, Any]) -> bool:
        """Update analysis schedule for a user."""
        try:
            async with self.connection_pool.acquire() as conn:
                # Update monitoring preferences
                query = """
                UPDATE users 
                SET preferences = preferences || $2
                WHERE firebase_uid = $1
                """
                monitoring_prefs = {"monitoring": new_schedule}
                await conn.execute(query, user_id, json.dumps(monitoring_prefs))
                
                # Reschedule existing analyses
                for analysis_type_str, config in new_schedule.items():
                    try:
                        analysis_type = AnalysisType(analysis_type_str)
                        await self.schedule_user_analysis(
                            user_id, 
                            analysis_type, 
                            config.get('priority', 5)
                        )
                    except ValueError:
                        self.logger.warning(f"Unknown analysis type: {analysis_type_str}")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating analysis schedule: {str(e)}")
            return False

    # === Budget Monitoring Methods ===

    async def check_budget_status(self, user_id: str, category: str = None) -> List[BudgetStatus]:
        """Check current budget status for user, optionally filtered by category."""
        try:
            async with self.connection_pool.acquire() as conn:
                # Build query based on whether category is specified
                if category:
                    category_filter = "AND bl.category = $2"
                    params = [user_id, category]
                else:
                    category_filter = ""
                    params = [user_id]
                
                query = f"""
                WITH current_spending AS (
                    SELECT 
                        category,
                        SUM(amount) as spent,
                        COUNT(*) as transaction_count,
                        AVG(amount) as avg_transaction
                    FROM transactions 
                    WHERE user_id = (SELECT user_id FROM users WHERE firebase_uid = $1)
                        AND transaction_date >= date_trunc('month', CURRENT_DATE)
                        AND deleted_at IS NULL
                    GROUP BY category
                ),
                daily_average AS (
                    SELECT 
                        category, 
                        AVG(daily_total) as avg_daily,
                        STDDEV(daily_total) as stddev_daily
                    FROM (
                        SELECT 
                            category, 
                            transaction_date, 
                            SUM(amount) as daily_total
                        FROM transactions
                        WHERE user_id = (SELECT user_id FROM users WHERE firebase_uid = $1)
                            AND transaction_date >= CURRENT_DATE - INTERVAL '30 days'
                            AND deleted_at IS NULL
                        GROUP BY category, transaction_date
                    ) daily_sums
                    GROUP BY category
                ),
                historical_average AS (
                    SELECT 
                        category,
                        AVG(monthly_total) as historical_monthly_avg
                    FROM (
                        SELECT 
                            category,
                            date_trunc('month', transaction_date) as month,
                            SUM(amount) as monthly_total
                        FROM transactions
                        WHERE user_id = (SELECT user_id FROM users WHERE firebase_uid = $1)
                            AND transaction_date >= CURRENT_DATE - INTERVAL '12 months'
                            AND deleted_at IS NULL
                        GROUP BY category, date_trunc('month', transaction_date)
                    ) monthly_sums
                    GROUP BY category
                )
                SELECT 
                    bl.category,
                    bl.limit_amount,
                    COALESCE(cs.spent, 0) as current_spent,
                    (COALESCE(cs.spent, 0) / bl.limit_amount * 100) as utilization_pct,
                    (DATE_PART('day', date_trunc('month', CURRENT_DATE) + INTERVAL '1 month' - INTERVAL '1 day') - DATE_PART('day', CURRENT_DATE)) as days_remaining,
                    COALESCE(da.avg_daily, 0) as spending_velocity,
                    COALESCE(ha.historical_monthly_avg, 0) as historical_average,
                    bl.last_calculated
                FROM budget_limits bl
                LEFT JOIN current_spending cs ON bl.category = cs.category
                LEFT JOIN daily_average da ON bl.category = da.category
                LEFT JOIN historical_average ha ON bl.category = ha.category
                WHERE bl.user_id = (SELECT user_id FROM users WHERE firebase_uid = $1)
                    AND bl.period_type = 'monthly'
                    AND (bl.effective_to IS NULL OR bl.effective_to > CURRENT_DATE)
                    {category_filter}
                """
                
                results = await conn.fetch(query, *params)
                
                budget_statuses = []
                for row in results:
                    # Calculate projected overage
                    current_spent = Decimal(str(row['current_spent']))
                    limit_amount = Decimal(str(row['limit_amount']))
                    spending_velocity = float(row['spending_velocity'] or 0)
                    days_remaining = int(row['days_remaining'])
                    
                    projected_spending = current_spent + Decimal(str(spending_velocity * days_remaining))
                    projected_overage = max(Decimal('0'), projected_spending - limit_amount)
                    
                    utilization_pct = float(row['utilization_pct'])
                    
                    # Determine risk level
                    if utilization_pct >= 95:
                        risk_level = RiskLevel.CRITICAL
                    elif utilization_pct >= 85:
                        risk_level = RiskLevel.HIGH
                    elif utilization_pct >= 70:
                        risk_level = RiskLevel.MEDIUM
                    else:
                        risk_level = RiskLevel.LOW
                    
                    budget_status = BudgetStatus(
                        user_id=user_id,
                        category=row['category'],
                        current_spent=current_spent,
                        budget_limit=limit_amount,
                        utilization_percentage=utilization_pct,
                        days_remaining=days_remaining,
                        projected_overage=projected_overage,
                        risk_level=risk_level,
                        last_updated=datetime.now(),
                        spending_velocity=spending_velocity,
                        historical_average=Decimal(str(row['historical_average'] or 0))
                    )
                    
                    budget_statuses.append(budget_status)
                
                return budget_statuses
                
        except Exception as e:
            self.logger.error(f"Error checking budget status: {str(e)}")
            return []

    async def calculate_spending_velocity(self, user_id: str, time_period: str = "30d") -> Dict[str, float]:
        """Calculate spending velocity (amount per day) for different categories."""
        try:
            async with self.connection_pool.acquire() as conn:
                # Parse time period
                if time_period.endswith('d'):
                    days = int(time_period[:-1])
                elif time_period.endswith('m'):
                    days = int(time_period[:-1]) * 30
                else:
                    days = 30  # Default to 30 days
                
                query = """
                SELECT 
                    category,
                    SUM(amount) as total_spent,
                    COUNT(DISTINCT transaction_date) as active_days
                FROM transactions
                WHERE user_id = (SELECT user_id FROM users WHERE firebase_uid = $1)
                    AND transaction_date >= CURRENT_DATE - INTERVAL '%s days'
                    AND deleted_at IS NULL
                GROUP BY category
                """ % days
                
                results = await conn.fetch(query, user_id)
                
                velocities = {}
                for row in results:
                    category = row['category']
                    total_spent = float(row['total_spent'])
                    active_days = max(1, int(row['active_days']))  # Avoid division by zero
                    
                    # Calculate velocity as average daily spending when active
                    velocities[category] = total_spent / active_days
                
                return velocities
                
        except Exception as e:
            self.logger.error(f"Error calculating spending velocity: {str(e)}")
            return {}

    async def predict_budget_overage(self, user_id: str, category: str) -> Optional[PredictionResult]:
        """Predict potential budget overage for a specific category."""
        try:
            budget_statuses = await self.check_budget_status(user_id, category)
            if not budget_statuses:
                return None
            
            budget_status = budget_statuses[0]
            
            if budget_status.spending_velocity <= 0:
                return None
            
            # Calculate days until budget is exceeded
            remaining_budget = budget_status.budget_limit - budget_status.current_spent
            days_until_overage = float(remaining_budget) / budget_status.spending_velocity
            
            # Calculate recommended daily limit to stay within budget
            recommended_daily_limit = float(remaining_budget) / max(1, budget_status.days_remaining)
            
            # Calculate confidence based on spending pattern consistency
            spending_velocities = await self.calculate_spending_velocity(user_id, "7d")
            recent_velocity = spending_velocities.get(category, 0)
            
            # Confidence is higher when recent velocity matches longer-term velocity
            velocity_ratio = min(recent_velocity, budget_status.spending_velocity) / max(recent_velocity, budget_status.spending_velocity, 0.01)
            confidence = min(0.95, velocity_ratio * 0.8 + 0.2)
            
            return PredictionResult(
                category=category,
                projected_overage=budget_status.projected_overage,
                confidence=confidence,
                days_until_overage=max(0, int(days_until_overage)),
                recommended_daily_limit=Decimal(str(recommended_daily_limit))
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting budget overage: {str(e)}")
            return None

    async def get_budget_utilization_trends(self, user_id: str) -> Dict[str, Any]:
        """Get budget utilization trends over time."""
        try:
            async with self.connection_pool.acquire() as conn:
                query = """
                WITH monthly_spending AS (
                    SELECT 
                        category,
                        date_trunc('month', transaction_date) as month,
                        SUM(amount) as monthly_total
                    FROM transactions
                    WHERE user_id = (SELECT user_id FROM users WHERE firebase_uid = $1)
                        AND transaction_date >= CURRENT_DATE - INTERVAL '6 months'
                        AND deleted_at IS NULL
                    GROUP BY category, date_trunc('month', transaction_date)
                ),
                monthly_budgets AS (
                    SELECT DISTINCT
                        category,
                        limit_amount
                    FROM budget_limits
                    WHERE user_id = (SELECT user_id FROM users WHERE firebase_uid = $1)
                        AND period_type = 'monthly'
                        AND (effective_to IS NULL OR effective_to > CURRENT_DATE - INTERVAL '6 months')
                ),
                utilization_by_month AS (
                    SELECT 
                        ms.category,
                        ms.month,
                        ms.monthly_total,
                        mb.limit_amount,
                        (ms.monthly_total / mb.limit_amount * 100) as utilization_pct
                    FROM monthly_spending ms
                    JOIN monthly_budgets mb ON ms.category = mb.category
                )
                SELECT 
                    category,
                    AVG(utilization_pct) as avg_utilization,
                    STDDEV(utilization_pct) as utilization_stddev,
                    MIN(utilization_pct) as min_utilization,
                    MAX(utilization_pct) as max_utilization,
                    COUNT(*) as months_data
                FROM utilization_by_month
                GROUP BY category
                """
                
                results = await conn.fetch(query, user_id)
                
                trends = {}
                for row in results:
                    category = row['category']
                    trends[category] = {
                        'average_utilization': float(row['avg_utilization'] or 0),
                        'utilization_volatility': float(row['utilization_stddev'] or 0),
                        'min_utilization': float(row['min_utilization'] or 0),
                        'max_utilization': float(row['max_utilization'] or 0),
                        'consistency_score': 1.0 - min(1.0, float(row['utilization_stddev'] or 0) / 100),
                        'months_of_data': int(row['months_data'])
                    }
                
                return trends
                
        except Exception as e:
            self.logger.error(f"Error getting budget utilization trends: {str(e)}")
            return {}

    # === Pattern Analysis Methods ===

    async def analyze_spending_patterns(self, user_id: str, lookback_days: int = 90) -> PatternAnalysis:
        """Analyze spending patterns for a user over a specified period."""
        try:
            async with self.connection_pool.acquire() as conn:
                end_date = date.today()
                start_date = end_date - timedelta(days=lookback_days)
                
                # Get spending data
                spending_data = await self._get_spending_data(conn, user_id, start_date, end_date)
                
                # Analyze patterns
                patterns = await self._detect_spending_patterns(spending_data)
                trend_indicators = await self._calculate_trend_indicators(spending_data)
                seasonal_factors = await self._analyze_seasonal_factors(conn, user_id, start_date, end_date)
                anomalies = await self._detect_pattern_anomalies(spending_data)
                
                # Calculate confidence scores
                confidence_scores = self._calculate_pattern_confidence(spending_data, patterns)
                
                # Generate recommendations
                recommendations = await self._generate_pattern_recommendations(patterns, trend_indicators, anomalies)
                
                return PatternAnalysis(
                    user_id=user_id,
                    analysis_period=(start_date, end_date),
                    spending_patterns=patterns,
                    trend_indicators=trend_indicators,
                    seasonal_factors=seasonal_factors,
                    anomalies_detected=anomalies,
                    confidence_scores=confidence_scores,
                    recommendations=recommendations
                )
                
        except Exception as e:
            self.logger.error(f"Error analyzing spending patterns: {str(e)}")
            return PatternAnalysis(
                user_id=user_id,
                analysis_period=(date.today() - timedelta(days=lookback_days), date.today()),
                spending_patterns={},
                trend_indicators={},
                seasonal_factors={},
                anomalies_detected=[],
                confidence_scores={},
                recommendations=[]
            )

    async def detect_pattern_changes(self, user_id: str, baseline_period: str = "3m") -> List[Dict[str, Any]]:
        """Detect changes in spending patterns compared to a baseline period."""
        try:
            async with self.connection_pool.acquire() as conn:
                # Parse baseline period
                if baseline_period.endswith('m'):
                    months = int(baseline_period[:-1])
                    baseline_days = months * 30
                else:
                    baseline_days = 90  # Default to 3 months
                
                query = """
                WITH weekly_spending AS (
                    SELECT 
                        date_trunc('week', transaction_date) as week_start,
                        category,
                        SUM(amount) as weekly_total
                    FROM transactions
                    WHERE user_id = (SELECT user_id FROM users WHERE firebase_uid = $1)
                        AND transaction_date >= CURRENT_DATE - INTERVAL '%s days'
                        AND deleted_at IS NULL
                    GROUP BY date_trunc('week', transaction_date), category
                ),
                pattern_stats AS (
                    SELECT 
                        category,
                        AVG(weekly_total) as avg_weekly,
                        STDDEV(weekly_total) as stddev_weekly,
                        COUNT(*) as weeks_count
                    FROM weekly_spending
                    WHERE week_start < date_trunc('week', CURRENT_DATE) - INTERVAL '4 weeks'
                    GROUP BY category
                ),
                recent_weeks AS (
                    SELECT 
                        category,
                        AVG(weekly_total) as recent_avg,
                        STDDEV(weekly_total) as recent_stddev
                    FROM weekly_spending
                    WHERE week_start >= date_trunc('week', CURRENT_DATE) - INTERVAL '4 weeks'
                    GROUP BY category
                )
                SELECT 
                    ps.category,
                    ps.avg_weekly as historical_avg,
                    rw.recent_avg,
                    ps.stddev_weekly as historical_stddev,
                    rw.recent_stddev,
                    ((rw.recent_avg - ps.avg_weekly) / NULLIF(ps.avg_weekly, 0) * 100) as change_percentage,
                    CASE 
                        WHEN ABS(rw.recent_avg - ps.avg_weekly) > 2 * COALESCE(ps.stddev_weekly, 0) THEN 'significant'
                        WHEN ABS(rw.recent_avg - ps.avg_weekly) > COALESCE(ps.stddev_weekly, 0) THEN 'notable'
                        ELSE 'normal'
                    END as change_significance,
                    ps.weeks_count
                FROM pattern_stats ps
                JOIN recent_weeks rw ON ps.category = rw.category
                WHERE ps.weeks_count >= 8
                """ % baseline_days
                
                results = await conn.fetch(query, user_id)
                
                pattern_changes = []
                for row in results:
                    if row['change_significance'] in ['significant', 'notable']:
                        change = {
                            'category': row['category'],
                            'change_type': 'increase' if row['change_percentage'] > 0 else 'decrease',
                            'change_percentage': float(row['change_percentage'] or 0),
                            'significance': row['change_significance'],
                            'historical_average': float(row['historical_avg'] or 0),
                            'recent_average': float(row['recent_avg'] or 0),
                            'confidence': min(1.0, float(row['weeks_count']) / 12.0)  # More weeks = higher confidence
                        }
                        pattern_changes.append(change)
                
                return pattern_changes
                
        except Exception as e:
            self.logger.error(f"Error detecting pattern changes: {str(e)}")
            return []

    async def compare_seasonal_patterns(self, user_id: str, current_period: str = "current_month") -> Dict[str, Any]:
        """Compare current spending to seasonal patterns from previous years."""
        try:
            async with self.connection_pool.acquire() as conn:
                current_month = datetime.now().month
                
                query = """
                WITH current_month_spending AS (
                    SELECT 
                        category,
                        SUM(amount) as current_amount
                    FROM transactions
                    WHERE user_id = (SELECT user_id FROM users WHERE firebase_uid = $1)
                        AND EXTRACT(month FROM transaction_date) = $2
                        AND EXTRACT(year FROM transaction_date) = EXTRACT(year FROM CURRENT_DATE)
                        AND deleted_at IS NULL
                    GROUP BY category
                ),
                historical_same_month AS (
                    SELECT 
                        category,
                        AVG(monthly_total) as historical_avg,
                        STDDEV(monthly_total) as historical_stddev,
                        COUNT(*) as years_data
                    FROM (
                        SELECT 
                            category,
                            EXTRACT(year FROM transaction_date) as year,
                            SUM(amount) as monthly_total
                        FROM transactions
                        WHERE user_id = (SELECT user_id FROM users WHERE firebase_uid = $1)
                            AND EXTRACT(month FROM transaction_date) = $2
                            AND EXTRACT(year FROM transaction_date) < EXTRACT(year FROM CURRENT_DATE)
                            AND deleted_at IS NULL
                        GROUP BY category, EXTRACT(year FROM transaction_date)
                    ) yearly_totals
                    GROUP BY category
                )
                SELECT 
                    COALESCE(cms.category, hsm.category) as category,
                    COALESCE(cms.current_amount, 0) as current_spending,
                    COALESCE(hsm.historical_avg, 0) as historical_average,
                    COALESCE(hsm.historical_stddev, 0) as historical_stddev,
                    hsm.years_data,
                    CASE 
                        WHEN hsm.historical_avg > 0 THEN
                            ((COALESCE(cms.current_amount, 0) - hsm.historical_avg) / hsm.historical_avg * 100)
                        ELSE 0
                    END as seasonal_variance_pct
                FROM current_month_spending cms
                FULL OUTER JOIN historical_same_month hsm ON cms.category = hsm.category
                WHERE COALESCE(hsm.years_data, 0) >= 1
                """
                
                results = await conn.fetch(query, user_id, current_month)
                
                seasonal_comparison = {}
                for row in results:
                    category = row['category']
                    seasonal_comparison[category] = {
                        'current_spending': float(row['current_spending'] or 0),
                        'historical_average': float(row['historical_average'] or 0),
                        'seasonal_variance_percentage': float(row['seasonal_variance_pct'] or 0),
                        'years_of_data': int(row['years_data'] or 0),
                        'is_seasonal_high': float(row['seasonal_variance_pct'] or 0) > 20,
                        'is_seasonal_low': float(row['seasonal_variance_pct'] or 0) < -20,
                        'confidence': min(1.0, float(row['years_data'] or 0) / 3.0)
                    }
                
                return seasonal_comparison
                
        except Exception as e:
            self.logger.error(f"Error comparing seasonal patterns: {str(e)}")
            return {}

    async def identify_emerging_trends(self, user_id: str, sensitivity: float = 0.7) -> List[Dict[str, Any]]:
        """Identify emerging spending trends based on recent data."""
        try:
            async with self.connection_pool.acquire() as conn:
                query = """
                WITH daily_spending AS (
                    SELECT 
                        transaction_date,
                        category,
                        SUM(amount) as daily_total
                    FROM transactions
                    WHERE user_id = (SELECT user_id FROM users WHERE firebase_uid = $1)
                        AND transaction_date >= CURRENT_DATE - INTERVAL '60 days'
                        AND deleted_at IS NULL
                    GROUP BY transaction_date, category
                ),
                trend_analysis AS (
                    SELECT 
                        category,
                        EXTRACT(epoch FROM transaction_date - (CURRENT_DATE - INTERVAL '60 days')) / 86400 as day_number,
                        daily_total
                    FROM daily_spending
                ),
                trend_coefficients AS (
                    SELECT 
                        category,
                        REGR_SLOPE(daily_total, day_number) as trend_slope,
                        REGR_R2(daily_total, day_number) as r_squared,
                        AVG(daily_total) as avg_daily,
                        STDDEV(daily_total) as stddev_daily,
                        COUNT(*) as data_points
                    FROM trend_analysis
                    GROUP BY category
                    HAVING COUNT(*) >= 10
                )
                SELECT 
                    category,
                    trend_slope,
                    r_squared,
                    avg_daily,
                    stddev_daily,
                    data_points,
                    CASE 
                        WHEN trend_slope > 0 AND r_squared > $2 THEN 'increasing'
                        WHEN trend_slope < 0 AND r_squared > $2 THEN 'decreasing'
                        ELSE 'stable'
                    END as trend_direction,
                    ABS(trend_slope) / NULLIF(avg_daily, 0) * 100 as trend_strength_pct
                FROM trend_coefficients
                WHERE r_squared > $2 AND ABS(trend_slope) > 0.1
                ORDER BY ABS(trend_slope) DESC
                """
                
                results = await conn.fetch(query, user_id, sensitivity)
                
                emerging_trends = []
                for row in results:
                    trend = {
                        'category': row['category'],
                        'trend_direction': row['trend_direction'],
                        'trend_strength_percentage': float(row['trend_strength_pct'] or 0),
                        'confidence': float(row['r_squared'] or 0),
                        'daily_change_rate': float(row['trend_slope'] or 0),
                        'average_daily_spending': float(row['avg_daily'] or 0),
                        'data_points': int(row['data_points']),
                        'significance': 'high' if float(row['trend_strength_pct'] or 0) > 20 else 'medium'
                    }
                    emerging_trends.append(trend)
                
                return emerging_trends
                
        except Exception as e:
            self.logger.error(f"Error identifying emerging trends: {str(e)}")
            return []

    # === Helper Methods ===

    async def _get_user_monitoring_preferences(self, conn, user_id: str) -> Dict[str, Any]:
        """Get user's monitoring preferences from database."""
        try:
            query = """
            SELECT preferences 
            FROM users 
            WHERE firebase_uid = $1
            """
            result = await conn.fetchval(query, user_id)
            
            if result:
                prefs = json.loads(result) if isinstance(result, str) else result
                return prefs.get('monitoring', {})
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting user monitoring preferences: {str(e)}")
            return {}

    def _determine_frequency(self, analysis_type: AnalysisType, user_prefs: Dict[str, Any], priority: int) -> MonitoringFrequency:
        """Determine monitoring frequency based on analysis type, user preferences, and priority."""
        # High priority users get more frequent monitoring
        if priority >= 8:
            frequency_map = {
                AnalysisType.BUDGET_CHECK: MonitoringFrequency.HOURLY,
                AnalysisType.ANOMALY_DETECTION: MonitoringFrequency.HOURLY,
                AnalysisType.GOAL_TRACKING: MonitoringFrequency.DAILY,
                AnalysisType.PATTERN_ANALYSIS: MonitoringFrequency.DAILY,
                AnalysisType.TREND_DETECTION: MonitoringFrequency.DAILY,
                AnalysisType.SEASONAL_REVIEW: MonitoringFrequency.WEEKLY
            }
        else:
            frequency_map = {
                AnalysisType.BUDGET_CHECK: MonitoringFrequency.DAILY,
                AnalysisType.ANOMALY_DETECTION: MonitoringFrequency.DAILY,
                AnalysisType.GOAL_TRACKING: MonitoringFrequency.DAILY,
                AnalysisType.PATTERN_ANALYSIS: MonitoringFrequency.WEEKLY,
                AnalysisType.TREND_DETECTION: MonitoringFrequency.WEEKLY,
                AnalysisType.SEASONAL_REVIEW: MonitoringFrequency.MONTHLY
            }
        
        # Check user preferences for overrides
        if user_prefs.get(analysis_type.value):
            pref_frequency = user_prefs[analysis_type.value].get('frequency')
            if pref_frequency:
                try:
                    return MonitoringFrequency(pref_frequency)
                except ValueError:
                    pass
        
        return frequency_map.get(analysis_type, MonitoringFrequency.DAILY)

    def _calculate_next_run_time(self, frequency: MonitoringFrequency, priority: int) -> datetime:
        """Calculate the next run time based on frequency and priority."""
        now = datetime.now()
        
        if frequency == MonitoringFrequency.HOURLY:
            base_interval = timedelta(hours=1)
        elif frequency == MonitoringFrequency.DAILY:
            base_interval = timedelta(days=1)
        elif frequency == MonitoringFrequency.WEEKLY:
            base_interval = timedelta(weeks=1)
        elif frequency == MonitoringFrequency.MONTHLY:
            base_interval = timedelta(days=30)
        else:
            base_interval = timedelta(days=1)
        
        # Adjust interval based on priority (higher priority = shorter interval)
        priority_multiplier = max(0.1, (11 - priority) / 10)
        adjusted_interval = base_interval * priority_multiplier
        
        return now + adjusted_interval

    async def _store_monitoring_schedule(self, conn, schedule: MonitoringSchedule) -> bool:
        """Store monitoring schedule in database."""
        try:
            query = """
            INSERT INTO monitoring_schedules 
            (user_id, analysis_type, frequency, next_run_time, last_run_time, priority, conditions, metadata, is_active, created_at)
            VALUES 
            ((SELECT user_id FROM users WHERE firebase_uid = $1), $2, $3, $4, $5, $6, $7, $8, true, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id, analysis_type) 
            DO UPDATE SET 
                frequency = EXCLUDED.frequency,
                next_run_time = EXCLUDED.next_run_time,
                priority = EXCLUDED.priority,
                conditions = EXCLUDED.conditions,
                metadata = EXCLUDED.metadata,
                updated_at = CURRENT_TIMESTAMP
            """
            
            await conn.execute(
                query,
                schedule.user_id,
                schedule.analysis_type.value,
                schedule.frequency.value,
                schedule.next_run_time,
                schedule.last_run_time,
                schedule.priority,
                json.dumps(schedule.conditions),
                json.dumps(schedule.metadata)
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing monitoring schedule: {str(e)}")
            return False

    async def _check_analysis_conditions(self, conn, user_id: str, analysis_type: AnalysisType, conditions: Dict[str, Any]) -> bool:
        """Check if additional conditions are met for running analysis."""
        try:
            if not conditions:
                return True
            
            # Check budget usage condition
            if 'budget_usage' in conditions:
                threshold = float(conditions['budget_usage'].replace('>', '').replace('%', ''))
                budget_statuses = await self.check_budget_status(user_id)
                
                for status in budget_statuses:
                    if status.utilization_percentage > threshold:
                        return True
                return False
            
            # Check if active goals exist
            if conditions.get('active_goals_exist'):
                query = """
                SELECT COUNT(*) FROM financial_goals 
                WHERE user_id = (SELECT user_id FROM users WHERE firebase_uid = $1)
                    AND status = 'active'
                    AND deleted_at IS NULL
                """
                goal_count = await conn.fetchval(query, user_id)
                return goal_count > 0
            
            # Check month boundary condition
            if conditions.get('month_boundary'):
                return datetime.now().day == 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking analysis conditions: {str(e)}")
            return True  # Default to running analysis on error

    async def _get_spending_data(self, conn, user_id: str, start_date: date, end_date: date) -> Dict[str, Any]:
        """Get spending data for pattern analysis."""
        try:
            query = """
            SELECT 
                transaction_date,
                category,
                merchant_name,
                amount,
                payment_method,
                EXTRACT(dow FROM transaction_date) as day_of_week,
                EXTRACT(hour FROM created_at) as hour_of_day
            FROM transactions
            WHERE user_id = (SELECT user_id FROM users WHERE firebase_uid = $1)
                AND transaction_date BETWEEN $2 AND $3
                AND deleted_at IS NULL
            ORDER BY transaction_date, created_at
            """
            
            results = await conn.fetch(query, user_id, start_date, end_date)
            
            # Organize data by category and time periods
            spending_data = {
                'transactions': [dict(row) for row in results],
                'by_category': {},
                'by_day_of_week': {},
                'by_hour': {},
                'by_merchant': {},
                'daily_totals': {}
            }
            
            for row in results:
                category = row['category']
                date_key = row['transaction_date'].isoformat()
                dow = int(row['day_of_week'])
                hour = int(row['hour_of_day'] or 12)
                merchant = row['merchant_name']
                amount = float(row['amount'])
                
                # By category
                if category not in spending_data['by_category']:
                    spending_data['by_category'][category] = []
                spending_data['by_category'][category].append(row)
                
                # By day of week
                if dow not in spending_data['by_day_of_week']:
                    spending_data['by_day_of_week'][dow] = 0
                spending_data['by_day_of_week'][dow] += amount
                
                # By hour
                if hour not in spending_data['by_hour']:
                    spending_data['by_hour'][hour] = 0
                spending_data['by_hour'][hour] += amount
                
                # By merchant
                if merchant not in spending_data['by_merchant']:
                    spending_data['by_merchant'][merchant] = 0
                spending_data['by_merchant'][merchant] += amount
                
                # Daily totals
                if date_key not in spending_data['daily_totals']:
                    spending_data['daily_totals'][date_key] = 0
                spending_data['daily_totals'][date_key] += amount
            
            return spending_data
            
        except Exception as e:
            self.logger.error(f"Error getting spending data: {str(e)}")
            return {}

    async def _detect_spending_patterns(self, spending_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect spending patterns from the data."""
        try:
            patterns = {}
            
            # Day of week patterns
            dow_data = spending_data.get('by_day_of_week', {})
            if dow_data:
                max_dow = max(dow_data.keys(), key=lambda k: dow_data[k])
                min_dow = min(dow_data.keys(), key=lambda k: dow_data[k])
                
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                patterns['day_of_week'] = {
                    'highest_spending_day': day_names[max_dow],
                    'lowest_spending_day': day_names[min_dow],
                    'weekend_vs_weekday_ratio': (dow_data.get(5, 0) + dow_data.get(6, 0)) / max(1, sum(dow_data.get(i, 0) for i in range(5))),
                    'daily_distribution': {day_names[k]: v for k, v in dow_data.items()}
                }
            
            # Hour of day patterns
            hour_data = spending_data.get('by_hour', {})
            if hour_data:
                patterns['time_of_day'] = {
                    'peak_spending_hour': max(hour_data.keys(), key=lambda k: hour_data[k]),
                    'morning_spending': sum(hour_data.get(h, 0) for h in range(6, 12)),
                    'afternoon_spending': sum(hour_data.get(h, 0) for h in range(12, 18)),
                    'evening_spending': sum(hour_data.get(h, 0) for h in range(18, 24)),
                    'late_night_spending': sum(hour_data.get(h, 0) for h in range(0, 6))
                }
            
            # Category patterns
            category_data = spending_data.get('by_category', {})
            if category_data:
                category_totals = {cat: sum(float(t['amount']) for t in transactions) 
                                 for cat, transactions in category_data.items()}
                
                patterns['categories'] = {
                    'top_categories': sorted(category_totals.items(), key=lambda x: x[1], reverse=True)[:5],
                    'category_diversity': len(category_totals),
                    'concentration_ratio': max(category_totals.values()) / sum(category_totals.values()) if category_totals else 0
                }
            
            # Merchant loyalty patterns
            merchant_data = spending_data.get('by_merchant', {})
            if merchant_data:
                patterns['merchant_loyalty'] = {
                    'top_merchants': sorted(merchant_data.items(), key=lambda x: x[1], reverse=True)[:10],
                    'merchant_diversity': len(merchant_data),
                    'loyalty_concentration': max(merchant_data.values()) / sum(merchant_data.values()) if merchant_data else 0
                }
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting spending patterns: {str(e)}")
            return {}

    async def _calculate_trend_indicators(self, spending_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate trend indicators from spending data."""
        try:
            daily_totals = spending_data.get('daily_totals', {})
            if not daily_totals:
                return {}
            
            # Convert to time series
            dates = sorted(daily_totals.keys())
            amounts = [daily_totals[date] for date in dates]
            
            if len(amounts) < 7:  # Need at least a week of data
                return {}
            
            # Calculate moving averages
            ma_7 = []
            ma_14 = []
            for i in range(len(amounts)):
                if i >= 6:
                    ma_7.append(statistics.mean(amounts[i-6:i+1]))
                if i >= 13:
                    ma_14.append(statistics.mean(amounts[i-13:i+1]))
            
            # Calculate trend direction
            recent_avg = statistics.mean(amounts[-7:]) if len(amounts) >= 7 else statistics.mean(amounts)
            older_avg = statistics.mean(amounts[-14:-7]) if len(amounts) >= 14 else statistics.mean(amounts[:-7]) if len(amounts) > 7 else recent_avg
            
            trend_direction = 'increasing' if recent_avg > older_avg * 1.05 else 'decreasing' if recent_avg < older_avg * 0.95 else 'stable'
            
            return {
                'overall_trend': trend_direction,
                'recent_average': recent_avg,
                'trend_strength': abs(recent_avg - older_avg) / older_avg if older_avg > 0 else 0,
                'volatility': statistics.stdev(amounts) if len(amounts) > 1 else 0,
                'data_points': len(amounts)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating trend indicators: {str(e)}")
            return {}

    async def _analyze_seasonal_factors(self, conn, user_id: str, start_date: date, end_date: date) -> Dict[str, Any]:
        """Analyze seasonal factors in spending."""
        try:
            query = """
            SELECT 
                EXTRACT(month FROM transaction_date) as month,
                EXTRACT(week FROM transaction_date) as week,
                category,
                SUM(amount) as total_amount
            FROM transactions
            WHERE user_id = (SELECT user_id FROM users WHERE firebase_uid = $1)
                AND transaction_date BETWEEN $2 AND $3
                AND deleted_at IS NULL
            GROUP BY EXTRACT(month FROM transaction_date), EXTRACT(week FROM transaction_date), category
            """
            
            results = await conn.fetch(query, user_id, start_date, end_date)
            
            monthly_patterns = {}
            weekly_patterns = {}
            
            for row in results:
                month = int(row['month'])
                week = int(row['week'])
                category = row['category']
                amount = float(row['total_amount'])
                
                # Monthly patterns
                if month not in monthly_patterns:
                    monthly_patterns[month] = {}
                if category not in monthly_patterns[month]:
                    monthly_patterns[month][category] = 0
                monthly_patterns[month][category] += amount
                
                # Weekly patterns
                if week not in weekly_patterns:
                    weekly_patterns[week] = 0
                weekly_patterns[week] += amount
            
            return {
                'monthly_patterns': monthly_patterns,
                'weekly_patterns': weekly_patterns,
                'peak_month': max(monthly_patterns.keys(), key=lambda m: sum(monthly_patterns[m].values())) if monthly_patterns else None,
                'seasonal_variance': statistics.stdev([sum(month_data.values()) for month_data in monthly_patterns.values()]) if len(monthly_patterns) > 1 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing seasonal factors: {str(e)}")
            return {}

    async def _detect_pattern_anomalies(self, spending_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in spending patterns."""
        try:
            anomalies = []
            daily_totals = spending_data.get('daily_totals', {})
            
            if len(daily_totals) < 7:
                return anomalies
            
            amounts = list(daily_totals.values())
            mean_amount = statistics.mean(amounts)
            std_amount = statistics.stdev(amounts) if len(amounts) > 1 else 0
            
            # Detect statistical outliers (> 2 standard deviations)
            for date, amount in daily_totals.items():
                if std_amount > 0 and abs(amount - mean_amount) > 2 * std_amount:
                    anomalies.append({
                        'type': 'statistical_outlier',
                        'date': date,
                        'amount': amount,
                        'deviation': abs(amount - mean_amount) / std_amount,
                        'severity': 'high' if abs(amount - mean_amount) > 3 * std_amount else 'medium'
                    })
            
            # Detect category anomalies
            category_data = spending_data.get('by_category', {})
            for category, transactions in category_data.items():
                if len(transactions) < 3:
                    continue
                
                amounts = [float(t['amount']) for t in transactions]
                cat_mean = statistics.mean(amounts)
                cat_std = statistics.stdev(amounts) if len(amounts) > 1 else 0
                
                for transaction in transactions:
                    amount = float(transaction['amount'])
                    if cat_std > 0 and abs(amount - cat_mean) > 2 * cat_std:
                        anomalies.append({
                            'type': 'category_outlier',
                            'category': category,
                            'date': transaction['transaction_date'].isoformat(),
                            'amount': amount,
                            'merchant': transaction['merchant_name'],
                            'deviation': abs(amount - cat_mean) / cat_std
                        })
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting pattern anomalies: {str(e)}")
            return []

    def _calculate_pattern_confidence(self, spending_data: Dict[str, Any], patterns: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for detected patterns."""
        try:
            confidence_scores = {}
            
            # Data quantity confidence
            transaction_count = len(spending_data.get('transactions', []))
            data_confidence = min(1.0, transaction_count / 100)  # Higher confidence with more data
            
            # Pattern stability confidence
            daily_totals = spending_data.get('daily_totals', {})
            if len(daily_totals) > 1:
                amounts = list(daily_totals.values())
                cv = statistics.stdev(amounts) / statistics.mean(amounts) if statistics.mean(amounts) > 0 else 1
                stability_confidence = max(0.1, 1 - cv)  # Lower confidence with high coefficient of variation
            else:
                stability_confidence = 0.5
            
            # Time coverage confidence
            date_range = len(daily_totals)
            coverage_confidence = min(1.0, date_range / 30)  # Higher confidence with more days
            
            # Overall confidence is weighted average
            overall_confidence = (data_confidence * 0.4 + stability_confidence * 0.4 + coverage_confidence * 0.2)
            
            confidence_scores.update({
                'overall': overall_confidence,
                'data_quantity': data_confidence,
                'pattern_stability': stability_confidence,
                'time_coverage': coverage_confidence
            })
            
            return confidence_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern confidence: {str(e)}")
            return {'overall': 0.5}

    async def _generate_pattern_recommendations(self, patterns: Dict[str, Any], trend_indicators: Dict[str, Any], anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recommendations based on pattern analysis."""
        try:
            recommendations = []
            
            # Day of week recommendations
            if 'day_of_week' in patterns:
                dow_patterns = patterns['day_of_week']
                weekend_ratio = dow_patterns.get('weekend_vs_weekday_ratio', 1)
                
                if weekend_ratio > 1.5:
                    recommendations.append({
                        'type': 'spending_timing',
                        'priority': 'medium',
                        'title': 'High Weekend Spending Detected',
                        'description': f'You spend {weekend_ratio:.1f}x more on weekends than weekdays.',
                        'suggestion': 'Consider setting a weekend spending limit or planning weekend activities in advance.',
                        'category': 'behavioral'
                    })
            
            # Trend-based recommendations
            if trend_indicators.get('overall_trend') == 'increasing':
                trend_strength = trend_indicators.get('trend_strength', 0)
                if trend_strength > 0.2:
                    recommendations.append({
                        'type': 'trend_alert',
                        'priority': 'high',
                        'title': 'Increasing Spending Trend',
                        'description': f'Your spending has increased by {trend_strength:.1%} recently.',
                        'suggestion': 'Review your recent purchases and consider adjusting your budget.',
                        'category': 'financial'
                    })
            
            # Anomaly-based recommendations
            high_anomalies = [a for a in anomalies if a.get('severity') == 'high']
            if len(high_anomalies) > 2:
                recommendations.append({
                    'type': 'anomaly_alert',
                    'priority': 'medium',
                    'title': 'Unusual Spending Patterns',
                    'description': f'Detected {len(high_anomalies)} unusual spending events recently.',
                    'suggestion': 'Review these transactions to ensure they align with your financial goals.',
                    'category': 'monitoring'
                })
            
            # Category concentration recommendations
            if 'categories' in patterns:
                concentration = patterns['categories'].get('concentration_ratio', 0)
                if concentration > 0.6:
                    recommendations.append({
                        'type': 'diversification',
                        'priority': 'low',
                        'title': 'Spending Concentration',
                        'description': f'{concentration:.1%} of your spending is in one category.',
                        'suggestion': 'Consider diversifying your spending or reviewing if this concentration is intentional.',
                        'category': 'optimization'
                    })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating pattern recommendations: {str(e)}")
            return []

    # === Cleanup and Maintenance Methods ===

    async def cleanup_old_schedules(self, days_old: int = 30) -> int:
        """Clean up old monitoring schedules that are no longer active."""
        try:
            async with self.connection_pool.acquire() as conn:
                query = """
                DELETE FROM monitoring_schedules 
                WHERE is_active = false 
                    AND updated_at < CURRENT_DATE - INTERVAL '%s days'
                """ % days_old
                
                result = await conn.execute(query)
                deleted_count = int(result.split()[-1])
                
                self.logger.info(f"Cleaned up {deleted_count} old monitoring schedules")
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old schedules: {str(e)}")
            return 0

    async def update_schedule_after_analysis(self, user_id: str, analysis_type: AnalysisType) -> bool:
        """Update schedule after completing an analysis."""
        try:
            async with self.connection_pool.acquire() as conn:
                # Get current schedule
                query = """
                SELECT frequency, priority, conditions 
                FROM monitoring_schedules 
                WHERE user_id = (SELECT user_id FROM users WHERE firebase_uid = $1)
                    AND analysis_type = $2 
                    AND is_active = true
                """
                result = await conn.fetchrow(query, user_id, analysis_type.value)
                
                if not result:
                    return False
                
                frequency = MonitoringFrequency(result['frequency'])
                priority = result['priority']
                
                # Calculate next run time
                next_run_time = self._calculate_next_run_time(frequency, priority)
                
                # Update schedule
                update_query = """
                UPDATE monitoring_schedules 
                SET last_run_time = CURRENT_TIMESTAMP,
                    next_run_time = $3,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = (SELECT user_id FROM users WHERE firebase_uid = $1)
                    AND analysis_type = $2
                """
                
                await conn.execute(update_query, user_id, analysis_type.value, next_run_time)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating schedule after analysis: {str(e)}")
            return False

    async def get_monitoring_status(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive monitoring status for a user."""
        try:
            async with self.connection_pool.acquire() as conn:
                query = """
                SELECT 
                    analysis_type,
                    frequency,
                    next_run_time,
                    last_run_time,
                    priority,
                    is_active
                FROM monitoring_schedules 
                WHERE user_id = (SELECT user_id FROM users WHERE firebase_uid = $1)
                ORDER BY priority DESC, next_run_time ASC
                """
                
                results = await conn.fetch(query, user_id)
                
                status = {
                    'user_id': user_id,
                    'active_schedules': len([r for r in results if r['is_active']]),
                    'next_analysis': min([r['next_run_time'] for r in results if r['is_active']], default=None),
                    'schedules': []
                }
                
                for row in results:
                    status['schedules'].append({
                        'analysis_type': row['analysis_type'],
                        'frequency': row['frequency'],
                        'next_run': row['next_run_time'],
                        'last_run': row['last_run_time'],
                        'priority': row['priority'],
                        'is_active': row['is_active']
                    })
                
                return status
                
        except Exception as e:
            self.logger.error(f"Error getting monitoring status: {str(e)}")
            return {'user_id': user_id, 'error': str(e)}