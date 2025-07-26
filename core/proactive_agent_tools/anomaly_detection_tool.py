import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import uuid
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json

from core.base_agent_tools.database_connector import DatabaseConnector
from core.base_agent_tools.config_manager import AgentConfig


class AnomalyType(Enum):
    """Types of spending anomalies that can be detected."""
    SPENDING_SPIKE = "spending_spike"
    SPENDING_DROP = "spending_drop"
    FREQUENCY_ANOMALY = "frequency_anomaly"
    MERCHANT_ANOMALY = "merchant_anomaly"
    CATEGORY_SHIFT = "category_shift"
    TIME_PATTERN_ANOMALY = "time_pattern_anomaly"
    AMOUNT_PATTERN_ANOMALY = "amount_pattern_anomaly"


class AnomalySeverity(Enum):
    """Severity levels for anomalies."""
    MINOR = "minor"          # 1-2 standard deviations
    MODERATE = "moderate"    # 2-3 standard deviations
    SIGNIFICANT = "significant"  # 3-4 standard deviations
    CRITICAL = "critical"    # >4 standard deviations


class AnomalyPolarity(Enum):
    """Whether the anomaly is positive or negative."""
    POSITIVE = "positive"    # Good anomaly (spending less, better habits)
    NEGATIVE = "negative"    # Bad anomaly (overspending, concerning patterns)
    NEUTRAL = "neutral"      # Unusual but not necessarily good or bad


@dataclass
class SpendingAnomaly:
    """Structure for a detected spending anomaly."""
    anomaly_id: str
    user_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    polarity: AnomalyPolarity
    
    # Core metrics
    z_score: float
    confidence_score: float
    impact_score: float
    
    # Context data
    category: Optional[str] = None
    merchant: Optional[str] = None
    time_period: Optional[str] = None
    
    # Anomaly details
    observed_value: float = 0.0
    expected_value: float = 0.0
    historical_mean: float = 0.0
    historical_std: float = 0.0
    
    # Explanatory context
    description: str = ""
    contributing_factors: List[str] = None
    recommendations: List[str] = None
    
    # Metadata
    detection_timestamp: datetime = None
    data_window_start: datetime = None
    data_window_end: datetime = None
    related_transactions: List[str] = None
    
    def __post_init__(self):
        if self.contributing_factors is None:
            self.contributing_factors = []
        if self.recommendations is None:
            self.recommendations = []
        if self.detection_timestamp is None:
            self.detection_timestamp = datetime.now()
        if self.related_transactions is None:
            self.related_transactions = []


class AnomalyDetectionTool:
    """
    Advanced anomaly detection tool for identifying unusual spending patterns.
    
    Uses multiple statistical and ML approaches to detect different types of anomalies:
    - Statistical methods (Z-score, IQR)
    - Machine learning (Isolation Forest)
    - Time series analysis
    - Pattern recognition
    """
    
    def __init__(self, db_connector: DatabaseConnector = None):
        self.db_connector = db_connector or DatabaseConnector()
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = AgentConfig.from_env()
        
        # Detection thresholds
        self.z_score_thresholds = {
            AnomalySeverity.MINOR: 1.5,
            AnomalySeverity.MODERATE: 2.0,
            AnomalySeverity.SIGNIFICANT: 3.0,
            AnomalySeverity.CRITICAL: 4.0
        }
        
        # ML models (initialized when needed)
        self.isolation_forest = None
        self.scaler = StandardScaler()
        
        self.logger.info("Anomaly Detection Tool initialized")
    
    async def detect_anomalies(
        self,
        user_id: str,
        analysis_window_days: int = 30,
        historical_window_days: int = 180,
        min_transactions: int = 10
    ) -> List[SpendingAnomaly]:
        """
        Main method to detect all types of spending anomalies for a user.
        
        Args:
            user_id: User to analyze
            analysis_window_days: Recent period to analyze for anomalies
            historical_window_days: Historical period to establish baseline
            min_transactions: Minimum transactions needed for reliable detection
            
        Returns:
            List of detected anomalies
        """
        try:
            self.logger.info(f"Starting anomaly detection for user {user_id}")
            
            # Get transaction data
            transactions_data = await self._get_user_transactions(
                user_id, historical_window_days
            )
            
            if len(transactions_data) < min_transactions:
                self.logger.warning(f"Insufficient transaction data for user {user_id}")
                return []
            
            # Prepare data for analysis
            df = pd.DataFrame(transactions_data)
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])
            
            # Define analysis windows
            end_date = df['transaction_date'].max()
            analysis_start = end_date - timedelta(days=analysis_window_days)
            historical_end = analysis_start
            historical_start = historical_end - timedelta(days=historical_window_days)
            
            # Split data into historical and recent
            historical_df = df[
                (df['transaction_date'] >= historical_start) & 
                (df['transaction_date'] < historical_end)
            ]
            recent_df = df[df['transaction_date'] >= analysis_start]
            
            if len(historical_df) < min_transactions // 2:
                self.logger.warning(f"Insufficient historical data for user {user_id}")
                return []
            
            # Detect different types of anomalies
            anomalies = []
            
            # 1. Amount-based anomalies
            anomalies.extend(
                await self._detect_amount_anomalies(user_id, historical_df, recent_df)
            )
            
            # 2. Frequency anomalies
            anomalies.extend(
                await self._detect_frequency_anomalies(user_id, historical_df, recent_df)
            )
            
            # 3. Category distribution anomalies
            anomalies.extend(
                await self._detect_category_anomalies(user_id, historical_df, recent_df)
            )
            
            # 4. Merchant pattern anomalies
            anomalies.extend(
                await self._detect_merchant_anomalies(user_id, historical_df, recent_df)
            )
            
            # 5. Time pattern anomalies
            anomalies.extend(
                await self._detect_time_pattern_anomalies(user_id, historical_df, recent_df)
            )
            
            # 6. ML-based anomalies (using Isolation Forest)
            anomalies.extend(
                await self._detect_ml_anomalies(user_id, historical_df, recent_df)
            )
            
            # Filter and rank anomalies
            filtered_anomalies = await self._filter_and_rank_anomalies(anomalies)
            
            self.logger.info(f"Detected {len(filtered_anomalies)} anomalies for user {user_id}")
            return filtered_anomalies
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {str(e)}")
            raise
    
    async def _get_user_transactions(self, user_id: str, days: int) -> List[Dict[str, Any]]:
        """Get user transaction data for the specified time period."""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            query = """
            SELECT 
                t.transaction_id,
                t.amount,
                t.transaction_date,
                t.category,
                t.subcategory,
                m.normalized_name as merchant_name,
                m.category as merchant_category,
                EXTRACT(HOUR FROM t.transaction_time) as hour_of_day,
                EXTRACT(DOW FROM t.transaction_date) as day_of_week,
                t.payment_method
            FROM transactions t
            LEFT JOIN merchants m ON t.merchant_id = m.merchant_id
            WHERE t.user_id = %s 
                AND t.transaction_date >= %s 
                AND t.transaction_date <= %s
                AND t.deleted_at IS NULL
            ORDER BY t.transaction_date DESC
            """
            
            result = await self.db_connector.execute_query(
                query, (user_id, start_date, end_date)
            )
            
            return [dict(row) for row in result]
            
        except Exception as e:
            self.logger.error(f"Error fetching transactions: {str(e)}")
            raise
    
    async def _detect_amount_anomalies(
        self, 
        user_id: str, 
        historical_df: pd.DataFrame, 
        recent_df: pd.DataFrame
    ) -> List[SpendingAnomaly]:
        """Detect anomalies in spending amounts."""
        anomalies = []
        
        try:
            # Overall spending anomalies
            historical_amounts = historical_df['amount'].values
            recent_amounts = recent_df['amount'].values
            
            if len(historical_amounts) > 0 and len(recent_amounts) > 0:
                # Statistical analysis
                hist_mean = np.mean(historical_amounts)
                hist_std = np.std(historical_amounts)
                
                if hist_std > 0:
                    # Check for individual transaction anomalies
                    for _, transaction in recent_df.iterrows():
                        z_score = abs((transaction['amount'] - hist_mean) / hist_std)
                        
                        if z_score >= self.z_score_thresholds[AnomalySeverity.MINOR]:
                            severity = self._calculate_severity_from_z_score(z_score)
                            polarity = AnomalyPolarity.NEGATIVE if transaction['amount'] > hist_mean else AnomalyPolarity.POSITIVE
                            
                            anomaly = SpendingAnomaly(
                                anomaly_id=str(uuid.uuid4()),
                                user_id=user_id,
                                anomaly_type=AnomalyType.SPENDING_SPIKE if transaction['amount'] > hist_mean else AnomalyType.SPENDING_DROP,
                                severity=severity,
                                polarity=polarity,
                                z_score=z_score,
                                confidence_score=min(z_score / 4.0, 1.0),
                                impact_score=abs(transaction['amount'] - hist_mean) / hist_mean,
                                category=transaction.get('category'),
                                merchant=transaction.get('merchant_name'),
                                observed_value=transaction['amount'],
                                expected_value=hist_mean,
                                historical_mean=hist_mean,
                                historical_std=hist_std,
                                description=self._generate_amount_anomaly_description(
                                    transaction['amount'], hist_mean, polarity
                                ),
                                related_transactions=[transaction['transaction_id']]
                            )
                            
                            # Add contributing factors and recommendations
                            await self._add_amount_anomaly_context(anomaly, transaction, historical_df)
                            anomalies.append(anomaly)
            
            # Category-specific amount anomalies
            for category in historical_df['category'].unique():
                if category:
                    cat_anomalies = await self._detect_category_amount_anomalies(
                        user_id, historical_df, recent_df, category
                    )
                    anomalies.extend(cat_anomalies)
            
        except Exception as e:
            self.logger.error(f"Error detecting amount anomalies: {str(e)}")
        
        return anomalies
    
    async def _detect_frequency_anomalies(
        self, 
        user_id: str, 
        historical_df: pd.DataFrame, 
        recent_df: pd.DataFrame
    ) -> List[SpendingAnomaly]:
        """Detect anomalies in transaction frequency."""
        anomalies = []
        
        try:
            # Calculate daily transaction frequencies
            historical_daily = historical_df.groupby('transaction_date').size()
            recent_daily = recent_df.groupby('transaction_date').size()
            
            if len(historical_daily) > 7 and len(recent_daily) > 0:
                hist_mean_freq = historical_daily.mean()
                hist_std_freq = historical_daily.std()
                
                if hist_std_freq > 0:
                    recent_mean_freq = recent_daily.mean()
                    z_score = abs((recent_mean_freq - hist_mean_freq) / hist_std_freq)
                    
                    if z_score >= self.z_score_thresholds[AnomalySeverity.MINOR]:
                        severity = self._calculate_severity_from_z_score(z_score)
                        polarity = AnomalyPolarity.NEGATIVE if recent_mean_freq > hist_mean_freq else AnomalyPolarity.POSITIVE
                        
                        anomaly = SpendingAnomaly(
                            anomaly_id=str(uuid.uuid4()),
                            user_id=user_id,
                            anomaly_type=AnomalyType.FREQUENCY_ANOMALY,
                            severity=severity,
                            polarity=polarity,
                            z_score=z_score,
                            confidence_score=min(z_score / 3.0, 1.0),
                            impact_score=abs(recent_mean_freq - hist_mean_freq) / hist_mean_freq,
                            observed_value=recent_mean_freq,
                            expected_value=hist_mean_freq,
                            historical_mean=hist_mean_freq,
                            historical_std=hist_std_freq,
                            description=self._generate_frequency_anomaly_description(
                                recent_mean_freq, hist_mean_freq, polarity
                            )
                        )
                        
                        await self._add_frequency_anomaly_context(anomaly, historical_df, recent_df)
                        anomalies.append(anomaly)
            
        except Exception as e:
            self.logger.error(f"Error detecting frequency anomalies: {str(e)}")
        
        return anomalies
    
    async def _detect_category_anomalies(
        self, 
        user_id: str, 
        historical_df: pd.DataFrame, 
        recent_df: pd.DataFrame
    ) -> List[SpendingAnomaly]:
        """Detect anomalies in category spending distribution."""
        anomalies = []
        
        try:
            # Calculate category distributions
            historical_cat_dist = historical_df.groupby('category')['amount'].sum() / historical_df['amount'].sum()
            recent_cat_dist = recent_df.groupby('category')['amount'].sum() / recent_df['amount'].sum()
            
            # Compare distributions
            for category in historical_cat_dist.index:
                if category in recent_cat_dist.index:
                    hist_pct = historical_cat_dist[category]
                    recent_pct = recent_cat_dist[category]
                    
                    # Calculate relative change
                    if hist_pct > 0:
                        relative_change = abs(recent_pct - hist_pct) / hist_pct
                        
                        if relative_change > 0.5:  # 50% change threshold
                            polarity = AnomalyPolarity.NEGATIVE if recent_pct > hist_pct else AnomalyPolarity.POSITIVE
                            
                            # Adjust polarity based on category type
                            if category.lower() in ['savings', 'investment', 'emergency']:
                                polarity = AnomalyPolarity.POSITIVE if recent_pct > hist_pct else AnomalyPolarity.NEGATIVE
                            
                            anomaly = SpendingAnomaly(
                                anomaly_id=str(uuid.uuid4()),
                                user_id=user_id,
                                anomaly_type=AnomalyType.CATEGORY_SHIFT,
                                severity=self._calculate_severity_from_change(relative_change),
                                polarity=polarity,
                                z_score=relative_change * 2,  # Approximate z-score
                                confidence_score=min(relative_change, 1.0),
                                impact_score=relative_change,
                                category=category,
                                observed_value=recent_pct,
                                expected_value=hist_pct,
                                description=self._generate_category_anomaly_description(
                                    category, recent_pct, hist_pct, polarity
                                )
                            )
                            
                            await self._add_category_anomaly_context(anomaly, category, historical_df, recent_df)
                            anomalies.append(anomaly)
            
        except Exception as e:
            self.logger.error(f"Error detecting category anomalies: {str(e)}")
        
        return anomalies
    
    async def _detect_merchant_anomalies(
        self, 
        user_id: str, 
        historical_df: pd.DataFrame, 
        recent_df: pd.DataFrame
    ) -> List[SpendingAnomaly]:
        """Detect anomalies in merchant spending patterns."""
        anomalies = []
        
        try:
            # New merchants (not seen in historical data)
            historical_merchants = set(historical_df['merchant_name'].dropna())
            recent_merchants = set(recent_df['merchant_name'].dropna())
            
            new_merchants = recent_merchants - historical_merchants
            
            for merchant in new_merchants:
                merchant_spending = recent_df[recent_df['merchant_name'] == merchant]['amount'].sum()
                total_recent_spending = recent_df['amount'].sum()
                
                if merchant_spending / total_recent_spending > 0.1:  # New merchant accounts for >10% of spending
                    anomaly = SpendingAnomaly(
                        anomaly_id=str(uuid.uuid4()),
                        user_id=user_id,
                        anomaly_type=AnomalyType.MERCHANT_ANOMALY,
                        severity=AnomalySeverity.MODERATE,
                        polarity=AnomalyPolarity.NEUTRAL,
                        z_score=2.0,  # Default for new merchant
                        confidence_score=0.7,
                        impact_score=merchant_spending / total_recent_spending,
                        merchant=merchant,
                        observed_value=merchant_spending,
                        expected_value=0.0,
                        description=f"New merchant detected: {merchant} (${merchant_spending:.2f} spent)"
                    )
                    
                    await self._add_merchant_anomaly_context(anomaly, merchant, recent_df)
                    anomalies.append(anomaly)
            
            # Unusual spending at existing merchants
            for merchant in historical_merchants.intersection(recent_merchants):
                hist_merchant_amounts = historical_df[historical_df['merchant_name'] == merchant]['amount']
                recent_merchant_amounts = recent_df[recent_df['merchant_name'] == merchant]['amount']
                
                if len(hist_merchant_amounts) > 2 and len(recent_merchant_amounts) > 0:
                    hist_mean = hist_merchant_amounts.mean()
                    hist_std = hist_merchant_amounts.std()
                    recent_total = recent_merchant_amounts.sum()
                    
                    if hist_std > 0:
                        z_score = abs((recent_total - hist_mean) / hist_std)
                        
                        if z_score >= self.z_score_thresholds[AnomalySeverity.MODERATE]:
                            severity = self._calculate_severity_from_z_score(z_score)
                            polarity = AnomalyPolarity.NEGATIVE if recent_total > hist_mean else AnomalyPolarity.POSITIVE
                            
                            anomaly = SpendingAnomaly(
                                anomaly_id=str(uuid.uuid4()),
                                user_id=user_id,
                                anomaly_type=AnomalyType.MERCHANT_ANOMALY,
                                severity=severity,
                                polarity=polarity,
                                z_score=z_score,
                                confidence_score=min(z_score / 3.0, 1.0),
                                impact_score=abs(recent_total - hist_mean) / hist_mean,
                                merchant=merchant,
                                observed_value=recent_total,
                                expected_value=hist_mean,
                                description=f"Unusual spending at {merchant}: ${recent_total:.2f} vs typical ${hist_mean:.2f}"
                            )
                            
                            await self._add_merchant_anomaly_context(anomaly, merchant, recent_df)
                            anomalies.append(anomaly)
            
        except Exception as e:
            self.logger.error(f"Error detecting merchant anomalies: {str(e)}")
        
        return anomalies
    
    async def _detect_time_pattern_anomalies(
        self, 
        user_id: str, 
        historical_df: pd.DataFrame, 
        recent_df: pd.DataFrame
    ) -> List[SpendingAnomaly]:
        """Detect anomalies in time-based spending patterns."""
        anomalies = []
        
        try:
            # Day of week patterns
            if 'day_of_week' in historical_df.columns and 'day_of_week' in recent_df.columns:
                hist_dow_dist = historical_df.groupby('day_of_week')['amount'].sum() / historical_df['amount'].sum()
                recent_dow_dist = recent_df.groupby('day_of_week')['amount'].sum() / recent_df['amount'].sum()
                
                for dow in hist_dow_dist.index:
                    if dow in recent_dow_dist.index:
                        hist_pct = hist_dow_dist[dow]
                        recent_pct = recent_dow_dist[dow]
                        
                        if hist_pct > 0:
                            relative_change = abs(recent_pct - hist_pct) / hist_pct
                            
                            if relative_change > 0.6:  # 60% change threshold
                                anomaly = SpendingAnomaly(
                                    anomaly_id=str(uuid.uuid4()),
                                    user_id=user_id,
                                    anomaly_type=AnomalyType.TIME_PATTERN_ANOMALY,
                                    severity=self._calculate_severity_from_change(relative_change),
                                    polarity=AnomalyPolarity.NEUTRAL,
                                    z_score=relative_change * 2,
                                    confidence_score=min(relative_change, 1.0),
                                    impact_score=relative_change,
                                    observed_value=recent_pct,
                                    expected_value=hist_pct,
                                    description=f"Unusual spending pattern on {self._get_day_name(dow)}: {recent_pct:.1%} vs typical {hist_pct:.1%}"
                                )
                                
                                anomalies.append(anomaly)
            
            # Hour of day patterns (if available)
            if 'hour_of_day' in historical_df.columns and 'hour_of_day' in recent_df.columns:
                # Similar analysis for hour patterns
                pass
            
        except Exception as e:
            self.logger.error(f"Error detecting time pattern anomalies: {str(e)}")
        
        return anomalies
    
    async def _detect_ml_anomalies(
        self, 
        user_id: str, 
        historical_df: pd.DataFrame, 
        recent_df: pd.DataFrame
    ) -> List[SpendingAnomaly]:
        """Detect anomalies using machine learning (Isolation Forest)."""
        anomalies = []
        
        try:
            # Prepare features for ML model
            features = ['amount', 'day_of_week']
            if 'hour_of_day' in historical_df.columns:
                features.append('hour_of_day')
            
            # Encode categorical features
            historical_ml_df = self._prepare_ml_features(historical_df)
            recent_ml_df = self._prepare_ml_features(recent_df)
            
            if len(historical_ml_df) > 10 and len(recent_ml_df) > 0:
                # Train Isolation Forest on historical data
                self.isolation_forest = IsolationForest(
                    contamination=0.1,
                    random_state=42,
                    n_estimators=100
                )
                
                # Fit scaler and model on historical data
                historical_features = self.scaler.fit_transform(historical_ml_df)
                self.isolation_forest.fit(historical_features)
                
                # Predict anomalies in recent data
                recent_features = self.scaler.transform(recent_ml_df)
                anomaly_scores = self.isolation_forest.decision_function(recent_features)
                is_anomaly = self.isolation_forest.predict(recent_features)
                
                # Process detected anomalies
                for idx, (score, is_anom) in enumerate(zip(anomaly_scores, is_anomaly)):
                    if is_anom == -1:  # Anomaly detected
                        transaction = recent_df.iloc[idx]
                        
                        # Convert isolation forest score to severity
                        normalized_score = abs(score)
                        severity = self._calculate_severity_from_isolation_score(normalized_score)
                        
                        anomaly = SpendingAnomaly(
                            anomaly_id=str(uuid.uuid4()),
                            user_id=user_id,
                            anomaly_type=AnomalyType.AMOUNT_PATTERN_ANOMALY,
                            severity=severity,
                            polarity=AnomalyPolarity.NEUTRAL,
                            z_score=normalized_score * 2,  # Approximate z-score
                            confidence_score=min(normalized_score, 1.0),
                            impact_score=normalized_score,
                            category=transaction.get('category'),
                            merchant=transaction.get('merchant_name'),
                            observed_value=transaction['amount'],
                            description=f"Unusual transaction pattern detected (ML): ${transaction['amount']:.2f} at {transaction.get('merchant_name', 'Unknown')}",
                            related_transactions=[transaction['transaction_id']]
                        )
                        
                        anomalies.append(anomaly)
            
        except Exception as e:
            self.logger.error(f"Error in ML anomaly detection: {str(e)}")
        
        return anomalies
    
    def _prepare_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning model."""
        ml_df = df.copy()
        
        # Select and create features
        features = ['amount']
        
        if 'day_of_week' in df.columns:
            features.append('day_of_week')
        
        if 'hour_of_day' in df.columns:
            features.append('hour_of_day')
        
        # Add derived features
        ml_df['log_amount'] = np.log1p(ml_df['amount'])
        features.append('log_amount')
        
        # Category encoding (simplified)
        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            ml_df['category_frequency'] = ml_df['category'].map(category_counts)
            features.append('category_frequency')
        
        return ml_df[features].fillna(0)
    
    async def _filter_and_rank_anomalies(self, anomalies: List[SpendingAnomaly]) -> List[SpendingAnomaly]:
        """Filter and rank anomalies by importance."""
        if not anomalies:
            return []
        
        # Remove duplicates based on similar characteristics
        unique_anomalies = self._remove_duplicate_anomalies(anomalies)
        
        # Calculate composite scores for ranking
        for anomaly in unique_anomalies:
            anomaly.composite_score = (
                anomaly.confidence_score * 0.4 +
                anomaly.impact_score * 0.3 +
                self._severity_to_score(anomaly.severity) * 0.3
            )
        
        # Sort by composite score (highest first)
        unique_anomalies.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Limit to top anomalies to avoid overwhelming user
        return unique_anomalies[:10]
    
    def _remove_duplicate_anomalies(self, anomalies: List[SpendingAnomaly]) -> List[SpendingAnomaly]:
        """Remove similar/duplicate anomalies."""
        if len(anomalies) <= 1:
            return anomalies
        
        unique_anomalies = []
        
        for anomaly in anomalies:
            is_duplicate = False
            
            for existing in unique_anomalies:
                if (anomaly.anomaly_type == existing.anomaly_type and
                    anomaly.category == existing.category and
                    anomaly.merchant == existing.merchant and
                    abs(anomaly.impact_score - existing.impact_score) < 0.1):
                    
                    # Keep the one with higher confidence
                    if anomaly.confidence_score > existing.confidence_score:
                        unique_anomalies.remove(existing)
                        unique_anomalies.append(anomaly)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_anomalies.append(anomaly)
        
        return unique_anomalies
    
    # Helper methods for severity calculation
    def _calculate_severity_from_z_score(self, z_score: float) -> AnomalySeverity:
        """Calculate severity based on z-score."""
        if z_score >= 4.0:
            return AnomalySeverity.CRITICAL
        elif z_score >= 3.0:
            return AnomalySeverity.SIGNIFICANT
        elif z_score >= 2.0:
            return AnomalySeverity.MODERATE
        else:
            return AnomalySeverity.MINOR
    
    def _calculate_severity_from_change(self, relative_change: float) -> AnomalySeverity:
        """Calculate severity based on relative change."""
        if relative_change >= 2.0:
            return AnomalySeverity.CRITICAL
        elif relative_change >= 1.0:
            return AnomalySeverity.SIGNIFICANT
        elif relative_change >= 0.5:
            return AnomalySeverity.MODERATE
        else:
            return AnomalySeverity.MINOR
    
    def _calculate_severity_from_isolation_score(self, score: float) -> AnomalySeverity:
        """Calculate severity based on isolation forest score."""
        if score >= 0.6:
            return AnomalySeverity.CRITICAL
        elif score >= 0.4:
            return AnomalySeverity.SIGNIFICANT
        elif score >= 0.2:
            return AnomalySeverity.MODERATE
        else:
            return AnomalySeverity.MINOR
    
    def _severity_to_score(self, severity: AnomalySeverity) -> float:
        """Convert severity enum to numeric score for ranking."""
        severity_scores = {
            AnomalySeverity.CRITICAL: 1.0,
            AnomalySeverity.SIGNIFICANT: 0.75,
            AnomalySeverity.MODERATE: 0.5,
            AnomalySeverity.MINOR: 0.25
        }
        return severity_scores.get(severity, 0.0)
    
    # Description generation methods
    def _generate_amount_anomaly_description(
        self, 
        observed_amount: float, 
        expected_amount: float, 
        polarity: AnomalyPolarity
    ) -> str:
        """Generate human-readable description for amount anomalies."""
        difference = abs(observed_amount - expected_amount)
        percentage_diff = (difference / expected_amount) * 100 if expected_amount > 0 else 0
        
        if polarity == AnomalyPolarity.NEGATIVE:
            return f"Unusually high transaction of ${observed_amount:.2f}, {percentage_diff:.1f}% above typical ${expected_amount:.2f}"
        else:
            return f"Unusually low transaction of ${observed_amount:.2f}, {percentage_diff:.1f}% below typical ${expected_amount:.2f}"
    
    def _generate_frequency_anomaly_description(
        self, 
        observed_freq: float, 
        expected_freq: float, 
        polarity: AnomalyPolarity
    ) -> str:
        """Generate description for frequency anomalies."""
        if polarity == AnomalyPolarity.NEGATIVE:
            return f"Increased transaction frequency: {observed_freq:.1f} vs typical {expected_freq:.1f} transactions per day"
        else:
            return f"Decreased transaction frequency: {observed_freq:.1f} vs typical {expected_freq:.1f} transactions per day"
    
    def _generate_category_anomaly_description(
        self, 
        category: str, 
        observed_pct: float, 
        expected_pct: float, 
        polarity: AnomalyPolarity
    ) -> str:
        """Generate description for category shift anomalies."""
        if polarity == AnomalyPolarity.NEGATIVE:
            return f"Significant increase in {category} spending: {observed_pct:.1%} vs typical {expected_pct:.1%} of total budget"
        else:
            return f"Notable decrease in {category} spending: {observed_pct:.1%} vs typical {expected_pct:.1%} of total budget"
    
    # Context addition methods
    async def _add_amount_anomaly_context(
        self, 
        anomaly: SpendingAnomaly, 
        transaction: pd.Series, 
        historical_df: pd.DataFrame
    ):
        """Add contextual information and recommendations for amount anomalies."""
        # Add contributing factors
        if transaction.get('category'):
            category_spending = historical_df[
                historical_df['category'] == transaction['category']
            ]['amount'].mean()
            
            if transaction['amount'] > category_spending * 2:
                anomaly.contributing_factors.append(f"Transaction significantly exceeds typical {transaction['category']} spending")
        
        # Time-based factors
        if 'day_of_week' in transaction:
            dow = transaction['day_of_week']
            if dow in [5, 6]:  # Weekend
                anomaly.contributing_factors.append("Weekend spending pattern")
        
        # Add recommendations based on polarity and severity
        if anomaly.polarity == AnomalyPolarity.NEGATIVE:
            if anomaly.severity in [AnomalySeverity.CRITICAL, AnomalySeverity.SIGNIFICANT]:
                anomaly.recommendations.extend([
                    "Review if this large expense was planned and necessary",
                    "Consider if this indicates a need to adjust your budget",
                    "Track similar transactions to identify spending pattern changes"
                ])
            else:
                anomaly.recommendations.append("Monitor future transactions in this category")
        else:
            anomaly.recommendations.append("Great job keeping spending low in this area!")
    
    async def _add_frequency_anomaly_context(
        self, 
        anomaly: SpendingAnomaly, 
        historical_df: pd.DataFrame, 
        recent_df: pd.DataFrame
    ):
        """Add context for frequency anomalies."""
        # Identify which merchants or categories are driving frequency changes
        recent_merchant_counts = recent_df['merchant_name'].value_counts()
        historical_merchant_counts = historical_df['merchant_name'].value_counts()
        
        # Find merchants with significantly different frequencies
        for merchant in recent_merchant_counts.index[:3]:
            recent_count = recent_merchant_counts[merchant]
            historical_avg = historical_merchant_counts.get(merchant, 0) / len(historical_df.groupby('transaction_date'))
            
            if recent_count > historical_avg * 1.5:
                anomaly.contributing_factors.append(f"Increased transactions at {merchant}")
        
        if anomaly.polarity == AnomalyPolarity.NEGATIVE:
            anomaly.recommendations.extend([
                "Review what's driving the increase in transaction frequency",
                "Consider consolidating purchases to reduce transaction fees",
                "Check if increased frequency aligns with your spending goals"
            ])
        else:
            anomaly.recommendations.append("Fewer transactions might indicate better spending discipline")
    
    async def _add_category_anomaly_context(
        self, 
        anomaly: SpendingAnomaly, 
        category: str, 
        historical_df: pd.DataFrame, 
        recent_df: pd.DataFrame
    ):
        """Add context for category shift anomalies."""
        # Analyze what merchants in this category are driving the change
        category_merchants_recent = recent_df[recent_df['category'] == category]['merchant_name'].value_counts()
        category_merchants_historical = historical_df[historical_df['category'] == category]['merchant_name'].value_counts()
        
        for merchant in category_merchants_recent.index[:2]:
            recent_spending = recent_df[
                (recent_df['category'] == category) & 
                (recent_df['merchant_name'] == merchant)
            ]['amount'].sum()
            
            historical_spending = historical_df[
                (historical_df['category'] == category) & 
                (historical_df['merchant_name'] == merchant)
            ]['amount'].sum() / len(historical_df.groupby('transaction_date')) * len(recent_df.groupby('transaction_date'))
            
            if recent_spending > historical_spending * 1.3:
                anomaly.contributing_factors.append(f"Increased spending at {merchant} in {category}")
        
        # Category-specific recommendations
        essential_categories = ['groceries', 'utilities', 'rent', 'healthcare']
        discretionary_categories = ['dining', 'entertainment', 'shopping', 'travel']
        
        if category.lower() in essential_categories:
            if anomaly.polarity == AnomalyPolarity.NEGATIVE:
                anomaly.recommendations.extend([
                    f"Review if increased {category} spending is due to price increases or usage changes",
                    "Consider ways to optimize essential spending without compromising quality"
                ])
        elif category.lower() in discretionary_categories:
            if anomaly.polarity == AnomalyPolarity.NEGATIVE:
                anomaly.recommendations.extend([
                    f"Consider reducing discretionary {category} spending to stay within budget",
                    "Set specific limits for discretionary categories"
                ])
            else:
                anomaly.recommendations.append(f"Great job reducing {category} spending!")
    
    async def _add_merchant_anomaly_context(
        self, 
        anomaly: SpendingAnomaly, 
        merchant: str, 
        recent_df: pd.DataFrame
    ):
        """Add context for merchant anomalies."""
        merchant_transactions = recent_df[recent_df['merchant_name'] == merchant]
        
        # Transaction frequency
        transaction_count = len(merchant_transactions)
        total_spending = merchant_transactions['amount'].sum()
        avg_transaction = total_spending / transaction_count if transaction_count > 0 else 0
        
        anomaly.contributing_factors.extend([
            f"{transaction_count} transactions totaling ${total_spending:.2f}",
            f"Average transaction: ${avg_transaction:.2f}"
        ])
        
        # Time pattern
        if 'day_of_week' in merchant_transactions.columns:
            frequent_days = merchant_transactions['day_of_week'].mode()
            if len(frequent_days) > 0:
                day_name = self._get_day_name(frequent_days.iloc[0])
                anomaly.contributing_factors.append(f"Most transactions on {day_name}")
        
        anomaly.recommendations.extend([
            f"Review your relationship with {merchant} and spending patterns",
            "Consider if this merchant aligns with your financial goals",
            "Set alerts for future spending at this merchant if concerning"
        ])
    
    def _get_day_name(self, day_of_week: int) -> str:
        """Convert day of week number to name."""
        days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        return days[day_of_week] if 0 <= day_of_week < 7 else 'Unknown'
    
    async def _detect_category_amount_anomalies(
        self, 
        user_id: str, 
        historical_df: pd.DataFrame, 
        recent_df: pd.DataFrame, 
        category: str
    ) -> List[SpendingAnomaly]:
        """Detect amount anomalies within a specific category."""
        anomalies = []
        
        try:
            # Filter data for the specific category
            hist_category = historical_df[historical_df['category'] == category]
            recent_category = recent_df[recent_df['category'] == category]
            
            if len(hist_category) < 3 or len(recent_category) == 0:
                return anomalies
            
            # Calculate category-specific statistics
            hist_amounts = hist_category['amount'].values
            hist_mean = np.mean(hist_amounts)
            hist_std = np.std(hist_amounts)
            
            if hist_std == 0:
                return anomalies
            
            # Check recent category spending
            recent_total = recent_category['amount'].sum()
            expected_total = hist_mean * len(recent_category)
            
            if expected_total > 0:
                z_score = abs((recent_total - expected_total) / (hist_std * np.sqrt(len(recent_category))))
                
                if z_score >= self.z_score_thresholds[AnomalySeverity.MODERATE]:
                    severity = self._calculate_severity_from_z_score(z_score)
                    polarity = AnomalyPolarity.NEGATIVE if recent_total > expected_total else AnomalyPolarity.POSITIVE
                    
                    anomaly = SpendingAnomaly(
                        anomaly_id=str(uuid.uuid4()),
                        user_id=user_id,
                        anomaly_type=AnomalyType.CATEGORY_SHIFT,
                        severity=severity,
                        polarity=polarity,
                        z_score=z_score,
                        confidence_score=min(z_score / 3.0, 1.0),
                        impact_score=abs(recent_total - expected_total) / expected_total,
                        category=category,
                        observed_value=recent_total,
                        expected_value=expected_total,
                        historical_mean=hist_mean,
                        historical_std=hist_std,
                        description=f"Category spending anomaly in {category}: ${recent_total:.2f} vs expected ${expected_total:.2f}",
                        related_transactions=recent_category['transaction_id'].tolist()
                    )
                    
                    await self._add_category_anomaly_context(anomaly, category, historical_df, recent_df)
                    anomalies.append(anomaly)
        
        except Exception as e:
            self.logger.error(f"Error detecting category amount anomalies for {category}: {str(e)}")
        
        return anomalies
    
    # Integration methods for Proactive Insights Agent
    async def get_anomalies_for_insights(
        self, 
        user_id: str, 
        analysis_window_days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Get anomalies formatted for the proactive insights agent.
        This method provides a simplified interface for integration.
        """
        try:
            anomalies = await self.detect_anomalies(
                user_id=user_id,
                analysis_window_days=analysis_window_days,
                historical_window_days=90,  # Shorter window for insights
                min_transactions=5  # Lower threshold for insights
            )
            
            # Convert to format expected by proactive insights agent
            insights_data = []
            for anomaly in anomalies:
                insight_data = {
                    "anomaly_type": anomaly.anomaly_type.value,
                    "severity": anomaly.severity.value,
                    "polarity": anomaly.polarity.value,
                    "category": anomaly.category,
                    "merchant": anomaly.merchant,
                    "z_score": anomaly.z_score,
                    "confidence_score": anomaly.confidence_score,
                    "impact_score": anomaly.impact_score,
                    "observed_value": anomaly.observed_value,
                    "expected_value": anomaly.expected_value,
                    "description": anomaly.description,
                    "contributing_factors": anomaly.contributing_factors,
                    "recommendations": anomaly.recommendations,
                    "detection_timestamp": anomaly.detection_timestamp.isoformat()
                }
                insights_data.append(insight_data)
            
            return insights_data
            
        except Exception as e:
            self.logger.error(f"Error getting anomalies for insights: {str(e)}")
            return []
    
    async def get_anomaly_summary(self, user_id: str) -> Dict[str, Any]:
        """Get a summary of anomalies for dashboard display."""
        try:
            anomalies = await self.detect_anomalies(user_id)
            
            summary = {
                "total_anomalies": len(anomalies),
                "by_severity": {},
                "by_type": {},
                "by_polarity": {},
                "high_impact_anomalies": [],
                "recent_anomaly_trend": "stable"  # This would be calculated based on historical data
            }
            
            # Aggregate by different dimensions
            for anomaly in anomalies:
                # By severity
                severity_key = anomaly.severity.value
                summary["by_severity"][severity_key] = summary["by_severity"].get(severity_key, 0) + 1
                
                # By type
                type_key = anomaly.anomaly_type.value
                summary["by_type"][type_key] = summary["by_type"].get(type_key, 0) + 1
                
                # By polarity
                polarity_key = anomaly.polarity.value
                summary["by_polarity"][polarity_key] = summary["by_polarity"].get(polarity_key, 0) + 1
                
                # High impact anomalies (top 3)
                if len(summary["high_impact_anomalies"]) < 3:
                    summary["high_impact_anomalies"].append({
                        "type": anomaly.anomaly_type.value,
                        "category": anomaly.category,
                        "impact_score": anomaly.impact_score,
                        "description": anomaly.description
                    })
            
            # Sort high impact anomalies by impact score
            summary["high_impact_anomalies"].sort(key=lambda x: x["impact_score"], reverse=True)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting anomaly summary: {str(e)}")
            return {"total_anomalies": 0, "error": str(e)}
    
    def set_custom_thresholds(self, user_id: str, thresholds: Dict[str, float]):
        """Allow users to customize anomaly detection thresholds."""
        # This would be stored in user preferences
        # For now, just updating instance thresholds
        if "z_score_minor" in thresholds:
            self.z_score_thresholds[AnomalySeverity.MINOR] = thresholds["z_score_minor"]
        if "z_score_moderate" in thresholds:
            self.z_score_thresholds[AnomalySeverity.MODERATE] = thresholds["z_score_moderate"]
        if "z_score_significant" in thresholds:
            self.z_score_thresholds[AnomalySeverity.SIGNIFICANT] = thresholds["z_score_significant"]
        if "z_score_critical" in thresholds:
            self.z_score_thresholds[AnomalySeverity.CRITICAL] = thresholds["z_score_critical"]
        
        self.logger.info(f"Updated anomaly thresholds for user {user_id}")
    
    async def explain_anomaly(self, anomaly_id: str) -> Dict[str, Any]:
        """Provide detailed explanation of a specific anomaly."""
        # This would retrieve the anomaly from storage and provide detailed analysis
        # For now, returning a template structure
        return {
            "anomaly_id": anomaly_id,
            "explanation": "Detailed explanation would be generated here",
            "statistical_context": {},
            "historical_comparison": {},
            "suggested_actions": []
        }