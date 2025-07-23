import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import numpy as np
from dataclasses import dataclass
from collections import defaultdict, Counter
import statistics
from enum import Enum


class PatternType(Enum):
    SPENDING_HABITS = "spending_habits"
    INCOME_PATTERNS = "income_patterns"
    CATEGORY_TRENDS = "category_trends"
    ANOMALY_DETECTION = "anomaly_detection"
    RECURRING_TRANSACTIONS = "recurring_transactions"
    SEASONAL_PATTERNS = "seasonal_patterns"
    LIFE_EVENT_IMPACT = "life_event_impact"
    OPTIMIZATION_OPPORTUNITIES = "optimization_opportunities"


class AnomalyType(Enum):
    AMOUNT_SPIKE = "amount_spike"
    FREQUENCY_CHANGE = "frequency_change"
    NEW_MERCHANT = "new_merchant"
    UNUSUAL_CATEGORY = "unusual_category"
    TIME_ANOMALY = "time_anomaly"
    LOCATION_ANOMALY = "location_anomaly"


@dataclass
class Pattern:
    """Represents a detected financial pattern."""
    pattern_id: str
    pattern_type: PatternType
    category: Optional[str]
    confidence: float
    frequency: str  # daily, weekly, monthly, etc.
    description: str
    supporting_data: Dict[str, Any]
    recommendations: List[str]
    detected_at: datetime
    time_range: Dict[str, datetime]


@dataclass
class Anomaly:
    """Represents a detected financial anomaly."""
    anomaly_id: str
    anomaly_type: AnomalyType
    severity: str  # low, medium, high, critical
    confidence: float
    description: str
    affected_data: Dict[str, Any]
    baseline_comparison: Dict[str, Any]
    recommendations: List[str]
    detected_at: datetime


class PatternRecognitionTool:
    """
    Advanced pattern recognition tool for financial behavior analysis.
    Identifies spending patterns, anomalies, and optimization opportunities.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the Pattern Recognition Tool.
        
        Args:
            logger: Logger instance for tracking operations
        """
        self.logger = logger or logging.getLogger(__name__)
        self.patterns_cache = {}
        self.anomalies_cache = {}
        
        # Configuration for pattern detection
        self.config = {
            "anomaly_thresholds": {
                "amount_multiplier": 2.5,  # Flag transactions > 2.5x normal
                "frequency_change": 0.5,   # Flag 50% change in frequency
                "new_merchant_threshold": 0.9,  # Confidence for new merchant patterns
                "category_variance": 3.0   # Standard deviations for category anomalies
            },
            "pattern_confidence_thresholds": {
                "minimum_confidence": 0.6,
                "high_confidence": 0.8,
                "minimum_data_points": 5
            },
            "seasonal_analysis": {
                "months_for_seasonality": 12,
                "weekly_cycle_threshold": 0.7
            }
        }
        
        self.logger.info("Pattern Recognition Tool initialized")
    
    async def recognize_patterns(
        self, 
        pattern_type: str, 
        time_window: str = "3_months",
        sensitivity: float = 0.5,
        categories: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main entry point for pattern recognition.
        
        Args:
            pattern_type: Type of pattern to recognize
            time_window: Time window for analysis
            sensitivity: Pattern detection sensitivity (0-1)
            categories: Specific categories to analyze
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing detected patterns and insights
        """
        try:
            self.logger.info(f"Starting pattern recognition: {pattern_type}")
            
            # Convert string to enum
            pattern_enum = PatternType(pattern_type)
            
            # Get sample data based on pattern type
            data = await self._get_sample_data(pattern_enum, time_window, categories)
            
            # Perform pattern recognition based on type
            if pattern_enum == PatternType.SPENDING_HABITS:
                result = await self._analyze_spending_habits(data, sensitivity)
            elif pattern_enum == PatternType.INCOME_PATTERNS:
                result = await self._analyze_income_patterns(data, sensitivity)
            elif pattern_enum == PatternType.CATEGORY_TRENDS:
                result = await self._analyze_category_trends(data, sensitivity)
            elif pattern_enum == PatternType.ANOMALY_DETECTION:
                result = await self._detect_anomalies(data, sensitivity)
            elif pattern_enum == PatternType.RECURRING_TRANSACTIONS:
                result = await self._find_recurring_transactions(data, sensitivity)
            elif pattern_enum == PatternType.SEASONAL_PATTERNS:
                result = await self._analyze_seasonal_patterns(data, sensitivity)
            elif pattern_enum == PatternType.LIFE_EVENT_IMPACT:
                result = await self._detect_life_event_impacts(data, sensitivity)
            elif pattern_enum == PatternType.OPTIMIZATION_OPPORTUNITIES:
                result = await self._find_optimization_opportunities(data, sensitivity)
            else:
                raise ValueError(f"Unsupported pattern type: {pattern_type}")
            
            # Add metadata
            result["analysis_metadata"] = {
                "pattern_type": pattern_type,
                "time_window": time_window,
                "sensitivity": sensitivity,
                "categories_analyzed": categories,
                "analysis_timestamp": datetime.now().isoformat(),
                "data_points_analyzed": len(data.get("transactions", []))
            }
            
            self.logger.info(f"Pattern recognition completed: {len(result.get('patterns', []))} patterns found")
            return result
            
        except Exception as e:
            self.logger.error(f"Pattern recognition failed: {e}")
            raise
    
    async def _get_sample_data(
        self, 
        pattern_type: PatternType, 
        time_window: str, 
        categories: Optional[List[str]]
    ) -> Dict[str, Any]:
        """
        Generate sample data for pattern analysis.
        In production, this would fetch real data from the database.
        """
        # Time window mapping
        time_deltas = {
            "1_month": timedelta(days=30),
            "3_months": timedelta(days=90),
            "6_months": timedelta(days=180),
            "1_year": timedelta(days=365),
            "all_time": timedelta(days=730)  # 2 years for sample
        }
        
        end_date = datetime.now()
        start_date = end_date - time_deltas.get(time_window, timedelta(days=90))
        
        # Generate sample transaction data
        transactions = []
        sample_categories = categories or ["groceries", "dining", "gas", "shopping", "utilities", "entertainment"]
        merchants = {
            "groceries": ["Whole Foods", "Safeway", "Trader Joes", "Kroger"],
            "dining": ["Starbucks", "McDonalds", "Local Cafe", "Restaurant ABC"],
            "gas": ["Shell", "Chevron", "BP", "Exxon"],
            "shopping": ["Amazon", "Target", "Walmart", "Best Buy"],
            "utilities": ["PG&E", "AT&T", "Comcast", "Water Dept"],
            "entertainment": ["Netflix", "Movie Theater", "Spotify", "Gaming Store"]
        }
        
        # Generate realistic transaction patterns
        current_date = start_date
        transaction_id = 1
        
        while current_date <= end_date:
            # Daily spending patterns (more spending on weekends for some categories)
            day_multiplier = 1.3 if current_date.weekday() >= 5 else 1.0
            
            # Monthly patterns (more spending at month beginning/end)
            month_day = current_date.day
            if month_day <= 5 or month_day >= 25:
                month_multiplier = 1.2
            else:
                month_multiplier = 1.0
            
            for category in sample_categories:
                # Category-specific patterns
                if category == "groceries":
                    # Weekly grocery shopping
                    if current_date.weekday() in [5, 6]:  # Weekend grocery runs
                        base_amount = 120
                        probability = 0.7
                    else:
                        base_amount = 25
                        probability = 0.3
                elif category == "dining":
                    # More dining on weekends and lunch hours
                    base_amount = 35 * day_multiplier
                    probability = 0.6
                elif category == "gas":
                    # Weekly gas fillups
                    if current_date.weekday() == 0:  # Monday gas fillup
                        base_amount = 45
                        probability = 0.8
                    else:
                        base_amount = 45
                        probability = 0.1
                elif category == "utilities":
                    # Monthly recurring
                    if month_day <= 5:
                        base_amount = 150
                        probability = 0.9
                    else:
                        probability = 0.0
                elif category == "shopping":
                    base_amount = 75 * month_multiplier
                    probability = 0.4
                else:
                    base_amount = 40
                    probability = 0.3
                
                # Generate transaction if probability hit
                if np.random.random() < probability:
                    # Add some variance to amounts
                    amount = base_amount * (0.7 + 0.6 * np.random.random())
                    
                    # Occasionally add anomalies
                    if np.random.random() < 0.02:  # 2% chance of anomaly
                        amount *= (3 + 2 * np.random.random())  # 3-5x normal amount
                    
                    merchant = np.random.choice(merchants[category])
                    
                    transaction = {
                        "transaction_id": f"txn_{transaction_id}",
                        "amount": round(amount, 2),
                        "category": category,
                        "merchant": merchant,
                        "date": current_date.date().isoformat(),
                        "time": f"{np.random.randint(8, 22):02d}:{np.random.randint(0, 60):02d}",
                        "day_of_week": current_date.strftime("%A"),
                        "month": current_date.month,
                        "week_of_year": current_date.isocalendar()[1]
                    }
                    
                    transactions.append(transaction)
                    transaction_id += 1
            
            current_date += timedelta(days=1)
        
        return {
            "transactions": transactions,
            "time_range": {
                "start_date": start_date,
                "end_date": end_date
            },
            "categories": sample_categories
        }
    
    async def _analyze_spending_habits(self, data: Dict[str, Any], sensitivity: float) -> Dict[str, Any]:
        """Analyze spending habits and identify patterns."""
        transactions = data["transactions"]
        patterns = []
        
        # Group transactions by various dimensions
        by_category = defaultdict(list)
        by_day_of_week = defaultdict(list)
        by_merchant = defaultdict(list)
        by_time_of_day = defaultdict(list)
        
        for txn in transactions:
            by_category[txn["category"]].append(txn)
            by_day_of_week[txn["day_of_week"]].append(txn)
            by_merchant[txn["merchant"]].append(txn)
            
            # Extract hour from time
            hour = int(txn["time"].split(":")[0])
            time_period = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
            by_time_of_day[time_period].append(txn)
        
        # Analyze weekly spending patterns
        weekly_pattern = self._analyze_weekly_patterns(by_day_of_week)
        if weekly_pattern["confidence"] >= sensitivity:
            patterns.append(Pattern(
                pattern_id=f"weekly_{len(patterns)}",
                pattern_type=PatternType.SPENDING_HABITS,
                category=None,
                confidence=weekly_pattern["confidence"],
                frequency="weekly",
                description=weekly_pattern["description"],
                supporting_data=weekly_pattern["data"],
                recommendations=weekly_pattern["recommendations"],
                detected_at=datetime.now(),
                time_range=data["time_range"]
            ))
        
        # Analyze category-specific habits
        for category, cat_transactions in by_category.items():
            if len(cat_transactions) >= 5:  # Minimum data points
                category_pattern = self._analyze_category_habits(category, cat_transactions)
                if category_pattern["confidence"] >= sensitivity:
                    patterns.append(Pattern(
                        pattern_id=f"category_{category}_{len(patterns)}",
                        pattern_type=PatternType.SPENDING_HABITS,
                        category=category,
                        confidence=category_pattern["confidence"],
                        frequency=category_pattern["frequency"],
                        description=category_pattern["description"],
                        supporting_data=category_pattern["data"],
                        recommendations=category_pattern["recommendations"],
                        detected_at=datetime.now(),
                        time_range=data["time_range"]
                    ))
        
        # Analyze merchant loyalty patterns
        merchant_patterns = self._analyze_merchant_loyalty(by_merchant)
        patterns.extend([Pattern(
            pattern_id=f"merchant_{p['merchant']}_{len(patterns)}",
            pattern_type=PatternType.SPENDING_HABITS,
            category=p.get("category"),
            confidence=p["confidence"],
            frequency="recurring",
            description=p["description"],
            supporting_data=p["data"],
            recommendations=p["recommendations"],
            detected_at=datetime.now(),
            time_range=data["time_range"]
        ) for p in merchant_patterns if p["confidence"] >= sensitivity])
        
        return {
            "patterns": [self._pattern_to_dict(p) for p in patterns],
            "summary": {
                "total_patterns": len(patterns),
                "high_confidence_patterns": len([p for p in patterns if p.confidence >= 0.8]),
                "categories_with_patterns": len(set(p.category for p in patterns if p.category)),
                "strongest_pattern": max(patterns, key=lambda p: p.confidence) if patterns else None
            }
        }
    
    async def _detect_anomalies(self, data: Dict[str, Any], sensitivity: float) -> Dict[str, Any]:
        """Detect anomalies in spending behavior."""
        transactions = data["transactions"]
        anomalies = []
        
        # Group by category for baseline analysis
        by_category = defaultdict(list)
        for txn in transactions:
            by_category[txn["category"]].append(float(txn["amount"]))
        
        # Detect amount anomalies
        for category, amounts in by_category.items():
            if len(amounts) >= 5:  # Need sufficient data
                mean_amount = statistics.mean(amounts)
                std_amount = statistics.stdev(amounts) if len(amounts) > 1 else 0
                threshold = mean_amount + (self.config["anomaly_thresholds"]["amount_multiplier"] * std_amount)
                
                # Find anomalous transactions
                category_txns = [txn for txn in transactions if txn["category"] == category]
                for txn in category_txns:
                    if float(txn["amount"]) > threshold:
                        severity = self._calculate_anomaly_severity(float(txn["amount"]), mean_amount, std_amount)
                        
                        anomaly = Anomaly(
                            anomaly_id=f"amount_{txn['transaction_id']}",
                            anomaly_type=AnomalyType.AMOUNT_SPIKE,
                            severity=severity,
                            confidence=min(1.0, (float(txn["amount"]) - mean_amount) / mean_amount),
                            description=f"Unusual {category} spending: ${txn['amount']} vs typical ${mean_amount:.2f}",
                            affected_data=txn,
                            baseline_comparison={
                                "mean": mean_amount,
                                "std_dev": std_amount,
                                "threshold": threshold
                            },
                            recommendations=self._get_anomaly_recommendations(AnomalyType.AMOUNT_SPIKE, category),
                            detected_at=datetime.now()
                        )
                        
                        if anomaly.confidence >= sensitivity:
                            anomalies.append(anomaly)
        
        # Detect frequency anomalies
        frequency_anomalies = await self._detect_frequency_anomalies(transactions, sensitivity)
        anomalies.extend(frequency_anomalies)
        
        # Detect new merchant patterns
        new_merchant_anomalies = await self._detect_new_merchant_anomalies(transactions, sensitivity)
        anomalies.extend(new_merchant_anomalies)
        
        return {
            "anomalies": [self._anomaly_to_dict(a) for a in anomalies],
            "summary": {
                "total_anomalies": len(anomalies),
                "high_severity": len([a for a in anomalies if a.severity in ["high", "critical"]]),
                "by_type": Counter([a.anomaly_type.value for a in anomalies]),
                "categories_affected": len(set(a.affected_data.get("category") for a in anomalies if isinstance(a.affected_data, dict)))
            }
        }
    
    async def _find_recurring_transactions(self, data: Dict[str, Any], sensitivity: float) -> Dict[str, Any]:
        """Find recurring transaction patterns."""
        transactions = data["transactions"]
        patterns = []
        
        # Group by merchant and amount (with tolerance)
        merchant_amounts = defaultdict(list)
        for txn in transactions:
            key = f"{txn['merchant']}"
            merchant_amounts[key].append({
                "amount": float(txn["amount"]),
                "date": txn["date"],
                "category": txn["category"]
            })
        
        # Analyze each merchant for recurring patterns
        for merchant, txn_list in merchant_amounts.items():
            if len(txn_list) >= 3:  # Need at least 3 occurrences
                recurring_pattern = self._analyze_recurring_pattern(merchant, txn_list)
                
                if recurring_pattern["confidence"] >= sensitivity:
                    patterns.append(Pattern(
                        pattern_id=f"recurring_{merchant.replace(' ', '_')}_{len(patterns)}",
                        pattern_type=PatternType.RECURRING_TRANSACTIONS,
                        category=recurring_pattern["category"],
                        confidence=recurring_pattern["confidence"],
                        frequency=recurring_pattern["frequency"],
                        description=recurring_pattern["description"],
                        supporting_data=recurring_pattern["data"],
                        recommendations=recurring_pattern["recommendations"],
                        detected_at=datetime.now(),
                        time_range=data["time_range"]
                    ))
        
        return {
            "patterns": [self._pattern_to_dict(p) for p in patterns],
            "summary": {
                "total_recurring_patterns": len(patterns),
                "merchants_with_patterns": len(set(p.supporting_data.get("merchant") for p in patterns)),
                "estimated_monthly_recurring": sum(p.supporting_data.get("avg_amount", 0) * 
                                                 self._frequency_to_monthly_multiplier(p.frequency) 
                                                 for p in patterns)
            }
        }
    
    async def _find_optimization_opportunities(self, data: Dict[str, Any], sensitivity: float) -> Dict[str, Any]:
        """Find cost-saving and optimization opportunities."""
        transactions = data["transactions"]
        opportunities = []
        patterns = []  # Initialize patterns list
        
        # Analyze spending by category for optimization
        by_category = defaultdict(list)
        for txn in transactions:
            by_category[txn["category"]].append(txn)
        
        for category, cat_transactions in by_category.items():
            if len(cat_transactions) >= 5:
                # Look for high-frequency, low-efficiency patterns
                merchant_analysis = self._analyze_merchant_efficiency(cat_transactions)
                
                for opp in merchant_analysis:
                    if opp["potential_savings"] > 0 and opp["confidence"] >= sensitivity:
                        # Create Pattern object instead of appending to wrong list
                        pattern = Pattern(
                            pattern_id=f"optimization_{category}_{len(patterns)}",
                            pattern_type=PatternType.OPTIMIZATION_OPPORTUNITIES,
                            category=category,
                            confidence=opp["confidence"],
                            frequency="ongoing",
                            description=opp["description"],
                            supporting_data=opp["data"],
                            recommendations=opp["recommendations"],
                            detected_at=datetime.now(),
                            time_range=data["time_range"]
                        )
                        patterns.append(pattern)
                        opportunities.append(opp)
        
        # Find bulk purchase opportunities
        bulk_opportunities = self._find_bulk_opportunities(by_category)
        for bulk_opp in bulk_opportunities:
            if bulk_opp["confidence"] >= sensitivity:
                pattern = Pattern(
                    pattern_id=f"bulk_{bulk_opp['category']}_{len(patterns)}",
                    pattern_type=PatternType.OPTIMIZATION_OPPORTUNITIES,
                    category=bulk_opp["category"],
                    confidence=bulk_opp["confidence"],
                    frequency="ongoing",
                    description=bulk_opp["description"],
                    supporting_data=bulk_opp["data"],
                    recommendations=bulk_opp["recommendations"],
                    detected_at=datetime.now(),
                    time_range=data["time_range"]
                )
                patterns.append(pattern)
        opportunities.extend(bulk_opportunities)
        
        # Find subscription optimization opportunities
        subscription_opportunities = self._find_subscription_opportunities(transactions)
        for sub_opp in subscription_opportunities:
            if sub_opp["confidence"] >= sensitivity:
                pattern = Pattern(
                    pattern_id=f"subscription_{sub_opp.get('merchant', 'unknown').replace(' ', '_')}_{len(patterns)}",
                    pattern_type=PatternType.OPTIMIZATION_OPPORTUNITIES,
                    category=sub_opp["category"],
                    confidence=sub_opp["confidence"],
                    frequency="recurring",
                    description=sub_opp["description"],
                    supporting_data=sub_opp["data"],
                    recommendations=sub_opp["recommendations"],
                    detected_at=datetime.now(),
                    time_range=data["time_range"]
                )
                patterns.append(pattern)
        opportunities.extend(subscription_opportunities)
        
        return {
            "patterns": [self._pattern_to_dict(p) for p in patterns],  # Use patterns list
            "opportunities": opportunities,  # Keep raw opportunities for backward compatibility
            "summary": {
                "total_opportunities": len(opportunities),
                "total_patterns": len(patterns),
                "potential_monthly_savings": sum(opp.get("potential_monthly_savings", 0) for opp in opportunities),
                "high_impact_opportunities": len([opp for opp in opportunities if opp.get("impact") == "high"]),
                "categories_with_opportunities": len(set(opp.get("category") for opp in opportunities if opp.get("category")))
            }
        }
    
    # Helper methods for pattern analysis
    def _analyze_weekly_patterns(self, by_day_of_week: Dict) -> Dict[str, Any]:
        """Analyze weekly spending patterns."""
        day_totals = {}
        for day, transactions in by_day_of_week.items():
            day_totals[day] = sum(float(txn["amount"]) for txn in transactions)
        
        # Check for clear weekly patterns
        if not day_totals:
            return {"confidence": 0}
        
        values = list(day_totals.values())
        if len(values) < 2:
            return {"confidence": 0}
        
        mean_spending = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        
        # Look for weekend vs weekday patterns
        weekend_days = ["Saturday", "Sunday"]
        weekday_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        
        weekend_avg = statistics.mean([day_totals.get(day, 0) for day in weekend_days])
        weekday_avg = statistics.mean([day_totals.get(day, 0) for day in weekday_days])
        
        ratio = weekend_avg / weekday_avg if weekday_avg > 0 else 0
        confidence = min(1.0, abs(ratio - 1) * 2)  # Higher confidence for bigger differences
        
        description = f"Weekly spending pattern detected: Weekend spending is {ratio:.1f}x weekday spending"
        
        return {
            "confidence": confidence,
            "description": description,
            "data": {
                "daily_averages": day_totals,
                "weekend_avg": weekend_avg,
                "weekday_avg": weekday_avg,
                "ratio": ratio
            },
            "recommendations": [
                "Consider budgeting differently for weekends vs weekdays",
                "Plan weekend activities to manage higher spending",
                "Look for weekday alternatives to expensive weekend activities"
            ]
        }
    
    def _analyze_category_habits(self, category: str, transactions: List[Dict]) -> Dict[str, Any]:
        """Analyze spending habits for a specific category."""
        amounts = [float(txn["amount"]) for txn in transactions]
        dates = [datetime.fromisoformat(txn["date"]) for txn in transactions]
        
        # Calculate frequency
        date_range = (max(dates) - min(dates)).days
        avg_frequency = len(transactions) / max(1, date_range) * 7  # Per week
        
        # Determine frequency description
        if avg_frequency >= 5:
            freq_desc = "daily"
        elif avg_frequency >= 2:
            freq_desc = "frequent"
        elif avg_frequency >= 0.5:
            freq_desc = "weekly"
        else:
            freq_desc = "occasional"
        
        # Calculate consistency
        std_dev = statistics.stdev(amounts) if len(amounts) > 1 else 0
        mean_amount = statistics.mean(amounts)
        consistency = 1 - min(1, std_dev / mean_amount) if mean_amount > 0 else 0
        
        confidence = (consistency + min(1, len(transactions) / 10)) / 2
        
        return {
            "confidence": confidence,
            "frequency": freq_desc,
            "description": f"{category.title()} spending habit: {freq_desc} purchases, avg ${mean_amount:.2f}",
            "data": {
                "avg_amount": mean_amount,
                "frequency_per_week": avg_frequency,
                "consistency_score": consistency,
                "total_transactions": len(transactions)
            },
            "recommendations": [
                f"Consider budgeting ${mean_amount * avg_frequency:.2f} per week for {category}",
                f"Look for bulk purchase opportunities in {category}" if avg_frequency > 2 else f"Monitor {category} spending patterns"
            ]
        }
    
    def _analyze_merchant_loyalty(self, by_merchant: Dict) -> List[Dict]:
        """Analyze merchant loyalty patterns."""
        patterns = []
        
        for merchant, transactions in by_merchant.items():
            if len(transactions) >= 3:  # Minimum for loyalty pattern
                category = transactions[0]["category"]
                amounts = [float(txn["amount"]) for txn in transactions]
                
                # Calculate loyalty metrics
                frequency = len(transactions)
                total_spent = sum(amounts)
                avg_amount = statistics.mean(amounts)
                
                # Simple confidence based on frequency and consistency
                confidence = min(1.0, frequency / 10)
                
                patterns.append({
                    "merchant": merchant,
                    "category": category,
                    "confidence": confidence,
                    "description": f"Loyalty to {merchant}: {frequency} visits, ${total_spent:.2f} total",
                    "data": {
                        "merchant": merchant,
                        "visit_count": frequency,
                        "total_spent": total_spent,
                        "avg_amount": avg_amount
                    },
                    "recommendations": [
                        f"Look for loyalty programs at {merchant}",
                        f"Compare prices with alternatives to {merchant}",
                        f"Consider bulk purchases at {merchant}" if avg_amount < 50 else f"Evaluate if {merchant} provides good value"
                    ]
                })
        
        return patterns
    
    def _calculate_anomaly_severity(self, amount: float, mean: float, std_dev: float) -> str:
        """Calculate severity of amount anomaly."""
        z_score = (amount - mean) / std_dev if std_dev > 0 else 0
        
        if z_score > 4:
            return "critical"
        elif z_score > 3:
            return "high"
        elif z_score > 2:
            return "medium"
        else:
            return "low"
    
    async def _detect_frequency_anomalies(self, transactions: List[Dict], sensitivity: float) -> List[Anomaly]:
        """Detect frequency-based anomalies."""
        # Group by category and merchant
        anomalies = []
        by_merchant_category = defaultdict(list)
        
        for txn in transactions:
            key = f"{txn['merchant']}_{txn['category']}"
            by_merchant_category[key].append(txn)
        
        # Look for sudden changes in frequency
        for key, txn_list in by_merchant_category.items():
            if len(txn_list) >= 6:  # Need enough data
                # Split into first and second half
                mid_point = len(txn_list) // 2
                first_half = txn_list[:mid_point]
                second_half = txn_list[mid_point:]
                
                first_freq = len(first_half) / max(1, (len(txn_list) // 2))
                second_freq = len(second_half) / max(1, (len(txn_list) // 2))
                
                if first_freq > 0:
                    freq_change = abs(second_freq - first_freq) / first_freq
                    
                    if freq_change >= self.config["anomaly_thresholds"]["frequency_change"]:
                        merchant, category = key.split("_", 1)
                        
                        anomaly = Anomaly(
                            anomaly_id=f"frequency_{key}",
                            anomaly_type=AnomalyType.FREQUENCY_CHANGE,
                            severity="medium" if freq_change < 1.0 else "high",
                            confidence=min(1.0, freq_change),
                            description=f"Frequency change at {merchant}: {freq_change:.1%} change",
                            affected_data={"merchant": merchant, "category": category},
                            baseline_comparison={
                                "previous_frequency": first_freq,
                                "recent_frequency": second_freq,
                                "change_percentage": freq_change
                            },
                            recommendations=self._get_anomaly_recommendations(AnomalyType.FREQUENCY_CHANGE, category),
                            detected_at=datetime.now()
                        )
                        
                        if anomaly.confidence >= sensitivity:
                            anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_new_merchant_anomalies(self, transactions: List[Dict], sensitivity: float) -> List[Anomaly]:
        """Detect new merchant patterns that might indicate changes."""
        merchant_first_seen = {}
        anomalies = []
        
        for txn in transactions:
            merchant = txn["merchant"]
            date = datetime.fromisoformat(txn["date"])
            
            if merchant not in merchant_first_seen:
                merchant_first_seen[merchant] = date
            else:
                merchant_first_seen[merchant] = min(merchant_first_seen[merchant], date)
        
        # Find merchants that appeared recently (last 30 days) with significant spending
        recent_threshold = datetime.now() - timedelta(days=30)
        
        for merchant, first_date in merchant_first_seen.items():
            if first_date > recent_threshold:
                merchant_txns = [txn for txn in transactions if txn["merchant"] == merchant]
                total_spent = sum(float(txn["amount"]) for txn in merchant_txns)

                if total_spent > 100:
                    confidence = min(1.0, total_spent / 500)
                    
                    anomaly = Anomaly(
                        anomaly_id=f"new_merchant_{merchant.replace(' ', '_')}",
                        anomaly_type=AnomalyType.NEW_MERCHANT,
                        severity="medium" if total_spent < 300 else "high",
                        confidence=confidence,
                        description=f"New merchant activity: ${total_spent:.2f} spent at {merchant} since {first_date.strftime('%Y-%m-%d')}",
                        affected_data={
                            "merchant": merchant,
                            "first_transaction_date": first_date.isoformat(),
                            "transaction_count": len(merchant_txns),
                            "total_amount": total_spent
                        },
                        baseline_comparison={
                            "days_since_first": (datetime.now() - first_date).days,
                            "spending_velocity": total_spent / max(1, (datetime.now() - first_date).days)
                        },
                        recommendations=self._get_anomaly_recommendations(AnomalyType.NEW_MERCHANT, merchant_txns[0]["category"]),
                        detected_at=datetime.now()
                    )
                    
                    if anomaly.confidence >= sensitivity:
                        anomalies.append(anomaly)
        
        return anomalies
    
    def _get_anomaly_recommendations(self, anomaly_type: AnomalyType, context: str) -> List[str]:
        """Get recommendations based on anomaly type."""
        recommendations = {
            AnomalyType.AMOUNT_SPIKE: [
                f"Review the unusual {context} transaction for accuracy",
                f"Consider if this represents a one-time expense or new spending pattern",
                f"Update your {context} budget if this becomes a recurring amount"
            ],
            AnomalyType.FREQUENCY_CHANGE: [
                f"Investigate why {context} spending frequency changed",
                f"Adjust budget allocations to reflect new {context} patterns",
                "Monitor if this represents a temporary or permanent change"
            ],
            AnomalyType.NEW_MERCHANT: [
                f"Evaluate if the new merchant offers better value than alternatives",
                f"Set up alerts for future transactions with this merchant",
                "Consider loyalty programs or bulk purchasing opportunities"
            ],
            AnomalyType.UNUSUAL_CATEGORY: [
                f"Review why spending occurred in an unusual category",
                "Consider if budget reallocation is needed",
                "Monitor for recurring activity in this category"
            ],
            AnomalyType.TIME_ANOMALY: [
                "Review transactions that occurred at unusual times",
                "Check for potential unauthorized activity",
                "Consider setting up time-based spending alerts"
            ],
            AnomalyType.LOCATION_ANOMALY: [
                "Verify transactions in unusual locations",
                "Consider travel-related spending adjustments",
                "Set up location-based fraud alerts"
            ]
        }
        
        return recommendations.get(anomaly_type, ["Review this anomaly and adjust spending accordingly"])
    
    def _analyze_recurring_pattern(self, merchant: str, transactions: List[Dict]) -> Dict[str, Any]:
        """Analyze a merchant's transactions for recurring patterns."""
        dates = [datetime.fromisoformat(txn["date"]) for txn in transactions]
        amounts = [txn["amount"] for txn in transactions]
        
        # Sort by date
        sorted_data = sorted(zip(dates, amounts), key=lambda x: x[0])
        dates, amounts = zip(*sorted_data)
        
        # Calculate intervals between transactions
        intervals = []
        for i in range(1, len(dates)):
            interval = (dates[i] - dates[i-1]).days
            intervals.append(interval)
        
        if not intervals:
            return {"confidence": 0}
        
        # Analyze interval consistency
        avg_interval = statistics.mean(intervals)
        interval_std = statistics.stdev(intervals) if len(intervals) > 1 else 0
        consistency = 1 - min(1, interval_std / avg_interval) if avg_interval > 0 else 0
        
        # Analyze amount consistency
        amounts_float = [float(a) for a in amounts]
        avg_amount = statistics.mean(amounts_float)
        amount_std = statistics.stdev(amounts_float) if len(amounts_float) > 1 else 0
        amount_consistency = 1 - min(1, amount_std / avg_amount) if avg_amount > 0 else 0
        
        # Determine frequency category
        if avg_interval <= 7:
            frequency = "weekly"
        elif avg_interval <= 31:
            frequency = "monthly"
        elif avg_interval <= 93:
            frequency = "quarterly"
        else:
            frequency = "irregular"
        
        # Overall confidence based on both consistencies
        confidence = (consistency + amount_consistency) / 2
        
        # Boost confidence for very regular patterns
        if consistency > 0.8 and amount_consistency > 0.8:
            confidence = min(1.0, confidence * 1.2)
        
        return {
            "confidence": confidence,
            "frequency": frequency,
            "description": f"Recurring {frequency} pattern at {merchant}: avg ${avg_amount:.2f} every {avg_interval:.0f} days",
            "category": transactions[0]["category"],
            "data": {
                "merchant": merchant,
                "avg_amount": avg_amount,
                "avg_interval_days": avg_interval,
                "interval_consistency": consistency,
                "amount_consistency": amount_consistency,
                "transaction_count": len(transactions)
            },
            "recommendations": [
                f"Budget ${avg_amount:.2f} {frequency} for {merchant}",
                f"Consider automating this {frequency} expense",
                f"Look for ways to optimize this recurring {transactions[0]['category']} expense"
            ]
        }
    
    def _frequency_to_monthly_multiplier(self, frequency: str) -> float:
        """Convert frequency string to monthly multiplier."""
        multipliers = {
            "daily": 30,
            "weekly": 4.33,  # Average weeks per month
            "monthly": 1,
            "quarterly": 0.33,
            "irregular": 0.5  # Conservative estimate
        }
        return multipliers.get(frequency, 1)
    
    def _analyze_merchant_efficiency(self, transactions: List[Dict]) -> List[Dict]:
        """Analyze merchant efficiency for optimization opportunities."""
        by_merchant = defaultdict(list)
        for txn in transactions:
            by_merchant[txn["merchant"]].append(float(txn["amount"]))
        
        opportunities = []
        
        if len(by_merchant) <= 1:
            return opportunities
        
        # Calculate average spending per merchant
        merchant_stats = {}
        for merchant, amounts in by_merchant.items():
            merchant_stats[merchant] = {
                "avg_amount": statistics.mean(amounts),
                "total_amount": sum(amounts),
                "transaction_count": len(amounts),
                "avg_per_transaction": statistics.mean(amounts)
            }
        
        # Find potential savings by switching to cheaper alternatives
        merchants_by_avg = sorted(merchant_stats.items(), key=lambda x: x[1]["avg_per_transaction"])
        cheapest_merchant = merchants_by_avg[0]
        
        for merchant, stats in merchant_stats.items():
            if merchant != cheapest_merchant[0]:
                potential_savings_per_txn = stats["avg_per_transaction"] - cheapest_merchant[1]["avg_per_transaction"]
                if potential_savings_per_txn > 5:  # Minimum $5 savings threshold
                    annual_savings = potential_savings_per_txn * stats["transaction_count"] * 12  # Extrapolate annually
                    
                    opportunities.append({
                        "type": "merchant_optimization",
                        "category": transactions[0]["category"],
                        "description": f"Potential savings by switching from {merchant} to {cheapest_merchant[0]}",
                        "potential_savings": potential_savings_per_txn,
                        "potential_monthly_savings": potential_savings_per_txn * stats["transaction_count"],
                        "confidence": 0.7,  # Medium confidence for merchant switching
                        "impact": "medium" if annual_savings < 200 else "high",
                        "data": {
                            "current_merchant": merchant,
                            "alternative_merchant": cheapest_merchant[0],
                            "savings_per_transaction": potential_savings_per_txn,
                            "estimated_annual_savings": annual_savings
                        },
                        "recommendations": [
                            f"Try {cheapest_merchant[0]} as an alternative to {merchant}",
                            f"Compare quality and convenience between {merchant} and {cheapest_merchant[0]}",
                            f"Track actual savings after switching"
                        ]
                    })
        
        return opportunities
    
    def _find_bulk_opportunities(self, by_category: Dict) -> List[Dict]:
        """Find bulk purchase opportunities."""
        opportunities = []
        
        for category, transactions in by_category.items():
            if len(transactions) >= 10 and category in ["groceries", "household", "supplies"]:
                amounts = [float(txn["amount"]) for txn in transactions]
                avg_amount = statistics.mean(amounts)
                
                # Look for frequent small purchases that could be bulked
                small_purchases = [a for a in amounts if a < avg_amount * 0.7]
                if len(small_purchases) >= len(amounts) * 0.6:  # 60% are small purchases
                    monthly_small_total = sum(small_purchases) * (30 / len(transactions))
                    potential_savings = monthly_small_total * 0.15  # Assume 15% bulk savings
                    
                    opportunities.append({
                        "type": "bulk_purchasing",
                        "category": category,
                        "description": f"Bulk purchasing opportunity in {category}",
                        "potential_monthly_savings": potential_savings,
                        "confidence": 0.6,
                        "impact": "medium" if potential_savings < 50 else "high",
                        "data": {
                            "category": category,
                            "small_purchase_ratio": len(small_purchases) / len(amounts),
                            "avg_small_purchase": statistics.mean(small_purchases),
                            "frequency": len(transactions)
                        },
                        "recommendations": [
                            f"Consider bulk purchasing for {category} items",
                            f"Look for warehouse stores or bulk suppliers",
                            f"Calculate storage costs vs savings"
                        ]
                    })
        
        return opportunities
    
    def _find_subscription_opportunities(self, transactions: List[Dict]) -> List[Dict]:
        """Find subscription optimization opportunities."""
        opportunities = []
        
        # Look for recurring payments that might be subscriptions
        recurring_merchants = defaultdict(list)
        for txn in transactions:
            if any(keyword in txn["merchant"].lower() for keyword in ["netflix", "spotify", "gym", "subscription", "monthly"]):
                recurring_merchants[txn["merchant"]].append(txn)
        
        for merchant, txns in recurring_merchants.items():
            if len(txns) >= 3:  # Likely recurring
                amounts = [float(txn["amount"]) for txn in txns]
                avg_amount = statistics.mean(amounts)
                monthly_cost = avg_amount * len(txns) / 3  # Rough monthly estimate
                
                if monthly_cost > 10:  # Worth optimizing
                    opportunities.append({
                        "type": "subscription_optimization",
                        "category": "subscriptions",
                        "description": f"Review {merchant} subscription (${monthly_cost:.2f}/month)",
                        "potential_monthly_savings": monthly_cost * 0.2,  # Assume potential 20% savings
                        "confidence": 0.5,
                        "impact": "low" if monthly_cost < 25 else "medium",
                        "data": {
                            "merchant": merchant,
                            "estimated_monthly_cost": monthly_cost,
                            "transaction_count": len(txns)
                        },
                        "recommendations": [
                            f"Review necessity of {merchant} subscription",
                            "Look for annual payment discounts",
                            "Compare with alternative services",
                            "Check for unused subscription features"
                        ]
                    })
        
        return opportunities
    
    async def _analyze_seasonal_patterns(self, data: Dict[str, Any], sensitivity: float) -> Dict[str, Any]:
        """Analyze seasonal spending patterns."""
        transactions = data["transactions"]
        patterns = []
        
        # Group by month and category
        by_month_category = defaultdict(lambda: defaultdict(float))
        for txn in transactions:
            month = txn["month"]
            category = txn["category"]
            by_month_category[category][month] += float(txn["amount"])
        
        # Analyze each category for seasonal patterns
        for category, monthly_data in by_month_category.items():
            if len(monthly_data) >= 6:  # Need at least 6 months of data
                months = list(monthly_data.keys())
                amounts = list(monthly_data.values())
                
                # Calculate seasonal variance
                mean_amount = statistics.mean(amounts)
                max_month = max(monthly_data.items(), key=lambda x: x[1])
                min_month = min(monthly_data.items(), key=lambda x: x[1])
                
                seasonal_variance = (max_month[1] - min_month[1]) / mean_amount if mean_amount > 0 else 0
                
                if seasonal_variance > 0.3:  # 30% variance threshold
                    confidence = min(1.0, seasonal_variance)
                    
                    pattern = Pattern(
                        pattern_id=f"seasonal_{category}_{len(patterns)}",
                        pattern_type=PatternType.SEASONAL_PATTERNS,
                        category=category,
                        confidence=confidence,
                        frequency="seasonal",
                        description=f"Seasonal pattern in {category}: peak in month {max_month[0]} (${max_month[1]:.2f}), low in month {min_month[0]} (${min_month[1]:.2f})",
                        supporting_data={
                            "category": category,
                            "peak_month": max_month[0],
                            "peak_amount": max_month[1],
                            "low_month": min_month[0],
                            "low_amount": min_month[1],
                            "seasonal_variance": seasonal_variance,
                            "monthly_data": dict(monthly_data)
                        },
                        recommendations=[
                            f"Budget more for {category} in month {max_month[0]}",
                            f"Take advantage of lower {category} costs in month {min_month[0]}",
                            f"Plan ahead for seasonal {category} expenses"
                        ],
                        detected_at=datetime.now(),
                        time_range=data["time_range"]
                    )
                    
                    if confidence >= sensitivity:
                        patterns.append(pattern)
        
        return {
            "patterns": [self._pattern_to_dict(p) for p in patterns],
            "summary": {
                "seasonal_categories": len(patterns),
                "strongest_seasonal_pattern": max(patterns, key=lambda p: p.confidence) if patterns else None,
                "avg_seasonal_variance": statistics.mean([p.supporting_data["seasonal_variance"] for p in patterns]) if patterns else 0
            }
        }
    
    async def _detect_life_event_impacts(self, data: Dict[str, Any], sensitivity: float) -> Dict[str, Any]:
        """Detect life event impacts on spending patterns."""
        transactions = data["transactions"]
        patterns = []
        
        # Group by month for trend analysis
        monthly_totals = defaultdict(float)
        monthly_categories = defaultdict(lambda: defaultdict(float))
        
        for txn in transactions:
            month_key = f"{datetime.fromisoformat(txn['date']).year}-{txn['month']:02d}"
            monthly_totals[month_key] += float(txn["amount"])
            monthly_categories[month_key][txn["category"]] += float(txn["amount"])
        
        # Look for significant month-to-month changes
        sorted_months = sorted(monthly_totals.keys())
        
        for i in range(1, len(sorted_months)):
            current_month = sorted_months[i]
            previous_month = sorted_months[i-1]
            
            current_total = monthly_totals[current_month]
            previous_total = monthly_totals[previous_month]
            
            if previous_total > 0:
                change_ratio = (current_total - previous_total) / previous_total
                
                if abs(change_ratio) > 0.4:  # 40% change threshold
                    # Analyze which categories drove the change
                    category_changes = {}
                    for category in set(list(monthly_categories[current_month].keys()) + 
                                      list(monthly_categories[previous_month].keys())):
                        current_cat = monthly_categories[current_month].get(category, 0)
                        previous_cat = monthly_categories[previous_month].get(category, 0)
                        
                        if previous_cat > 0:
                            cat_change = (current_cat - previous_cat) / previous_cat
                            if abs(cat_change) > 0.3:
                                category_changes[category] = cat_change
                    
                    confidence = min(1.0, abs(change_ratio) * 1.5)
                    
                    if confidence >= sensitivity:
                        pattern = Pattern(
                            pattern_id=f"life_event_{current_month}_{len(patterns)}",
                            pattern_type=PatternType.LIFE_EVENT_IMPACT,
                            category=None,
                            confidence=confidence,
                            frequency="one-time",
                            description=f"Significant spending change in {current_month}: {change_ratio:+.1%} vs previous month",
                            supporting_data={
                                "month": current_month,
                                "previous_month": previous_month,
                                "total_change_ratio": change_ratio,
                                "current_total": current_total,
                                "previous_total": previous_total,
                                "category_changes": category_changes
                            },
                            recommendations=[
                                f"Review what caused the {abs(change_ratio):.1%} spending change in {current_month}",
                                "Consider if this represents a permanent lifestyle change",
                                "Adjust future budgets if this change is expected to continue"
                            ],
                            detected_at=datetime.now(),
                            time_range=data["time_range"]
                        )
                        
                        patterns.append(pattern)
        
        return {
            "patterns": [self._pattern_to_dict(p) for p in patterns],
            "summary": {
                "life_events_detected": len(patterns),
                "months_with_changes": len(set(p.supporting_data["month"] for p in patterns)),
                "biggest_change": max(patterns, key=lambda p: abs(p.supporting_data["total_change_ratio"])) if patterns else None
            }
        }
    
    async def _analyze_income_patterns(self, data: Dict[str, Any], sensitivity: float) -> Dict[str, Any]:
        """Analyze income patterns (placeholder for income data)."""
        # This would analyze income transactions in a real implementation
        # For now, return a basic structure
        return {
            "patterns": [],
            "summary": {
                "note": "Income pattern analysis requires income transaction data"
            }
        }
    
    async def _analyze_category_trends(self, data: Dict[str, Any], sensitivity: float) -> Dict[str, Any]:
        """Analyze spending trends by category over time."""
        transactions = data["transactions"]
        patterns = []
        
        # Group by category and time period (weekly)
        by_category_week = defaultdict(lambda: defaultdict(float))
        
        for txn in transactions:
            week_key = txn["week_of_year"]
            category = txn["category"]
            by_category_week[category][week_key] += float(txn["amount"])
        
        # Analyze trends for each category
        for category, weekly_data in by_category_week.items():
            if len(weekly_data) >= 4:  # Need at least 4 weeks
                weeks = sorted(weekly_data.keys())
                amounts = [weekly_data[week] for week in weeks]
                
                # Simple trend analysis
                if len(amounts) >= 2:
                    # Calculate trend using linear regression slope
                    x = list(range(len(amounts)))
                    n = len(amounts)
                    sum_x = sum(x)
                    sum_y = sum(amounts)
                    sum_xy = sum(x[i] * amounts[i] for i in range(n))
                    sum_x2 = sum(x[i] ** 2 for i in range(n))
                    
                    if n * sum_x2 - sum_x ** 2 != 0:
                        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
                        
                        # Normalize slope by average amount
                        avg_amount = statistics.mean(amounts)
                        normalized_slope = slope / avg_amount if avg_amount > 0 else 0
                        
                        if abs(normalized_slope) > 0.1:  # 10% trend per period
                            confidence = min(1.0, abs(normalized_slope) * 5)
                            
                            trend_direction = "increasing" if slope > 0 else "decreasing"
                            
                            if confidence >= sensitivity:
                                pattern = Pattern(
                                    pattern_id=f"trend_{category}_{len(patterns)}",
                                    pattern_type=PatternType.CATEGORY_TRENDS,
                                    category=category,
                                    confidence=confidence,
                                    frequency="trending",
                                    description=f"{category.title()} spending is {trend_direction} by {abs(normalized_slope):.1%} per week",
                                    supporting_data={
                                        "category": category,
                                        "trend_direction": trend_direction,
                                        "slope": slope,
                                        "normalized_slope": normalized_slope,
                                        "weeks_analyzed": len(weeks),
                                        "weekly_data": dict(weekly_data)
                                    },
                                    recommendations=[
                                        f"Monitor {category} spending trend",
                                        f"Investigate causes of {trend_direction} {category} spending",
                                        f"Adjust {category} budget to account for trend" if abs(normalized_slope) > 0.2 else f"Continue tracking {category} trend"
                                    ],
                                    detected_at=datetime.now(),
                                    time_range=data["time_range"]
                                )
                                
                                patterns.append(pattern)
        
        return {
            "patterns": [self._pattern_to_dict(p) for p in patterns],
            "summary": {
                "trending_categories": len(patterns),
                "increasing_trends": len([p for p in patterns if p.supporting_data["trend_direction"] == "increasing"]),
                "decreasing_trends": len([p for p in patterns if p.supporting_data["trend_direction"] == "decreasing"]),
                "strongest_trend": max(patterns, key=lambda p: abs(p.supporting_data["normalized_slope"])) if patterns else None
            }
        }
    
    def _pattern_to_dict(self, pattern: Pattern) -> Dict[str, Any]:
        """Convert Pattern object to dictionary for JSON serialization."""
        return {
            "pattern_id": pattern.pattern_id,
            "pattern_type": pattern.pattern_type.value,
            "category": pattern.category,
            "confidence": pattern.confidence,
            "frequency": pattern.frequency,
            "description": pattern.description,
            "supporting_data": pattern.supporting_data,
            "recommendations": pattern.recommendations,
            "detected_at": pattern.detected_at.isoformat(),
            "time_range": {
                "start_date": pattern.time_range["start_date"].isoformat() if isinstance(pattern.time_range["start_date"], datetime) else pattern.time_range["start_date"],
                "end_date": pattern.time_range["end_date"].isoformat() if isinstance(pattern.time_range["end_date"], datetime) else pattern.time_range["end_date"]
            }
        }
    
    def _anomaly_to_dict(self, anomaly: Anomaly) -> Dict[str, Any]:
        """Convert Anomaly object to dictionary for JSON serialization."""
        return {
            "anomaly_id": anomaly.anomaly_id,
            "anomaly_type": anomaly.anomaly_type.value,
            "severity": anomaly.severity,
            "confidence": anomaly.confidence,
            "description": anomaly.description,
            "affected_data": anomaly.affected_data,
            "baseline_comparison": anomaly.baseline_comparison,
            "recommendations": anomaly.recommendations,
            "detected_at": anomaly.detected_at.isoformat()
        }
    
    def get_pattern_summary(self, patterns: List[Dict]) -> Dict[str, Any]:
        """Generate a summary of detected patterns."""
        if not patterns:
            return {"message": "No patterns detected"}
        
        return {
            "total_patterns": len(patterns),
            "by_type": Counter([p["pattern_type"] for p in patterns]),
            "high_confidence": len([p for p in patterns if p["confidence"] >= 0.8]),
            "categories_involved": len(set(p["category"] for p in patterns if p["category"])),
            "top_recommendations": self._get_top_recommendations(patterns)
        }
    
    def _get_top_recommendations(self, patterns: List[Dict]) -> List[str]:
        """Extract top recommendations from patterns."""
        all_recommendations = []
        for pattern in patterns:
            all_recommendations.extend(pattern.get("recommendations", []))
        
        # Count recommendation frequency and return top ones
        rec_counter = Counter(all_recommendations)
        return [rec for rec, count in rec_counter.most_common(5)]