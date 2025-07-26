import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import uuid
from core.proactive_agent_tools.anomaly_detection_tool import AnomalyDetectionTool, AnomalyType as AnomalyToolType, AnomalySeverity as AnomalyToolSeverity, AnomalyPolarity
from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration
from core.base_agent_tools.database_connector import DatabaseConnector
from core.base_agent_tools.config_manager import AgentConfig
from core.base_agent_tools.vertex_initializer import VertexAIInitializer
from core.base_agent_tools.integration_coordinator import IntegrationCoordinator
from core.base_agent_tools.error_handler import ErrorHandler
from core.base_agent_tools.user_profile_manager import UserProfileManager
from core.proactive_agent_tools.continuous_monitoring_tool import ContinuousMonitoringTool, AnalysisType, MonitoringFrequency, RiskLevel
from core.proactive_agent_tools.trend_prediction_tool import TrendPredictionTool, TrendType, TrendSeverity as TrendToolSeverity, PredictionConfidence
from core.proactive_agent_tools.timing_optimization_tool import TimingOptimizationTool, NotificationUrgency
from core.proactive_agent_tools.notification_generation_tool import (
    NotificationGenerationTool, 
    NotificationType as NotifType
)

class InsightType(Enum):
    """Types of proactive insights that can be generated."""
    BUDGET_ALERT = "budget_alert"
    SPENDING_ANOMALY = "spending_anomaly"
    TREND_WARNING = "trend_warning"
    GOAL_UPDATE = "goal_update"
    OPPORTUNITY = "opportunity"
    SEASONAL_REMINDER = "seasonal_reminder"
    CELEBRATION = "celebration"


class InsightSeverity(Enum):
    """Severity levels for insights."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ProactiveInsight:
    """Structure for a proactive insight."""
    insight_id: str
    user_id: str
    insight_type: InsightType
    severity: InsightSeverity
    title: str
    message: str
    data_points: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    confidence_score: float
    relevance_score: float
    optimal_timing: datetime
    expires_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MonitoringResult:
    """Result from continuous monitoring analysis."""
    user_id: str
    analysis_type: str
    insights: List[ProactiveInsight]
    execution_time: float
    data_coverage: float
    next_analysis_time: datetime


class ProactiveInsightsAgent:
    """
    Agent responsible for continuous monitoring and proactive insight generation.
    
    This agent runs background analysis to identify spending patterns, anomalies,
    and opportunities, then generates timely insights and alerts for users.
    """
    
    def __init__(
        self,
        agent_name: str = "proactive_insights_agent",
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model_name: Optional[str] = None,
        db_connection_string: Optional[str] = None
    ):
        # Initialize configuration
        self.agent_name = agent_name
        self.system_config = AgentConfig.from_env()
        self.project_id = project_id or self.system_config.project_id
        self.location = location or self.system_config.location
        self.model_name = model_name or self.system_config.model_name
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Initialize shared tools
        self.db_connector = DatabaseConnector()
        self.integration_coordinator = IntegrationCoordinator()
        self.error_handler = ErrorHandler(self.logger)
        self.user_profile_manager = UserProfileManager()
        
        # Initialize Tools
        self.db_connection_string = db_connection_string or self.system_config.database_url
        self.monitoring_tool = ContinuousMonitoringTool(self.db_connection_string)
        self.anomaly_detection_tool = AnomalyDetectionTool(self.db_connector)
        self.trend_prediction_tool = TrendPredictionTool(
            db_connector=self.db_connector,
            config=self.system_config
        )
        self.timing_optimization_tool = TimingOptimizationTool(
            project_id=self.project_id,
            location=self.location,
            model_name=self.model_name
        )
        self.notification_generation_tool = NotificationGenerationTool(
            db_connection_string=self.db_connection_string,
            project_id=self.project_id,
            location=self.location,
            model_name=self.model_name
        )
        
        # Initialize Vertex AI
        VertexAIInitializer.initialize(self.project_id, self.location)
        self.model = GenerativeModel(model_name=self.model_name)
        
        # Tool registry - will be populated with individual tools
        self.tools = {}
        
        # Agent state
        self.monitoring_active = False
        self.last_analysis_run = {}  # Track per-user analysis times
        
        self.logger.info(f"Proactive Insights Agent initialized: {agent_name}")

    async def initialize(self):
        """Initialize the agent and its tools."""
        try:
            await self.monitoring_tool.initialize()
            self.logger.info("Continuous monitoring tool initialized successfully")
            await self.notification_generation_tool.initialize()
            self.logger.info("Notification generation tool initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize continuous monitoring tool: {str(e)}")
            raise

    async def close(self):
        """Close agent and cleanup resources."""
        try:
            await self.monitoring_tool.close()
            await self.notification_generation_tool.close()
            self.logger.info("Agent resources cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up structured logging for the agent."""
        logger = logging.getLogger(f"financial_agent.{self.agent_name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def register_tool(self, tool_name: str, tool_instance: Any):
        """Register a tool with the agent."""
        self.tools[tool_name] = tool_instance
        self.logger.info(f"Registered tool: {tool_name}")
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method called by the orchestrator.
        
        Args:
            input_data: Dictionary containing:
                - historical_analysis: Results from financial analysis agent
                - prediction_context: Context for prediction generation
                - user_id: Target user ID
                - analysis_type: Type of proactive analysis to perform
                
        Returns:
            Dictionary containing generated predictions and insights
        """
        try:
            self.logger.info(f"Processing proactive insights request for user: {input_data.get('user_id')}")
            
            # Extract input parameters
            user_id = input_data.get('user_id')
            historical_analysis = input_data.get('historical_analysis', {})
            prediction_context = input_data.get('prediction_context', {})
            analysis_type = input_data.get('analysis_type', 'general')
            
            if not user_id:
                raise ValueError("user_id is required for proactive insights generation")
            
            # Get user profile and preferences
            user_profile = await self.user_profile_manager.get_profile(user_id)
            
            # Perform different types of proactive analysis
            insights = []
            
            if analysis_type == 'budget_monitoring':
                insights.extend(await self._monitor_budget_thresholds(user_id, historical_analysis))
            elif analysis_type == 'anomaly_detection':
                insights.extend(await self._detect_spending_anomalies(user_id, historical_analysis))
            elif analysis_type == 'trend_prediction':
                insights.extend(await self._predict_spending_trends(user_id, historical_analysis, prediction_context))
            elif analysis_type == 'seasonal_analysis':
                insights.extend(await self._analyze_seasonal_patterns(user_id, historical_analysis))
            elif analysis_type == 'goal_tracking':
                insights.extend(await self._track_financial_goals(user_id, historical_analysis))
            else:
                # General comprehensive analysis
                insights.extend(await self._comprehensive_analysis(user_id, historical_analysis, prediction_context))
            
            # Filter and prioritize insights based on user preferences
            filtered_insights = await self._filter_and_prioritize_insights(insights, user_profile)
            
            # Optimize timing for insights
            optimized_insights = await self._optimize_insight_timing(filtered_insights, user_profile)
            
            notifications = await self._generate_notifications_from_insights(optimized_insights)
        
            # Generate predictions based on insights
            predictions = await self._generate_predictions_from_insights(optimized_insights, prediction_context)
            
            # Store insights for future reference
            await self._store_insights(optimized_insights)
            
            return {
                "predictions": predictions,
                "insights": [self._insight_to_dict(insight) for insight in optimized_insights],
                "notifications": notifications,  # Add this line
                "analysis_metadata": {
                    "analysis_type": analysis_type,
                    "insights_generated": len(optimized_insights),
                    "notifications_generated": len(notifications),  # Add this line
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "agent": self.agent_name
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in proactive insights processing: {str(e)}")
            return await self.error_handler.handle_error(e, {
                "agent": self.agent_name,
                "input_data": input_data,
                "operation": "process"
            })
    
    async def _generate_notifications_from_insights(self, insights: List[ProactiveInsight]) -> List[Dict[str, Any]]:
        """Generate notifications from proactive insights using the notification generation tool."""
        notifications = []
        
        try:
            # Map insight types to notification types
            insight_to_notification_mapping = {
                InsightType.BUDGET_ALERT: NotifType.BUDGET_ALERT,
                InsightType.SPENDING_ANOMALY: NotifType.SPENDING_INSIGHT,
                InsightType.TREND_WARNING: NotifType.SPENDING_INSIGHT,
                InsightType.GOAL_UPDATE: NotifType.GOAL_UPDATE,
                InsightType.OPPORTUNITY: NotifType.RECOMMENDATION,
                InsightType.SEASONAL_REMINDER: NotifType.SEASONAL_REMINDER,
                InsightType.CELEBRATION: NotifType.CELEBRATION
            }
            
            # Prepare notification generation requests
            notification_requests = []
            for insight in insights:
                notification_type = insight_to_notification_mapping.get(
                    insight.insight_type, 
                    NotifType.SPENDING_INSIGHT
                )
                
                # Determine template based on insight specifics
                template_id = self._select_template_id(insight)
                
                notification_request = {
                    'user_id': insight.user_id,
                    'notification_type': notification_type.value,
                    'template_id': template_id,
                    'insight_data': {
                        'insight_id': insight.insight_id,
                        'insight_type': insight.insight_type.value,
                        'severity': insight.severity.value,
                        'title': insight.title,
                        'message': insight.message,
                        'data_points': insight.data_points,
                        'recommendations': insight.recommendations,
                        'confidence_score': insight.confidence_score,
                        'relevance_score': insight.relevance_score
                    },
                    'custom_data': {
                        'optimal_timing': insight.optimal_timing.isoformat(),
                        'expires_at': insight.expires_at.isoformat() if insight.expires_at else None,
                        'metadata': insight.metadata or {}
                    }
                }
                notification_requests.append(notification_request)
            
            # Generate notifications in batch for efficiency
            if notification_requests:
                generated_notifications = await self.notification_generation_tool.generate_batch_notifications(
                    notification_requests
                )
                
                # Convert to dict format for return
                for notif in generated_notifications:
                    notifications.append({
                        'notification_id': notif.notification_id,
                        'user_id': notif.user_id,
                        'notification_type': notif.notification_type.value,
                        'title': notif.title,
                        'message': notif.message,
                        'rich_content': notif.rich_content,
                        'priority': notif.priority.value,
                        'channels': [ch.value for ch in notif.channels],
                        'action_buttons': notif.action_buttons,
                        'data_visualization': notif.data_visualization,
                        'quick_actions': notif.quick_actions,
                        'expires_at': notif.expires_at.isoformat() if notif.expires_at else None,
                        'metadata': notif.metadata
                    })
                
                self.logger.info(f"Generated {len(notifications)} notifications from {len(insights)} insights")
            
        except Exception as e:
            self.logger.error(f"Error generating notifications from insights: {str(e)}")
            # Continue without notifications - insights are still valuable
        
        return notifications
    
    def _select_template_id(self, insight: ProactiveInsight) -> str:
        """Select appropriate notification template based on insight characteristics."""
        
        # Budget alerts
        if insight.insight_type == InsightType.BUDGET_ALERT:
            usage_pct = insight.data_points.get('usage_percentage', 0)
            if usage_pct > 100:
                return "budget_exceeded"
            else:
                return "budget_warning"
        
        # Spending insights
        elif insight.insight_type == InsightType.SPENDING_ANOMALY:
            polarity = insight.data_points.get('polarity', 'neutral')
            if polarity == 'positive':
                return "spending_improvement"
            else:
                return "spending_spike"
        
        # Trend warnings
        elif insight.insight_type == InsightType.TREND_WARNING:
            return "spending_spike"  # Use spending spike template for trends
        
        # Goal updates
        elif insight.insight_type == InsightType.GOAL_UPDATE:
            return "goal_progress"
        
        # Celebrations
        elif insight.insight_type == InsightType.CELEBRATION:
            return "spending_improvement"
        
        # Opportunities and recommendations
        elif insight.insight_type == InsightType.OPPORTUNITY:
            return "merchant_alternative"
        
        # Seasonal reminders
        elif insight.insight_type == InsightType.SEASONAL_REMINDER:
            return "seasonal_preparation"
        
        # Default fallback
        return "spending_spike"
    
    async def _monitor_budget_thresholds(self, user_id: str, historical_analysis: Dict[str, Any]) -> List[ProactiveInsight]:
        """Monitor budget thresholds and generate alerts using the monitoring tool."""
        insights = []
        
        try:
            # Use the continuous monitoring tool to check budget status
            budget_statuses = await self.monitoring_tool.check_budget_status(user_id)
            
            for budget_status in budget_statuses:
                # Determine severity based on risk level from monitoring tool
                if budget_status.risk_level == RiskLevel.CRITICAL:
                    severity = InsightSeverity.CRITICAL
                    message = f"Critical: You've used {budget_status.utilization_percentage:.1f}% of your {budget_status.category} budget. Projected overage: ${budget_status.projected_overage:.2f}"
                elif budget_status.risk_level == RiskLevel.HIGH:
                    severity = InsightSeverity.HIGH
                    message = f"Warning: You're at {budget_status.utilization_percentage:.1f}% of your {budget_status.category} budget with {budget_status.days_remaining} days remaining."
                elif budget_status.risk_level == RiskLevel.MEDIUM:
                    severity = InsightSeverity.MEDIUM
                    message = f"Notice: You've reached {budget_status.utilization_percentage:.1f}% of your {budget_status.category} budget."
                else:
                    continue  # Skip low risk items
                
                # Get prediction data for recommendations
                prediction = await self.monitoring_tool.predict_budget_overage(user_id, budget_status.category)
                
                recommendations = []
                if prediction and prediction.projected_overage > 0:
                    recommendations.append({
                        "type": "spending_limit",
                        "description": f"Consider limiting daily spending to ${prediction.recommended_daily_limit:.2f} to stay within budget",
                        "confidence": prediction.confidence
                    })
                
                insight = ProactiveInsight(
                    insight_id=str(uuid.uuid4()),
                    user_id=user_id,
                    insight_type=InsightType.BUDGET_ALERT,
                    severity=severity,
                    title=f"{budget_status.category.title()} Budget Alert",
                    message=message,
                    data_points={
                        "category": budget_status.category,
                        "spent_amount": float(budget_status.current_spent),
                        "budget_limit": float(budget_status.budget_limit),
                        "usage_percentage": budget_status.utilization_percentage,
                        "days_remaining": budget_status.days_remaining,
                        "projected_overage": float(budget_status.projected_overage),
                        "spending_velocity": budget_status.spending_velocity,
                        "risk_level": budget_status.risk_level.value
                    },
                    recommendations=recommendations,
                    confidence_score=0.95,
                    relevance_score=0.9,
                    optimal_timing=datetime.now()
                )
                insights.append(insight)
            
        except Exception as e:
            self.logger.error(f"Error monitoring budget thresholds: {str(e)}")
        
        return insights
    
    async def _detect_spending_anomalies(self, user_id: str, historical_analysis: Dict[str, Any]) -> List[ProactiveInsight]:
        """Detect unusual spending patterns using the advanced anomaly detection tool."""
        insights = []
        
        try:
            # Use the anomaly detection tool to get detailed anomalies
            detected_anomalies = await self.anomaly_detection_tool.get_anomalies_for_insights(
                user_id=user_id,
                analysis_window_days=7  # Look at last 7 days for recent anomalies
            )
            
            for anomaly_data in detected_anomalies:
                # Map anomaly tool severity to insight severity
                severity_mapping = {
                    'critical': InsightSeverity.CRITICAL,
                    'significant': InsightSeverity.HIGH,
                    'moderate': InsightSeverity.MEDIUM,
                    'minor': InsightSeverity.LOW
                }
                
                # Map anomaly type to insight type
                if anomaly_data['anomaly_type'] in ['spending_spike', 'spending_drop', 'amount_pattern_anomaly']:
                    insight_type = InsightType.SPENDING_ANOMALY
                elif anomaly_data['anomaly_type'] == 'frequency_anomaly':
                    insight_type = InsightType.TREND_WARNING
                elif anomaly_data['anomaly_type'] == 'category_shift':
                    insight_type = InsightType.TREND_WARNING
                else:
                    insight_type = InsightType.SPENDING_ANOMALY
                
                # Determine if this is positive (celebration) or negative (warning)
                if anomaly_data.get('polarity') == 'positive' and anomaly_data.get('severity') in ['minor', 'moderate']:
                    insight_type = InsightType.CELEBRATION
                
                severity = severity_mapping.get(anomaly_data.get('severity', 'minor'), InsightSeverity.LOW)
                
                # Create insight from anomaly data
                insight = ProactiveInsight(
                    insight_id=str(uuid.uuid4()),
                    user_id=user_id,
                    insight_type=insight_type,
                    severity=severity,
                    title=self._generate_anomaly_insight_title(anomaly_data),
                    message=anomaly_data.get('description', 'Unusual spending pattern detected'),
                    data_points={
                        "anomaly_type": anomaly_data.get('anomaly_type'),
                        "category": anomaly_data.get('category'),
                        "merchant": anomaly_data.get('merchant'),
                        "observed_value": anomaly_data.get('observed_value', 0),
                        "expected_value": anomaly_data.get('expected_value', 0),
                        "z_score": anomaly_data.get('z_score', 0),
                        "impact_score": anomaly_data.get('impact_score', 0),
                        "polarity": anomaly_data.get('polarity'),
                        "contributing_factors": anomaly_data.get('contributing_factors', []),
                        "detection_timestamp": anomaly_data.get('detection_timestamp')
                    },
                    recommendations=self._convert_anomaly_recommendations(anomaly_data.get('recommendations', [])),
                    confidence_score=anomaly_data.get('confidence_score', 0.8),
                    relevance_score=min(anomaly_data.get('impact_score', 0.5) + 0.3, 1.0),
                    optimal_timing=datetime.now()
                )
                insights.append(insight)
            
            self.logger.info(f"Generated {len(insights)} anomaly-based insights for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Error detecting spending anomalies with anomaly detection tool: {str(e)}")
            # Fallback to basic anomaly detection if the tool fails
            insights.extend(await self._basic_anomaly_detection_fallback(user_id, historical_analysis))
        
        return insights

    def _generate_anomaly_insight_title(self, anomaly_data: Dict[str, Any]) -> str:
        """Generate user-friendly titles for anomaly insights."""
        anomaly_type = anomaly_data.get('anomaly_type', '')
        category = anomaly_data.get('category', '')
        merchant = anomaly_data.get('merchant', '')
        polarity = anomaly_data.get('polarity', 'neutral')
        
        if anomaly_type == 'spending_spike':
            if category:
                return f"High {category.title()} Spending Detected"
            elif merchant:
                return f"Unusual Spending at {merchant}"
            else:
                return "Spending Spike Alert"
        
        elif anomaly_type == 'spending_drop':
            if polarity == 'positive':
                return f"Great Job! Reduced {category.title() if category else 'Overall'} Spending"
            else:
                return f"Decreased {category.title() if category else 'Overall'} Spending Noticed"
        
        elif anomaly_type == 'frequency_anomaly':
            return "Transaction Frequency Change"
        
        elif anomaly_type == 'category_shift':
            return f"Spending Pattern Change in {category.title() if category else 'Categories'}"
        
        elif anomaly_type == 'merchant_anomaly':
            if merchant:
                return f"New Spending Pattern at {merchant}"
            else:
                return "Merchant Spending Change"
        
        elif anomaly_type == 'time_pattern_anomaly':
            return "Timing Pattern Change"
        
        else:
            return "Unusual Spending Pattern"

    def _convert_anomaly_recommendations(self, anomaly_recommendations: List[str]) -> List[Dict[str, Any]]:
        """Convert anomaly tool recommendations to insight format."""
        recommendations = []
        
        for rec in anomaly_recommendations:
            recommendations.append({
                "type": "anomaly_response",
                "description": rec,
                "confidence": 0.8,
                "priority": "medium"
            })
        
        return recommendations

    async def _basic_anomaly_detection_fallback(self, user_id: str, historical_analysis: Dict[str, Any]) -> List[ProactiveInsight]:
        """Fallback basic anomaly detection if the advanced tool fails."""
        insights = []
        
        try:
            # Basic implementation using historical analysis data
            spending_patterns = historical_analysis.get('spending_patterns', {})
            recent_spending = historical_analysis.get('recent_spending', {})
            
            for category, recent_amount in recent_spending.items():
                pattern = spending_patterns.get(category, {})
                avg_amount = pattern.get('average', 0)
                std_dev = pattern.get('std_deviation', 0)
                
                if avg_amount > 0 and std_dev > 0:
                    # Calculate z-score for anomaly detection
                    z_score = abs((recent_amount - avg_amount) / std_dev)
                    
                    if z_score > 2.0:  # Significant anomaly
                        if recent_amount > avg_amount:
                            insight_type = InsightType.SPENDING_ANOMALY
                            severity = InsightSeverity.HIGH if z_score > 3.0 else InsightSeverity.MEDIUM
                            message = f"Unusually high spending in {category}: ${recent_amount:.2f} vs typical ${avg_amount:.2f}"
                        else:
                            insight_type = InsightType.CELEBRATION
                            severity = InsightSeverity.LOW
                            message = f"Great job! You've spent less than usual on {category} this period."
                        
                        insight = ProactiveInsight(
                            insight_id=str(uuid.uuid4()),
                            user_id=user_id,
                            insight_type=insight_type,
                            severity=severity,
                            title=f"{category.title()} Spending Anomaly",
                            message=message,
                            data_points={
                                "category": category,
                                "recent_amount": recent_amount,
                                "average_amount": avg_amount,
                                "z_score": z_score,
                                "std_deviation": std_dev,
                                "source": "fallback_detection"
                            },
                            recommendations=[{
                                "type": "basic_anomaly_response",
                                "description": "Monitor this category for continued unusual patterns",
                                "confidence": 0.6
                            }],
                            confidence_score=0.6,  # Lower confidence for fallback
                            relevance_score=0.7,
                            optimal_timing=datetime.now()
                        )
                        insights.append(insight)
            
            self.logger.info(f"Used fallback anomaly detection, generated {len(insights)} insights")
            
        except Exception as e:
            self.logger.error(f"Error in fallback anomaly detection: {str(e)}")
        
        return insights
    
    async def _predict_spending_trends(self, user_id: str, historical_analysis: Dict[str, Any], prediction_context: Dict[str, Any]) -> List[ProactiveInsight]:
        """Predict future spending trends using the TrendPredictionTool."""
        insights = []
        
        try:
            # Extract parameters from context
            categories = prediction_context.get('categories', None)
            prediction_types = prediction_context.get('prediction_types', None)
            horizon_days = prediction_context.get('horizon_days', 90)
            
            # Get trend predictions from the tool
            trend_predictions = await self.trend_prediction_tool.predict_trends(
                user_id=user_id,
                categories=categories,
                prediction_types=prediction_types,
                horizon_days=horizon_days
            )
            
            # Convert trend predictions to proactive insights
            for prediction in trend_predictions:
                # Map severity levels
                severity_mapping = {
                    TrendToolSeverity.CRITICAL: InsightSeverity.CRITICAL,
                    TrendToolSeverity.HIGH: InsightSeverity.HIGH,
                    TrendToolSeverity.MEDIUM: InsightSeverity.MEDIUM,
                    TrendToolSeverity.LOW: InsightSeverity.LOW
                }
                
                # Map trend types to insight types
                insight_type_mapping = {
                    TrendType.SPENDING_INCREASE: InsightType.TREND_WARNING,
                    TrendType.SPENDING_DECREASE: InsightType.OPPORTUNITY,
                    TrendType.SEASONAL_PATTERN: InsightType.SEASONAL_REMINDER,
                    TrendType.BUDGET_CHALLENGE: InsightType.BUDGET_ALERT,
                    TrendType.EMERGING_CATEGORY: InsightType.OPPORTUNITY,
                    TrendType.BEHAVIOR_SHIFT: InsightType.TREND_WARNING,
                    TrendType.MERCHANT_LOYALTY: InsightType.OPPORTUNITY,
                    TrendType.PAYMENT_METHOD_SHIFT: InsightType.OPPORTUNITY
                }
                
                # Generate recommendations from prediction actions
                recommendations = []
                
                # Add preparation actions
                for action in prediction.preparation_actions:
                    recommendations.append({
                        "type": "preparation",
                        "action": action.get("action", ""),
                        "description": action.get("description", ""),
                        "priority": action.get("priority", "medium"),
                        "estimated_impact": action.get("estimated_impact", "")
                    })
                
                # Add mitigation strategies
                for strategy in prediction.mitigation_strategies:
                    recommendations.append({
                        "type": "mitigation",
                        "strategy": strategy.get("strategy", ""),
                        "description": strategy.get("description", ""),
                        "effort": strategy.get("effort", "medium"),
                        "potential_savings": strategy.get("potential_savings", "")
                    })
                
                # Add opportunities
                for opportunity in prediction.opportunities:
                    recommendations.append({
                        "type": "opportunity",
                        "opportunity": opportunity.get("opportunity", ""),
                        "description": opportunity.get("description", ""),
                        "potential_benefit": opportunity.get("potential_benefit", ""),
                        "action_required": opportunity.get("action_required", "")
                    })
                
                insight = ProactiveInsight(
                    insight_id=str(uuid.uuid4()),
                    user_id=user_id,
                    insight_type=insight_type_mapping.get(prediction.trend_type, InsightType.TREND_WARNING),
                    severity=severity_mapping.get(prediction.severity, InsightSeverity.MEDIUM),
                    title=prediction.title,
                    message=prediction.description,
                    data_points={
                        "category": prediction.category,
                        "trend_type": prediction.trend_type.value,
                        "predicted_impact": prediction.predicted_impact,
                        "timeline": prediction.timeline,
                        "confidence_score": prediction.confidence_score,
                        "statistical_evidence": prediction.statistical_evidence,
                        "model_metrics": prediction.model_metrics
                    },
                    recommendations=recommendations,
                    confidence_score=prediction.confidence_score,
                    relevance_score=self._calculate_relevance_score(prediction),
                    optimal_timing=datetime.now(),
                    expires_at=prediction.valid_until,
                    metadata={
                        "prediction_id": prediction.prediction_id,
                        "model_used": prediction.model_used,
                        "data_quality_score": prediction.data_quality_score,
                        "historical_data": prediction.historical_data
                    }
                )
                insights.append(insight)
            
        except Exception as e:
            self.logger.error(f"Error predicting spending trends with TrendPredictionTool: {str(e)}")
        
        return insights
    
    def _calculate_relevance_score(self, prediction) -> float:
        """Calculate relevance score based on prediction characteristics."""
        base_score = 0.7
        
        # Higher relevance for critical/high severity
        if prediction.severity in [TrendToolSeverity.CRITICAL, TrendToolSeverity.HIGH]:
            base_score += 0.2
        
        # Higher relevance for high confidence predictions
        if prediction.confidence_score > 0.8:
            base_score += 0.1
        
        # Higher relevance for budget-related predictions
        if prediction.trend_type == TrendType.BUDGET_CHALLENGE:
            base_score += 0.15
        
        # Higher relevance based on predicted impact
        impact = prediction.predicted_impact
        if isinstance(impact, dict):
            if impact.get('increase_percentage', 0) > 30:
                base_score += 0.1
            if impact.get('overage_amount', 0) > 100:
                base_score += 0.1
        
        return min(base_score, 1.0)
    
    async def _analyze_seasonal_patterns(self, user_id: str, historical_analysis: Dict[str, Any]) -> List[ProactiveInsight]:
        """Analyze seasonal spending patterns and generate reminders."""
        insights = []
        
        try:
            # This would be enhanced with seasonal pattern analysis
            current_month = datetime.now().month
            
            # Example seasonal insights based on month
            seasonal_insights = {
                11: "Holiday shopping season is approaching. Consider setting aside extra budget for gifts.",
                12: "End of year - great time to review your annual spending and plan for next year.",
                1: "New year, new budget! Consider adjusting your spending categories based on last year's patterns.",
                4: "Tax season - don't forget to track deductible expenses.",
                9: "Back to school season may increase certain spending categories."
            }
            
            if current_month in seasonal_insights:
                insight = ProactiveInsight(
                    insight_id=str(uuid.uuid4()),
                    user_id=user_id,
                    insight_type=InsightType.SEASONAL_REMINDER,
                    severity=InsightSeverity.LOW,
                    title="Seasonal Spending Reminder",
                    message=seasonal_insights[current_month],
                    data_points={"month": current_month, "seasonal_context": True},
                    recommendations=[],
                    confidence_score=0.7,
                    relevance_score=0.6,
                    optimal_timing=datetime.now()
                )
                insights.append(insight)
                
        except Exception as e:
            self.logger.error(f"Error analyzing seasonal patterns: {str(e)}")
        
        return insights
    
    async def _track_financial_goals(self, user_id: str, historical_analysis: Dict[str, Any]) -> List[ProactiveInsight]:
        """Track progress on financial goals."""
        insights = []
        
        try:
            goal_progress = historical_analysis.get('goal_progress', {})
            
            for goal_id, progress_data in goal_progress.items():
                progress_percentage = progress_data.get('progress_percentage', 0)
                goal_type = progress_data.get('goal_type', 'saving')
                target_amount = progress_data.get('target_amount', 0)
                current_amount = progress_data.get('current_amount', 0)
                
                if progress_percentage >= 90:
                    insight_type = InsightType.CELEBRATION
                    severity = InsightSeverity.LOW
                    message = f"Amazing! You're at {progress_percentage:.1f}% of your {goal_type} goal!"
                elif progress_percentage >= 75:
                    insight_type = InsightType.GOAL_UPDATE
                    severity = InsightSeverity.LOW
                    message = f"Great progress! You're {progress_percentage:.1f}% towards your {goal_type} goal."
                elif progress_percentage < 25:
                    insight_type = InsightType.GOAL_UPDATE
                    severity = InsightSeverity.MEDIUM
                    message = f"Consider increasing efforts towards your {goal_type} goal. Currently at {progress_percentage:.1f}%."
                else:
                    continue
                
                insight = ProactiveInsight(
                    insight_id=str(uuid.uuid4()),
                    user_id=user_id,
                    insight_type=insight_type,
                    severity=severity,
                    title=f"Goal Progress Update",
                    message=message,
                    data_points={
                        "goal_id": goal_id,
                        "goal_type": goal_type,
                        "progress_percentage": progress_percentage,
                        "target_amount": target_amount,
                        "current_amount": current_amount
                    },
                    recommendations=[],
                    confidence_score=0.9,
                    relevance_score=0.85,
                    optimal_timing=datetime.now()
                )
                insights.append(insight)
                
        except Exception as e:
            self.logger.error(f"Error tracking financial goals: {str(e)}")
        
        return insights
    
    async def _comprehensive_analysis(self, user_id: str, historical_analysis: Dict[str, Any], prediction_context: Dict[str, Any]) -> List[ProactiveInsight]:
        """Perform comprehensive analysis combining all insight types."""
        insights = []
        
        # Run all analysis types
        insights.extend(await self._monitor_budget_thresholds(user_id, historical_analysis))
        insights.extend(await self._detect_spending_anomalies(user_id, historical_analysis))
        insights.extend(await self._predict_spending_trends(user_id, historical_analysis, prediction_context))
        insights.extend(await self._analyze_seasonal_patterns(user_id, historical_analysis))
        insights.extend(await self._track_financial_goals(user_id, historical_analysis))
        
        return insights
    
    async def _filter_and_prioritize_insights(self, insights: List[ProactiveInsight], user_profile: Dict[str, Any]) -> List[ProactiveInsight]:
        """Filter and prioritize insights based on user preferences."""
        try:
            # Filter based on user notification preferences
            user_prefs = user_profile.get('preferences', {}).get('notifications', {})
            
            filtered_insights = []
            for insight in insights:
                # Check if user wants this type of insight
                if insight.insight_type == InsightType.BUDGET_ALERT and not user_prefs.get('budgetAlerts', True):
                    continue
                if insight.insight_type in [InsightType.SPENDING_ANOMALY, InsightType.TREND_WARNING] and not user_prefs.get('spendingInsights', True):
                    continue
                if insight.insight_type == InsightType.GOAL_UPDATE and not user_prefs.get('proactiveInsights', True):
                    continue
                
                filtered_insights.append(insight)
            
            # Sort by priority (severity + relevance)
            filtered_insights.sort(
                key=lambda x: (x.severity.value, x.relevance_score, x.confidence_score),
                reverse=True
            )
            
            # Limit to top insights to avoid overwhelming the user
            max_insights = user_prefs.get('maxInsightsPerSession', 5)
            return filtered_insights[:max_insights]
            
        except Exception as e:
            self.logger.error(f"Error filtering insights: {str(e)}")
            return insights[:3]  # Return top 3 as fallback
    
    async def _optimize_insight_timing(self, insights: List[ProactiveInsight], user_profile: Dict[str, Any]) -> List[ProactiveInsight]:
        """Optimize timing for insight delivery using the TimingOptimizationTool."""
        try:
            if not insights:
                return insights
            
            self.logger.info(f"Optimizing timing for {len(insights)} insights")
            
            # Prepare notifications data for batch optimization
            notifications_data = []
            for insight in insights:
                # Map insight severity to notification urgency
                urgency_mapping = {
                    InsightSeverity.CRITICAL: NotificationUrgency.IMMEDIATE,
                    InsightSeverity.HIGH: NotificationUrgency.HIGH,
                    InsightSeverity.MEDIUM: NotificationUrgency.MEDIUM,
                    InsightSeverity.LOW: NotificationUrgency.LOW
                }
                
                urgency = urgency_mapping.get(insight.severity, NotificationUrgency.MEDIUM)
                
                notification_data = {
                    "notification_id": insight.insight_id,
                    "insight_type": insight.insight_type.value,
                    "severity": insight.severity.value,
                    "title": insight.title,
                    "message": insight.message,
                    "urgency": urgency.value,
                    "priority_score": insight.relevance_score,
                    "confidence_score": insight.confidence_score,
                    "original_timing": insight.optimal_timing.isoformat(),
                    "data_points": insight.data_points
                }
                notifications_data.append(notification_data)
            
            # Use batch optimization for better distribution
            user_id = insights[0].user_id
            optimized_schedules = await self.timing_optimization_tool.batch_optimize_notifications(
                user_id=user_id,
                notifications=notifications_data
            )
            
            # Apply optimized timing back to insights
            schedule_map = {schedule.notification_id: schedule for schedule in optimized_schedules}
            
            for insight in insights:
                schedule = schedule_map.get(insight.insight_id)
                if schedule:
                    # Update insight timing with optimized schedule
                    insight.optimal_timing = schedule.optimized_time
                    
                    # Set expiration based on urgency and delay
                    delay_hours = (schedule.optimized_time - schedule.original_time).total_seconds() / 3600
                    
                    if schedule.urgency == NotificationUrgency.IMMEDIATE:
                        insight.expires_at = schedule.optimized_time + timedelta(hours=24)
                    elif schedule.urgency == NotificationUrgency.HIGH:
                        insight.expires_at = schedule.optimized_time + timedelta(hours=48)
                    elif insight.insight_type == InsightType.BUDGET_ALERT:
                        insight.expires_at = schedule.optimized_time + timedelta(days=7)
                    elif insight.insight_type == InsightType.CELEBRATION:
                        insight.expires_at = schedule.optimized_time + timedelta(days=3)
                    else:
                        insight.expires_at = schedule.optimized_time + timedelta(days=14)
                    
                    # Add timing metadata
                    if not insight.metadata:
                        insight.metadata = {}
                    
                    insight.metadata.update({
                        "timing_optimization": {
                            "original_time": schedule.original_time.isoformat(),
                            "optimized_time": schedule.optimized_time.isoformat(),
                            "delay_reason": schedule.delay_reason,
                            "timing_confidence": schedule.confidence_score,
                            "urgency": schedule.urgency.value,
                            "reschedule_count": schedule.reschedule_count
                        }
                    })
            
            # Sort insights by optimized timing for delivery order
            insights.sort(key=lambda x: x.optimal_timing)
            
            self.logger.info(f"Successfully optimized timing for {len(insights)} insights")
            return insights
            
        except Exception as e:
            self.logger.error(f"Error optimizing insight timing with TimingOptimizationTool: {str(e)}")
    
    async def _generate_predictions_from_insights(self, insights: List[ProactiveInsight], prediction_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions based on the insights."""
        try:
            predictions = {
                "budget_projections": {},
                "spending_forecasts": {},
                "goal_timelines": {},
                "risk_assessments": {}
            }
            
            for insight in insights:
                if insight.insight_type == InsightType.BUDGET_ALERT:
                    category = insight.data_points.get('category')
                    usage_percentage = insight.data_points.get('usage_percentage', 0)
                    
                    if usage_percentage > 90:
                        predictions["risk_assessments"][category] = {
                            "risk_level": "high",
                            "likelihood_of_overspend": 0.8,
                            "recommendation": "immediate_action_required"
                        }
                
                elif insight.insight_type == InsightType.TREND_WARNING:
                    category = insight.data_points.get('category')
                    projected_increase = insight.data_points.get('projected_increase', 0)
                    
                    predictions["spending_forecasts"][category] = {
                        "trend": "increasing",
                        "projected_monthly_increase": projected_increase,
                        "confidence": insight.confidence_score
                    }
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error generating predictions: {str(e)}")
            return {}
    
    async def _store_insights(self, insights: List[ProactiveInsight]):
        """Store insights in the database for tracking and history."""
        try:
            if not insights:
                return
            
            # This would integrate with the database to store insights
            # For now, just logging that insights were generated
            self.logger.info(f"Generated {len(insights)} proactive insights")
            
            # Store in shared state for other agents to access
            insights_data = {
                "insights": [self._insight_to_dict(insight) for insight in insights],
                "generated_at": datetime.now().isoformat(),
                "agent": self.agent_name
            }
            
            self.integration_coordinator.update_shared_state({
                f"proactive_insights_{insights[0].user_id}": insights_data
            })
            
        except Exception as e:
            self.logger.error(f"Error storing insights: {str(e)}")
    
    def _insight_to_dict(self, insight: ProactiveInsight) -> Dict[str, Any]:
        """Convert ProactiveInsight to dictionary format."""
        return {
            "insight_id": insight.insight_id,
            "user_id": insight.user_id,
            "insight_type": insight.insight_type.value,
            "severity": insight.severity.value,
            "title": insight.title,
            "message": insight.message,
            "data_points": insight.data_points,
            "recommendations": insight.recommendations,
            "confidence_score": insight.confidence_score,
            "relevance_score": insight.relevance_score,
            "optimal_timing": insight.optimal_timing.isoformat(),
            "expires_at": insight.expires_at.isoformat() if insight.expires_at else None,
            "metadata": insight.metadata or {}
        }
    
    async def start_continuous_monitoring(self, user_ids: List[str] = None):
        """Start continuous monitoring for specified users or all active users."""
        self.logger.info("Starting continuous monitoring...")
        self.monitoring_active = True
        
        try:
            if user_ids:
                # Schedule monitoring for specific users
                for user_id in user_ids:
                    await self._schedule_user_monitoring(user_id)
            else:
                # This would typically get all active users from database
                self.logger.info("Continuous monitoring started for all active users")
                
        except Exception as e:
            self.logger.error(f"Error starting continuous monitoring: {str(e)}")
            self.monitoring_active = False

    async def _schedule_user_monitoring(self, user_id: str):
        """Schedule monitoring for a specific user."""
        try:
            # Schedule different types of analysis
            monitoring_types = [
                (AnalysisType.BUDGET_CHECK, 8),  # High priority
                (AnalysisType.ANOMALY_DETECTION, 7),
                (AnalysisType.GOAL_TRACKING, 6),
                (AnalysisType.PATTERN_ANALYSIS, 5),
                (AnalysisType.TREND_DETECTION, 5),
                (AnalysisType.SEASONAL_REVIEW, 4)
            ]
            
            for analysis_type, priority in monitoring_types:
                success = await self.monitoring_tool.schedule_user_analysis(
                    user_id=user_id,
                    analysis_type=analysis_type,
                    priority=priority
                )
                
                if success:
                    self.logger.info(f"Scheduled {analysis_type.value} for user {user_id}")
                else:
                    self.logger.warning(f"Failed to schedule {analysis_type.value} for user {user_id}")
                    
        except Exception as e:
            self.logger.error(f"Error scheduling monitoring for user {user_id}: {str(e)}")

    async def should_run_analysis(self, user_id: str, analysis_type: str) -> bool:
        """Check if analysis should be run for a user."""
        try:
            analysis_enum = AnalysisType(analysis_type)
            return await self.monitoring_tool.should_run_analysis(user_id, analysis_enum)
        except (ValueError, Exception) as e:
            self.logger.error(f"Error checking if analysis should run: {str(e)}")
            return False

    async def update_monitoring_schedule(self, user_id: str, new_schedule: Dict[str, Any]) -> bool:
        """Update monitoring schedule for a user."""
        try:
            return await self.monitoring_tool.update_analysis_schedule(user_id, new_schedule)
        except Exception as e:
            self.logger.error(f"Error updating monitoring schedule: {str(e)}")
            return False

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            "monitoring_active": self.monitoring_active,
            "registered_tools": list(self.tools.keys()),
            "last_analysis_times": self.last_analysis_run,
            "agent_name": self.agent_name,
            "monitoring_tool_available": self.monitoring_tool is not None
        }

    async def get_user_monitoring_status(self, user_id: str) -> Dict[str, Any]:
        """Get monitoring status for a specific user."""
        try:
            return await self.monitoring_tool.get_monitoring_status(user_id)
        except Exception as e:
            self.logger.error(f"Error getting user monitoring status: {str(e)}")
            return {"user_id": user_id, "error": str(e)}
    
    async def stop_continuous_monitoring(self):
        """Stop continuous monitoring."""
        self.logger.info("Stopping continuous monitoring...")
        self.monitoring_active = False
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            "monitoring_active": self.monitoring_active,
            "registered_tools": list(self.tools.keys()),
            "last_analysis_times": self.last_analysis_run,
            "agent_name": self.agent_name
        }
    
    async def get_anomaly_insights_summary(self, user_id: str) -> Dict[str, Any]:
        """Get a summary of anomaly-based insights for reporting."""
        try:
            anomaly_summary = await self.anomaly_detection_tool.get_anomaly_summary(user_id)
            
            # Transform the summary for proactive insights context
            insights_summary = {
                "total_anomalies": anomaly_summary.get("total_anomalies", 0),
                "critical_anomalies": anomaly_summary.get("by_severity", {}).get("critical", 0),
                "positive_anomalies": anomaly_summary.get("by_polarity", {}).get("positive", 0),
                "negative_anomalies": anomaly_summary.get("by_polarity", {}).get("negative", 0),
                "top_concerns": anomaly_summary.get("high_impact_anomalies", [])[:3],
                "trend": anomaly_summary.get("recent_anomaly_trend", "stable"),
                "last_updated": datetime.now().isoformat()
            }
            
            return insights_summary
            
        except Exception as e:
            self.logger.error(f"Error getting anomaly insights summary: {str(e)}")
            return {
                "total_anomalies": 0,
                "error": str(e),
                "last_updated": datetime.now().isoformat()
            }