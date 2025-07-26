import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import asyncpg
from pathlib import Path

# Import your existing modules
from core.base_agent_tools.database_connector import DatabaseConnector
from core.base_agent_tools.config_manager import AgentConfig
from core.base_agent_tools.vertex_initializer import VertexAIInitializer
from vertexai.generative_models import GenerativeModel


class NotificationType(Enum):
    """Types of notifications that can be generated."""
    BUDGET_ALERT = "budget_alert"
    SPENDING_INSIGHT = "spending_insight" 
    GOAL_UPDATE = "goal_update"
    TIP = "tip"
    CELEBRATION = "celebration"
    WARNING = "warning"
    RECOMMENDATION = "recommendation"
    SEASONAL_REMINDER = "seasonal_reminder"
    ANOMALY_ALERT = "anomaly_alert"


class NotificationPriority(Enum):
    """Priority levels for notifications."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    URGENT = "urgent"


class NotificationChannel(Enum):
    """Available notification channels."""
    PUSH = "push"
    EMAIL = "email"
    IN_APP = "in_app"
    SMS = "sms"


class NotificationTone(Enum):
    """Tone styles for notifications."""
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    ENCOURAGING = "encouraging"
    URGENT = "urgent"
    CELEBRATORY = "celebratory"


@dataclass
class NotificationTemplate:
    """Template for generating notifications."""
    template_id: str
    notification_type: NotificationType
    title_template: str
    message_template: str
    tone: NotificationTone
    priority: NotificationPriority
    channels: List[NotificationChannel]
    personalization_fields: List[str]
    action_buttons: List[Dict[str, str]]
    expiry_hours: int = 24
    max_frequency_per_day: int = 1


@dataclass
class NotificationContent:
    """Generated notification content."""
    notification_id: str
    user_id: str
    notification_type: NotificationType
    title: str
    message: str
    rich_content: Dict[str, Any]
    priority: NotificationPriority
    channels: List[NotificationChannel]
    action_buttons: List[Dict[str, str]]
    data_visualization: Optional[Dict[str, Any]] = None
    quick_actions: List[Dict[str, str]] = None
    expires_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class UserPersonalizationData:
    """User personalization information for notifications."""
    user_id: str
    display_name: str
    preferred_tone: NotificationTone
    notification_preferences: Dict[str, bool]
    timezone: str
    spending_personality: str  # conservative, moderate, spender
    primary_concerns: List[str]  # budget, goals, savings, etc.
    communication_style: str  # brief, detailed, visual
    financial_literacy_level: str  # beginner, intermediate, advanced


class NotificationGenerationTool:
    """
    Tool for generating personalized, engaging notifications for financial insights.
    """
    
    def __init__(
        self,
        db_connection_string: str,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.db_connection_string = db_connection_string
        self.db_connector = DatabaseConnector()
        self.config = AgentConfig.from_env()
        
        # Vertex AI setup
        self.project_id = project_id or self.config.project_id
        self.location = location or self.config.location
        self.model_name = model_name or self.config.model_name
        
        VertexAIInitializer.initialize(self.project_id, self.location)
        self.model = GenerativeModel(model_name=self.model_name)
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Database connection pool
        self.db_pool = None
        
        # Template storage
        self.templates = {}
        self._load_notification_templates()
        
        # Personalization cache
        self.personalization_cache = {}
        
        self.logger.info("Notification Generation Tool initialized")

    def _setup_logging(self) -> logging.Logger:
        """Set up structured logging."""
        logger = logging.getLogger("financial_agent.notification_generation_tool")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    async def initialize(self):
        """Initialize database connections and load templates."""
        try:
            # Initialize database pool
            self.db_pool = await asyncpg.create_pool(
                self.db_connection_string,
                min_size=2,
                max_size=10,
                command_timeout=30
            )
            
            self.logger.info("Notification Generation Tool initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize: {str(e)}")
            raise

    async def close(self):
        """Close database connections."""
        if self.db_pool:
            await self.db_pool.close()

    def _load_notification_templates(self):
        """Load notification templates."""
        # Budget Alert Templates
        self.templates["budget_warning"] = NotificationTemplate(
            template_id="budget_warning",
            notification_type=NotificationType.BUDGET_ALERT,
            title_template="Budget Alert: {category} at {percentage}%",
            message_template="You've used {percentage}% of your {category} budget ({spent} of {limit}). {time_remaining} remaining this {period}.",
            tone=NotificationTone.URGENT,
            priority=NotificationPriority.HIGH,
            channels=[NotificationChannel.PUSH, NotificationChannel.IN_APP],
            personalization_fields=["category", "percentage", "spent", "limit", "time_remaining", "period"],
            action_buttons=[
                {"text": "View Details", "action": "view_budget_details"},
                {"text": "Adjust Budget", "action": "adjust_budget"}
            ],
            expiry_hours=48,
            max_frequency_per_day=2
        )
        
        self.templates["budget_exceeded"] = NotificationTemplate(
            template_id="budget_exceeded",
            notification_type=NotificationType.BUDGET_ALERT,
            title_template="Budget Exceeded: {category}",
            message_template="You've exceeded your {category} budget by {overage}. Consider adjusting your spending or budget for the remainder of this {period}.",
            tone=NotificationTone.URGENT,
            priority=NotificationPriority.URGENT,
            channels=[NotificationChannel.PUSH, NotificationChannel.IN_APP, NotificationChannel.EMAIL],
            personalization_fields=["category", "overage", "period"],
            action_buttons=[
                {"text": "See Breakdown", "action": "view_spending_breakdown"},
                {"text": "Get Tips", "action": "get_saving_tips"}
            ],
            expiry_hours=72
        )
        
        # Spending Insight Templates
        self.templates["spending_spike"] = NotificationTemplate(
            template_id="spending_spike",
            notification_type=NotificationType.SPENDING_INSIGHT,
            title_template="Unusual Spending Detected",
            message_template="You've spent {amount} more than usual on {category} this {period}. This is {percentage}% above your typical spending.",
            tone=NotificationTone.FRIENDLY,
            priority=NotificationPriority.MEDIUM,
            channels=[NotificationChannel.PUSH, NotificationChannel.IN_APP],
            personalization_fields=["amount", "category", "period", "percentage"],
            action_buttons=[
                {"text": "View Pattern", "action": "view_spending_pattern"},
                {"text": "Set Alert", "action": "set_spending_alert"}
            ]
        )
        
        self.templates["spending_improvement"] = NotificationTemplate(
            template_id="spending_improvement",
            notification_type=NotificationType.CELEBRATION,
            title_template="Great Job! Spending Down",
            message_template="You've reduced your {category} spending by {amount} ({percentage}%) this {period}. Keep it up!",
            tone=NotificationTone.CELEBRATORY,
            priority=NotificationPriority.LOW,
            channels=[NotificationChannel.PUSH, NotificationChannel.IN_APP],
            personalization_fields=["category", "amount", "percentage", "period"],
            action_buttons=[
                {"text": "See Progress", "action": "view_progress"},
                {"text": "Share Achievement", "action": "share_achievement"}
            ]
        )
        
        # Goal Update Templates
        self.templates["goal_progress"] = NotificationTemplate(
            template_id="goal_progress",
            notification_type=NotificationType.GOAL_UPDATE,
            title_template="Goal Progress: {goal_name}",
            message_template="You're {percentage}% of the way to your {goal_name} goal! {progress_message}",
            tone=NotificationTone.ENCOURAGING,
            priority=NotificationPriority.MEDIUM,
            channels=[NotificationChannel.PUSH, NotificationChannel.IN_APP],
            personalization_fields=["goal_name", "percentage", "progress_message"],
            action_buttons=[
                {"text": "View Goal", "action": "view_goal_details"},
                {"text": "Update Goal", "action": "update_goal"}
            ]
        )
        
        # Recommendation Templates
        self.templates["merchant_alternative"] = NotificationTemplate(
            template_id="merchant_alternative",
            notification_type=NotificationType.RECOMMENDATION,
            title_template="Save Money: Alternative to {merchant}",
            message_template="Based on your spending at {merchant}, you could save {savings} by shopping at {alternative}. Similar products, better prices!",
            tone=NotificationTone.FRIENDLY,
            priority=NotificationPriority.LOW,
            channels=[NotificationChannel.IN_APP],
            personalization_fields=["merchant", "savings", "alternative"],
            action_buttons=[
                {"text": "View Alternatives", "action": "view_alternatives"},
                {"text": "Get Directions", "action": "get_directions"}
            ]
        )
        
        # Seasonal Templates
        self.templates["seasonal_preparation"] = NotificationTemplate(
            template_id="seasonal_preparation",
            notification_type=NotificationType.SEASONAL_REMINDER,
            title_template="Seasonal Spending Ahead",
            message_template="Based on your history, you typically spend {amount} more on {category} during {season}. Consider budgeting an extra {suggestion} this month.",
            tone=NotificationTone.PROFESSIONAL,
            priority=NotificationPriority.LOW,
            channels=[NotificationChannel.IN_APP],
            personalization_fields=["amount", "category", "season", "suggestion"],
            action_buttons=[
                {"text": "Adjust Budget", "action": "adjust_seasonal_budget"},
                {"text": "View History", "action": "view_seasonal_history"}
            ]
        )

    async def generate_notification(
        self,
        user_id: str,
        insight_data: Dict[str, Any],
        notification_type: NotificationType,
        template_id: Optional[str] = None,
        custom_data: Optional[Dict[str, Any]] = None
    ) -> NotificationContent:
        """
        Generate a personalized notification based on insight data.
        
        Args:
            user_id: Target user ID
            insight_data: Data from the proactive insights agent
            notification_type: Type of notification to generate
            template_id: Specific template to use (optional)
            custom_data: Additional data for personalization
            
        Returns:
            NotificationContent object with generated content
        """
        try:
            self.logger.info(f"Generating {notification_type.value} notification for user {user_id}")
            
            # Get user personalization data
            user_data = await self._get_user_personalization_data(user_id)
            
            # Select appropriate template
            template = await self._select_template(
                notification_type, 
                template_id, 
                insight_data, 
                user_data
            )
            
            # Generate personalized content
            content = await self._generate_personalized_content(
                template,
                user_data,
                insight_data,
                custom_data or {}
            )
            
            # Add data visualization if appropriate
            if self._should_include_visualization(template, insight_data):
                content.data_visualization = await self._generate_data_visualization(
                    insight_data, user_data
                )
            
            # Add quick actions
            content.quick_actions = await self._generate_quick_actions(
                template, insight_data, user_data
            )
            
            # Store notification in database
            await self._store_notification(content)
            
            self.logger.info(f"Successfully generated notification {content.notification_id}")
            return content
            
        except Exception as e:
            self.logger.error(f"Error generating notification: {str(e)}")
            raise

    async def generate_batch_notifications(
        self,
        notifications_data: List[Dict[str, Any]]
    ) -> List[NotificationContent]:
        """
        Generate multiple notifications efficiently.
        
        Args:
            notifications_data: List of notification generation requests
            
        Returns:
            List of generated NotificationContent objects
        """
        try:
            self.logger.info(f"Generating batch of {len(notifications_data)} notifications")
            
            # Group by user for efficiency
            user_groups = {}
            for notif_data in notifications_data:
                user_id = notif_data['user_id']
                if user_id not in user_groups:
                    user_groups[user_id] = []
                user_groups[user_id].append(notif_data)
            
            # Pre-load user personalization data
            user_data_cache = {}
            for user_id in user_groups.keys():
                user_data_cache[user_id] = await self._get_user_personalization_data(user_id)
            
            # Generate notifications
            generated_notifications = []
            for user_id, user_notifications in user_groups.items():
                user_data = user_data_cache[user_id]
                
                # Apply frequency limits
                filtered_notifications = await self._apply_frequency_limits(
                    user_id, user_notifications
                )
                
                for notif_data in filtered_notifications:
                    try:
                        content = await self._generate_single_notification(
                            notif_data, user_data
                        )
                        generated_notifications.append(content)
                    except Exception as e:
                        self.logger.error(f"Error generating notification for user {user_id}: {str(e)}")
                        continue
            
            # Batch store notifications
            await self._batch_store_notifications(generated_notifications)
            
            self.logger.info(f"Successfully generated {len(generated_notifications)} notifications")
            return generated_notifications
            
        except Exception as e:
            self.logger.error(f"Error in batch notification generation: {str(e)}")
            raise

    async def _get_user_personalization_data(self, user_id: str) -> UserPersonalizationData:
        """Get user personalization data from database."""
        # Check cache first
        if user_id in self.personalization_cache:
            cached_data = self.personalization_cache[user_id]
            if (datetime.now() - cached_data['timestamp']).seconds < 3600:  # 1 hour cache
                return cached_data['data']
        
        try:
            async with self.db_pool.acquire() as conn:
                # Get user basic info and preferences
                user_query = """
                    SELECT 
                        u.user_id,
                        u.display_name,
                        u.preferences,
                        u.spending_patterns,
                        COALESCE(u.monthly_income, 0) as monthly_income,
                        u.risk_tolerance
                    FROM users u 
                    WHERE u.user_id = $1
                """
                
                user_row = await conn.fetchrow(user_query, uuid.UUID(user_id))
                
                if not user_row:
                    raise ValueError(f"User {user_id} not found")
                
                preferences = user_row['preferences'] or {}
                
                # Determine spending personality from patterns
                spending_patterns = user_row['spending_patterns'] or {}
                spending_personality = self._determine_spending_personality(
                    spending_patterns, user_row['monthly_income'], user_row['risk_tolerance']
                )
                
                # Get primary concerns from recent activity
                primary_concerns = await self._get_user_primary_concerns(conn, user_id)
                
                # Create personalization data
                personalization_data = UserPersonalizationData(
                    user_id=user_id,
                    display_name=user_row['display_name'] or "there",
                    preferred_tone=NotificationTone(preferences.get('notification_tone', 'friendly')),
                    notification_preferences=preferences.get('notifications', {}),
                    timezone=preferences.get('timezone', 'UTC'),
                    spending_personality=spending_personality,
                    primary_concerns=primary_concerns,
                    communication_style=preferences.get('communication_style', 'balanced'),
                    financial_literacy_level=preferences.get('financial_literacy', 'intermediate')
                )
                
                # Cache the data
                self.personalization_cache[user_id] = {
                    'data': personalization_data,
                    'timestamp': datetime.now()
                }
                
                return personalization_data
                
        except Exception as e:
            self.logger.error(f"Error getting user personalization data: {str(e)}")
            # Return default personalization data
            return UserPersonalizationData(
                user_id=user_id,
                display_name="there",
                preferred_tone=NotificationTone.FRIENDLY,
                notification_preferences={},
                timezone="UTC",
                spending_personality="moderate",
                primary_concerns=["budget"],
                communication_style="balanced",
                financial_literacy_level="intermediate"
            )

    def _determine_spending_personality(
        self, 
        spending_patterns: Dict[str, Any], 
        monthly_income: float, 
        risk_tolerance: str
    ) -> str:
        """Determine user's spending personality from patterns."""
        # Analyze spending patterns to determine personality
        # This is a simplified version - you could make this more sophisticated
        
        if risk_tolerance == "conservative":
            return "conservative"
        elif risk_tolerance == "aggressive":
            return "spender"
        
        # Look at spending patterns
        pattern_indicators = spending_patterns.get('personality_indicators', {})
        impulse_score = pattern_indicators.get('impulse_score', 0.5)
        budget_adherence = pattern_indicators.get('budget_adherence', 0.5)
        
        if budget_adherence > 0.8 and impulse_score < 0.3:
            return "conservative"
        elif budget_adherence < 0.4 or impulse_score > 0.7:
            return "spender"
        else:
            return "moderate"

    async def _get_user_primary_concerns(self, conn, user_id: str) -> List[str]:
        """Get user's primary financial concerns from recent activity."""
        concerns = []
        
        try:
            # Check for budget-related concerns
            budget_query = """
                SELECT COUNT(*) as exceeded_budgets
                FROM budget_limits bl
                WHERE bl.user_id = $1 
                AND bl.current_spent > bl.limit_amount
                AND bl.effective_to IS NULL
            """
            
            exceeded_count = await conn.fetchval(budget_query, uuid.UUID(user_id))
            if exceeded_count > 0:
                concerns.append("budget")
            
            # Check for goal-related concerns
            goal_query = """
                SELECT COUNT(*) as behind_goals
                FROM financial_goals fg
                WHERE fg.user_id = $1 
                AND fg.status = 'active'
                AND fg.target_date < CURRENT_DATE + INTERVAL '30 days'
                AND fg.progress_percentage < 75
            """
            
            behind_goals = await conn.fetchval(goal_query, uuid.UUID(user_id))
            if behind_goals > 0:
                concerns.append("goals")
            
            # Check for recent high spending
            spending_query = """
                SELECT 
                    SUM(amount) as recent_spending,
                    AVG(amount) as avg_transaction
                FROM transactions t
                WHERE t.user_id = $1 
                AND t.transaction_date >= CURRENT_DATE - INTERVAL '7 days'
                AND t.deleted_at IS NULL
            """
            
            spending_row = await conn.fetchrow(spending_query, uuid.UUID(user_id))
            if spending_row and spending_row['avg_transaction']:
                if spending_row['avg_transaction'] > 100:  # High average transaction
                    concerns.append("spending_control")
            
            # Default concerns if none identified
            if not concerns:
                concerns = ["budget", "savings"]
                
        except Exception as e:
            self.logger.error(f"Error getting user concerns: {str(e)}")
            concerns = ["budget"]
        
        return concerns

    async def _select_template(
        self,
        notification_type: NotificationType,
        template_id: Optional[str],
        insight_data: Dict[str, Any],
        user_data: UserPersonalizationData
    ) -> NotificationTemplate:
        """Select the most appropriate template for the notification."""
        
        if template_id and template_id in self.templates:
            return self.templates[template_id]
        
        # Select based on notification type and context
        if notification_type == NotificationType.BUDGET_ALERT:
            budget_utilization = insight_data.get('data_points', {}).get('usage_percentage', 0)
            if budget_utilization > 100:
                return self.templates["budget_exceeded"]
            else:
                return self.templates["budget_warning"]
        
        elif notification_type == NotificationType.SPENDING_INSIGHT:
            polarity = insight_data.get('data_points', {}).get('polarity', 'neutral')
            if polarity == 'positive':
                return self.templates["spending_improvement"]
            else:
                return self.templates["spending_spike"]
        
        elif notification_type == NotificationType.GOAL_UPDATE:
            return self.templates["goal_progress"]
        
        elif notification_type == NotificationType.RECOMMENDATION:
            return self.templates["merchant_alternative"]
        
        elif notification_type == NotificationType.SEASONAL_REMINDER:
            return self.templates["seasonal_preparation"]
        
        # Default fallback
        return self.templates.get("spending_spike", list(self.templates.values())[0])

    async def _generate_personalized_content(
        self,
        template: NotificationTemplate,
        user_data: UserPersonalizationData,
        insight_data: Dict[str, Any],
        custom_data: Dict[str, Any]
    ) -> NotificationContent:
        """Generate personalized notification content using AI."""
        
        try:
            # Prepare data for template filling
            template_data = self._prepare_template_data(insight_data, custom_data, user_data)
            
            # Use AI to enhance the message with personalization
            enhanced_content = await self._enhance_with_ai(
                template, template_data, user_data
            )
            
            # Create notification content
            notification_content = NotificationContent(
                notification_id=str(uuid.uuid4()),
                user_id=user_data.user_id,
                notification_type=template.notification_type,
                title=enhanced_content['title'],
                message=enhanced_content['message'],
                rich_content=enhanced_content.get('rich_content', {}),
                priority=template.priority,
                channels=self._select_channels(template, user_data),
                action_buttons=template.action_buttons.copy(),
                expires_at=datetime.now() + timedelta(hours=template.expiry_hours),
                metadata={
                    'template_id': template.template_id,
                    'user_tone': user_data.preferred_tone.value,
                    'generation_timestamp': datetime.now().isoformat(),
                    'insight_data': insight_data
                }
            )
            
            return notification_content
            
        except Exception as e:
            self.logger.error(f"Error generating personalized content: {str(e)}")
            raise

    def _prepare_template_data(
        self, 
        insight_data: Dict[str, Any], 
        custom_data: Dict[str, Any],
        user_data: UserPersonalizationData
    ) -> Dict[str, Any]:
        """Prepare data for template filling."""
        
        template_data = {}
        data_points = insight_data.get('data_points', {})
        
        # Extract common template variables
        template_data.update({
            'user_name': user_data.display_name,
            'category': data_points.get('category', '').title(),
            'amount': self._format_currency(data_points.get('spent_amount', 0)),
            'percentage': f"{data_points.get('usage_percentage', 0):.0f}",
            'period': 'month',  # Could be dynamic
            'limit': self._format_currency(data_points.get('budget_limit', 0)),
            'spent': self._format_currency(data_points.get('spent_amount', 0)),
            'overage': self._format_currency(abs(data_points.get('projected_overage', 0))),
            'savings': self._format_currency(data_points.get('potential_savings', 0)),
            'time_remaining': f"{data_points.get('days_remaining', 0)} days"
        })
        
        # Add goal-specific data if available
        if 'goal_type' in data_points:
            template_data.update({
                'goal_name': data_points.get('goal_type', '').title(),
                'progress_percentage': f"{data_points.get('progress_percentage', 0):.0f}"
            })
        
        # Add custom data
        template_data.update(custom_data)
        
        return template_data

    async def _enhance_with_ai(
        self,
        template: NotificationTemplate,
        template_data: Dict[str, Any],
        user_data: UserPersonalizationData
    ) -> Dict[str, Any]:
        """Use AI to enhance notification content with personalization."""
        
        try:
            # Create prompt for AI enhancement
            prompt = self._create_enhancement_prompt(template, template_data, user_data)
            
            # Generate enhanced content
            response = await self.model.generate_content_async(prompt)
            
            # Parse response
            enhanced_content = self._parse_ai_response(response.text)
            
            return enhanced_content
            
        except Exception as e:
            self.logger.error(f"AI enhancement failed, using template: {str(e)}")
            # Fallback to template-based generation
            return self._generate_template_content(template, template_data)

    def _create_enhancement_prompt(
        self,
        template: NotificationTemplate,
        template_data: Dict[str, Any],
        user_data: UserPersonalizationData
    ) -> str:
        """Create prompt for AI content enhancement."""
        
        prompt = f"""
        Generate a personalized financial notification with the following parameters:

        User Profile:
        - Name: {user_data.display_name}
        - Spending Personality: {user_data.spending_personality}
        - Communication Style: {user_data.communication_style}
        - Financial Literacy: {user_data.financial_literacy_level}
        - Primary Concerns: {', '.join(user_data.primary_concerns)}
        - Preferred Tone: {user_data.preferred_tone.value}

        Notification Details:
        - Type: {template.notification_type.value}
        - Priority: {template.priority.value}
        - Template Tone: {template.tone.value}

        Base Template:
        - Title: {template.title_template}
        - Message: {template.message_template}

        Data to Include:
        {json.dumps(template_data, indent=2)}

        Requirements:
        1. Personalize the title and message using the template and data
        2. Match the user's preferred tone and communication style
        3. Keep the title under 60 characters
        4. Keep the message under 200 characters for mobile notifications
        5. Make it actionable and relevant
        6. Use the user's name naturally if appropriate
        7. Include specific numbers and data points
        8. For celebrations, be enthusiastic; for warnings, be helpful not alarming

        Return the response as JSON with the following structure:
        {{
            "title": "Enhanced notification title",
            "message": "Enhanced notification message",
            "rich_content": {{
                "summary": "Brief summary for rich notifications",
                "emphasis": "Key point to emphasize"
            }}
        }}
        """
        
        return prompt

    def _parse_ai_response(self, response_text: str) -> Dict[str, Any]:
        """Parse AI response and extract content."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # If no JSON found, try to parse structured text
                lines = response_text.strip().split('\n')
                return {
                    "title": lines[0] if lines else "Financial Update",
                    "message": lines[1] if len(lines) > 1 else "Check your spending activity",
                    "rich_content": {}
                }
        except Exception as e:
            self.logger.error(f"Error parsing AI response: {str(e)}")
            return {
                "title": "Financial Update",
                "message": "Please check your financial activity",
                "rich_content": {}
            }

    def _generate_template_content(
        self, 
        template: NotificationTemplate, 
        template_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate content using template fallback."""
        try:
            title = template.title_template.format(**template_data)
            message = template.message_template.format(**template_data)
            
            return {
                "title": title,
                "message": message,
                "rich_content": {}
            }
        except KeyError as e:
            self.logger.error(f"Missing template data key: {str(e)}")
            return {
                "title": "Financial Update",
                "message": "Please review your recent financial activity",
                "rich_content": {}
            }

    def _select_channels(
        self, 
        template: NotificationTemplate, 
        user_data: UserPersonalizationData
    ) -> List[NotificationChannel]:
        """Select appropriate notification channels based on user preferences and urgency."""
        user_prefs = user_data.notification_preferences
        available_channels = template.channels.copy()
        
        # Filter based on user preferences
        selected_channels = []
        
        for channel in available_channels:
            if channel == NotificationChannel.PUSH and user_prefs.get('pushEnabled', True):
                selected_channels.append(channel)
            elif channel == NotificationChannel.EMAIL and user_prefs.get('emailEnabled', False):
                selected_channels.append(channel)
            elif channel == NotificationChannel.IN_APP:
                # Always include in-app notifications
                selected_channels.append(channel)
            elif channel == NotificationChannel.SMS and user_prefs.get('smsEnabled', False):
                selected_channels.append(channel)
        
        # Ensure we have at least in-app notifications
        if not selected_channels:
            selected_channels = [NotificationChannel.IN_APP]
        
        # For urgent notifications, ensure push is included if enabled
        if template.priority == NotificationPriority.URGENT and user_prefs.get('pushEnabled', True):
            if NotificationChannel.PUSH not in selected_channels:
                selected_channels.append(NotificationChannel.PUSH)
        
        return selected_channels

    def _should_include_visualization(
        self, 
        template: NotificationTemplate, 
        insight_data: Dict[str, Any]
    ) -> bool:
        """Determine if visualization should be included in the notification."""
        
        # Include visualizations for certain types
        visualization_types = [
            NotificationType.BUDGET_ALERT,
            NotificationType.SPENDING_INSIGHT,
            NotificationType.GOAL_UPDATE
        ]
        
        if template.notification_type not in visualization_types:
            return False
        
        # Check if we have sufficient data for visualization
        data_points = insight_data.get('data_points', {})
        
        # Budget alerts need spending vs limit data
        if template.notification_type == NotificationType.BUDGET_ALERT:
            return 'spent_amount' in data_points and 'budget_limit' in data_points
        
        # Spending insights need historical comparison
        elif template.notification_type == NotificationType.SPENDING_INSIGHT:
            return 'amount' in data_points and ('average_amount' in data_points or 'previous_amount' in data_points)
        
        # Goal updates need progress data
        elif template.notification_type == NotificationType.GOAL_UPDATE:
            return 'progress_percentage' in data_points or 'current_amount' in data_points
        
        return False

    async def _generate_data_visualization(
        self, 
        insight_data: Dict[str, Any], 
        user_data: UserPersonalizationData
    ) -> Dict[str, Any]:
        """Generate data visualization configuration for the notification."""
        
        data_points = insight_data.get('data_points', {})
        visualization = {
            "type": "progress_bar",  # Default type
            "data": {},
            "config": {
                "colorScheme": "default",
                "showLabels": True,
                "animated": True
            }
        }
        
        # Budget visualization
        if 'budget_limit' in data_points and 'spent_amount' in data_points:
            spent = float(data_points['spent_amount'])
            limit = float(data_points['budget_limit'])
            usage_pct = (spent / limit * 100) if limit > 0 else 0
            
            visualization.update({
                "type": "budget_gauge",
                "data": {
                    "spent": spent,
                    "limit": limit,
                    "percentage": min(usage_pct, 100),
                    "category": data_points.get('category', 'Total')
                },
                "config": {
                    "colorScheme": "budget" if usage_pct < 80 else "warning" if usage_pct < 100 else "danger",
                    "showTarget": True,
                    "formatCurrency": True
                }
            })
        
        # Spending comparison visualization
        elif 'amount' in data_points and 'average_amount' in data_points:
            current = float(data_points['amount'])
            average = float(data_points['average_amount'])
            change_pct = ((current - average) / average * 100) if average > 0 else 0
            
            visualization.update({
                "type": "comparison_chart",
                "data": {
                    "current": current,
                    "average": average,
                    "change_percentage": change_pct,
                    "category": data_points.get('category', 'Spending')
                },
                "config": {
                    "colorScheme": "positive" if change_pct <= 0 else "negative",
                    "showComparison": True,
                    "formatCurrency": True
                }
            })
        
        # Goal progress visualization
        elif 'progress_percentage' in data_points:
            progress = float(data_points.get('progress_percentage', 0))
            
            visualization.update({
                "type": "progress_ring",
                "data": {
                    "percentage": progress,
                    "current": data_points.get('current_amount', 0),
                    "target": data_points.get('target_amount', 0),
                    "goal_name": data_points.get('goal_type', 'Goal')
                },
                "config": {
                    "colorScheme": "progress",
                    "showPercentage": True,
                    "formatCurrency": True
                }
            })
        
        return visualization

    async def _generate_quick_actions(
        self, 
        template: NotificationTemplate, 
        insight_data: Dict[str, Any], 
        user_data: UserPersonalizationData
    ) -> List[Dict[str, str]]:
        """Generate contextual quick actions for the notification."""
        
        quick_actions = []
        data_points = insight_data.get('data_points', {})
        
        # Budget-related quick actions
        if template.notification_type == NotificationType.BUDGET_ALERT:
            usage_pct = data_points.get('usage_percentage', 0)
            category = data_points.get('category', '')
            
            if usage_pct > 100:
                quick_actions.extend([
                    {
                        "text": "View Transactions",
                        "action": "view_category_transactions",
                        "data": {"category": category},
                        "icon": "list"
                    },
                    {
                        "text": "Adjust Budget",
                        "action": "adjust_budget",
                        "data": {"category": category},
                        "icon": "edit"
                    }
                ])
            else:
                quick_actions.extend([
                    {
                        "text": "Set Alert",
                        "action": "set_spending_alert",
                        "data": {"category": category, "threshold": 90},
                        "icon": "bell"
                    },
                    {
                        "text": "Get Tips",
                        "action": "get_saving_tips",
                        "data": {"category": category},
                        "icon": "lightbulb"
                    }
                ])
        
        # Spending insight quick actions
        elif template.notification_type == NotificationType.SPENDING_INSIGHT:
            category = data_points.get('category', '')
            polarity = data_points.get('polarity', 'neutral')
            
            if polarity == 'negative':  # Increased spending
                quick_actions.extend([
                    {
                        "text": "Analyze Pattern",
                        "action": "view_spending_pattern",
                        "data": {"category": category},
                        "icon": "chart"
                    },
                    {
                        "text": "Find Alternatives",
                        "action": "find_merchant_alternatives",
                        "data": {"category": category},
                        "icon": "map"
                    }
                ])
            else:  # Positive change
                quick_actions.extend([
                    {
                        "text": "View Progress",
                        "action": "view_progress_details",
                        "data": {"category": category},
                        "icon": "trophy"
                    },
                    {
                        "text": "Set New Goal",
                        "action": "create_savings_goal",
                        "data": {"category": category},
                        "icon": "target"
                    }
                ])
        
        # Goal-related quick actions
        elif template.notification_type == NotificationType.GOAL_UPDATE:
            goal_id = data_points.get('goal_id', '')
            progress = data_points.get('progress_percentage', 0)
            
            if progress >= 90:
                quick_actions.extend([
                    {
                        "text": "Complete Goal",
                        "action": "complete_goal",
                        "data": {"goal_id": goal_id},
                        "icon": "check"
                    },
                    {
                        "text": "Create New Goal",
                        "action": "create_goal",
                        "data": {},
                        "icon": "plus"
                    }
                ])
            else:
                quick_actions.extend([
                    {
                        "text": "View Goal",
                        "action": "view_goal_details",
                        "data": {"goal_id": goal_id},
                        "icon": "eye"
                    },
                    {
                        "text": "Boost Progress",
                        "action": "get_goal_tips",
                        "data": {"goal_id": goal_id},
                        "icon": "rocket"
                    }
                ])
        
        # Recommendation quick actions
        elif template.notification_type == NotificationType.RECOMMENDATION:
            quick_actions.extend([
                {
                    "text": "Learn More",
                    "action": "view_recommendation_details",
                    "data": insight_data,
                    "icon": "info"
                },
                {
                    "text": "Not Interested",
                    "action": "dismiss_recommendation",
                    "data": {"recommendation_id": insight_data.get('insight_id')},
                    "icon": "x"
                }
            ])
        
        # Add universal quick actions
        quick_actions.append({
            "text": "View Dashboard",
            "action": "open_dashboard",
            "data": {},
            "icon": "home"
        })
        
        return quick_actions

    async def _apply_frequency_limits(
        self, 
        user_id: str, 
        notifications_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply frequency limits to prevent notification spam."""
        
        try:
            async with self.db_pool.acquire() as conn:
                # Get today's sent notifications for this user
                today_notifications = await conn.fetch("""
                    SELECT notification_type, COUNT(*) as count
                    FROM notifications n
                    WHERE n.user_id = $1 
                    AND n.sent_at::date = CURRENT_DATE
                    AND n.status = 'sent'
                    GROUP BY notification_type
                """, uuid.UUID(user_id))
                
                today_counts = {row['notification_type']: row['count'] for row in today_notifications}
                
                # Filter notifications based on frequency limits
                filtered_notifications = []
                
                for notif_data in notifications_data:
                    notif_type = notif_data.get('notification_type', 'unknown')
                    template_id = notif_data.get('template_id')
                    
                    # Get template for frequency limits
                    template = self.templates.get(template_id)
                    if not template:
                        continue
                    
                    current_count = today_counts.get(notif_type, 0)
                    max_frequency = template.max_frequency_per_day
                    
                    # Allow if under frequency limit
                    if current_count < max_frequency:
                        filtered_notifications.append(notif_data)
                    else:
                        self.logger.info(f"Skipping notification due to frequency limit: {notif_type} for user {user_id}")
                
                return filtered_notifications
                
        except Exception as e:
            self.logger.error(f"Error applying frequency limits: {str(e)}")
            return notifications_data  # Return all if error

    async def _generate_single_notification(
        self, 
        notif_data: Dict[str, Any], 
        user_data: UserPersonalizationData
    ) -> NotificationContent:
        """Generate a single notification from notification data."""
        
        return await self.generate_notification(
            user_id=user_data.user_id,
            insight_data=notif_data.get('insight_data', {}),
            notification_type=NotificationType(notif_data.get('notification_type')),
            template_id=notif_data.get('template_id'),
            custom_data=notif_data.get('custom_data', {})
        )

    async def _store_notification(self, notification: NotificationContent):
        """Store notification in the database."""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO notifications (
                        notification_id, user_id, notification_type, title, message,
                        data_points, priority, channels, action_buttons, 
                        data_visualization, quick_actions, expires_at, metadata,
                        status, created_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, 'pending', CURRENT_TIMESTAMP
                    )
                """, 
                uuid.UUID(notification.notification_id),
                uuid.UUID(notification.user_id),
                notification.notification_type.value,
                notification.title,
                notification.message,
                json.dumps(notification.rich_content),
                notification.priority.value,
                json.dumps([ch.value for ch in notification.channels]),
                json.dumps(notification.action_buttons),
                json.dumps(notification.data_visualization) if notification.data_visualization else None,
                json.dumps(notification.quick_actions) if notification.quick_actions else None,
                notification.expires_at,
                json.dumps(notification.metadata) if notification.metadata else None
                )
                
        except Exception as e:
            self.logger.error(f"Error storing notification: {str(e)}")
            raise

    async def _batch_store_notifications(self, notifications: List[NotificationContent]):
        """Store multiple notifications efficiently."""
        if not notifications:
            return
            
        try:
            async with self.db_pool.acquire() as conn:
                # Prepare batch insert data
                notification_data = []
                for notif in notifications:
                    notification_data.append((
                        uuid.UUID(notif.notification_id),
                        uuid.UUID(notif.user_id),
                        notif.notification_type.value,
                        notif.title,
                        notif.message,
                        json.dumps(notif.rich_content),
                        notif.priority.value,
                        json.dumps([ch.value for ch in notif.channels]),
                        json.dumps(notif.action_buttons),
                        json.dumps(notif.data_visualization) if notif.data_visualization else None,
                        json.dumps(notif.quick_actions) if notif.quick_actions else None,
                        notif.expires_at,
                        json.dumps(notif.metadata) if notif.metadata else None
                    ))
                
                # Batch insert
                await conn.executemany("""
                    INSERT INTO notifications (
                        notification_id, user_id, notification_type, title, message,
                        data_points, priority, channels, action_buttons, 
                        data_visualization, quick_actions, expires_at, metadata,
                        status, created_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, 'pending', CURRENT_TIMESTAMP
                    )
                """, notification_data)
                
                self.logger.info(f"Batch stored {len(notifications)} notifications")
                
        except Exception as e:
            self.logger.error(f"Error batch storing notifications: {str(e)}")
            raise

    def _format_currency(self, amount: float, currency: str = "USD") -> str:
        """Format currency amounts for display."""
        if currency == "USD":
            return f"${amount:,.2f}"
        else:
            return f"{amount:,.2f} {currency}"

    async def get_notification_templates(self) -> Dict[str, NotificationTemplate]:
        """Get all available notification templates."""
        return self.templates.copy()

    async def add_custom_template(self, template: NotificationTemplate):
        """Add a custom notification template."""
        self.templates[template.template_id] = template
        self.logger.info(f"Added custom template: {template.template_id}")

    async def get_user_notification_history(
        self, 
        user_id: str, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get notification history for a user."""
        try:
            async with self.db_pool.acquire() as conn:
                notifications = await conn.fetch("""
                    SELECT 
                        notification_id,
                        notification_type,
                        title,
                        message,
                        priority,
                        status,
                        created_at,
                        sent_at,
                        viewed_at
                    FROM notifications 
                    WHERE user_id = $1 
                    ORDER BY created_at DESC 
                    LIMIT $2
                """, uuid.UUID(user_id), limit)
                
                return [dict(row) for row in notifications]
                
        except Exception as e:
            self.logger.error(f"Error getting notification history: {str(e)}")
            return []

    async def mark_notification_as_sent(self, notification_id: str):
        """Mark a notification as sent."""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE notifications 
                    SET status = 'sent', sent_at = CURRENT_TIMESTAMP
                    WHERE notification_id = $1
                """, uuid.UUID(notification_id))
                
        except Exception as e:
            self.logger.error(f"Error marking notification as sent: {str(e)}")

    async def mark_notification_as_viewed(self, notification_id: str):
        """Mark a notification as viewed by the user."""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE notifications 
                    SET status = 'viewed', viewed_at = CURRENT_TIMESTAMP
                    WHERE notification_id = $1 AND status = 'sent'
                """, uuid.UUID(notification_id))
                
        except Exception as e:
            self.logger.error(f"Error marking notification as viewed: {str(e)}")

    async def get_pending_notifications(self, user_id: str) -> List[Dict[str, Any]]:
        """Get pending notifications for a user."""
        try:
            async with self.db_pool.acquire() as conn:
                notifications = await conn.fetch("""
                    SELECT 
                        notification_id,
                        notification_type,
                        title,
                        message,
                        data_points,
                        priority,
                        channels,
                        action_buttons,
                        data_visualization,
                        quick_actions,
                        expires_at,
                        metadata
                    FROM notifications 
                    WHERE user_id = $1 
                    AND status = 'pending'
                    AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                    ORDER BY priority DESC, created_at ASC
                """, uuid.UUID(user_id))
                
                # Parse JSON fields
                result = []
                for row in notifications:
                    notif_dict = dict(row)
                    
                    # Parse JSON fields
                    if notif_dict['data_points']:
                        notif_dict['data_points'] = json.loads(notif_dict['data_points'])
                    if notif_dict['channels']:
                        notif_dict['channels'] = json.loads(notif_dict['channels'])
                    if notif_dict['action_buttons']:
                        notif_dict['action_buttons'] = json.loads(notif_dict['action_buttons'])
                    if notif_dict['data_visualization']:
                        notif_dict['data_visualization'] = json.loads(notif_dict['data_visualization'])
                    if notif_dict['quick_actions']:
                        notif_dict['quick_actions'] = json.loads(notif_dict['quick_actions'])
                    if notif_dict['metadata']:
                        notif_dict['metadata'] = json.loads(notif_dict['metadata'])
                    
                    result.append(notif_dict)
                
                return result
                
        except Exception as e:
            self.logger.error(f"Error getting pending notifications: {str(e)}")
            return []

    async def clean_expired_notifications(self):
        """Clean up expired notifications."""
        try:
            async with self.db_pool.acquire() as conn:
                deleted_count = await conn.fetchval("""
                    UPDATE notifications 
                    SET status = 'expired'
                    WHERE status IN ('pending', 'sent')
                    AND expires_at IS NOT NULL 
                    AND expires_at < CURRENT_TIMESTAMP
                    RETURNING COUNT(*)
                """)
                
                if deleted_count:
                    self.logger.info(f"Marked {deleted_count} notifications as expired")
                    
        except Exception as e:
            self.logger.error(f"Error cleaning expired notifications: {str(e)}")

    async def get_notification_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get notification analytics for a user."""
        try:
            async with self.db_pool.acquire() as conn:
                # Get notification stats
                stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_notifications,
                        COUNT(*) FILTER (WHERE status = 'sent') as sent_count,
                        COUNT(*) FILTER (WHERE status = 'viewed') as viewed_count,
                        COUNT(*) FILTER (WHERE status = 'dismissed') as dismissed_count,
                        AVG(CASE WHEN viewed_at IS NOT NULL AND sent_at IS NOT NULL 
                            THEN EXTRACT(EPOCH FROM (viewed_at - sent_at))/60 END) as avg_view_time_minutes
                    FROM notifications 
                    WHERE user_id = $1 
                    AND created_at >= CURRENT_TIMESTAMP - INTERVAL '%s days'
                """, uuid.UUID(user_id), days)
                
                # Get notification breakdown by type
                type_breakdown = await conn.fetch("""
                    SELECT 
                        notification_type,
                        COUNT(*) as count,
                        COUNT(*) FILTER (WHERE status = 'viewed') as viewed_count
                    FROM notifications 
                    WHERE user_id = $1 
                    AND created_at >= CURRENT_TIMESTAMP - INTERVAL '%s days'
                    GROUP BY notification_type
                    ORDER BY count DESC
                """, uuid.UUID(user_id), days)
                
                # Calculate engagement rate
                total = stats['total_notifications'] or 0
                viewed = stats['viewed_count'] or 0
                engagement_rate = (viewed / total) if total > 0 else 0
                
                return {
                    "total_notifications": total,
                    "sent_count": stats['sent_count'] or 0,
                    "viewed_count": viewed,
                    "dismissed_count": stats['dismissed_count'] or 0,
                    "engagement_rate": engagement_rate,
                    "avg_view_time_minutes": float(stats['avg_view_time_minutes'] or 0),
                    "type_breakdown": [dict(row) for row in type_breakdown],
                    "analysis_period_days": days
                }
                
        except Exception as e:
            self.logger.error(f"Error getting notification analytics: {str(e)}")
            return {}

    def get_tool_status(self) -> Dict[str, Any]:
        """Get current tool status and configuration."""
        return {
            "tool_name": "notification_generation_tool",
            "templates_loaded": len(self.templates),
            "available_templates": list(self.templates.keys()),
            "database_connected": self.db_pool is not None,
            "cache_size": len(self.personalization_cache),
            "supported_channels": [channel.value for channel in NotificationChannel],
            "supported_types": [ntype.value for ntype in NotificationType]
        }