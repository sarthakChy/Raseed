import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, time
from dataclasses import dataclass
from enum import Enum
import uuid
import statistics
from collections import defaultdict, Counter

import pytz
from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration
from core.base_agent_tools.database_connector import DatabaseConnector
from core.base_agent_tools.config_manager import AgentConfig
from core.base_agent_tools.vertex_initializer import VertexAIInitializer
from core.base_agent_tools.integration_coordinator import IntegrationCoordinator
from core.base_agent_tools.error_handler import ErrorHandler


class NotificationUrgency(Enum):
    """Urgency levels for notifications."""
    IMMEDIATE = "immediate"      # Send now regardless of timing
    HIGH = "high"               # Send within 1-2 hours
    MEDIUM = "medium"           # Send within optimal window today
    LOW = "low"                 # Can wait for next optimal window
    BACKGROUND = "background"   # Can wait days/weeks


class UserActivityLevel(Enum):
    """User activity levels for timing optimization."""
    HIGHLY_ACTIVE = "highly_active"     # Multiple sessions daily
    MODERATELY_ACTIVE = "moderately_active"  # Daily usage
    OCCASIONALLY_ACTIVE = "occasionally_active"  # Few times per week
    INACTIVE = "inactive"               # Rare usage


@dataclass
class OptimalTimeWindow:
    """Represents an optimal time window for notifications."""
    start_time: time
    end_time: time
    timezone: str
    confidence_score: float
    day_of_week: Optional[int] = None  # 0=Monday, 6=Sunday, None=any day
    priority_level: int = 1  # 1=highest priority window


@dataclass
class NotificationSchedule:
    """Scheduled notification with optimized timing."""
    notification_id: str
    user_id: str
    original_time: datetime
    optimized_time: datetime
    urgency: NotificationUrgency
    delay_reason: str
    confidence_score: float
    reschedule_count: int = 0
    max_reschedules: int = 3


@dataclass
class UserTimingProfile:
    """User's timing preferences and patterns."""
    user_id: str
    timezone: str
    optimal_windows: List[OptimalTimeWindow]
    avoid_windows: List[Tuple[time, time]]  # Times to avoid notifications
    preferred_frequency: str  # daily, weekly, real_time
    max_daily_notifications: int
    activity_level: UserActivityLevel
    sleep_schedule: Optional[Tuple[time, time]] = None
    work_schedule: Optional[Tuple[time, time]] = None
    weekend_preference: bool = True  # Whether user accepts weekend notifications
    last_updated: datetime = None


class TimingOptimizationTool:
    """
    Tool for optimizing notification timing based on user patterns and preferences.
    
    This tool analyzes user behavior, engagement patterns, and preferences to determine
    the optimal timing for delivering proactive insights and notifications.
    """
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        # Initialize configuration
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
        
        # Initialize Vertex AI for ML-based timing predictions
        VertexAIInitializer.initialize(self.project_id, self.location)
        self.model = GenerativeModel(model_name=self.model_name)
        
        # Timing optimization state
        self.user_timing_profiles = {}  # Cache for user timing profiles
        self.notification_queue = {}    # Per-user notification queues
        self.engagement_history = {}    # Track notification engagement
        
        # Default timing windows (will be personalized per user)
        self.default_optimal_windows = [
            OptimalTimeWindow(time(9, 0), time(10, 30), "UTC", 0.7, priority_level=1),   # Morning
            OptimalTimeWindow(time(13, 0), time(14, 0), "UTC", 0.6, priority_level=2),   # Lunch
            OptimalTimeWindow(time(18, 0), time(20, 0), "UTC", 0.8, priority_level=1),   # Evening
        ]
        
        self.logger.info("Timing Optimization Tool initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up structured logging for the tool."""
        logger = logging.getLogger("financial_agent.timing_optimization_tool")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def optimize_notification_timing(
        self,
        user_id: str,
        notification_data: Dict[str, Any],
        urgency: NotificationUrgency = NotificationUrgency.MEDIUM
    ) -> NotificationSchedule:
        """
        Optimize timing for a single notification.
        
        Args:
            user_id: Target user ID
            notification_data: Notification content and metadata
            urgency: Urgency level affecting timing flexibility
            
        Returns:
            NotificationSchedule with optimized timing
        """
        try:
            self.logger.info(f"Optimizing notification timing for user {user_id}")
            
            # Get user timing profile
            timing_profile = await self.get_user_timing_profile(user_id)
            
            # Current time in user's timezone
            user_tz = pytz.timezone(timing_profile.timezone)
            current_time = datetime.now(user_tz)
            
            # Determine optimal delivery time based on urgency
            if urgency == NotificationUrgency.IMMEDIATE:
                optimized_time = current_time
                delay_reason = "immediate_delivery_required"
                confidence = 1.0
            else:
                optimized_time, delay_reason, confidence = await self._calculate_optimal_time(
                    timing_profile, current_time, urgency, notification_data
                )
            
            # Check for notification fatigue
            if await self._would_cause_fatigue(user_id, optimized_time):
                optimized_time, delay_reason = await self._handle_notification_fatigue(
                    timing_profile, optimized_time, urgency
                )
                confidence *= 0.8  # Reduce confidence due to fatigue adjustment
            
            # Create notification schedule
            schedule = NotificationSchedule(
                notification_id=notification_data.get('notification_id', str(uuid.uuid4())),
                user_id=user_id,
                original_time=current_time,
                optimized_time=optimized_time,
                urgency=urgency,
                delay_reason=delay_reason,
                confidence_score=confidence
            )
            
            # Add to user's notification queue
            await self._add_to_notification_queue(schedule)
            
            self.logger.info(f"Optimized notification timing: {current_time} -> {optimized_time}")
            return schedule
            
        except Exception as e:
            self.logger.error(f"Error optimizing notification timing: {str(e)}")
            # Fallback to immediate delivery with low confidence
            return NotificationSchedule(
                notification_id=str(uuid.uuid4()),
                user_id=user_id,
                original_time=datetime.now(),
                optimized_time=datetime.now(),
                urgency=urgency,
                delay_reason="optimization_failed",
                confidence_score=0.1
            )
    
    async def batch_optimize_notifications(
        self,
        user_id: str,
        notifications: List[Dict[str, Any]]
    ) -> List[NotificationSchedule]:
        """
        Optimize timing for multiple notifications, considering batch effects.
        
        Args:
            user_id: Target user ID
            notifications: List of notification data
            
        Returns:
            List of optimized notification schedules
        """
        try:
            if not notifications:
                return []
            
            timing_profile = await self.get_user_timing_profile(user_id)
            schedules = []
            
            # Sort notifications by urgency and priority
            sorted_notifications = sorted(
                notifications,
                key=lambda x: (
                    self._urgency_priority(x.get('urgency', 'medium')),
                    x.get('priority_score', 0.5)
                ),
                reverse=True
            )
            
            user_tz = pytz.timezone(timing_profile.timezone)
            current_time = datetime.now(user_tz)
            
            # Track notifications scheduled for each time window to avoid clustering
            window_usage = defaultdict(int)
            
            for notification in sorted_notifications:
                urgency = NotificationUrgency(notification.get('urgency', 'medium'))
                
                # Get base optimal time
                base_time, reason, confidence = await self._calculate_optimal_time(
                    timing_profile, current_time, urgency, notification
                )
                
                # Adjust for batch distribution
                optimized_time = await self._distribute_in_batch(
                    base_time, timing_profile, window_usage, urgency
                )
                
                schedule = NotificationSchedule(
                    notification_id=notification.get('notification_id', str(uuid.uuid4())),
                    user_id=user_id,
                    original_time=current_time,
                    optimized_time=optimized_time,
                    urgency=urgency,
                    delay_reason=reason,
                    confidence_score=confidence
                )
                
                schedules.append(schedule)
                
                # Update window usage tracking
                window_key = self._get_time_window_key(optimized_time)
                window_usage[window_key] += 1
            
            # Add all schedules to queue
            for schedule in schedules:
                await self._add_to_notification_queue(schedule)
            
            self.logger.info(f"Batch optimized {len(schedules)} notifications for user {user_id}")
            return schedules
            
        except Exception as e:
            self.logger.error(f"Error in batch optimization: {str(e)}")
            return []
    
    async def get_user_timing_profile(self, user_id: str) -> UserTimingProfile:
        """Get or create user timing profile."""
        try:
            # Check cache first
            if user_id in self.user_timing_profiles:
                profile = self.user_timing_profiles[user_id]
                # Refresh if older than 24 hours
                if profile.last_updated and (datetime.now() - profile.last_updated).hours < 24:
                    return profile
            
            # Build profile from database and behavioral analysis
            profile = await self._build_user_timing_profile(user_id)
            
            # Cache the profile
            self.user_timing_profiles[user_id] = profile
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Error getting user timing profile: {str(e)}")
            return await self._create_default_timing_profile(user_id)
    
    async def _build_user_timing_profile(self, user_id: str) -> UserTimingProfile:
        """Build comprehensive timing profile from user data."""
        try:
            # Get user preferences from database
            user_data = await self._get_user_preferences(user_id)
            
            # Analyze historical engagement patterns
            engagement_patterns = await self._analyze_engagement_patterns(user_id)
            
            # Analyze chat activity patterns
            activity_patterns = await self._analyze_chat_activity_patterns(user_id)
            
            # Determine activity level
            activity_level = self._determine_activity_level(engagement_patterns, activity_patterns)
            
            # Extract timezone and basic preferences
            timezone = user_data.get('preferences', {}).get('timezone', 'UTC')
            notification_prefs = user_data.get('preferences', {}).get('notifications', {})
            
            # Build optimal time windows based on patterns
            optimal_windows = await self._identify_optimal_windows(
                engagement_patterns, activity_patterns, timezone
            )
            
            # Identify times to avoid
            avoid_windows = await self._identify_avoid_windows(
                engagement_patterns, activity_patterns
            )
            
            # Determine notification frequency limits
            max_daily = self._calculate_max_daily_notifications(
                notification_prefs, activity_level
            )
            
            profile = UserTimingProfile(
                user_id=user_id,
                timezone=timezone,
                optimal_windows=optimal_windows,
                avoid_windows=avoid_windows,
                preferred_frequency=notification_prefs.get('frequency', 'daily'),
                max_daily_notifications=max_daily,
                activity_level=activity_level,
                sleep_schedule=self._infer_sleep_schedule(activity_patterns),
                work_schedule=self._infer_work_schedule(activity_patterns),
                weekend_preference=notification_prefs.get('weekendNotifications', True),
                last_updated=datetime.now()
            )
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Error building timing profile: {str(e)}")
            return await self._create_default_timing_profile(user_id)
    
    async def _calculate_optimal_time(
        self,
        timing_profile: UserTimingProfile,
        current_time: datetime,
        urgency: NotificationUrgency,
        notification_data: Dict[str, Any]
    ) -> Tuple[datetime, str, float]:
        """Calculate optimal delivery time for a notification."""
        try:
            # Handle immediate urgency
            if urgency == NotificationUrgency.IMMEDIATE:
                return current_time, "immediate_urgency", 1.0
            
            # Check if current time is in an optimal window
            if self._is_in_optimal_window(current_time, timing_profile):
                if urgency in [NotificationUrgency.HIGH, NotificationUrgency.MEDIUM]:
                    return current_time, "current_time_optimal", 0.9
            
            # Find next optimal window
            next_window = await self._find_next_optimal_window(
                current_time, timing_profile, urgency
            )
            
            if next_window:
                # Calculate specific time within the window
                optimal_time = await self._calculate_time_in_window(
                    next_window, timing_profile, notification_data
                )
                
                delay_hours = (optimal_time - current_time).total_seconds() / 3600
                
                # Check if delay is acceptable for urgency level
                max_delay = self._get_max_delay_hours(urgency)
                if delay_hours <= max_delay:
                    return optimal_time, f"next_optimal_window_{delay_hours:.1f}h", 0.8
            
            # Fallback based on urgency
            if urgency == NotificationUrgency.HIGH:
                # Send within 2 hours, even if not optimal
                fallback_time = current_time + timedelta(hours=2)
                return fallback_time, "urgency_override", 0.6
            elif urgency == NotificationUrgency.LOW:
                # Wait up to 24 hours for optimal window
                tomorrow_window = await self._find_next_day_optimal_window(
                    current_time, timing_profile
                )
                if tomorrow_window:
                    return tomorrow_window, "next_day_optimal", 0.7
            
            # Default fallback
            return current_time + timedelta(hours=1), "default_delay", 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal time: {str(e)}")
            return current_time, "calculation_error", 0.1
    
    async def _would_cause_fatigue(self, user_id: str, proposed_time: datetime) -> bool:
        """Check if sending notification at proposed time would cause fatigue."""
        try:
            # Get recent notifications for this user
            recent_notifications = await self._get_recent_notifications(user_id, hours=24)
            
            # Get user's fatigue thresholds
            timing_profile = await self.get_user_timing_profile(user_id)
            max_daily = timing_profile.max_daily_notifications
            
            # Count notifications in same day
            same_day_count = sum(
                1 for notif_time in recent_notifications
                if notif_time.date() == proposed_time.date()
            )
            
            if same_day_count >= max_daily:
                return True
            
            # Check for clustering (multiple notifications in short time)
            time_window = timedelta(hours=2)
            nearby_notifications = sum(
                1 for notif_time in recent_notifications
                if abs((notif_time - proposed_time).total_seconds()) < time_window.total_seconds()
            )
            
            if nearby_notifications >= 2:
                return True
            
            # Check for rapid succession
            last_notification = max(recent_notifications, default=None)
            if last_notification:
                time_since_last = proposed_time - last_notification
                if time_since_last < timedelta(minutes=30):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking notification fatigue: {str(e)}")
            return False  # Default to allowing notification
    
    async def _handle_notification_fatigue(
        self,
        timing_profile: UserTimingProfile,
        proposed_time: datetime,
        urgency: NotificationUrgency
    ) -> Tuple[datetime, str]:
        """Handle notification fatigue by rescheduling."""
        try:
            if urgency == NotificationUrgency.IMMEDIATE:
                # Must send immediately despite fatigue
                return proposed_time, "immediate_despite_fatigue"
            
            # Find next available slot
            current_check = proposed_time + timedelta(hours=2)
            
            for _ in range(48):  # Check up to 48 hours ahead
                if not await self._would_cause_fatigue(timing_profile.user_id, current_check):
                    if self._is_in_optimal_window(current_check, timing_profile):
                        return current_check, "rescheduled_for_fatigue"
                
                current_check += timedelta(hours=1)
            
            # If no good slot found, schedule for next day at first optimal window
            next_day = proposed_time + timedelta(days=1)
            first_window = min(timing_profile.optimal_windows, key=lambda w: w.start_time)
            
            optimal_next_day = next_day.replace(
                hour=first_window.start_time.hour,
                minute=first_window.start_time.minute,
                second=0,
                microsecond=0
            )
            
            return optimal_next_day, "next_day_due_to_fatigue"
            
        except Exception as e:
            self.logger.error(f"Error handling notification fatigue: {str(e)}")
            return proposed_time + timedelta(hours=4), "fatigue_error_delay"
    
    async def _analyze_engagement_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze user's historical notification engagement patterns."""
        try:
            # Query notification engagement from database
            query = """
            SELECT 
                n.sent_at,
                n.read_at,
                n.dismissed_at,
                n.acted_upon_at,
                n.type,
                n.priority,
                EXTRACT(HOUR FROM n.sent_at) as hour_sent,
                EXTRACT(DOW FROM n.sent_at) as day_of_week,
                CASE 
                    WHEN n.read_at IS NOT NULL THEN 1 
                    ELSE 0 
                END as was_read,
                CASE 
                    WHEN n.acted_upon_at IS NOT NULL THEN 1 
                    ELSE 0 
                END as was_acted_upon
            FROM notifications n
            WHERE n.user_id = %s 
                AND n.sent_at >= %s
                AND n.status = 'sent'
            ORDER BY n.sent_at DESC
            """
            
            thirty_days_ago = datetime.now() - timedelta(days=30)
            results = await self.db_connector.fetch_all(query, [user_id, thirty_days_ago])
            
            if not results:
                return {"total_notifications": 0, "engagement_rate": 0.0}
            
            # Analyze patterns
            total_notifications = len(results)
            read_notifications = sum(r['was_read'] for r in results)
            acted_notifications = sum(r['was_acted_upon'] for r in results)
            
            # Engagement by hour
            hourly_engagement = defaultdict(list)
            for result in results:
                hour = result['hour_sent']
                hourly_engagement[hour].append(result['was_read'])
            
            # Calculate hourly engagement rates
            hourly_rates = {}
            for hour, engagements in hourly_engagement.items():
                hourly_rates[hour] = sum(engagements) / len(engagements) if engagements else 0
            
            # Engagement by day of week
            daily_engagement = defaultdict(list)
            for result in results:
                day = result['day_of_week']
                daily_engagement[day].append(result['was_read'])
            
            daily_rates = {}
            for day, engagements in daily_engagement.items():
                daily_rates[day] = sum(engagements) / len(engagements) if engagements else 0
            
            return {
                "total_notifications": total_notifications,
                "engagement_rate": read_notifications / total_notifications,
                "action_rate": acted_notifications / total_notifications,
                "hourly_engagement": hourly_rates,
                "daily_engagement": daily_rates,
                "best_hours": sorted(hourly_rates.items(), key=lambda x: x[1], reverse=True)[:3],
                "best_days": sorted(daily_rates.items(), key=lambda x: x[1], reverse=True)[:3]
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing engagement patterns: {str(e)}")
            return {"total_notifications": 0, "engagement_rate": 0.0}
    
    async def _analyze_chat_activity_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze user's chat activity patterns to infer optimal timing."""
        try:
            # Query chat activity from both PostgreSQL and Firestore
            # PostgreSQL for historical queries
            pg_query = """
            SELECT 
                created_at,
                EXTRACT(HOUR FROM created_at) as hour_active,
                EXTRACT(DOW FROM created_at) as day_of_week
            FROM chat_queries
            WHERE user_id = %s 
                AND created_at >= %s
            ORDER BY created_at DESC
            """
            
            thirty_days_ago = datetime.now() - timedelta(days=30)
            chat_results = await self.db_connector.fetch_all(pg_query, [user_id, thirty_days_ago])
            
            if not chat_results:
                return {"activity_level": "low", "peak_hours": []}
            
            # Analyze activity patterns
            hourly_activity = Counter(r['hour_active'] for r in chat_results)
            daily_activity = Counter(r['day_of_week'] for r in chat_results)
            
            # Identify peak activity hours
            total_sessions = len(chat_results)
            peak_threshold = total_sessions * 0.1  # Hours with >10% of activity
            
            peak_hours = [
                hour for hour, count in hourly_activity.items()
                if count >= peak_threshold
            ]
            
            # Calculate activity level
            days_with_activity = len(set(r['created_at'].date() for r in chat_results))
            days_analyzed = min(30, (datetime.now() - thirty_days_ago).days)
            activity_frequency = days_with_activity / days_analyzed if days_analyzed > 0 else 0
            
            if activity_frequency > 0.8:
                activity_level = UserActivityLevel.HIGHLY_ACTIVE
            elif activity_frequency > 0.4:
                activity_level = UserActivityLevel.MODERATELY_ACTIVE
            elif activity_frequency > 0.1:
                activity_level = UserActivityLevel.OCCASIONALLY_ACTIVE
            else:
                activity_level = UserActivityLevel.INACTIVE
            
            return {
                "activity_level": activity_level,
                "total_sessions": total_sessions,
                "activity_frequency": activity_frequency,
                "peak_hours": sorted(peak_hours),
                "hourly_distribution": dict(hourly_activity),
                "daily_distribution": dict(daily_activity),
                "most_active_day": max(daily_activity.items(), key=lambda x: x[1])[0] if daily_activity else 1
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing chat activity: {str(e)}")
            return {"activity_level": UserActivityLevel.OCCASIONALLY_ACTIVE, "peak_hours": []}
    
    async def _identify_optimal_windows(
        self,
        engagement_patterns: Dict[str, Any],
        activity_patterns: Dict[str, Any],
        timezone: str
    ) -> List[OptimalTimeWindow]:
        """Identify optimal notification windows based on user patterns."""
        try:
            optimal_windows = []
            
            # Get best engagement hours
            best_hours = engagement_patterns.get('best_hours', [])
            peak_activity_hours = activity_patterns.get('peak_hours', [])
            
            # Combine engagement and activity data
            hour_scores = {}
            
            # Score based on engagement
            for hour, engagement_rate in best_hours:
                hour_scores[hour] = engagement_rate * 0.7  # 70% weight for engagement
            
            # Add activity score
            hourly_activity = activity_patterns.get('hourly_distribution', {})
            max_activity = max(hourly_activity.values()) if hourly_activity else 1
            
            for hour, activity_count in hourly_activity.items():
                activity_score = (activity_count / max_activity) * 0.3  # 30% weight for activity
                hour_scores[hour] = hour_scores.get(hour, 0) + activity_score
            
            # Create windows from high-scoring consecutive hours
            if hour_scores:
                sorted_hours = sorted(hour_scores.items(), key=lambda x: x[1], reverse=True)
                
                # Group consecutive high-scoring hours into windows
                processed_hours = set()
                
                for hour, score in sorted_hours:
                    if hour in processed_hours or score < 0.3:  # Minimum threshold
                        continue
                    
                    # Find consecutive hours around this peak
                    window_start = hour
                    window_end = hour + 1
                    
                    # Extend backwards
                    while (window_start - 1) in hour_scores and (window_start - 1) not in processed_hours:
                        if hour_scores[window_start - 1] >= score * 0.7:  # 70% of peak score
                            window_start -= 1
                        else:
                            break
                    
                    # Extend forwards
                    while (window_end % 24) in hour_scores and (window_end % 24) not in processed_hours:
                        if hour_scores[window_end % 24] >= score * 0.7:
                            window_end += 1
                        else:
                            break
                    
                    # Create window
                    if window_end - window_start >= 1:  # At least 1 hour window
                        optimal_windows.append(OptimalTimeWindow(
                            start_time=time(window_start, 0),
                            end_time=time(window_end % 24, 0),
                            timezone=timezone,
                            confidence_score=score,
                            priority_level=len(optimal_windows) + 1
                        ))
                        
                        # Mark hours as processed
                        for h in range(window_start, window_end):
                            processed_hours.add(h % 24)
                    
                    if len(optimal_windows) >= 4:  # Limit to 4 windows
                        break
            
            # Fallback to default windows if no patterns found
            if not optimal_windows:
                for window in self.default_optimal_windows:
                    optimal_windows.append(OptimalTimeWindow(
                        start_time=window.start_time,
                        end_time=window.end_time,
                        timezone=timezone,
                        confidence_score=0.5,
                        priority_level=window.priority_level
                    ))
            
            return optimal_windows
            
        except Exception as e:
            self.logger.error(f"Error identifying optimal windows: {str(e)}")
            return [OptimalTimeWindow(time(9, 0), time(10, 0), timezone, 0.5)]
    
    async def update_engagement_feedback(
        self,
        notification_id: str,
        user_id: str,
        engagement_type: str,
        timestamp: datetime
    ):
        """Update timing optimization based on engagement feedback."""
        try:
            # Record engagement in history
            if user_id not in self.engagement_history:
                self.engagement_history[user_id] = []
            
            self.engagement_history[user_id].append({
                "notification_id": notification_id,
                "engagement_type": engagement_type,  # read, dismissed, acted_upon
                "timestamp": timestamp,
                "hour": timestamp.hour,
                "day_of_week": timestamp.weekday()
            })
            
            # Trigger profile refresh for this user
            if user_id in self.user_timing_profiles:
                del self.user_timing_profiles[user_id]
            
            self.logger.info(f"Updated engagement feedback for user {user_id}: {engagement_type}")
            
        except Exception as e:
            self.logger.error(f"Error updating engagement feedback: {str(e)}")
    
    def get_notification_queue_status(self, user_id: str) -> Dict[str, Any]:
        """Get status of notification queue for a user."""
        try:
            queue = self.notification_queue.get(user_id, [])
            
            now = datetime.now()
            pending = [n for n in queue if n.optimized_time > now]
            overdue = [n for n in queue if n.optimized_time <= now]
            
            return {
                "user_id": user_id,
                "total_queued": len(queue),
                "pending": len(pending),
                "overdue": len(overdue),
                "next_notification": min([n.optimized_time for n in pending], default=None),
                "queue_health": "healthy" if len(overdue) == 0 else "needs_attention"
            }
            
        except Exception as e:
            self.logger.error(f"Error getting queue status: {str(e)}")
            return {"user_id": user_id, "error": str(e)}
    
    async def _get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences from database."""
        try:
            query = """
            SELECT 
                preferences,
                firebase_uid,
                monthly_income,
                risk_tolerance,
                spending_patterns
            FROM users 
            WHERE user_id = %s AND is_active = true
            """
            
            result = await self.db_connector.fetch_one(query, [user_id])
            
            if not result:
                # Create default user entry if doesn't exist
                return {
                    "preferences": {
                        "timezone": "UTC",
                        "notifications": {
                            "frequency": "daily",
                            "weekendNotifications": True,
                            "pushEnabled": True
                        }
                    }
                }
            
            return {
                "preferences": result.get('preferences', {}),
                "monthly_income": result.get('monthly_income', 0),
                "risk_tolerance": result.get('risk_tolerance', 'moderate'),
                "spending_patterns": result.get('spending_patterns', {})
            }
            
        except Exception as e:
            self.logger.error(f"Error getting user preferences: {str(e)}")
            return {"preferences": {"timezone": "UTC", "notifications": {"frequency": "daily"}}}
    
    def _determine_activity_level(
        self, 
        engagement_patterns: Dict[str, Any], 
        activity_patterns: Dict[str, Any]
    ) -> UserActivityLevel:
        """Determine user activity level based on patterns."""
        try:
            engagement_rate = engagement_patterns.get('engagement_rate', 0)
            activity_frequency = activity_patterns.get('activity_frequency', 0)
            total_sessions = activity_patterns.get('total_sessions', 0)
            
            # Weighted scoring
            engagement_weight = 0.4
            frequency_weight = 0.4
            volume_weight = 0.2
            
            # Normalize session count (assume 30+ sessions in 30 days is high)
            volume_score = min(total_sessions / 30, 1.0)
            
            combined_score = (
                engagement_rate * engagement_weight +
                activity_frequency * frequency_weight +
                volume_score * volume_weight
            )
            
            if combined_score >= 0.7:
                return UserActivityLevel.HIGHLY_ACTIVE
            elif combined_score >= 0.4:
                return UserActivityLevel.MODERATELY_ACTIVE
            elif combined_score >= 0.1:
                return UserActivityLevel.OCCASIONALLY_ACTIVE
            else:
                return UserActivityLevel.INACTIVE
                
        except Exception as e:
            self.logger.error(f"Error determining activity level: {str(e)}")
            return UserActivityLevel.MODERATELY_ACTIVE
    
    def _calculate_max_daily_notifications(
        self, 
        notification_prefs: Dict[str, Any], 
        activity_level: UserActivityLevel
    ) -> int:
        """Calculate maximum daily notifications based on preferences and activity."""
        try:
            # Base limits by activity level
            base_limits = {
                UserActivityLevel.HIGHLY_ACTIVE: 8,
                UserActivityLevel.MODERATELY_ACTIVE: 5,
                UserActivityLevel.OCCASIONALLY_ACTIVE: 3,
                UserActivityLevel.INACTIVE: 1
            }
            
            base_limit = base_limits.get(activity_level, 3)
            
            # Adjust based on user preferences
            frequency_pref = notification_prefs.get('frequency', 'daily')
            if frequency_pref == 'real_time':
                base_limit = int(base_limit * 1.5)
            elif frequency_pref == 'weekly':
                base_limit = max(1, int(base_limit * 0.3))
            
            # Respect explicit user settings if provided
            if 'maxDaily' in notification_prefs:
                user_max = notification_prefs['maxDaily']
                base_limit = min(base_limit, user_max)
            
            return max(1, base_limit)  # At least 1 notification allowed
            
        except Exception as e:
            self.logger.error(f"Error calculating max daily notifications: {str(e)}")
            return 3
    
    def _infer_sleep_schedule(self, activity_patterns: Dict[str, Any]) -> Optional[Tuple[time, time]]:
        """Infer sleep schedule from activity patterns."""
        try:
            hourly_activity = activity_patterns.get('hourly_distribution', {})
            
            if not hourly_activity:
                return None
            
            # Find hours with minimal activity (likely sleep hours)
            min_activity_threshold = max(hourly_activity.values()) * 0.1
            
            quiet_hours = [
                hour for hour, activity in hourly_activity.items()
                if activity <= min_activity_threshold
            ]
            
            if len(quiet_hours) < 4:  # Need at least 4 hours of sleep
                return None
            
            # Find the longest consecutive stretch of quiet hours
            quiet_hours.sort()
            longest_stretch = []
            current_stretch = [quiet_hours[0]]
            
            for i in range(1, len(quiet_hours)):
                if quiet_hours[i] == quiet_hours[i-1] + 1 or (quiet_hours[i-1] == 23 and quiet_hours[i] == 0):
                    current_stretch.append(quiet_hours[i])
                else:
                    if len(current_stretch) > len(longest_stretch):
                        longest_stretch = current_stretch
                    current_stretch = [quiet_hours[i]]
            
            if len(current_stretch) > len(longest_stretch):
                longest_stretch = current_stretch
            
            if len(longest_stretch) >= 4:
                sleep_start = time(min(longest_stretch), 0)
                sleep_end = time((max(longest_stretch) + 1) % 24, 0)
                return (sleep_start, sleep_end)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error inferring sleep schedule: {str(e)}")
            return None
    
    def _infer_work_schedule(self, activity_patterns: Dict[str, Any]) -> Optional[Tuple[time, time]]:
        """Infer work schedule from activity patterns."""
        try:
            hourly_activity = activity_patterns.get('hourly_distribution', {})
            
            if not hourly_activity:
                return None
            
            # Look for consistent high activity during typical work hours (8-18)
            work_hours_activity = {
                hour: activity for hour, activity in hourly_activity.items()
                if 8 <= hour <= 18
            }
            
            if not work_hours_activity:
                return None
            
            # Find peak activity hours during work period
            avg_work_activity = sum(work_hours_activity.values()) / len(work_hours_activity)
            peak_threshold = avg_work_activity * 0.8
            
            consistent_work_hours = [
                hour for hour, activity in work_hours_activity.items()
                if activity >= peak_threshold
            ]
            
            if len(consistent_work_hours) >= 6:  # At least 6 hours of consistent activity
                work_start = time(min(consistent_work_hours), 0)
                work_end = time(max(consistent_work_hours) + 1, 0)
                return (work_start, work_end)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error inferring work schedule: {str(e)}")
            return None
    
    async def _identify_avoid_windows(
        self,
        engagement_patterns: Dict[str, Any],
        activity_patterns: Dict[str, Any]
    ) -> List[Tuple[time, time]]:
        """Identify time windows to avoid for notifications."""
        try:
            avoid_windows = []
            
            # Get hourly engagement and activity data
            hourly_engagement = engagement_patterns.get('hourly_engagement', {})
            hourly_activity = activity_patterns.get('hourly_distribution', {})
            
            # Find hours with very low engagement rates
            if hourly_engagement:
                avg_engagement = sum(hourly_engagement.values()) / len(hourly_engagement)
                low_engagement_threshold = avg_engagement * 0.3
                
                low_engagement_hours = [
                    hour for hour, rate in hourly_engagement.items()
                    if rate <= low_engagement_threshold
                ]
                
                # Group consecutive low engagement hours
                if low_engagement_hours:
                    low_engagement_hours.sort()
                    current_window_start = low_engagement_hours[0]
                    current_window_end = low_engagement_hours[0]
                    
                    for hour in low_engagement_hours[1:]:
                        if hour == current_window_end + 1:
                            current_window_end = hour
                        else:
                            # End current window
                            if current_window_end - current_window_start >= 1:  # At least 2 hours
                                avoid_windows.append((
                                    time(current_window_start, 0),
                                    time((current_window_end + 1) % 24, 0)
                                ))
                            current_window_start = hour
                            current_window_end = hour
                    
                    # Handle last window
                    if current_window_end - current_window_start >= 1:
                        avoid_windows.append((
                            time(current_window_start, 0),
                            time((current_window_end + 1) % 24, 0)
                        ))
            
            # Add common avoid periods if no specific data
            if not avoid_windows:
                # Late night/early morning (1 AM - 6 AM)
                avoid_windows.append((time(1, 0), time(6, 0)))
                
                # Late dinner time (21:30 - 23:00)
                avoid_windows.append((time(21, 30), time(23, 0)))
            
            return avoid_windows
            
        except Exception as e:
            self.logger.error(f"Error identifying avoid windows: {str(e)}")
            return [(time(1, 0), time(6, 0))]  # Default avoid early morning
    
    async def _create_default_timing_profile(self, user_id: str) -> UserTimingProfile:
        """Create default timing profile for new users."""
        try:
            return UserTimingProfile(
                user_id=user_id,
                timezone="UTC",
                optimal_windows=self.default_optimal_windows.copy(),
                avoid_windows=[(time(1, 0), time(6, 0))],
                preferred_frequency="daily",
                max_daily_notifications=3,
                activity_level=UserActivityLevel.MODERATELY_ACTIVE,
                weekend_preference=True,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error creating default profile: {str(e)}")
            raise
    
    def _is_in_optimal_window(self, check_time: datetime, timing_profile: UserTimingProfile) -> bool:
        """Check if given time falls within any optimal window."""
        try:
            current_time = check_time.time()
            current_day = check_time.weekday()
            
            for window in timing_profile.optimal_windows:
                # Check day of week constraint
                if window.day_of_week is not None and window.day_of_week != current_day:
                    continue
                
                # Check time range
                if window.start_time <= window.end_time:
                    # Normal time range (e.g., 9:00 - 17:00)
                    if window.start_time <= current_time <= window.end_time:
                        return True
                else:
                    # Overnight range (e.g., 22:00 - 6:00)
                    if current_time >= window.start_time or current_time <= window.end_time:
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking optimal window: {str(e)}")
            return False
    
    async def _find_next_optimal_window(
        self,
        current_time: datetime,
        timing_profile: UserTimingProfile,
        urgency: NotificationUrgency
    ) -> Optional[datetime]:
        """Find the next optimal window for notification delivery."""
        try:
            # Sort windows by priority
            sorted_windows = sorted(timing_profile.optimal_windows, key=lambda w: w.priority_level)
            
            # Look ahead based on urgency
            max_days_ahead = {
                NotificationUrgency.HIGH: 1,
                NotificationUrgency.MEDIUM: 2,
                NotificationUrgency.LOW: 7,
                NotificationUrgency.BACKGROUND: 14
            }.get(urgency, 2)
            
            for days_ahead in range(max_days_ahead):
                check_date = current_time.date() + timedelta(days=days_ahead)
                check_day = check_date.weekday()
                
                # Skip weekends if user doesn't prefer them
                if not timing_profile.weekend_preference and check_day >= 5:
                    continue
                
                for window in sorted_windows:
                    # Check day of week constraint
                    if window.day_of_week is not None and window.day_of_week != check_day:
                        continue
                    
                    # Calculate window start time for this date
                    window_start = datetime.combine(check_date, window.start_time)
                    window_start = window_start.replace(tzinfo=current_time.tzinfo)
                    
                    # Skip if this window is in the past
                    if window_start <= current_time:
                        continue
                    
                    # Check if this time conflicts with avoid windows
                    if not self._conflicts_with_avoid_windows(window_start.time(), timing_profile):
                        return window_start
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding next optimal window: {str(e)}")
            return None
    
    def _conflicts_with_avoid_windows(self, check_time: time, timing_profile: UserTimingProfile) -> bool:
        """Check if time conflicts with avoid windows."""
        try:
            for avoid_start, avoid_end in timing_profile.avoid_windows:
                if avoid_start <= avoid_end:
                    # Normal time range
                    if avoid_start <= check_time <= avoid_end:
                        return True
                else:
                    # Overnight range
                    if check_time >= avoid_start or check_time <= avoid_end:
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking avoid windows: {str(e)}")
            return False
    
    async def _calculate_time_in_window(
        self,
        window_start: datetime,
        timing_profile: UserTimingProfile,
        notification_data: Dict[str, Any]
    ) -> datetime:
        """Calculate specific time within an optimal window."""
        try:
            # Find the matching window configuration
            matching_window = None
            window_time = window_start.time()
            
            for window in timing_profile.optimal_windows:
                if window.start_time <= window_time:
                    matching_window = window
                    break
            
            if not matching_window:
                return window_start
            
            # Calculate window duration
            window_end_dt = datetime.combine(window_start.date(), matching_window.end_time)
            if matching_window.end_time <= matching_window.start_time:
                # Overnight window
                window_end_dt += timedelta(days=1)
            
            window_duration = (window_end_dt - window_start).total_seconds()
            
            # Position within window based on notification priority and type
            notification_priority = notification_data.get('priority_score', 0.5)
            
            # High priority notifications go earlier in window
            position_factor = 1 - notification_priority  # 0.0 = start, 1.0 = end
            
            # Add some randomization to avoid clustering
            import random
            position_factor += random.uniform(-0.1, 0.1)
            position_factor = max(0, min(1, position_factor))
            
            # Calculate final time
            offset_seconds = window_duration * position_factor
            final_time = window_start + timedelta(seconds=offset_seconds)
            
            return final_time
            
        except Exception as e:
            self.logger.error(f"Error calculating time in window: {str(e)}")
            return window_start
    
    def _get_max_delay_hours(self, urgency: NotificationUrgency) -> float:
        """Get maximum delay hours for each urgency level."""
        return {
            NotificationUrgency.IMMEDIATE: 0,
            NotificationUrgency.HIGH: 2,
            NotificationUrgency.MEDIUM: 12,
            NotificationUrgency.LOW: 48,
            NotificationUrgency.BACKGROUND: 168  # 1 week
        }.get(urgency, 12)
    
    async def _find_next_day_optimal_window(
        self,
        current_time: datetime,
        timing_profile: UserTimingProfile
    ) -> Optional[datetime]:
        """Find optimal window in the next day."""
        try:
            tomorrow = current_time + timedelta(days=1)
            tomorrow_day = tomorrow.weekday()
            
            # Skip weekend if user doesn't prefer
            if not timing_profile.weekend_preference and tomorrow_day >= 5:
                # Find next weekday
                days_to_add = 1
                while (tomorrow + timedelta(days=days_to_add)).weekday() >= 5:
                    days_to_add += 1
                tomorrow = tomorrow + timedelta(days=days_to_add)
            
            # Get highest priority window
            best_window = min(timing_profile.optimal_windows, key=lambda w: w.priority_level)
            
            optimal_time = datetime.combine(tomorrow.date(), best_window.start_time)
            optimal_time = optimal_time.replace(tzinfo=current_time.tzinfo)
            
            return optimal_time
            
        except Exception as e:
            self.logger.error(f"Error finding next day window: {str(e)}")
            return None
    
    async def _get_recent_notifications(self, user_id: str, hours: int = 24) -> List[datetime]:
        """Get recent notification times for fatigue checking."""
        try:
            # Check in-memory queue first
            queue = self.notification_queue.get(user_id, [])
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            recent_times = [
                n.optimized_time for n in queue
                if n.optimized_time >= cutoff_time
            ]
            
            # Also check database for actually sent notifications
            query = """
            SELECT sent_at
            FROM notifications
            WHERE user_id = %s 
                AND sent_at >= %s
                AND status = 'sent'
            ORDER BY sent_at DESC
            """
            
            db_results = await self.db_connector.fetch_all(query, [user_id, cutoff_time])
            db_times = [r['sent_at'] for r in db_results]
            
            # Combine and deduplicate
            all_times = list(set(recent_times + db_times))
            return sorted(all_times)
            
        except Exception as e:
            self.logger.error(f"Error getting recent notifications: {str(e)}")
            return []
    
    async def _add_to_notification_queue(self, schedule: NotificationSchedule):
        """Add notification schedule to user's queue."""
        try:
            if schedule.user_id not in self.notification_queue:
                self.notification_queue[schedule.user_id] = []
            
            self.notification_queue[schedule.user_id].append(schedule)
            
            # Sort queue by optimized time
            self.notification_queue[schedule.user_id].sort(key=lambda x: x.optimized_time)
            
            # Limit queue size (remove oldest low-priority notifications if needed)
            max_queue_size = 50
            if len(self.notification_queue[schedule.user_id]) > max_queue_size:
                # Remove oldest low-priority notifications
                queue = self.notification_queue[schedule.user_id]
                queue.sort(key=lambda x: (x.urgency.value, x.optimized_time))
                self.notification_queue[schedule.user_id] = queue[-max_queue_size:]
            
        except Exception as e:
            self.logger.error(f"Error adding to notification queue: {str(e)}")
    
    def _urgency_priority(self, urgency: str) -> int:
        """Convert urgency to priority number for sorting."""
        priority_map = {
            'immediate': 4,
            'high': 3,
            'medium': 2,
            'low': 1,
            'background': 0
        }
        return priority_map.get(urgency.lower(), 2)
    
    async def _distribute_in_batch(
        self,
        base_time: datetime,
        timing_profile: UserTimingProfile,
        window_usage: Dict[str, int],
        urgency: NotificationUrgency
    ) -> datetime:
        """Distribute notifications in batch to avoid clustering."""
        try:
            window_key = self._get_time_window_key(base_time)
            current_usage = window_usage.get(window_key, 0)
            
            # If window is getting crowded, try to distribute
            if current_usage >= 2 and urgency not in [NotificationUrgency.IMMEDIATE, NotificationUrgency.HIGH]:
                # Try to find a nearby less crowded window
                alternatives = []
                
                for minutes_offset in [-30, 30, -60, 60, -90, 90]:
                    alt_time = base_time + timedelta(minutes=minutes_offset)
                    alt_key = self._get_time_window_key(alt_time)
                    alt_usage = window_usage.get(alt_key, 0)
                    
                    # Check if alternative time is still in optimal window
                    if (self._is_in_optimal_window(alt_time, timing_profile) and 
                        alt_usage < current_usage):
                        alternatives.append((alt_time, alt_usage))
                
                # Pick the least crowded alternative
                if alternatives:
                    alternatives.sort(key=lambda x: x[1])
                    return alternatives[0][0]
            
            return base_time
            
        except Exception as e:
            self.logger.error(f"Error distributing in batch: {str(e)}")
            return base_time
    
    def _get_time_window_key(self, dt: datetime) -> str:
        """Get time window key for tracking usage."""
        # Group into 30-minute windows
        window_minute = (dt.minute // 30) * 30
        return f"{dt.date()}_{dt.hour:02d}:{window_minute:02d}"
    
    async def cleanup_expired_schedules(self):
        """Clean up expired notification schedules."""
        try:
            current_time = datetime.now()
            cleaned_count = 0
            
            for user_id, queue in list(self.notification_queue.items()):
                # Remove schedules that are significantly overdue (>24 hours)
                cutoff_time = current_time - timedelta(hours=24)
                
                original_length = len(queue)
                self.notification_queue[user_id] = [
                    schedule for schedule in queue
                    if schedule.optimized_time > cutoff_time or 
                       schedule.urgency == NotificationUrgency.IMMEDIATE
                ]
                
                cleaned_count += original_length - len(self.notification_queue[user_id])
            
            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} expired notification schedules")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up expired schedules: {str(e)}")
    
    async def get_optimization_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get analytics on timing optimization performance."""
        try:
            # Get recent optimization data
            recent_schedules = []
            if user_id in self.notification_queue:
                recent_schedules = self.notification_queue[user_id][-20:]  # Last 20 notifications
            
            if not recent_schedules:
                return {"user_id": user_id, "insufficient_data": True}
            
            # Calculate metrics
            delays = []
            confidence_scores = []
            reschedule_counts = []
            
            for schedule in recent_schedules:
                delay_hours = (schedule.optimized_time - schedule.original_time).total_seconds() / 3600
                delays.append(delay_hours)
                confidence_scores.append(schedule.confidence_score)
                reschedule_counts.append(schedule.reschedule_count)
            
            # Calculate statistics
            avg_delay = statistics.mean(delays) if delays else 0
            avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0
            avg_reschedules = statistics.mean(reschedule_counts) if reschedule_counts else 0
            
            return {
                "user_id": user_id,
                "total_optimizations": len(recent_schedules),
                "average_delay_hours": round(avg_delay, 2),
                "average_confidence": round(avg_confidence, 3),
                "average_reschedules": round(avg_reschedules, 2),
                "optimization_quality": "good" if avg_confidence > 0.7 else "needs_improvement"
            }
            
        except Exception as e:
            self.logger.error(f"Error getting optimization analytics: {str(e)}")
            return {"user_id": user_id, "error": str(e)}