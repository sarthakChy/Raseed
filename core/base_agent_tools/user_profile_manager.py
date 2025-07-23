import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from utils.google_services_utils import initialize_firestore
from google.cloud import firestore


class UserProfileManager:
    """
    Manages user profiles, preferences, financial goals, and personalization data
    stored in Firestore under users/{userId}.
    Used by all agents for contextual personalization and session continuity.
    """

    def __init__(self, project_id: Optional[str] = None):
        self.db = initialize_firestore()
        self.collection = self.db.collection("users")
        self.logger = logging.getLogger("user_profile_manager")

    def _convert_datetime_to_string(self, data: Any) -> Any:
        if isinstance(data, datetime):
            return data.isoformat()
        elif isinstance(data, dict):
            return {k: self._convert_datetime_to_string(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_datetime_to_string(i) for i in data]
        return data

    async def get_profile(self, user_id: str) -> Dict[str, Any]:
        """Get the full user profile stored under users/{userId}"""
        try:
            doc = self.collection.document(user_id).get()
            if doc.exists:
                return self._convert_datetime_to_string(doc.to_dict())
            else:
                return await self._create_default_profile(user_id)
        except Exception as e:
            self.logger.error(f"Failed to retrieve user profile: {e}")
            return {}

    async def update_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update fields in the user's profile document"""
        try:
            updates["lastUpdated"] = datetime.now()
            self.collection.document(user_id).set(updates, merge=True)
            return True
        except Exception as e:
            self.logger.error(f"Failed to update user profile: {e}")
            return False

    async def get_preferences(self, user_id: str) -> Dict[str, Any]:
        """Retrieve user's preferences section"""
        profile = await self.get_profile(user_id)
        return profile.get("preferences", {})

    async def update_preferences(self, user_id: str, prefs: Dict[str, Any]) -> bool:
        """Update user's preferences"""
        return await self.update_profile(user_id, {"preferences": prefs})

    async def get_financial_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user's financial profile (budget, goals, income, etc.)"""
        profile = await self.get_profile(user_id)
        return profile.get("financialProfile", {})

    async def update_financial_profile(self, user_id: str, data: Dict[str, Any]) -> bool:
        """Update user's financial profile"""
        return await self.update_profile(user_id, {"financialProfile": data})

    async def get_privacy_settings(self, user_id: str) -> Dict[str, Any]:
        """Get user's privacy settings"""
        prefs = await self.get_preferences(user_id)
        return prefs.get("privacySettings", {})

    async def update_privacy_settings(self, user_id: str, privacy: Dict[str, Any]) -> bool:
        """Update only the privacy settings"""
        prefs = await self.get_preferences(user_id)
        prefs["privacySettings"] = privacy
        return await self.update_preferences(user_id, prefs)

    async def record_interaction(self, user_id: str, interaction: Dict[str, Any]) -> bool:
        """
        Record a user interaction (agent, query, feedback, timestamp).
        Adds to 'interactionHistory' list under users/{userId}.
        """
        try:
            interaction["timestamp"] = datetime.now()
            doc_ref = self.collection.document(user_id)
            doc_ref.update({
                "interactionHistory": firestore.ArrayUnion([interaction]),
                "lastInteraction": datetime.now()
            })
            return True
        except Exception as e:
            self.logger.error(f"Failed to record interaction: {e}")
            return False

    async def get_recent_interactions(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch recent interactions from embedded array"""
        profile = await self.get_profile(user_id)
        history = profile.get("interactionHistory", [])
        history = sorted(history, key=lambda x: x.get("timestamp", ""), reverse=True)
        return history[:limit]

    async def get_personalization_context(self, user_id: str) -> Dict[str, Any]:
        """Returns the personalization-relevant fields from the user's profile"""
        profile = await self.get_profile(user_id)
        context = {
            "user_id": user_id,
            "preferences": profile.get("preferences", {}),
            "financialProfile": profile.get("financialProfile", {}),
            "privacySettings": profile.get("preferences", {}).get("privacySettings", {}),
            "lastInteraction": profile.get("lastInteraction", None),
            "recentInteractions": await self.get_recent_interactions(user_id, limit=5)
        }
        return context

    async def _create_default_profile(self, user_id: str) -> Dict[str, Any]:
        """Create a default user document"""
        now = datetime.now()
        default_doc = {
            "userId": user_id,
            "createdAt": now,
            "lastLoginAt": now,
            "preferences": {
                "currency": "USD",
                "language": "en",
                "timezone": "UTC",
                "notifications": {
                    "pushEnabled": True,
                    "emailEnabled": False,
                    "budgetAlerts": True,
                    "spendingInsights": True,
                    "weeklyReports": False,
                    "proactiveInsights": True,
                    "frequency": "weekly"
                },
                "privacySettings": {
                    "shareData": False,
                    "anonymousAnalytics": True
                }
            },
            "financialProfile": {
                "monthlyIncome": 0,
                "budgetLimits": {},
                "financialGoals": [],
                "riskTolerance": "moderate"
            },
            "interactionHistory": [],
            "lastInteraction": now
        }
        self.collection.document(user_id).set(default_doc)
        return self._convert_datetime_to_string(default_doc)