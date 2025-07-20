import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

from google.cloud import firestore


class UserProfileManager:
    """
    Manages user profiles, preferences, and interaction history.
    Provides personalization context for all agents.
    """
    
    def __init__(self, project_id: Optional[str] = None):
        """
        Initialize the User Profile Manager.
        
        Args:
            project_id: Google Cloud project ID for Firestore
        """
        self.project_id = project_id
        self.logger = logging.getLogger("financial_agent.user_profile")
        
        # Initialize Firestore client
        if project_id:
            self.db = firestore.Client(project=project_id)
        else:
            self.db = firestore.Client()
        
        # Collection references
        self.profiles_collection = self.db.collection('user_profiles')
        self.preferences_collection = self.db.collection('user_preferences')
        self.interaction_history_collection = self.db.collection('interaction_history')
    
    async def get_profile(
        self, 
        user_id: str, 
        sections: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve user profile information.
        
        Args:
            user_id: User identifier
            sections: Specific profile sections to retrieve
            
        Returns:
            Dictionary containing user profile data
        """
        try:
            # Get base profile
            profile_doc = self.profiles_collection.document(user_id).get()
            
            if not profile_doc.exists:
                # Create default profile for new user
                return await self._create_default_profile(user_id)
            
            profile_data = profile_doc.to_dict()
            
            # If specific sections requested, filter the data
            if sections:
                filtered_data = {}
                for section in sections:
                    if section in profile_data:
                        filtered_data[section] = profile_data[section]
                return filtered_data
            
            # Get preferences
            preferences = await self.get_preferences(user_id)
            profile_data['preferences'] = preferences
            
            # Add interaction context
            recent_interactions = await self.get_recent_interactions(user_id, limit=10)
            profile_data['recent_interactions'] = recent_interactions
            
            return profile_data
            
        except Exception as e:
            self.logger.error(f"Error retrieving profile for user {user_id}: {str(e)}")
            return await self._create_default_profile(user_id)
    
    async def update_profile(
        self, 
        user_id: str, 
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update user profile information.
        
        Args:
            user_id: User identifier
            updates: Dictionary of fields to update
            
        Returns:
            Updated profile data
        """
        try:
            # Add timestamp
            updates['last_updated'] = datetime.now()
            updates['updated_by'] = 'system'
            
            # Update the document
            self.profiles_collection.document(user_id).update(updates)
            
            self.logger.info(f"Updated profile for user {user_id}: {list(updates.keys())}")
            
            # Return updated profile
            return await self.get_profile(user_id)
            
        except Exception as e:
            self.logger.error(f"Error updating profile for user {user_id}: {str(e)}")
            raise
    
    async def get_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get user preferences and settings.
        
        Args:
            user_id: User identifier
            
        Returns:
            User preferences dictionary
        """
        try:
            prefs_doc = self.preferences_collection.document(user_id).get()
            
            if prefs_doc.exists:
                return prefs_doc.to_dict()
            else:
                # Return default preferences
                return self._get_default_preferences()
                
        except Exception as e:
            self.logger.error(f"Error retrieving preferences for user {user_id}: {str(e)}")
            return self._get_default_preferences()
    
    async def update_preferences(
        self, 
        user_id: str, 
        preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update user preferences.
        
        Args:
            user_id: User identifier
            preferences: New preferences to set
            
        Returns:
            Updated preferences
        """
        try:
            preferences['last_updated'] = datetime.now()
            
            # Merge with existing preferences
            existing_prefs = await self.get_preferences(user_id)
            existing_prefs.update(preferences)
            
            # Save to Firestore
            self.preferences_collection.document(user_id).set(existing_prefs)
            
            self.logger.info(f"Updated preferences for user {user_id}")
            return existing_prefs
            
        except Exception as e:
            self.logger.error(f"Error updating preferences for user {user_id}: {str(e)}")
            raise
    
    async def record_interaction(
        self, 
        user_id: str, 
        interaction_data: Dict[str, Any]
    ) -> str:
        """
        Record a user interaction for context and personalization.
        
        Args:
            user_id: User identifier
            interaction_data: Data about the interaction
            
        Returns:
            Interaction ID
        """
        try:
            interaction = {
                'user_id': user_id,
                'timestamp': datetime.now(),
                'agent_name': interaction_data.get('agent_name'),
                'query': interaction_data.get('query'),
                'intent': interaction_data.get('intent'),
                'response_type': interaction_data.get('response_type'),
                'satisfaction_score': interaction_data.get('satisfaction_score'),
                'context': interaction_data.get('context', {}),
                'metadata': interaction_data.get('metadata', {})
            }
            
            # Add to Firestore
            doc_ref = self.interaction_history_collection.add(interaction)
            interaction_id = doc_ref[1].id
            
            # Update user profile with latest interaction
            await self.update_profile(user_id, {
                'last_interaction': datetime.now(),
                'total_interactions': firestore.Increment(1)
            })
            
            return interaction_id
            
        except Exception as e:
            self.logger.error(f"Error recording interaction for user {user_id}: {str(e)}")
            raise
    
    async def get_recent_interactions(
        self, 
        user_id: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent user interactions for context.
        
        Args:
            user_id: User identifier
            limit: Maximum number of interactions to retrieve
            
        Returns:
            List of recent interactions
        """
        try:
            query = (self.interaction_history_collection
                    .where('user_id', '==', user_id)
                    .order_by('timestamp', direction=firestore.Query.DESCENDING)
                    .limit(limit))
            
            docs = query.stream()
            interactions = []
            
            for doc in docs:
                interaction_data = doc.to_dict()
                interaction_data['id'] = doc.id
                interactions.append(interaction_data)
            
            return interactions
            
        except Exception as e:
            self.logger.error(f"Error retrieving interactions for user {user_id}: {str(e)}")
            return []
    
    async def get_user_financial_goals(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get user's financial goals and objectives.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of financial goals
        """
        profile = await self.get_profile(user_id, sections=['financial_goals'])
        return profile.get('financial_goals', [])
    
    async def update_financial_goals(
        self, 
        user_id: str, 
        goals: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Update user's financial goals.
        
        Args:
            user_id: User identifier
            goals: List of financial goals
            
        Returns:
            Updated profile section
        """
        return await self.update_profile(user_id, {'financial_goals': goals})
    
    async def _create_default_profile(self, user_id: str) -> Dict[str, Any]:
        """Create a default profile for a new user."""
        default_profile = {
            'user_id': user_id,
            'created_at': datetime.now(),
            'last_updated': datetime.now(),
            'total_interactions': 0,
            'communication_style': 'balanced',  # casual, professional, balanced
            'detail_level': 'medium',  # low, medium, high
            'notification_preferences': {
                'enabled': True,
                'frequency': 'weekly',
                'categories': ['budgets', 'anomalies', 'insights']
            },
            'financial_goals': [],
            'spending_categories': {
                'priority_categories': [],
                'ignored_categories': []
            },
            'analysis_preferences': {
                'time_periods': ['monthly', 'quarterly'],
                'comparison_types': ['month_over_month', 'year_over_year'],
                'chart_preferences': ['trends', 'distributions']
            }
        }
        
        # Save to Firestore
        self.profiles_collection.document(user_id).set(default_profile)
        
        # Create default preferences
        default_preferences = self._get_default_preferences()
        await self.update_preferences(user_id, default_preferences)
        
        self.logger.info(f"Created default profile for new user: {user_id}")
        return default_profile
    
    def _get_default_preferences(self) -> Dict[str, Any]:
        """Get default user preferences."""
        return {
            'timezone': 'UTC',
            'currency': 'USD',
            'date_format': 'MM/DD/YYYY',
            'number_format': 'US',
            'language': 'en',
            'theme': 'light',
            'notifications': {
                'push_enabled': True,
                'email_enabled': True,
                'sms_enabled': False,
                'budget_alerts': True,
                'anomaly_alerts': True,
                'weekly_summary': True
            },
            'privacy': {
                'data_sharing': False,
                'analytics_tracking': True,
                'personalized_insights': True
            }
        }
    
    async def get_personalization_context(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive personalization context for agents.
        
        Args:
            user_id: User identifier
            
        Returns:
            Personalization context dictionary
        """
        profile = await self.get_profile(user_id)
        recent_interactions = await self.get_recent_interactions(user_id, limit=5)
        
        return {
            'user_profile': profile,
            'communication_style': profile.get('communication_style', 'balanced'),
            'detail_level': profile.get('detail_level', 'medium'),
            'financial_goals': profile.get('financial_goals', []),
            'preferences': profile.get('preferences', {}),
            'recent_context': [
                {
                    'query': interaction.get('query'),
                    'intent': interaction.get('intent'),
                    'timestamp': interaction.get('timestamp')
                }
                for interaction in recent_interactions
            ],
            'spending_patterns': profile.get('spending_categories', {}),
            'analysis_preferences': profile.get('analysis_preferences', {})
        }