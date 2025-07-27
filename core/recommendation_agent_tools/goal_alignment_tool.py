import json
import logging
from typing import Dict, Any, Optional, List
from vertexai.generative_models import GenerativeModel
from core.recommendation_agent_tools.tools_instructions import goal_alignment_synthesis_instruction


class GoalAlignmentTool:
    """
    Tool for evaluating how financial recommendations align with user's stated goals.
    Provides alignment scoring and suggestions for improvement.
    """
    
    def __init__(self, project_id: str, logger: logging.Logger):
        """
        Initialize the Goal Alignment Tool.
        
        Args:
            project_id: Google Cloud project ID
            logger: Logger instance for this tool
        """
        self.project_id = project_id
        self.logger = logger
        self.model_name = "gemini-2.0-flash-001"
        
        # Initialize synthesis model with specific instructions
        self.synthesis_model = GenerativeModel(
            self.model_name,
            system_instruction=goal_alignment_synthesis_instruction
        )
        
        self.logger.info("GoalAlignmentTool initialized")
    
    async def align_with_goals(
        self,
        user_id: str,
        recommendation_details: str,
        impact_estimate: Optional[str] = None,
        relevant_goals: Optional[List[str]] = None,
        goal_timeframe: Optional[str] = None,
        user_profile_manager = None
    ) -> Dict[str, Any]:
        """
        Evaluate how a recommendation aligns with user's financial goals.
        
        Args:
            user_id: User identifier
            recommendation_details: Description of the recommendation to evaluate
            impact_estimate: Estimated financial impact
            relevant_goals: Specific goals to evaluate against
            goal_timeframe: Timeframe for the goals
            user_profile_manager: User profile manager (injected by agent)
            
        Returns:
            Dictionary containing goal alignment analysis
        """
        self.logger.info(f"Evaluating goal alignment for user: {user_id}")
        
        try:
            # Fetch user's financial goals from profile
            user_financial_goals = []
            user_profile_data = {}
            
            if user_profile_manager:
                try:
                    user_profile = await user_profile_manager.get_profile(
                        user_id, 
                        ["financial_goals", "preferences", "risk_tolerance"]
                    )
                    user_financial_goals = getattr(user_profile, "financial_goals", [])
                    user_profile_data = {
                        "financial_goals": user_financial_goals,
                        "preferences": getattr(user_profile, "preferences", {}),
                        "risk_tolerance": getattr(user_profile, "risk_tolerance", "medium")
                    }
                except Exception as e:
                    self.logger.warning(f"Could not fetch user profile: {e}")
            
            # Use provided goals or fallback to profile goals
            goals_to_evaluate = relevant_goals if relevant_goals else [
                goal.get('description', goal) if isinstance(goal, dict) else str(goal) 
                for goal in user_financial_goals
            ]
            
            if not goals_to_evaluate:
                # Provide default financial goals if none available
                goals_to_evaluate = [
                    "build emergency fund",
                    "reduce monthly expenses",
                    "increase savings rate",
                    "improve financial stability"
                ]
            
            # Prepare analysis data
            alignment_data = {
                "user_id": user_id,
                "recommendation": recommendation_details,
                "impact_estimate": impact_estimate,
                "relevant_goals": goals_to_evaluate,
                "goal_timeframe": goal_timeframe,
                "user_profile": user_profile_data
            }
            
            # Create synthesis prompt
            prompt = f"""
            Evaluate how this financial recommendation aligns with the user's goals:
            
            Recommendation Details:
            ```json
            {json.dumps({
                "description": recommendation_details,
                "estimated_impact": impact_estimate,
                "timeframe": goal_timeframe
            }, indent=2)}
            ```
            
            User's Financial Goals:
            ```json
            {json.dumps(goals_to_evaluate, indent=2)}
            ```
            
            User Profile Context:
            ```json
            {json.dumps(user_profile_data, indent=2)}
            ```
            
            Please evaluate the alignment and provide structured analysis following the system instructions.
            """

            # Generate synthesis
            response = await self.synthesis_model.generate_content_async(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )

            result = json.loads(response.text)
            
            # Enhance result with additional metadata
            result["success"] = True
            result["evaluation_metadata"] = {
                "goals_evaluated": len(goals_to_evaluate),
                "has_user_profile": bool(user_profile_data),
                "impact_quantified": bool(impact_estimate),
                "recommendation_complexity": self._assess_recommendation_complexity(recommendation_details)
            }
            
            return result

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error in goal alignment: {e}")
            return {
                "success": False,
                "error": str(e),
                "alignment_score": 0.0,
                "goal_contribution": "Error evaluating goal alignment",
                "alignment_suggestions": ["Analysis unavailable due to processing error"],
                "risk_assessment": "unknown"
            }
        except Exception as e:
            self.logger.error(f"Error in goal alignment analysis: {e}")
            return {
                "success": False,
                "error": str(e),
                "alignment_score": 0.0,
                "goal_contribution": "Error evaluating goal alignment",
                "alignment_suggestions": ["Analysis unavailable due to system error"],
                "risk_assessment": "unknown"
            }
    
    def _assess_recommendation_complexity(self, recommendation_details: str) -> str:
        """
        Assess the complexity of a recommendation based on its description.
        
        Args:
            recommendation_details: Description of the recommendation
            
        Returns:
            Complexity level (low/medium/high)
        """
        try:
            details_lower = recommendation_details.lower()
            
            # High complexity indicators
            high_complexity_terms = [
                'investment', 'portfolio', 'diversify', 'market', 'stocks', 'bonds',
                'refinance', 'mortgage', 'loan consolidation', 'tax strategy'
            ]
            
            # Medium complexity indicators
            medium_complexity_terms = [
                'budget reallocation', 'subscription audit', 'negotiate', 'switch providers',
                'automatic savings', 'debt payment strategy'
            ]
            
            # Low complexity indicators
            low_complexity_terms = [
                'reduce spending', 'cancel subscription', 'cook at home', 'use coupons',
                'waiting period', 'price comparison'
            ]
            
            if any(term in details_lower for term in high_complexity_terms):
                return "high"
            elif any(term in details_lower for term in medium_complexity_terms):
                return "medium"
            elif any(term in details_lower for term in low_complexity_terms):
                return "low"
            else:
                return "medium"  # Default
                
        except Exception as e:
            self.logger.error(f"Error assessing recommendation complexity: {e}")
            return "medium"
    
    def _calculate_goal_alignment_score(
        self, 
        recommendation: str, 
        goals: List[str], 
        impact_estimate: Optional[str] = None
    ) -> float:
        """
        Calculate a basic alignment score between recommendation and goals.
        
        Args:
            recommendation: Recommendation description
            goals: List of user goals
            impact_estimate: Estimated financial impact
            
        Returns:
            Alignment score between 0.0 and 1.0
        """
        try:
            if not goals:
                return 0.5  # Neutral score if no goals
            
            recommendation_lower = recommendation.lower()
            alignment_scores = []
            
            for goal in goals:
                goal_lower = str(goal).lower()
                
                # Calculate basic text similarity/relevance
                score = 0.0
                
                # Keywords that indicate financial improvement
                improvement_keywords = [
                    'save', 'reduce', 'optimize', 'improve', 'increase', 'build',
                    'pay off', 'eliminate', 'grow', 'invest'
                ]
                
                # Check for keyword overlap
                rec_words = set(recommendation_lower.split())
                goal_words = set(goal_lower.split())
                
                # Basic word overlap
                overlap = len(rec_words.intersection(goal_words))
                score += overlap * 0.1
                
                # Improvement action alignment
                if any(keyword in recommendation_lower for keyword in improvement_keywords):
                    if any(keyword in goal_lower for keyword in improvement_keywords):
                        score += 0.3
                
                # Category alignment (savings, debt, emergency fund, etc.)
                if 'save' in goal_lower and ('save' in recommendation_lower or 'reduce' in recommendation_lower):
                    score += 0.4
                if 'debt' in goal_lower and ('pay' in recommendation_lower or 'reduce' in recommendation_lower):
                    score += 0.4
                if 'emergency' in goal_lower and 'save' in recommendation_lower:
                    score += 0.4
                
                alignment_scores.append(min(1.0, score))
            
            # Return average alignment score
            return sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating alignment score: {e}")
            return 0.5
    
    def _generate_alignment_suggestions(
        self, 
        recommendation: str, 
        goals: List[str], 
        current_score: float
    ) -> List[str]:
        """
        Generate suggestions to improve goal alignment.
        
        Args:
            recommendation: Current recommendation
            goals: User's financial goals
            current_score: Current alignment score
            
        Returns:
            List of alignment improvement suggestions
        """
        suggestions = []
        
        try:
            if current_score < 0.3:
                suggestions.append("Consider how this recommendation specifically contributes to your stated financial goals")
                suggestions.append("Quantify the expected financial impact in terms of your goal targets")
            
            if current_score < 0.5:
                suggestions.append("Set specific milestones to track progress toward your goals")
                suggestions.append("Consider adjusting the timeline to better match your goal deadlines")
            
            if current_score < 0.7:
                suggestions.append("Evaluate if this recommendation should be prioritized over other goal-aligned actions")
                suggestions.append("Consider combining this with other strategies for maximum goal impact")
            
            # Goal-specific suggestions
            for goal in goals:
                goal_lower = str(goal).lower()
                if 'emergency' in goal_lower and current_score < 0.6:
                    suggestions.append("Prioritize liquid savings that can be accessed quickly for emergencies")
                elif 'debt' in goal_lower and current_score < 0.6:
                    suggestions.append("Focus on recommendations that free up cash flow for debt payments")
                elif 'retirement' in goal_lower and current_score < 0.6:
                    suggestions.append("Consider the long-term compound effect of this recommendation on retirement savings")
                    
            return list(set(suggestions))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"Error generating alignment suggestions: {e}")
            return ["Review how this recommendation supports your overall financial objectives"]