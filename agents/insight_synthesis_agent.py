from agents.base_agent import BaseAgent
from core.insights_agent_tools.insight_summary import InsightSummaryTool
from core.insights_agent_tools.visualisation import VisualisationTool
from core.insights_agent_tools.explanation import ExplanationTool
from vertexai.preview.generative_models import FunctionDeclaration
from typing import Dict, Any


class InsightSynthesisAgent(BaseAgent):
    def __init__(self, agent_name: str = "insight_synthesis_agent", project_id: str = "massive-incline-466204-t5",location: str = "us-central1", user_id: str = None, model=None):
        super().__init__(
            agent_name="insight_synthesis_agent",
            project_id=project_id,
            user_id=user_id,
            location = "us-central1",
            model_name=model
        )

        # Pass the shared model from BaseAgent to all tools
        self.insight_tool = InsightSummaryTool(model=self.model)
        self.visualisation_tool = VisualisationTool(model=self.model)
        self.explanation_tool = ExplanationTool(model=self.model)

        # Register tools with BaseAgent
        self.register_tool(
            tool_declaration=FunctionDeclaration(
                name="generate_insight_summary",
                description="Generate summarized insights from raw analysis results.",
                parameters={
                    "type": "object",
                    "properties": {
                        "analysis_results": {"type": "object"},
                        "original_query": {"type": "string"},
                        "user_id": {"type": "string"},
                        "recommendations": {"type" : "object"},
                    },
                    "required": ["analysis_results"]
                }
            ),
            executor_func=lambda analysis_results, recommendations, original_query="", user_id="" : self.insight_tool.run(
                analysis_results, original_query, user_id, recommendations
            )
        )

        self.register_tool(
            tool_declaration=FunctionDeclaration(
                name="suggest_chart_type_and_values",
                description="Suggest an appropriate chart type for the analysis results and appropriate requried values to visualise it.",
                parameters={
                    "type": "object",
                    "properties": {
                        "analysis_results": {"type": "object"},
                        "original_query": {"type": "string"},
                        "user_id": {"type": "string"}
                    },
                    "required": ["analysis_results"]
                }
            ),
            executor_func=lambda analysis_results, original_query="", user_id="": self.chart_tool.run(
                analysis_results, original_query, user_id
            )
        )

        self.register_tool(
            tool_declaration=FunctionDeclaration(
                name="generate_explanation",
                description="Generate a narrative explanation for the chart using results.",
                parameters={
                    "type": "object",
                    "properties": {
                        "chart_type": {"type": "string"},
                        "axes": {"type": "object"},
                        "analysis_results": {"type": "object"},
                        "original_query": {"type": "string"},
                        "user_id": {"type": "string"}
                    },
                    "required": ["chart_type", "axes", "analysis_results"]
                }
            ),
            executor_func=lambda chart_type, axes, analysis_results, original_query="", user_id="": self.explanation_tool.run(
                chart_type, axes, analysis_results, original_query, user_id
            )
        )

    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        analysis_results = request.get("analysis_results")
        original_query = request.get("original_query", "")
        user_id = request.get("user_id", "")
        recommendations = request.get("recommendations")

        # Step 1: Generate insight summary
        insight_summary = self.insight_tool.run(analysis_results, original_query, user_id, recommendations)

        # Step 2: Suggest chart type and values
        visualisation = self.visualisation_tool.run(analysis_results, original_query, user_id)

        # Step 3: Generate explanation for chart
        explanation = self.explanation_tool.run(visualisation, analysis_results, original_query, user_id)

        return {
            "insights": insight_summary,
            "visualization": visualisation,
            "explanation": explanation
        }
