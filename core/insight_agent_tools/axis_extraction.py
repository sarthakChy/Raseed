import json

class AxisExtractionTool:
    def __init__(self, model):
        self.model = model

    def run(self, analysis_results: str, query: str, user_id: str) -> dict:
        prompt = f"""
You are a data visualization assistant.

Given this query and analysis:
Query: "{query}"
Results: {analysis_results}

Suggest appropriate X and Y axes for a chart.

Format:
{{
  "x_axis": "...",
  "y_axis": "..."
}}
"""
        try:
            response = self.model.generate_content(prompt).text
            return json.loads(response)
        except Exception:
            return {"x_axis": "", "y_axis": ""}
