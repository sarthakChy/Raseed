import json
class ExplanationTool:
    def __init__(self, model):
        self.model = model

    def run(self, visualization_spec: dict, analysis_results: str, query: str, user_id: str) -> str:
        chart_type = visualization_spec.get("type", "table")
        fields = visualization_spec.get("fields", {})

        prompt = f"""
You're a data visualization assistant.

User ID: {user_id}
Query: "{query}"
Chart Type: {chart_type}
Fields: {json.dumps(fields, indent=2)}
Analysis Results:
{analysis_results}

Explain in 1-2 lines why this chart type is suitable for this data and query.
"""
        try:
            return self.model.generate_content(prompt).text.strip()
        except Exception:
            return "No explanation could be generated."
