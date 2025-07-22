class InsightSummaryTool:
    def __init__(self, model):
        self.model = model

    def run(self, analysis_results: str, query: str, user_id: str) -> str:
        prompt = f"""
You are a financial insights assistant. Your task is to analyze a structured summary of the user's spending data and generate an insightful, natural language report. This should help the user understand not just what they spent on, but also **why it matters**, **how it compares**, and **what patterns emerge**.

## User Info
- User ID: {user_id}
- Query: "{query}"

## Data Summary
{analysis_results}

## Instructions:
- Focus on top spending categories, major contributors, and percentages.
- Mention anything unusually high or low compared to other categories.
- Highlight spending habits or patterns (e.g. concentration, diversity, skew).
- Provide useful takeaways in a **clear, friendly tone**.
- Avoid repeating the query verbatim or rephrasing the structure; just deliver insights.
- Output 4-5 lines maximum.

Respond in plain English:
"""
        try:
            return self.model.generate_content(prompt).text.strip()
        except Exception:
            return "No insight summary could be generated."
