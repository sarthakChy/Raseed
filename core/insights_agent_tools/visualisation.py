import json
import re

class VisualisationTool:
    def __init__(self, model):
        self.model = model

    def run(self, analysis_results: str, query: str, user_id: str) -> dict:
        prompt = f"""
You are a financial visualization assistant.

Your job is to return a valid JSON object with:
1. the best chart type for the given analysis,
2. the required data fields for that chart,
3. a one-sentence caption.

ONLY respond with the JSON object. Do NOT explain anything. Do NOT wrap your output in triple backticks.

Supported chart types and their exact JSON formats are:

---

1. Bar chart:
{{
  "type": "bar_chart",
  "fields": {{
    "x_axis": ["<label1>", "<label2>", ...],
    "y_axis": [<number1>, <number2>, ...]
  }},
  "caption": "..."
}}

2. Line chart:
{{
  "type": "line_chart",
  "fields": {{
    "x_axis": ["<label1>", "<label2>", ...],
    "y_axis": [<number1>, <number2>, ...],
    "series": ["<series_name1>", "<series_name2>", ...]  // optional
  }},
  "caption": "..."
}}

3. Pie chart:
{{
  "type": "pie_chart",
  "fields": {{
    "labels": ["<label1>", "<label2>", ...],
    "values": [<number1>, <number2>, ...]
  }},
  "caption": "..."
}}

---

Do not include "table" or any other chart type. Your output must be valid JSON with no extra text.

Query: "{query}"
User ID: {user_id}

Analysis Results:
{analysis_results}
"""
        try:
            raw_response = self.model.generate_content(prompt).text.strip()

            # Clean markdown/code block wrappers if present
            cleaned = re.sub(r"^```(json)?\s*|```$", "", raw_response.strip(), flags=re.IGNORECASE).strip()
            parsed = json.loads(cleaned)

            valid_types = {"bar_chart", "line_chart", "pie_chart"}
            if (
                isinstance(parsed, dict)
                and parsed.get("type") in valid_types
                and isinstance(parsed.get("fields"), dict)
            ):
                return parsed

            raise ValueError(f"Invalid chart type or structure: {parsed}")

        except Exception as e:
            print("❌ ChartTypeTool failed to produce valid visualization JSON:", e)
            raise e  # Fail loudly so the caller can handle fallback
