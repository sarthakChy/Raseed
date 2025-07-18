import json
import re

def parse_json(raw_text):
    """
    Extracts and parses a JSON object from a Markdown code block (e.g., ```json ... ```).

    Args:
        raw_text (str): Raw text that includes a Markdown-style JSON block.

    Returns:
        dict: Parsed JSON data as a Python dictionary.

    Raises:
        ValueError: If no valid JSON block is found or JSON parsing fails.
    """
    if not isinstance(raw_text, str):
        raise ValueError("Input must be a string")

    # Match ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON block found in the input text")

    json_text = match.group(1).strip()

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")
