import json

# This is the main system instruction that will be passed to the Gemini model.
# It defines the agent's role, rules, and the required JSON schema.
SYSTEM_INSTRUCTION = """
You are an expert receipt processing agent. Your primary task is to analyze a receipt image and extract key information into a structured JSON format.

**Rules of Operation:**
1.  You MUST return a single, valid JSON object and nothing else. Do not wrap the JSON in markdown backticks or any other text.
2.  The JSON object must strictly adhere to the following schema:
    - `merchantName`: The name of the store or merchant (string).
    - `transactionDate`: The date of the transaction in YYYY-MM-DD format (string).
    - `items`: An array of objects, where each object has `name` (string) and `price` (number).
    - `total`: The final total amount of the receipt (number).
    - `tax`: The total tax amount, if available (number).
3.  If any value cannot be found in the receipt, its corresponding JSON value MUST be `null`.
4.  Item names should be cleaned, concise, and in uppercase if possible.

You will be shown a few examples of an image followed by the perfect JSON output. Learn from these examples, then analyze the final user-provided image and generate its corresponding JSON object.
"""


# This list holds the few-shot examples.
# You need to replace the placeholder strings with the actual base64 data
# from your sample images in Google Cloud Storage.
FEW_SHOT_EXAMPLES = [
    {
        #"image_base64": "PASTE_YOUR_BASE64_STRING_FOR_COFFEE_RECEIPT_HERE", 
        #"mime_type": "image/jpeg", # Or image/png, etc.
        "expected_json": {
            "merchantName": "The Coffee House",
            "transactionDate": "2025-07-17",
            "items": [
                {"name": "ESPRESSO", "price": 3.50},
                {"name": "MUFFIN", "price": 2.75}
            ],
            "total": 6.80,
            "tax": 0.55
        }
    },
    {
        #"image_base64": "PASTE_YOUR_BASE64_STRING_FOR_GROCERY_RECEIPT_HERE",
        #"mime_type": "image/png",
        "expected_json": {
            "merchantName": "Quick Mart",
            "transactionDate": "2025-07-16",
            "items": [
                {"name": "SODA 12PK", "price": 8.99},
                {"name": "CHIPS LRG", "price": 4.29},
                {"name": "ICE CREAM", "price": 5.49}
            ],
            "total": 18.77,
            "tax": 1.50
        }
    }
]
