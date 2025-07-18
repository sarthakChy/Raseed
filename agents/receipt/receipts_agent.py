import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
from fastapi import (
    FastAPI, HTTPException, Query, Body, Request,
    File, UploadFile, Form, Depends, Header
)
from fastapi.responses import JSONResponse
import json
from agents.receipt.prompt import SYSTEM_INSTRUCTION, FEW_SHOT_EXAMPLES

class ReceiptAgent:
    """
        Initialize the Vertex AI agent for receipt analysis.

        Args:
            project_id: Google Cloud project ID
            location: Vertex AI location
            file_bytes: The actual bytes of the receipt image (read asynchronously by caller)
            file_content_type: The content type of the image file
    """
    def __init__(self, project_id: str, location: str, file_bytes: bytes, file_content_type: str, system_instruction:str):
        
        self.project_id = project_id
        self.location = location
        vertexai.init(project=project_id, location=location)
        
        self.model = GenerativeModel("gemini-2.0-flash",system_instruction=SYSTEM_INSTRUCTION,)

        self.conversation_history = []
        self.user_image_bytes = file_bytes
        self.file_content_type = file_content_type
        
    def analyze(self):
        if not self.file_content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File provided is not an image.")

        try:
            
            for example in FEW_SHOT_EXAMPLES:
                json_part = Part.from_text(json.dumps(example["expected_json"]))
                self.conversation_history.append(json_part)

            user_image_part = Part.from_data(data=self.user_image_bytes, mime_type=self.file_content_type)
            self.conversation_history.append(user_image_part)


            self.response = self.model.generate_content( # Use await here
                contents=self.conversation_history,
                generation_config=GenerationConfig(
                    temperature=0.3,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=2048,
                ),
            )

            if not self.response or not self.response.text:
                raise HTTPException(status_code=500, detail="No response from the AI model.")
            else:
                return self.response
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during receipt analysis: {str(e)}")