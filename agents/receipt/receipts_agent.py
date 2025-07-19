import json
import logging
from typing import List, Dict, Any, Union

import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig, HarmCategory, HarmBlockThreshold, Content
from agents.receipt.prompt import SYSTEM_INSTRUCTION, USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class ReceiptAgent:
    """
    Vertex AI agent for extracting structured data from receipt images.
    Assumes `vertexai.init()` has been called globally before instantiation.
    """
    def __init__(self, file_bytes: bytes, file_content_type: str):
        """
        Initializes the ReceiptAgent.

        Args:
            file_bytes: The actual bytes of the receipt image.
            file_content_type: The MIME type of the image file (e.g., "image/jpeg").
        """
        self.model = GenerativeModel("gemini-2.0-flash",system_instruction=SYSTEM_INSTRUCTION)

        self.user_image_bytes = file_bytes
        self.file_content_type = file_content_type
        
        logger.info(f"ReceiptAgent instantiated. Model: {self.model._model_name}.")

    def _build_contents(self) -> List[Content]:
        """
        Builds the 'contents' list for the Vertex AI generate_content method.
        """
        contents = [
            USER_PROMPT_TEMPLATE,
            Part.from_data(data=self.user_image_bytes, mime_type=self.file_content_type),
        ]

        return contents

    def analyze(self):
        """
        Analyzes the receipt image using the configured Gemini model.

        Returns:
            vertexai.generative_models.GenerationResponse: The response object from the AI model.

        Raises:
            HTTPException: If the file is not an image, or if the AI model
                           returns an invalid or empty response.
        """
        if not self.file_content_type.startswith("image/"):
            logger.error(f"File provided is not an image: {self.file_content_type}")

        try:
            contents = self._build_contents()
            logger.info("Contents built for generate_content call.")

            response = self.model.generate_content(
                contents=contents,
                generation_config=GenerationConfig(
                    temperature=0.3,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=2048,
                ),
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            logger.info("generate_content call completed.")

            if not response or not response.text:
                logger.error("No response or empty text received from the AI model.")
            else:
                return response
        except Exception as e:
            logger.exception(f"Error within ReceiptAgent.analyze: {str(e)}")