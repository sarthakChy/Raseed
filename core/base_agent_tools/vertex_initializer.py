# core/base_agent_tools/vertex_initializer.py
import vertexai
import logging
from typing import Optional

class VertexAIInitializer:
    _initialized_projects = set()
    
    @classmethod
    def initialize(cls, project_id: str, location: str = "us-central1") -> None:
        """Initialize Vertex AI if not already done for this project."""
        project_key = f"{project_id}:{location}"
        
        if project_key not in cls._initialized_projects:
            try:
                vertexai.init(project=project_id, location=location)
                cls._initialized_projects.add(project_key)
                logging.getLogger("financial_agent").info(
                    f"Vertex AI initialized for project: {project_id}"
                )
            except Exception as e:
                logging.getLogger("financial_agent").error(
                    f"Failed to initialize Vertex AI: {e}"
                )
                raise