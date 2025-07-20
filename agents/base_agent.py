import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json

from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration
import vertexai

from core.base_agent_tools.user_profile_manager import UserProfileManager
from core.base_agent_tools.error_handler import ErrorHandler
from core.base_agent_tools.integration_coordinator import IntegrationCoordinator
from core.base_agent_tools.database_connector import DatabaseConnector


class BaseAgent(ABC):
    """
    Base class for all financial analysis agents.
    Provides common functionality including Vertex AI setup, tool management,
    and shared operations.
    """
    
    def __init__(
        self,
        agent_name: str,
        project_id: str,
        location: str = "us-central1",
        model_name: str = "gemini-1.5-pro",
        user_id: Optional[str] = None
    ):
        """
        Initialize the base agent with common setup.
        
        Args:
            agent_name: Unique identifier for this agent
            project_id: Google Cloud project ID
            location: Vertex AI location
            model_name: Generative model to use
            user_id: Current user identifier
        """
        self.agent_name = agent_name
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.user_id = user_id
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # Initialize Vertex AI
        self._initialize_vertex_ai()
        
        # Initialize shared tools
        self.user_profile_manager = UserProfileManager()
        self.error_handler = ErrorHandler(self.logger)
        self.integration_coordinator = IntegrationCoordinator()
        self.db_connector = DatabaseConnector(project_id)
        
        # Registry for all tools
        self.tools_registry: Dict[str, Any] = {}
        self.vertex_tools: List[Tool] = []
        
        # Register base tools
        self._register_base_tools()
        
        # Initialize the generative model
        self.model = None
        self._initialize_model()
        
        self.logger.info(f"Initialized {self.agent_name} agent")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up structured logging for the agent."""
        logger = logging.getLogger(f"financial_agent.{self.agent_name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_vertex_ai(self):
        """Initialize Vertex AI client."""
        try:
            vertexai.init(project=self.project_id, location=self.location)
            self.logger.info("Vertex AI initialized successfully")
        except Exception as e:
            self.error_handler.handle_error(
                error=e,
                context=f"Failed to initialize Vertex AI for {self.agent_name}",
                user_id=self.user_id
            )
            raise
    
    def _register_base_tools(self):
        """Register tools that are common to all agents."""
        
        # User Profile Management Tool
        user_profile_tool = FunctionDeclaration(
            name="get_user_profile",
            description="Retrieve user profile information including preferences and settings",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier"
                    },
                    "profile_sections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific profile sections to retrieve (optional)"
                    }
                },
                "required": ["user_id"]
            }
        )
        
        # Database Query Tool
        database_tool = FunctionDeclaration(
            name="execute_database_query",
            description="Execute SQL queries against the financial database",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL query to execute"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Query parameters for parameterized queries"
                    },
                    "cache_key": {
                        "type": "string",
                        "description": "Optional cache key for query results"
                    }
                },
                "required": ["query"]
            }
        )
        
        # Error Logging Tool
        error_logging_tool = FunctionDeclaration(
            name="log_error",
            description="Log errors and system events for monitoring",
            parameters={
                "type": "object",
                "properties": {
                    "error_type": {
                        "type": "string",
                        "description": "Type of error (validation, processing, system)"
                    },
                    "message": {
                        "type": "string",
                        "description": "Error message"
                    },
                    "context": {
                        "type": "object",
                        "description": "Additional context about the error"
                    }
                },
                "required": ["error_type", "message"]
            }
        )
        
        # Create tools and add to registry
        base_tools = [user_profile_tool, database_tool, error_logging_tool]
        self.vertex_tools.extend(base_tools)
        
        # Register tool execution functions
        self.tools_registry.update({
            "get_user_profile": self._execute_get_user_profile,
            "execute_database_query": self._execute_database_query,
            "log_error": self._execute_log_error,
        })
    
    def _initialize_model(self):
        """Initialize the generative model with tools."""
        try:
            if self.vertex_tools:
                tools = [Tool(function_declarations=self.vertex_tools)]
                self.model = GenerativeModel(
                    model_name=self.model_name,
                    tools=tools
                )
            else:
                self.model = GenerativeModel(model_name=self.model_name)
                
            self.logger.info(f"Model initialized with {len(self.vertex_tools)} tools")
        except Exception as e:
            self.error_handler.handle_error(
                error=e,
                context=f"Failed to initialize model for {self.agent_name}",
                user_id=self.user_id
            )
            raise
    
    def register_tool(self, tool_declaration: FunctionDeclaration, executor_func):
        """
        Register a new tool with the agent.
        
        Args:
            tool_declaration: Vertex AI function declaration
            executor_func: Function to execute when tool is called
        """
        self.vertex_tools.append(tool_declaration)
        self.tools_registry[tool_declaration.name] = executor_func
        
        # Reinitialize model with updated tools
        self._initialize_model()
        
        self.logger.info(f"Registered tool: {tool_declaration.name}")
    
    async def execute_tool_call(self, function_call) -> Dict[str, Any]:
        """
        Execute a tool function call.
        
        Args:
            function_call: Function call from the model response
            
        Returns:
            Dictionary with execution results
        """
        function_name = function_call.name
        function_args = dict(function_call.args) if function_call.args else {}
        
        try:
            if function_name in self.tools_registry:
                executor = self.tools_registry[function_name]
                result = await executor(**function_args)
                
                self.logger.info(f"Executed tool: {function_name}")
                return {
                    "success": True,
                    "result": result,
                    "function_name": function_name
                }
            else:
                raise ValueError(f"Unknown function: {function_name}")
                
        except Exception as e:
            self.logger.error(f"Error executing {function_name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "function_name": function_name
            }
    
    # Base tool executors
    async def _execute_get_user_profile(self, user_id: str, profile_sections: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute user profile retrieval."""
        return await self.user_profile_manager.get_profile(user_id, profile_sections)
    
    async def _execute_database_query(self, query: str, parameters: Optional[Dict] = None, cache_key: Optional[str] = None) -> Dict[str, Any]:
        """Execute database query."""
        return await self.db_connector.execute_query(query, parameters, cache_key)
    
    async def _execute_log_error(self, error_type: str, message: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute error logging."""
        self.error_handler.log_error(error_type, message, context or {})
        return {"logged": True, "timestamp": datetime.now().isoformat()}
    
    @abstractmethod
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request using this agent's specialized capabilities.
        Must be implemented by each specialized agent.
        
        Args:
            request: Processing request with query and context
            
        Returns:
            Dictionary with processing results
        """
        pass
    
    async def generate_response(self, prompt: str, context: Optional[Dict] = None) -> str:
        """
        Generate a response using the model.
        
        Args:
            prompt: Input prompt for the model
            context: Additional context for the conversation
            
        Returns:
            Generated response text
        """
        try:
            # Add user context if available
            if context and self.user_id:
                user_profile = await self.user_profile_manager.get_profile(self.user_id)
                context.update({"user_profile": user_profile})
            
            # Generate response
            response = await self.model.generate_content_async(prompt)
            
            # Handle function calls if present
            if response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        tool_result = await self.execute_tool_call(part.function_call)
                        # Continue conversation with tool results if needed
                        
            return response.text if response.text else "No response generated"
            
        except Exception as e:
            self.error_handler.handle_error(
                error=e,
                context=f"Failed to generate response in {self.agent_name}",
                user_id=self.user_id
            )
            return "I apologize, but I encountered an error processing your request."
    
    def set_user_context(self, user_id: str):
        """Set the current user context for the agent."""
        self.user_id = user_id
        self.logger.info(f"User context set to: {user_id}")
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Return information about this agent."""
        return {
            "name": self.agent_name,
            "model": self.model_name,
            "tools_count": len(self.vertex_tools),
            "user_id": self.user_id,
            "status": "active"
        }