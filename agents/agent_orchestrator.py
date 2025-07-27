import logging
import asyncio
import yaml
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
import uuid
from pydantic import BaseModel, Field
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration
import vertexai
from core.base_agent_tools.config_manager import AgentConfig
from core.base_agent_tools.vertex_initializer import VertexAIInitializer
from core.base_agent_tools.integration_coordinator import IntegrationCoordinator
from core.base_agent_tools.error_handler import ErrorHandler
from core.base_agent_tools.user_profile_manager import UserProfileManager
from agents.translate import TranslationService 


class IntentType(Enum):
    """Intent classification types for routing queries."""
    ANALYTICAL = "analytical"
    EXPLORATORY = "exploratory"
    ACTIONABLE = "actionable"
    COMPARATIVE = "comparative"
    PREDICTIVE = "predictive"
    UNKNOWN = "unknown"


class WorkflowStatus(Enum):
    """Status of workflow execution."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL_SUCCESS = "partial_success"


@dataclass
class AgentDefinition:
    """Configuration for an agent."""
    name: str
    class_path: str
    description: str
    capabilities: List[str]
    required_for_intents: List[str]
    optional_for_intents: List[str]
    retry_count: int = 1
    timeout_seconds: int = 30


@dataclass
class WorkflowStep:
    """Individual step in a workflow."""
    agent_name: str
    step_id: str
    depends_on: List[str]  # List of step_ids this step depends on
    input_mapping: Dict[str, str]  # How to map previous outputs to this step's input
    required: bool = True
    retry_count: int = 1


@dataclass
class WorkflowDefinition:
    """Complete workflow definition."""
    intent_type: IntentType
    steps: List[WorkflowStep]
    aggregation_agent: Optional[str] = "synthesis_agent"
    description: str = ""


@dataclass
class StepResult:
    """Result from executing a workflow step."""
    step_id: str
    agent_name: str
    success: bool
    result: Any = None
    error: str = ""
    execution_time: float = 0.0
    retry_attempt: int = 0


@dataclass
class WorkflowResult:
    """Final result from workflow execution."""
    workflow_id: str
    intent_type: IntentType
    status: WorkflowStatus
    results: Dict[str, StepResult]
    final_response: str = ""
    execution_time: float = 0.0
    error_summary: str = ""


class StepResultResponse(BaseModel):
    step_id: str
    agent_name: str
    success: bool
    result: Optional[Any] = None
    error: str = ""
    execution_time: float = 0.0
    retry_attempt: int = 0


class OrchestratorResponse(BaseModel):
    success: bool
    workflow_id: Optional[str] = None
    intent: Optional[str] = None
    status: Optional[str] = None
    response: str = ""
    execution_time: float = 0.0
    step_results: Dict[str, Dict[str, Any]] = {}
    error_summary: str = ""
    error: Optional[str] = None
    query: Optional[str] = None
    # Add translation metadata
    original_language: Optional[str] = None
    translation_confidence: Optional[float] = None
    was_translated: bool = False

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            uuid.UUID: lambda u: str(u)
        }


class MasterOrchestrator:
    def __init__(
        self,
        project_id: Optional[str] = None,
        config_path: str = "config/agent_config.yaml",
        location: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        # Load system configuration
        self.system_config = AgentConfig.from_env()
        
        # Use provided values or fall back to system config defaults
        self.project_id = project_id or self.system_config.project_id
        self.location = location or self.system_config.location
        self.model_name = model_name or self.system_config.model_name
        self.logger = self._setup_logging()
        
        # Initialize shared tools
        self.integration_coordinator = IntegrationCoordinator()
        self.error_handler = ErrorHandler(self.logger)
        self.user_profile_manager = UserProfileManager()
        
        # Initialize translation service
        try:
            self.translation_service = TranslationService(project_id=self.project_id)
            self.logger.info("Translation service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize translation service: {e}")
            self.translation_service = None

        VertexAIInitializer.initialize(self.project_id, self.location)
        
        # Load configurations - Initialize these before loading
        self.agent_configs: Dict[str, AgentConfig] = {}
        self.agent_definitions: Dict[str, AgentDefinition] = {}
        self.workflow_definitions: Dict[IntentType, WorkflowDefinition] = {}
        self._load_configurations(config_path)
        
        # Registry of loaded agents
        self.agents_registry: Dict[str, Any] = {}
        
        # Initialize intent classification model
        self.intent_model = GenerativeModel(model_name=self.model_name)
        
        self.logger.info("Master Orchestrator initialized successfully")

    def _setup_logging(self) -> logging.Logger:
        """Set up structured logging."""
        logger = logging.getLogger("financial_agent.orchestrator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_configurations(self, config_path: str):
        """Load agent and workflow configurations."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        
            # Load agent definitions
            for agent_data in config.get('agents', []):
                agent_def = AgentDefinition(**agent_data)
                self.agent_definitions[agent_def.name] = agent_def
            
            # Load workflow definitions
            for workflow_data in config.get('workflows', []):
                intent_type = IntentType(workflow_data['intent_type'])
                steps = [WorkflowStep(**step) for step in workflow_data['steps']]
                workflow = WorkflowDefinition(
                    intent_type=intent_type,
                    steps=steps,
                    aggregation_agent=workflow_data.get('aggregation_agent', 'synthesis_agent'),
                    description=workflow_data.get('description', '')
                )
                self.workflow_definitions[intent_type] = workflow
            
            self.logger.info(f"Loaded {len(self.agent_definitions)} agents and {len(self.workflow_definitions)} workflows")
            
        except Exception as e:
            self.logger.error(f"Failed to load configurations: {e}")
            raise

    async def translate_query_to_english(self, query: str) -> Dict[str, Any]:
        """
        Translate user query to English if needed.
        
        Args:
            query: User's query in any language
            
        Returns:
            Dictionary with translation result and metadata
        """
        if not self.translation_service:
            return {
                'success': False,
                'translated_text': query,
                'original_text': query,
                'source_language': 'unknown',
                'was_translated': False,
                'error': 'Translation service not available'
            }
        
        try:
            # First detect if it's already English
            if self.translation_service.is_english(query):
                return {
                    'success': True,
                    'translated_text': query,
                    'original_text': query,
                    'source_language': 'en',
                    'was_translated': False,
                    'detection_confidence': 1.0
                }
            
            # Translate to English
            result = self.translation_service.translate_to_english(query)
            return result
            
        except Exception as e:
            self.logger.error(f"Query translation failed: {e}")
            return {
                'success': False,
                'translated_text': query,
                'original_text': query,
                'source_language': 'unknown',
                'was_translated': False,
                'error': str(e)
            }

    async def translate_response_to_user_language(self, response: str, target_language: str) -> Dict[str, Any]:
        """
        Translate response back to user's language.
        
        Args:
            response: English response text
            target_language: Target language code
            
        Returns:
            Dictionary with translation result
        """
        if not self.translation_service or target_language == 'en':
            return {
                'success': True,
                'translated_text': response,
                'original_text': response,
                'target_language': target_language or 'en',
                'was_translated': False
            }
        
        try:
            result = self.translation_service.translate_from_english(response, target_language)
            return result
            
        except Exception as e:
            self.logger.error(f"Response translation failed: {e}")
            return {
                'success': False,
                'translated_text': response,
                'original_text': response,
                'target_language': target_language,
                'was_translated': False,
                'error': str(e)
            }
    
    async def classify_intent(self, query: str) -> IntentType:
        """
        Classify user query intent using Vertex AI.
        
        Args:
            query: User's natural language query (should be in English)
            
        Returns:
            Classified intent type
        """
        try:
            prompt = f"""
Classify the following financial query into one of these intent types:

1. ANALYTICAL: "How much did I spend?" - Queries requesting specific calculations, amounts, or data analysis
2. EXPLORATORY: "What are my spending patterns?" - Queries exploring trends, patterns, or general insights
3. ACTIONABLE: "Help me save money" - Queries seeking recommendations, advice, or actionable steps
4. COMPARATIVE: "Compare this month vs last" - Queries comparing different time periods, categories, or metrics
5. PREDICTIVE: "Will I exceed my budget?" - Queries about future projections, forecasts, or predictions

Query: "{query}"

Respond with just the intent type in uppercase (ANALYTICAL, EXPLORATORY, ACTIONABLE, COMPARATIVE, or PREDICTIVE).
If the query doesn't clearly fit any category, respond with UNKNOWN.
"""
            
            response = await self.intent_model.generate_content_async(prompt)
            intent_text = response.text.strip().upper()
            
            try:
                return IntentType(intent_text.lower())
            except ValueError:
                self.logger.warning(f"Unknown intent classification: {intent_text}")
                return IntentType.UNKNOWN
                
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return IntentType.UNKNOWN
    
    async def load_agent(self, agent_name: str) -> Any:
        """Dynamically load an agent if not already loaded."""
        if agent_name in self.agents_registry:
            return self.agents_registry[agent_name]
        
        if agent_name not in self.agent_definitions:
            raise ValueError(f"Agent configuration not found: {agent_name}")
        
        try:
            agent_def = self.agent_definitions[agent_name]
            
            # Dynamic import of agent class
            module_path, class_name = agent_def.class_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            agent_class = getattr(module, class_name)
            
            # Initialize agent with system config
            agent_instance = agent_class(
                agent_name=agent_name,
                project_id=self.project_id,
                location=self.location
            )
            
            # Register with integration coordinator
            self.integration_coordinator.register_agent(agent_name, agent_instance)
            self.agents_registry[agent_name] = agent_instance
            
            self.logger.info(f"Loaded agent: {agent_name}")
            return agent_instance
            
        except Exception as e:
            self.logger.error(f"Failed to load agent {agent_name}: {e}")
            raise
    
    async def execute_workflow_step(
        self,
        step: WorkflowStep,
        workflow_context: Dict[str, Any],
        previous_results: Dict[str, StepResult]
    ) -> StepResult:
        """
        Execute a single workflow step.
        
        Args:
            step: Step to execute
            workflow_context: Overall workflow context
            previous_results: Results from previous steps
            
        Returns:
            Step execution result
        """
        start_time = datetime.now()
        
        try:
            # Load the agent
            agent = await self.load_agent(step.agent_name)
            
            # Prepare input for the step
            step_input = {"query": workflow_context.get("translated_query", workflow_context.get("original_query", ""))}
            
            # Map outputs from previous steps
            for input_key, source_path in step.input_mapping.items():
                if '.' in source_path:
                    step_id, result_key = source_path.split('.', 1)
                    if step_id in previous_results and previous_results[step_id].success:
                        if hasattr(previous_results[step_id].result, result_key):
                            step_input[input_key] = getattr(previous_results[step_id].result, result_key)
                        elif isinstance(previous_results[step_id].result, dict):
                            step_input[input_key] = previous_results[step_id].result.get(result_key)
                else:
                    # Direct mapping from workflow context
                    step_input[input_key] = workflow_context.get(source_path)
            
            # Add user context
            if workflow_context.get("user_id"):
                step_input["user_id"] = workflow_context["user_id"]
            
            # Execute the agent
            result = await agent.process(step_input)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return StepResult(
                step_id=step.step_id,
                agent_name=step.agent_name,
                success=True,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_message = str(e)
            
            self.logger.error(f"Step {step.step_id} failed: {error_message}")
            
            return StepResult(
                step_id=step.step_id,
                agent_name=step.agent_name,
                success=False,
                error=error_message,
                execution_time=execution_time
            )
    
    async def execute_workflow(
        self,
        workflow: WorkflowDefinition,
        context: Dict[str, Any]
    ) -> WorkflowResult:
        """
        Execute a complete workflow.
        
        Args:
            workflow: Workflow definition to execute
            context: Execution context
            
        Returns:
            Workflow execution result
        """
        workflow_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # Store workflow state
        workflow_state = {
            "id": workflow_id,
            "status": WorkflowStatus.IN_PROGRESS.value,
            "context": context,
            "steps": [asdict(step) for step in workflow.steps]
        }
        self.integration_coordinator.update_shared_state({f"workflow_{workflow_id}": workflow_state})
        
        step_results: Dict[str, StepResult] = {}
        completed_steps: List[str] = []
        
        try:
            # Execute steps based on dependencies
            remaining_steps = workflow.steps.copy()
            
            while remaining_steps:
                # Find steps that can be executed (dependencies met)
                executable_steps = []
                for step in remaining_steps:
                    if all(dep in completed_steps for dep in step.depends_on):
                        executable_steps.append(step)
                
                if not executable_steps:
                    raise Exception("Circular dependency detected in workflow")
                
                # Execute steps (sequential for now, parallel later)
                for step in executable_steps:
                    retry_count = 0
                    step_result = None
                    
                    # Retry logic
                    while retry_count <= step.retry_count:
                        step_result = await self.execute_workflow_step(step, context, step_results)
                        step_result.retry_attempt = retry_count
                        
                        if step_result.success:
                            break
                        
                        retry_count += 1
                        if retry_count <= step.retry_count:
                            self.logger.info(f"Retrying step {step.step_id}, attempt {retry_count}")
                    
                    step_results[step.step_id] = step_result
                    
                    # Check if required step failed
                    if not step_result.success and step.required:
                        # Continue with partial results but mark as partial success
                        pass
                    
                    completed_steps.append(step.step_id)
                    remaining_steps.remove(step)
            
            # Determine workflow status
            failed_required_steps = [
                result for result in step_results.values()
                if not result.success and any(s.required for s in workflow.steps if s.step_id == result.step_id)
            ]
            
            if failed_required_steps:
                status = WorkflowStatus.PARTIAL_SUCCESS
            else:
                status = WorkflowStatus.COMPLETED

            # Aggregate results using synthesis agent if specified
            final_response = ""
            if workflow.aggregation_agent and workflow.aggregation_agent in self.agent_definitions:
                try:
                    synthesis_agent = await self.load_agent(workflow.aggregation_agent)
                    
                    # Prepare synthesis input with successful results
                    successful_results = {}
                    for step_id, result in step_results.items():
                        if result.success and result.result:
                            # Check if result has analysis or response field
                            if isinstance(result.result, dict):
                                if "analysis" in result.result:
                                    successful_results[step_id] = result.result["analysis"]
                                elif "response" in result.result:
                                    successful_results[step_id] = result.result["response"]
                                else:
                                    successful_results[step_id] = result.result
                            else:
                                successful_results[step_id] = result.result
                    
                    synthesis_input = {
                        "step_results": successful_results,
                        "original_query": context.get("original_query", ""),
                        "user_id": context.get("user_id"),
                        "workflow_context": context
                    }
                    
                    synthesis_result = await synthesis_agent.process(synthesis_input)
                    
                    # Extract response from synthesis result
                    if isinstance(synthesis_result, dict):
                        final_response = (
                            synthesis_result.get("response", "") or 
                            synthesis_result.get("analysis", "") or 
                            synthesis_result.get("synthesis", "") or
                            str(synthesis_result)
                        )
                    else:
                        final_response = str(synthesis_result)
                        
                except Exception as e:
                    self.logger.error(f"Synthesis failed: {e}")
                    # Fallback: use the last successful step result
                    for step_id, result in reversed(list(step_results.items())):
                        if result.success and result.result:
                            if isinstance(result.result, dict):
                                final_response = (
                                    result.result.get("analysis", "") or
                                    result.result.get("response", "") or
                                    "Analysis completed successfully."
                                )
                                break
                    
                    if not final_response:
                        final_response = "Results processed, but final synthesis failed."

            # If no aggregation agent, try to extract response from the last successful step
            elif step_results:
                for step_id, result in reversed(list(step_results.items())):
                    if result.success and result.result:
                        if isinstance(result.result, dict):
                            final_response = (
                                result.result.get("analysis", "") or
                                result.result.get("response", "") or
                                "Analysis completed successfully."
                            )
                            break
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update workflow state to completed
            workflow_state["status"] = status.value
            self.integration_coordinator.update_shared_state({f"workflow_{workflow_id}": workflow_state})
            
            return WorkflowResult(
                workflow_id=workflow_id,
                intent_type=workflow.intent_type,
                status=status,
                results=step_results,
                final_response=final_response,
                execution_time=execution_time,
                error_summary=""
            )
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_message = str(e)
            
            # Update workflow state
            workflow_state["status"] = WorkflowStatus.FAILED.value
            self.integration_coordinator.update_shared_state({f"workflow_{workflow_id}": workflow_state})
            
            return WorkflowResult(
                workflow_id=workflow_id,
                intent_type=workflow.intent_type,
                status=WorkflowStatus.FAILED,
                results=step_results,
                error_summary=error_message,
                execution_time=execution_time
            )
    
    async def process_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        additional_context: Optional[Dict] = None,
        preserve_user_language: bool = True
    ) -> OrchestratorResponse:
        """
        Process a user query through the complete orchestration pipeline with multilingual support.
        
        Args:
            query: User's natural language query in any language
            user_id: User identifier
            additional_context: Additional context for processing
            preserve_user_language: Whether to translate response back to user's language
            
        Returns:
            Complete response with results and metadata
        """
        start_time = datetime.now()
        original_query = query
        translation_metadata = {}
        
        try:
            # Step 1: Translate query to English if needed
            translation_result = await self.translate_query_to_english(query)
            
            if not translation_result['success']:
                self.logger.warning(f"Query translation failed: {translation_result.get('error', 'Unknown error')}")
                # Continue with original query if translation fails
                english_query = query
                translation_metadata = {
                    'original_language': 'unknown',
                    'translation_confidence': 0.0,
                    'was_translated': False
                }
            else:
                english_query = translation_result['translated_text']
                translation_metadata = {
                    'original_language': translation_result.get('source_language', 'unknown'),
                    'translation_confidence': translation_result.get('detection_confidence', translation_result.get('confidence', 0.0)),
                    'was_translated': translation_result.get('was_translated', False)
                }
            
            self.logger.info(f"Query translation: {translation_metadata}")
            
            # Step 2: Classify intent (using English query)
            intent = await self.classify_intent(english_query)
            self.logger.info(f"Classified intent: {intent.value} for query: {english_query[:50]}...")
            
            # Step 3: Check if we have a workflow for this intent
            if intent not in self.workflow_definitions:
                response_text = f"No workflow defined for intent: {intent.value}"
                
                # Translate error message back to user's language if needed
                if (preserve_user_language and 
                    translation_metadata.get('was_translated') and 
                    translation_metadata.get('original_language') != 'en'):
                    
                    error_translation = await self.translate_response_to_user_language(
                        response_text, 
                        translation_metadata['original_language']
                    )
                    if error_translation['success']:
                        response_text = error_translation['translated_text']
                
                return OrchestratorResponse(
                    success=False,
                    error=response_text,
                    intent=intent.value,
                    **translation_metadata
                )
            
            # Step 4: Prepare workflow context
            workflow_context = {
                "original_query": original_query,
                "translated_query": english_query,
                "user_id": user_id,
                "intent": intent.value,
                "timestamp": datetime.now().isoformat(),
                "translation_metadata": translation_metadata
            }
            
            if additional_context:
                workflow_context.update(additional_context)
            
            # Step 5: Execute workflow
            workflow = self.workflow_definitions[intent]
            result = await self.execute_workflow(workflow, workflow_context)
            
            # Step 6: Translate response back to user's language if needed
            final_response = result.final_response
            if (preserve_user_language and 
                translation_metadata.get('was_translated') and 
                translation_metadata.get('original_language') != 'en' and
                final_response):
                
                response_translation = await self.translate_response_to_user_language(
                    final_response, 
                    translation_metadata['original_language']
                )
                
                if response_translation['success']:
                    final_response = response_translation['translated_text']
                    self.logger.info(f"Response translated back to {translation_metadata['original_language']}")
                else:
                    self.logger.warning(f"Failed to translate response back to user language: {response_translation.get('error')}")
            
            # Step 7: Return comprehensive response
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return OrchestratorResponse(
                success=result.status in [WorkflowStatus.COMPLETED, WorkflowStatus.PARTIAL_SUCCESS],
                workflow_id=result.workflow_id,
                intent=intent.value,
                status=result.status.value,
                response=final_response,
                execution_time=execution_time,
                step_results={k: asdict(v) for k, v in result.results.items()},
                error_summary=result.error_summary,
                **translation_metadata
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_message = str(e)
            
            # Try to translate error message to user's language
            if (preserve_user_language and 
                translation_metadata.get('was_translated') and 
                translation_metadata.get('original_language') != 'en'):
                
                error_translation = await self.translate_response_to_user_language(
                    error_message, 
                    translation_metadata['original_language']
                )
                if error_translation['success']:
                    error_message = error_translation['translated_text']
            
            self.logger.error(f"Query processing failed: {e}")
            return OrchestratorResponse(
                success=False,
                error=error_message,
                query=original_query,
                execution_time=execution_time,
                **translation_metadata
            )
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages for translation."""
        if not self.translation_service:
            return []
        
        return self.translation_service.get_supported_languages()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and metrics."""
        status = {
            "loaded_agents": list(self.agents_registry.keys()),
            "available_agents": list(self.agent_definitions.keys()),
            "workflow_definitions": len(self.workflow_definitions),
            "system_stats": self.integration_coordinator.get_system_stats(),
            "translation_service_available": self.translation_service is not None
        }
        
        if self.translation_service:
            try:
                supported_languages_count = len(self.get_supported_languages())
                status["supported_languages_count"] = supported_languages_count
            except Exception as e:
                status["translation_service_error"] = str(e)
        
        return status