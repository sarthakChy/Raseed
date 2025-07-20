import logging
import asyncio
import yaml
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
import uuid

from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration
import vertexai

from core.base_agent_tools.integration_coordinator import IntegrationCoordinator
from core.base_agent_tools.error_handler import ErrorHandler
from core.base_agent_tools.user_profile_manager import UserProfileManager


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
class AgentConfig:
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


class MasterOrchestrator:
    """
    Master orchestrator that routes queries to appropriate agents and manages workflows.
    """
    
    def __init__(
        self,
        project_id: str,
        config_path: str = "config/agent_config.yaml",
        location: str = "us-central1",
        model_name: str = "gemini-1.5-pro"
    ):
        """
        Initialize the master orchestrator.
        
        Args:
            project_id: Google Cloud project ID
            config_path: Path to agent configuration file
            location: Vertex AI location
            model_name: Model for intent classification
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # Initialize Vertex AI
        self._initialize_vertex_ai()
        
        # Initialize shared tools
        self.integration_coordinator = IntegrationCoordinator()
        self.error_handler = ErrorHandler(self.logger)
        self.user_profile_manager = UserProfileManager()
        
        # Load configurations
        self.agent_configs: Dict[str, AgentConfig] = {}
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
    
    def _initialize_vertex_ai(self):
        """Initialize Vertex AI client."""
        try:
            vertexai.init(project=self.project_id, location=self.location)
            self.logger.info("Vertex AI initialized for orchestrator")
        except Exception as e:
            self.logger.error(f"Failed to initialize Vertex AI: {e}")
            raise
    
    def _load_configurations(self, config_path: str):
        """Load agent and workflow configurations."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Load agent configurations
            for agent_data in config.get('agents', []):
                agent_config = AgentConfig(**agent_data)
                self.agent_configs[agent_config.name] = agent_config
            
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
            
            self.logger.info(f"Loaded {len(self.agent_configs)} agents and {len(self.workflow_definitions)} workflows")
            
        except Exception as e:
            self.logger.error(f"Failed to load configurations: {e}")
            raise
    
    async def classify_intent(self, query: str, user_context: Optional[Dict] = None) -> IntentType:
        """
        Classify user query intent using Vertex AI.
        
        Args:
            query: User's natural language query
            user_context: Additional context about the user
            
        Returns:
            Classified intent type
        """
        try:
            # Prepare classification prompt
            context_str = ""
            if user_context:
                context_str = f"\nUser Context: {json.dumps(user_context, indent=2)}"
            
            prompt = f"""
Classify the following financial query into one of these intent types:

1. ANALYTICAL: "How much did I spend?" - Queries requesting specific calculations, amounts, or data analysis
2. EXPLORATORY: "What are my spending patterns?" - Queries exploring trends, patterns, or general insights
3. ACTIONABLE: "Help me save money" - Queries seeking recommendations, advice, or actionable steps
4. COMPARATIVE: "Compare this month vs last" - Queries comparing different time periods, categories, or metrics
5. PREDICTIVE: "Will I exceed my budget?" - Queries about future projections, forecasts, or predictions

Query: "{query}"{context_str}

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
        """
        Dynamically load an agent if not already loaded.
        
        Args:
            agent_name: Name of the agent to load
            
        Returns:
            Loaded agent instance
        """
        if agent_name in self.agents_registry:
            return self.agents_registry[agent_name]
        
        if agent_name not in self.agent_configs:
            raise ValueError(f"Agent configuration not found: {agent_name}")
        
        try:
            config = self.agent_configs[agent_name]
            
            # Dynamic import of agent class
            module_path, class_name = config.class_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            agent_class = getattr(module, class_name)
            
            # Initialize agent
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
            step_input = {"query": workflow_context.get("original_query", "")}
            
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
            if workflow.aggregation_agent and workflow.aggregation_agent in self.agent_configs:
                try:
                    synthesis_agent = await self.load_agent(workflow.aggregation_agent)
                    synthesis_input = {
                        "step_results": {k: v.result for k, v in step_results.items() if v.success},
                        "original_query": context.get("original_query", ""),
                        "user_id": context.get("user_id")
                    }
                    synthesis_result = await synthesis_agent.process(synthesis_input)
                    final_response = synthesis_result.get("response", "")
                except Exception as e:
                    self.logger.error(f"Synthesis failed: {e}")
                    final_response = "Results processed, but final synthesis failed."
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update workflow state
            workflow_state["status"] = status.value
            self.integration_coordinator.update_shared_state({f"workflow_{workflow_id}": workflow_state})
            
            return WorkflowResult(
                workflow_id=workflow_id,
                intent_type=workflow.intent_type,
                status=status,
                results=step_results,
                final_response=final_response,
                execution_time=execution_time
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
        additional_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the complete orchestration pipeline.
        
        Args:
            query: User's natural language query
            user_id: User identifier
            additional_context: Additional context for processing
            
        Returns:
            Complete response with results and metadata
        """
        try:
            # Get user context
            user_context = {}
            if user_id:
                user_profile = await self.user_profile_manager.get_personalization_context(user_id)
                user_context.update(user_profile)
            
            if additional_context:
                user_context.update(additional_context)
            
            # Classify intent
            intent = await self.classify_intent(query, user_context)
            self.logger.info(f"Classified intent: {intent.value} for query: {query[:50]}...")
            
            # Check if we have a workflow for this intent
            if intent not in self.workflow_definitions:
                return {
                    "success": False,
                    "error": f"No workflow defined for intent: {intent.value}",
                    "intent": intent.value
                }
            
            # Prepare workflow context
            workflow_context = {
                "original_query": query,
                "user_id": user_id,
                "intent": intent.value,
                "user_context": user_context,
                "timestamp": datetime.now().isoformat()
            }
            
            # Execute workflow
            workflow = self.workflow_definitions[intent]
            result = await self.execute_workflow(workflow, workflow_context)
            
            # Return comprehensive response
            return {
                "success": result.status in [WorkflowStatus.COMPLETED, WorkflowStatus.PARTIAL_SUCCESS],
                "workflow_id": result.workflow_id,
                "intent": intent.value,
                "status": result.status.value,
                "response": result.final_response,
                "execution_time": result.execution_time,
                "step_results": {k: asdict(v) for k, v in result.results.items()},
                "error_summary": result.error_summary
            }
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and metrics."""
        return {
            "loaded_agents": list(self.agents_registry.keys()),
            "available_agents": list(self.agent_configs.keys()),
            "workflow_definitions": len(self.workflow_definitions),
            "system_stats": self.integration_coordinator.get_system_stats()
        }