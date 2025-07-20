import asyncio
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum
import uuid
from dataclasses import dataclass


class WorkflowStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentTask:
    """Represents a task to be executed by an agent."""
    task_id: str
    agent_name: str
    function_name: str
    parameters: Dict[str, Any]
    priority: AgentPriority = AgentPriority.MEDIUM
    dependencies: List[str] = None  # Task IDs this task depends on
    timeout: int = 300  # Timeout in seconds
    created_at: datetime = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class WorkflowResult:
    """Represents the result of a workflow execution."""
    workflow_id: str
    status: WorkflowStatus
    results: Dict[str, Any]
    errors: List[Dict[str, Any]]
    execution_time: float
    completed_tasks: List[str]
    failed_tasks: List[str]


class IntegrationCoordinator:
    """
    Coordinates workflows between multiple agents, manages handoffs,
    and orchestrates complex multi-agent analytical processes.
    """
    
    def __init__(self):
        """Initialize the integration coordinator."""
        self.active_workflows: Dict[str, Dict] = {}
        self.agent_registry: Dict[str, Any] = {}
        self.task_queue: List[AgentTask] = []
        self.completed_workflows: List[WorkflowResult] = []
        self.shared_state: Dict[str, Any] = {}
        
    def register_agent(self, agent_name: str, agent_instance: Any):
        """
        Register an agent with the coordinator.
        
        Args:
            agent_name: Name of the agent
            agent_instance: Agent instance
        """
        self.agent_registry[agent_name] = agent_instance
        
    def create_workflow(self, workflow_name: str, tasks: List[AgentTask]) -> str:
        """
        Create a new workflow with multiple tasks.
        
        Args:
            workflow_name: Name of the workflow
            tasks: List of tasks to execute
            
        Returns:
            Workflow ID
        """
        workflow_id = str(uuid.uuid4())
        
        self.active_workflows[workflow_id] = {
            "name": workflow_name,
            "id": workflow_id,
            "tasks": {task.task_id: task for task in tasks},
            "status": WorkflowStatus.PENDING,
            "results": {},
            "errors": [],
            "started_at": None,
            "completed_at": None,
            "dependency_graph": self._build_dependency_graph(tasks)
        }
        
        return workflow_id
    
    async def execute_workflow(self, workflow_id: str) -> WorkflowResult:
        """
        Execute a workflow by processing all its tasks.
        
        Args:
            workflow_id: ID of the workflow to execute
            
        Returns:
            WorkflowResult with execution details
        """
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.active_workflows[workflow_id]
        workflow["status"] = WorkflowStatus.IN_PROGRESS
        workflow["started_at"] = datetime.now()
        
        try:
            # Execute tasks based on dependency order
            execution_order = self._get_execution_order(workflow["tasks"], workflow["dependency_graph"])
            
            for task_batch in execution_order:
                # Execute tasks in parallel where possible
                batch_results = await self._execute_task_batch(task_batch, workflow)
                
                # Update workflow results
                for task_id, result in batch_results.items():
                    workflow["results"][task_id] = result
                    if not result.get("success", False):
                        workflow["errors"].append({
                            "task_id": task_id,
                            "error": result.get("error", "Unknown error"),
                            "timestamp": datetime.now().isoformat()
                        })
            
            # Determine final status
            failed_tasks = [task_id for task_id, result in workflow["results"].items() 
                           if not result.get("success", False)]
            
            if failed_tasks:
                workflow["status"] = WorkflowStatus.FAILED
            else:
                workflow["status"] = WorkflowStatus.COMPLETED
            
            workflow["completed_at"] = datetime.now()
            
            # Create result object
            execution_time = (workflow["completed_at"] - workflow["started_at"]).total_seconds()
            result = WorkflowResult(
                workflow_id=workflow_id,
                status=workflow["status"],
                results=workflow["results"],
                errors=workflow["errors"],
                execution_time=execution_time,
                completed_tasks=[task_id for task_id, result in workflow["results"].items() 
                               if result.get("success", False)],
                failed_tasks=failed_tasks
            )
            
            # Move to completed workflows
            self.completed_workflows.append(result)
            if len(self.completed_workflows) > 100:  # Keep last 100 workflows
                self.completed_workflows.pop(0)
            
            del self.active_workflows[workflow_id]
            
            return result
            
        except Exception as e:
            workflow["status"] = WorkflowStatus.FAILED
            workflow["completed_at"] = datetime.now()
            workflow["errors"].append({
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "context": "Workflow execution failed"
            })
            
            execution_time = (workflow["completed_at"] - workflow["started_at"]).total_seconds()
            result = WorkflowResult(
                workflow_id=workflow_id,
                status=WorkflowStatus.FAILED,
                results=workflow["results"],
                errors=workflow["errors"],
                execution_time=execution_time,
                completed_tasks=[],
                failed_tasks=list(workflow["tasks"].keys())
            )
            
            self.completed_workflows.append(result)
            del self.active_workflows[workflow_id]
            
            return result
    
    async def _execute_task_batch(self, task_ids: List[str], workflow: Dict) -> Dict[str, Any]:
        """Execute a batch of tasks in parallel."""
        tasks = [workflow["tasks"][task_id] for task_id in task_ids]
        
        # Create coroutines for each task
        coroutines = [
            self._execute_single_task(task, workflow)
            for task in tasks
        ]
        
        # Execute in parallel with timeout
        try:
            results = await asyncio.gather(*coroutines, return_exceptions=True)
            
            # Process results
            batch_results = {}
            for i, (task, result) in enumerate(zip(tasks, results)):
                if isinstance(result, Exception):
                    batch_results[task.task_id] = {
                        "success": False,
                        "error": str(result),
                        "task_id": task.task_id
                    }
                else:
                    batch_results[task.task_id] = result
            
            return batch_results
            
        except Exception as e:
            # If gather fails, return error for all tasks
            return {
                task.task_id: {
                    "success": False,
                    "error": f"Batch execution failed: {str(e)}",
                    "task_id": task.task_id
                }
                for task in tasks
            }
    
    async def _execute_single_task(self, task: AgentTask, workflow: Dict) -> Dict[str, Any]:
        """Execute a single agent task."""
        try:
            # Get the agent
            if task.agent_name not in self.agent_registry:
                raise ValueError(f"Agent {task.agent_name} not registered")
            
            agent = self.agent_registry[task.agent_name]
            
            # Add shared state to parameters
            enhanced_params = {
                **task.parameters,
                "shared_state": self.shared_state.copy(),
                "workflow_context": {
                    "workflow_id": workflow["id"],
                    "workflow_name": workflow["name"]
                }
            }
            
            # Execute the task with timeout
            result = await asyncio.wait_for(
                agent.process(enhanced_params),
                timeout=task.timeout
            )
            
            # Update shared state if result contains state updates
            if isinstance(result, dict) and "shared_state_updates" in result:
                self.shared_state.update(result["shared_state_updates"])
            
            return {
                "success": True,
                "result": result,
                "task_id": task.task_id,
                "agent_name": task.agent_name,
                "execution_time": datetime.now().isoformat()
            }
            
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Task timed out after {task.timeout} seconds",
                "task_id": task.task_id,
                "agent_name": task.agent_name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "task_id": task.task_id,
                "agent_name": task.agent_name
            }
    
    def _build_dependency_graph(self, tasks: List[AgentTask]) -> Dict[str, List[str]]:
        """Build a dependency graph from tasks."""
        graph = {}
        for task in tasks:
            graph[task.task_id] = task.dependencies.copy()
        return graph
    
    def _get_execution_order(self, tasks: Dict[str, AgentTask], dependency_graph: Dict[str, List[str]]) -> List[List[str]]:
        """Get the execution order for tasks based on dependencies."""
        # Topological sort to determine execution order
        execution_order = []
        remaining_tasks = set(tasks.keys())
        completed_tasks = set()
        
        while remaining_tasks:
            # Find tasks with no unmet dependencies
            ready_tasks = [
                task_id for task_id in remaining_tasks
                if all(dep in completed_tasks for dep in dependency_graph[task_id])
            ]
            
            if not ready_tasks:
                # Circular dependency or missing dependency
                raise ValueError("Circular dependency detected or missing dependency")
            
            # Sort by priority (higher priority first)
            ready_tasks.sort(key=lambda task_id: tasks[task_id].priority.value, reverse=True)
            
            execution_order.append(ready_tasks)
            remaining_tasks -= set(ready_tasks)
            completed_tasks.update(ready_tasks)
        
        return execution_order
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get the current status of a workflow."""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            return {
                "workflow_id": workflow_id,
                "status": workflow["status"].value,
                "progress": {
                    "total_tasks": len(workflow["tasks"]),
                    "completed_tasks": len(workflow["results"]),
                    "failed_tasks": len(workflow["errors"])
                },
                "started_at": workflow.get("started_at"),
                "is_active": True
            }
        else:
            # Check completed workflows
            for completed in self.completed_workflows:
                if completed.workflow_id == workflow_id:
                    return {
                        "workflow_id": workflow_id,
                        "status": completed.status.value,
                        "execution_time": completed.execution_time,
                        "completed_tasks": len(completed.completed_tasks),
                        "failed_tasks": len(completed.failed_tasks),
                        "is_active": False
                    }
            
            return {"error": f"Workflow {workflow_id} not found"}
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel an active workflow."""
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]["status"] = WorkflowStatus.CANCELLED
            return True
        return False
    
    def get_shared_state(self, key: Optional[str] = None) -> Any:
        """Get shared state value(s)."""
        if key:
            return self.shared_state.get(key)
        return self.shared_state.copy()
    
    def update_shared_state(self, updates: Dict[str, Any]):
        """Update shared state with new values."""
        self.shared_state.update(updates)
    
    def clear_shared_state(self):
        """Clear all shared state."""
        self.shared_state.clear()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics and health information."""
        return {
            "active_workflows": len(self.active_workflows),
            "registered_agents": len(self.agent_registry),
            "completed_workflows": len(self.completed_workflows),
            "shared_state_size": len(self.shared_state),
            "agent_list": list(self.agent_registry.keys())
        }