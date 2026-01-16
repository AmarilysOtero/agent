"""Workflow Persistence - Database integration for workflows and execution history"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum

from .graph_schema import GraphDefinition
from .workflow_state import WorkflowState
from .performance_metrics import WorkflowMetrics

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowRecord:
    """Database record for a workflow definition"""
    workflow_id: str
    name: str
    description: Optional[str] = None
    graph_definition: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    created_by: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "graph_definition": self.graph_definition,
            "version": self.version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_by": self.created_by,
            "tags": self.tags,
            "is_active": self.is_active
        }


@dataclass
class ExecutionRecord:
    """Database record for a workflow execution"""
    execution_id: str
    workflow_id: str
    run_id: str
    goal: str
    status: WorkflowStatus
    result: Optional[str] = None
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    state_snapshot: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "execution_id": self.execution_id,
            "workflow_id": self.workflow_id,
            "run_id": self.run_id,
            "goal": self.goal,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "metrics": self.metrics,
            "state_snapshot": self.state_snapshot,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_by": self.created_by
        }


class WorkflowPersistence:
    """Manages persistence of workflows and execution history"""
    
    def __init__(self, storage_backend: Optional[Any] = None):
        """
        Initialize persistence layer.
        
        Args:
            storage_backend: Optional storage backend (database, file system, etc.)
                           If None, uses in-memory storage
        """
        self.storage_backend = storage_backend
        self._workflows: Dict[str, WorkflowRecord] = {}
        self._executions: Dict[str, ExecutionRecord] = {}
        self._workflow_executions: Dict[str, List[str]] = {}  # workflow_id -> [execution_id]
    
    def save_workflow(self, workflow: WorkflowRecord) -> None:
        """Save a workflow definition"""
        workflow.updated_at = datetime.now()
        if not workflow.created_at:
            workflow.created_at = datetime.now()
        
        self._workflows[workflow.workflow_id] = workflow
        
        if self.storage_backend:
            # Would persist to actual database here
            logger.debug(f"Persisting workflow {workflow.workflow_id} to backend")
        
        logger.info(f"Saved workflow: {workflow.workflow_id}")
    
    def get_workflow(self, workflow_id: str) -> Optional[WorkflowRecord]:
        """Get a workflow by ID"""
        return self._workflows.get(workflow_id)
    
    def list_workflows(
        self,
        tags: Optional[List[str]] = None,
        is_active: Optional[bool] = None
    ) -> List[WorkflowRecord]:
        """List workflows with optional filtering"""
        workflows = list(self._workflows.values())
        
        if tags:
            workflows = [w for w in workflows if any(tag in w.tags for tag in tags)]
        
        if is_active is not None:
            workflows = [w for w in workflows if w.is_active == is_active]
        
        return workflows
    
    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow (soft delete by setting is_active=False)"""
        if workflow_id in self._workflows:
            self._workflows[workflow_id].is_active = False
            logger.info(f"Deleted workflow: {workflow_id}")
            return True
        return False
    
    def save_execution(self, execution: ExecutionRecord) -> None:
        """Save an execution record"""
        if not execution.started_at:
            execution.started_at = datetime.now()
        
        self._executions[execution.execution_id] = execution
        
        # Track executions per workflow
        if execution.workflow_id not in self._workflow_executions:
            self._workflow_executions[execution.workflow_id] = []
        self._workflow_executions[execution.workflow_id].append(execution.execution_id)
        
        if self.storage_backend:
            # Would persist to actual database here
            logger.debug(f"Persisting execution {execution.execution_id} to backend")
        
        logger.info(f"Saved execution: {execution.execution_id}")
    
    def get_execution(self, execution_id: str) -> Optional[ExecutionRecord]:
        """Get an execution by ID"""
        return self._executions.get(execution_id)
    
    def list_executions(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[WorkflowStatus] = None,
        limit: int = 100
    ) -> List[ExecutionRecord]:
        """List executions with optional filtering"""
        executions = list(self._executions.values())
        
        if workflow_id:
            executions = [e for e in executions if e.workflow_id == workflow_id]
        
        if status:
            executions = [e for e in executions if e.status == status]
        
        # Sort by started_at descending
        executions.sort(key=lambda e: e.started_at or datetime.min, reverse=True)
        
        return executions[:limit]
    
    def update_execution_status(
        self,
        execution_id: str,
        status: WorkflowStatus,
        result: Optional[str] = None,
        error: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update execution status and results"""
        execution = self._executions.get(execution_id)
        if not execution:
            return False
        
        execution.status = status
        execution.completed_at = datetime.now()
        
        if result is not None:
            execution.result = result
        if error is not None:
            execution.error = error
        if metrics is not None:
            execution.metrics = metrics
        
        logger.info(f"Updated execution {execution_id} status to {status.value}")
        return True
    
    def get_workflow_execution_history(
        self,
        workflow_id: str,
        limit: int = 50
    ) -> List[ExecutionRecord]:
        """Get execution history for a workflow"""
        execution_ids = self._workflow_executions.get(workflow_id, [])
        executions = [self._executions[eid] for eid in execution_ids if eid in self._executions]
        
        # Sort by started_at descending
        executions.sort(key=lambda e: e.started_at or datetime.min, reverse=True)
        
        return executions[:limit]
    
    def export_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Export workflow definition as JSON"""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return None
        
        return workflow.to_dict()
    
    def import_workflow(self, data: Dict[str, Any]) -> WorkflowRecord:
        """Import workflow definition from JSON"""
        workflow = WorkflowRecord(
            workflow_id=data["workflow_id"],
            name=data["name"],
            description=data.get("description"),
            graph_definition=data.get("graph_definition", {}),
            version=data.get("version", "1.0.0"),
            tags=data.get("tags", []),
            is_active=data.get("is_active", True)
        )
        
        if "created_at" in data and data["created_at"]:
            workflow.created_at = datetime.fromisoformat(data["created_at"])
        if "created_by" in data:
            workflow.created_by = data["created_by"]
        
        self.save_workflow(workflow)
        return workflow


# Global persistence instance
_global_persistence = WorkflowPersistence()


def get_workflow_persistence() -> WorkflowPersistence:
    """Get the global workflow persistence instance"""
    return _global_persistence
