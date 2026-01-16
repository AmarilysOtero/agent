"""NodeResult - Structured output for nodes"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import time


class NodeStatus(str, Enum):
    """Status of node execution"""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    IN_PROGRESS = "in_progress"


@dataclass
class NodeResult:
    """
    Structured output for nodes (replaces raw dict).
    
    Contains:
    - state_updates: Changes to make to WorkflowState
    - artifacts: Additional data produced (not stored in state)
    - next_nodes: Which nodes to execute next (for dynamic routing)
    - status: Execution status
    - metrics: Performance and other metrics
    """
    
    # State updates (dict of state paths -> values)
    state_updates: Dict[str, Any] = field(default_factory=dict)
    
    # Artifacts (data not stored in state, but available for downstream nodes)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    
    # Next nodes to execute (for dynamic routing)
    # If empty, executor uses graph edges to determine next nodes
    next_nodes: List[str] = field(default_factory=list)
    
    # Execution status
    status: NodeStatus = NodeStatus.SUCCESS
    
    # Metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Error information (if status is FAILED)
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    # Timestamps
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    def mark_complete(self) -> None:
        """Mark execution as complete"""
        self.end_time = time.time()
        if self.status == NodeStatus.IN_PROGRESS:
            self.status = NodeStatus.SUCCESS
    
    def mark_failed(self, error: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Mark execution as failed"""
        self.end_time = time.time()
        self.status = NodeStatus.FAILED
        self.error = error
        self.error_details = details or {}
    
    def mark_skipped(self, reason: Optional[str] = None) -> None:
        """Mark execution as skipped"""
        self.end_time = time.time()
        self.status = NodeStatus.SKIPPED
        if reason:
            self.metrics["skip_reason"] = reason
    
    def add_metric(self, key: str, value: Any) -> None:
        """Add a metric"""
        self.metrics[key] = value
    
    def add_state_update(self, path: str, value: Any) -> None:
        """Add a state update"""
        self.state_updates[path] = value
    
    def add_artifact(self, key: str, value: Any) -> None:
        """Add an artifact"""
        self.artifacts[key] = value
    
    def get_duration(self) -> Optional[float]:
        """Get execution duration in seconds"""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "state_updates": self.state_updates,
            "artifacts": self.artifacts,
            "next_nodes": self.next_nodes,
            "status": self.status.value,
            "metrics": self.metrics,
            "error": self.error,
            "error_details": self.error_details,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.get_duration()
        }
    
    @classmethod
    def success(
        cls,
        state_updates: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        next_nodes: Optional[List[str]] = None
    ) -> NodeResult:
        """Create a successful result"""
        result = cls(
            state_updates=state_updates or {},
            artifacts=artifacts or {},
            next_nodes=next_nodes or [],
            status=NodeStatus.SUCCESS
        )
        result.mark_complete()
        return result
    
    @classmethod
    def failed(cls, error: str, details: Optional[Dict[str, Any]] = None) -> NodeResult:
        """Create a failed result"""
        result = cls(status=NodeStatus.FAILED)
        result.mark_failed(error, details)
        return result
    
    @classmethod
    def skipped(cls, reason: Optional[str] = None) -> NodeResult:
        """Create a skipped result"""
        result = cls(status=NodeStatus.SKIPPED)
        result.mark_skipped(reason)
        return result
