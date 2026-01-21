"""Execution Monitor - Real-time monitoring and streaming of workflow execution"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Callable, AsyncIterator
import asyncio
import time
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum

from .workflow_state import WorkflowState
from .execution_context import ExecutionContext
from .node_result import NodeResult, NodeStatus

logger = logging.getLogger(__name__)


class ExecutionEventType(str, Enum):
    """Types of execution events"""
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    NODE_SKIPPED = "node_skipped"
    STATE_UPDATED = "state_updated"
    BRANCH_CREATED = "branch_created"
    BRANCH_COMPLETED = "branch_completed"
    LOOP_ITERATION = "loop_iteration"
    MERGE_WAITING = "merge_waiting"
    MERGE_COMPLETED = "merge_completed"


@dataclass
class ExecutionEvent:
    """An execution event"""
    event_type: ExecutionEventType
    run_id: str
    timestamp: float
    node_id: Optional[str] = None
    branch_id: Optional[str] = None
    iteration: int = 0
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_type": self.event_type.value,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "branch_id": self.branch_id,
            "iteration": self.iteration,
            "data": self.data
        }


class ExecutionMonitor:
    """Monitors workflow execution and streams events"""
    
    def __init__(self):
        self.subscribers: List[Callable[[ExecutionEvent], None]] = []
        self.event_history: Dict[str, List[ExecutionEvent]] = {}  # run_id -> events
        self.active_runs: Dict[str, Dict[str, Any]] = {}  # run_id -> run info
    
    def subscribe(self, callback: Callable[[ExecutionEvent], None]) -> None:
        """Subscribe to execution events"""
        self.subscribers.append(callback)
        logger.info(f"New subscriber registered. Total subscribers: {len(self.subscribers)}")
    
    def unsubscribe(self, callback: Callable[[ExecutionEvent], None]) -> None:
        """Unsubscribe from execution events"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def _emit(self, event: ExecutionEvent) -> None:
        """Emit an event to all subscribers"""
        # Store in history
        if event.run_id not in self.event_history:
            self.event_history[event.run_id] = []
        self.event_history[event.run_id].append(event)
        
        # Notify subscribers
        for callback in self.subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event subscriber: {e}", exc_info=True)
    
    def workflow_started(self, run_id: str, goal: str, graph_id: Optional[str] = None) -> None:
        """Emit workflow started event"""
        self.active_runs[run_id] = {
            "goal": goal,
            "graph_id": graph_id,
            "start_time": time.time(),
            "status": "running"
        }
        self._emit(ExecutionEvent(
            event_type=ExecutionEventType.WORKFLOW_STARTED,
            run_id=run_id,
            timestamp=time.time(),
            data={"goal": goal, "graph_id": graph_id}
        ))
    
    def workflow_completed(self, run_id: str, result: str, duration_ms: float) -> None:
        """Emit workflow completed event"""
        if run_id in self.active_runs:
            self.active_runs[run_id]["status"] = "completed"
            self.active_runs[run_id]["duration_ms"] = duration_ms
        
        self._emit(ExecutionEvent(
            event_type=ExecutionEventType.WORKFLOW_COMPLETED,
            run_id=run_id,
            timestamp=time.time(),
            data={"result": result[:100], "duration_ms": duration_ms}
        ))
    
    def workflow_failed(self, run_id: str, error: str) -> None:
        """Emit workflow failed event"""
        if run_id in self.active_runs:
            self.active_runs[run_id]["status"] = "failed"
            self.active_runs[run_id]["error"] = error
        
        self._emit(ExecutionEvent(
            event_type=ExecutionEventType.WORKFLOW_FAILED,
            run_id=run_id,
            timestamp=time.time(),
            data={"error": error}
        ))
    
    def node_started(
        self,
        run_id: str,
        node_id: str,
        node_type: str,
        context: Optional[ExecutionContext] = None
    ) -> None:
        """Emit node started event"""
        self._emit(ExecutionEvent(
            event_type=ExecutionEventType.NODE_STARTED,
            run_id=run_id,
            timestamp=time.time(),
            node_id=node_id,
            branch_id=context.branch_id if context else None,
            iteration=context.iteration if context else 0,
            data={"node_type": node_type}
        ))
    
    def node_completed(
        self,
        run_id: str,
        node_id: str,
        result: NodeResult,
        context: Optional[ExecutionContext] = None
    ) -> None:
        """Emit node completed event"""
        self._emit(ExecutionEvent(
            event_type=ExecutionEventType.NODE_COMPLETED,
            run_id=run_id,
            timestamp=time.time(),
            node_id=node_id,
            branch_id=context.branch_id if context else None,
            iteration=context.iteration if context else 0,
            data={
                "status": result.status.value,
                "duration_ms": (result.end_time - result.start_time) * 1000 if result.end_time and result.start_time else 0,
                "cache_hit": result.artifacts.get("cache_hit", False),
                "retry_count": result.metrics.get("retry_count", 0)
            }
        ))
    
    def node_failed(
        self,
        run_id: str,
        node_id: str,
        error: str,
        context: Optional[ExecutionContext] = None
    ) -> None:
        """Emit node failed event"""
        self._emit(ExecutionEvent(
            event_type=ExecutionEventType.NODE_FAILED,
            run_id=run_id,
            timestamp=time.time(),
            node_id=node_id,
            branch_id=context.branch_id if context else None,
            data={"error": error}
        ))
    
    def state_updated(self, run_id: str, updates: Dict[str, Any]) -> None:
        """Emit state updated event"""
        self._emit(ExecutionEvent(
            event_type=ExecutionEventType.STATE_UPDATED,
            run_id=run_id,
            timestamp=time.time(),
            data={"updates": updates}
        ))
    
    def get_event_history(self, run_id: str) -> List[ExecutionEvent]:
        """Get event history for a run"""
        return self.event_history.get(run_id, [])
    
    def get_active_runs(self) -> Dict[str, Dict[str, Any]]:
        """Get all active runs"""
        return self.active_runs.copy()
    
    def get_run_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific run"""
        return self.active_runs.get(run_id)


# Global monitor instance
_global_monitor = ExecutionMonitor()


def get_execution_monitor() -> ExecutionMonitor:
    """Get the global execution monitor"""
    return _global_monitor


class EventStream:
    """Async event stream for real-time monitoring"""
    
    def __init__(self, monitor: ExecutionMonitor, run_id: Optional[str] = None):
        self.monitor = monitor
        self.run_id = run_id
        self.queue: asyncio.Queue = asyncio.Queue()
        self.subscribed = False
    
    async def __aenter__(self):
        """Context manager entry"""
        self.monitor.subscribe(self._on_event)
        self.subscribed = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.subscribed:
            self.monitor.unsubscribe(self._on_event)
    
    def _on_event(self, event: ExecutionEvent) -> None:
        """Handle event from monitor"""
        if self.run_id is None or event.run_id == self.run_id:
            try:
                self.queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Event queue full, dropping event")
    
    async def stream(self) -> AsyncIterator[ExecutionEvent]:
        """Stream events asynchronously"""
        while True:
            try:
                event = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                yield event
            except asyncio.TimeoutError:
                # Yield None to keep connection alive
                continue
            except Exception as e:
                logger.error(f"Error in event stream: {e}")
                break
