"""Workflow Debugger - Advanced debugging and troubleshooting tools"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .execution_context import ExecutionContext
from .node_result import NodeResult, NodeStatus
from .workflow_state import WorkflowState

logger = logging.getLogger(__name__)


class BreakpointType(str, Enum):
    """Types of breakpoints"""
    NODE = "node"  # Break before/after node execution
    CONDITION = "condition"  # Break on condition evaluation
    STATE = "state"  # Break when state matches condition
    ERROR = "error"  # Break on error


@dataclass
class Breakpoint:
    """A breakpoint for debugging"""
    breakpoint_id: str
    type: BreakpointType
    node_id: Optional[str] = None
    condition: Optional[str] = None  # For state breakpoints
    enabled: bool = True
    hit_count: int = 0


@dataclass
class DebugTrace:
    """A trace entry for debugging"""
    trace_id: str
    timestamp: datetime
    node_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    state_snapshot: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    event_type: str = "execution"  # execution, breakpoint, error, etc.


class WorkflowDebugger:
    """Advanced debugging and troubleshooting for workflows"""
    
    def __init__(self):
        self.breakpoints: Dict[str, Breakpoint] = {}
        self.traces: List[DebugTrace] = []
        self.watch_expressions: List[str] = []  # State expressions to watch
        self.is_debugging: bool = False
        self._trace_counter = 0
    
    def add_breakpoint(
        self,
        breakpoint_id: str,
        type: BreakpointType,
        node_id: Optional[str] = None,
        condition: Optional[str] = None
    ) -> Breakpoint:
        """Add a breakpoint"""
        bp = Breakpoint(
            breakpoint_id=breakpoint_id,
            type=type,
            node_id=node_id,
            condition=condition
        )
        self.breakpoints[breakpoint_id] = bp
        logger.info(f"Added breakpoint: {breakpoint_id}")
        return bp
    
    def remove_breakpoint(self, breakpoint_id: str) -> bool:
        """Remove a breakpoint"""
        if breakpoint_id in self.breakpoints:
            del self.breakpoints[breakpoint_id]
            logger.info(f"Removed breakpoint: {breakpoint_id}")
            return True
        return False
    
    def add_watch_expression(self, expression: str) -> None:
        """Add a watch expression (state path to monitor)"""
        if expression not in self.watch_expressions:
            self.watch_expressions.append(expression)
            logger.info(f"Added watch expression: {expression}")
    
    def check_breakpoint(
        self,
        node_id: str,
        context: ExecutionContext,
        state: WorkflowState,
        result: Optional[NodeResult] = None
    ) -> Optional[Breakpoint]:
        """Check if a breakpoint should trigger"""
        for bp in self.breakpoints.values():
            if not bp.enabled:
                continue
            
            if bp.type == BreakpointType.NODE and bp.node_id == node_id:
                bp.hit_count += 1
                return bp
            
            elif bp.type == BreakpointType.ERROR and result and result.status == NodeStatus.FAILED:
                bp.hit_count += 1
                return bp
            
            elif bp.type == BreakpointType.STATE and bp.condition:
                # Evaluate condition against state
                try:
                    if self._evaluate_condition(bp.condition, state):
                        bp.hit_count += 1
                        return bp
                except Exception as e:
                    logger.warning(f"Error evaluating breakpoint condition: {e}")
        
        return None
    
    def _evaluate_condition(self, condition: str, state: WorkflowState) -> bool:
        """Evaluate a condition against workflow state (simplified)"""
        # Simplified evaluation - would use proper expression evaluator
        # For now, just check if condition string appears in state
        state_dict = state.to_dict()
        return condition in str(state_dict)
    
    def add_trace(
        self,
        node_id: Optional[str] = None,
        context: Optional[ExecutionContext] = None,
        state: Optional[WorkflowState] = None,
        result: Optional[NodeResult] = None,
        event_type: str = "execution"
    ) -> DebugTrace:
        """Add a trace entry"""
        trace = DebugTrace(
            trace_id=f"trace_{self._trace_counter}",
            timestamp=datetime.now(),
            node_id=node_id,
            context=context.__dict__ if context else None,
            state_snapshot=state.model_dump() if state else None,
            result=result.to_dict() if result else None,
            event_type=event_type
        )
        
        self._trace_counter += 1
        self.traces.append(trace)
        
        # Keep only last 10000 traces
        if len(self.traces) > 10000:
            self.traces = self.traces[-10000:]
        
        return trace
    
    def get_traces(
        self,
        node_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[DebugTrace]:
        """Get traces with optional filtering"""
        traces = list(self.traces)
        
        if node_id:
            traces = [t for t in traces if t.node_id == node_id]
        
        if event_type:
            traces = [t for t in traces if t.event_type == event_type]
        
        # Sort by timestamp descending
        traces.sort(key=lambda t: t.timestamp, reverse=True)
        
        return traces[:limit]
    
    def get_watch_values(self, state: WorkflowState) -> Dict[str, Any]:
        """Get values for watch expressions"""
        watch_values = {}
        state_dict = state.to_dict()
        
        for expr in self.watch_expressions:
            try:
                # Simplified - would use proper path evaluation
                if expr in state_dict:
                    watch_values[expr] = state_dict[expr]
                else:
                    watch_values[expr] = None
            except Exception as e:
                watch_values[expr] = f"Error: {e}"
        
        return watch_values
    
    def analyze_execution_path(
        self,
        workflow_id: str,
        execution_id: str
    ) -> Dict[str, Any]:
        """Analyze execution path for debugging"""
        # Get traces for this execution
        execution_traces = [t for t in self.traces if t.context and t.context.get("run_id") == execution_id]
        
        # Build execution path
        path = []
        for trace in sorted(execution_traces, key=lambda t: t.timestamp):
            path.append({
                "node_id": trace.node_id,
                "timestamp": trace.timestamp.isoformat(),
                "event_type": trace.event_type,
                "status": trace.result.get("status") if trace.result else None
            })
        
        # Find issues
        issues = []
        for trace in execution_traces:
            if trace.result and trace.result.get("status") == "failed":
                issues.append({
                    "node_id": trace.node_id,
                    "error": trace.result.get("error"),
                    "timestamp": trace.timestamp.isoformat()
                })
        
        return {
            "execution_id": execution_id,
            "path": path,
            "issues": issues,
            "total_steps": len(path),
            "failed_steps": len(issues)
        }
    
    def clear_traces(self) -> None:
        """Clear all traces"""
        self.traces.clear()
        self._trace_counter = 0
        logger.info("Cleared all traces")


# Global debugger instance
_global_debugger = WorkflowDebugger()


def get_workflow_debugger() -> WorkflowDebugger:
    """Get the global workflow debugger"""
    return _global_debugger
