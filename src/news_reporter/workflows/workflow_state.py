"""WorkflowState - Shared state object for graph execution"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import time
import logging

logger = logging.getLogger(__name__)


class WorkflowState(BaseModel):
    """
    Shared state object that every node reads/writes during graph execution.
    Makes graph execution deterministic and enables state propagation.
    """
    
    # Input
    goal: str
    
    # Triage results
    triage: Optional[Dict[str, Any]] = None
    selected_search: Optional[str] = None  # "sql"|"aisearch"|"neo4j"|None
    database_id: Optional[str] = None
    targets: List[str] = Field(default_factory=list)
    
    # Search results
    latest: str = ""
    
    # Reporter drafts (per reporter_id)
    drafts: Dict[str, str] = Field(default_factory=dict)  # reporter_id -> draft
    
    # Final outputs (per reporter_id)
    final: Dict[str, str] = Field(default_factory=dict)  # reporter_id -> final
    
    # Review verdicts (per reporter_id)
    verdicts: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)  # reporter_id -> list of verdicts
    
    # Conditional node results
    conditional: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # node_id -> {result: bool, ...}
    
    # Logging and telemetry
    logs: List[Dict[str, Any]] = Field(default_factory=list)
    execution_trace: List[Dict[str, Any]] = Field(default_factory=list)
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get nested value from state using dot notation.
        
        Examples:
            state.get("triage.preferred_agent")
            state.get("drafts.reporter_1")
            state.get("final.reporter_2", "")
        """
        parts = path.split(".")
        value = self.model_dump()
        
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
                if value is None:
                    return default
            else:
                return default
        
        return value if value is not None else default
    
    def set(self, path: str, value: Any) -> None:
        """
        Set nested value in state using dot notation.
        
        Examples:
            state.set("triage.preferred_agent", "sql")
            state.set("drafts.reporter_1", "draft text")
            state.set("conditional.select_search.result", True)
        """
        parts = path.split(".")
        if len(parts) == 1:
            # Direct attribute
            if hasattr(self, parts[0]):
                setattr(self, parts[0], value)
            else:
                logger.warning(f"Attempted to set unknown attribute: {parts[0]}")
            return
        
        # Nested path - navigate to parent dict
        root_key = parts[0]
        final_key = parts[-1]
        
        # Check if root key exists and is a dict
        if not hasattr(self, root_key):
            logger.warning(f"Attempted to set nested value on unknown root attribute: {root_key}")
            return
        
        root_value = getattr(self, root_key)
        
        # If root value is None, initialize as dict
        if root_value is None:
            root_value = {}
            setattr(self, root_key, root_value)
        
        # Navigate/create nested structure
        current = root_value
        for part in parts[1:-1]:  # All parts except first and last
            if not isinstance(current, dict):
                logger.warning(f"Cannot set nested value: {path} (parent '{part}' is not a dict)")
                return
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set the final value
        if isinstance(current, dict):
            current[final_key] = value
            # Update the root attribute to ensure Pydantic sees the change
            setattr(self, root_key, root_value)
        else:
            logger.warning(f"Cannot set nested value: {path} (parent is not a dict)")
    
    def append_log(self, level: str, message: str, node_id: Optional[str] = None, **kwargs: Any) -> None:
        """Add a log entry to state.logs"""
        log_entry = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            "node_id": node_id,
            **kwargs
        }
        self.logs.append(log_entry)
        logger.log(getattr(logging, level.upper(), logging.INFO), f"[{node_id}] {message}")
    
    def add_trace(
        self,
        node_id: str,
        start_time: float,
        end_time: float,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """Add execution trace entry for a node"""
        trace_entry = {
            "node_id": node_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "inputs_summary": self._summarize_data(inputs) if inputs else None,
            "outputs_summary": self._summarize_data(outputs) if outputs else None,
            "error": error
        }
        self.execution_trace.append(trace_entry)
    
    @staticmethod
    def _summarize_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of data (sizes, types, ids) for trace"""
        if not data:
            return {}
        
        summary = {}
        for key, value in data.items():
            if isinstance(value, str):
                summary[key] = {"type": "str", "length": len(value)}
            elif isinstance(value, (list, dict)):
                summary[key] = {"type": type(value).__name__, "size": len(value)}
            elif isinstance(value, (int, float, bool)):
                summary[key] = {"type": type(value).__name__, "value": value}
            else:
                summary[key] = {"type": type(value).__name__}
        
        return summary
