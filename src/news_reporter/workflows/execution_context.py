"""Execution Context - Thread-safe context for node execution (Phase 4 enhancement)"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import uuid


@dataclass
class ExecutionContext:
    """
    Execution context for tracking execution flow.
    
    This tracks lineage, branches, iterations, and loop membership for proper routing.
    Phase 4: Added retry tracking and performance metrics.
    Phase 5: Added parent_loop_id for loop feedback routing.
    """
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Unique ID for entire workflow run
    branch_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Unique ID for this execution branch
    parent_branch_id: Optional[str] = None  # Parent branch (for fanout tracking)
    node_id: Optional[str] = None  # Current node being executed
    iteration: int = 0  # Iteration number (for loops/retries)
    depth: int = 0  # Execution depth (for cycle detection)
    retry_count: int = 0  # Number of retries attempted (Phase 4)
    parent_loop_id: Optional[str] = None  # Loop node that spawned this execution (Phase 5)
    
    # Execution path tracking
    path: List[str] = field(default_factory=list)  # Nodes visited in this branch
    
    # User-defined metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def create_child_branch(self, node_id: str) -> ExecutionContext:
        """Create a child branch context for fanout/parallel execution"""
        child_branch_id = str(uuid.uuid4())
        return ExecutionContext(
            run_id=self.run_id,
            branch_id=child_branch_id,
            parent_branch_id=self.branch_id,
            node_id=node_id,
            iteration=0,
            depth=self.depth + 1,
            retry_count=0,
            parent_loop_id=self.parent_loop_id,  # Inherit loop membership
            path=self.path + [node_id],
            metadata=self.metadata.copy()
        )
    
    def create_loop_iteration(self, iteration: int) -> ExecutionContext:
        """Create context for next loop iteration (same branch, incremented iteration)"""
        return ExecutionContext(
            run_id=self.run_id,
            branch_id=self.branch_id,
            parent_branch_id=self.parent_branch_id,
            node_id=self.node_id,
            iteration=iteration,
            depth=self.depth,
            retry_count=0,
            parent_loop_id=self.parent_loop_id,
            path=self.path,
            metadata=self.metadata.copy()
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "run_id": self.run_id,
            "branch_id": self.branch_id,
            "parent_branch_id": self.parent_branch_id,
            "node_id": self.node_id,
            "iteration": self.iteration,
            "depth": self.depth
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> ExecutionContext:
        """Create from dictionary"""
        return cls(
            run_id=data.get("run_id", str(uuid.uuid4())),
            branch_id=data.get("branch_id", str(uuid.uuid4())),
            parent_branch_id=data.get("parent_branch_id"),
            node_id=data.get("node_id"),
            iteration=data.get("iteration", 0),
            depth=data.get("depth", 0)
        )
