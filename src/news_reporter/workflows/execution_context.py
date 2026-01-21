"""Execution Context - Tracks branch identity in parallel/looping workflows"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import uuid


@dataclass
class ExecutionContext:
    """
    Context object for tracking branch identity in parallel/looping workflows.
    
    Used to:
    - Track which branch of execution a node belongs to
    - Enable proper state isolation/merging in fanout scenarios
    - Support loop iterations with proper state tracking
    - Enable debugging and tracing of execution paths
    """
    
    # Unique identifiers
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Unique ID for entire workflow run
    branch_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Unique ID for this branch
    parent_branch_id: Optional[str] = None  # Parent branch (for nested fanouts/loops)
    node_id: Optional[str] = None  # Current node being executed
    
    # Metadata
    iteration: int = 0  # Current iteration (for loops)
    depth: int = 0  # Nesting depth (for debugging)
    
    def create_child_branch(self, node_id: str) -> ExecutionContext:
        """Create a child branch context (for fanout)"""
        return ExecutionContext(
            run_id=self.run_id,
            branch_id=str(uuid.uuid4()),
            parent_branch_id=self.branch_id,
            node_id=node_id,
            iteration=self.iteration,
            depth=self.depth + 1
        )
    
    def create_iteration(self, node_id: str) -> ExecutionContext:
        """Create a new iteration context (for loops)"""
        return ExecutionContext(
            run_id=self.run_id,
            branch_id=self.branch_id,  # Same branch, new iteration
            parent_branch_id=self.parent_branch_id,
            node_id=node_id,
            iteration=self.iteration + 1,
            depth=self.depth
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
