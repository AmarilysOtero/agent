"""Execution Tracker - Tracks fanout branches and loop iterations for coordination"""

from __future__ import annotations
from typing import Dict, Set, List, Optional
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class BranchTracker:
    """Tracks a single fanout branch"""
    branch_id: str
    fanout_node_id: str
    item: str  # The item this branch is processing (e.g., reporter_id)
    branch_node_id: str  # The branch node being executed
    started_at: float = field(default_factory=time.time)
    completed: bool = False
    result: Optional[Any] = None


@dataclass
class FanoutTracker:
    """Tracks a fanout operation and its branches"""
    fanout_node_id: str
    items: List[str]  # Items to fan out over
    branch_node_ids: List[str]  # Branch nodes to execute
    branches: Dict[str, BranchTracker] = field(default_factory=dict)  # item -> BranchTracker
    merge_node_id: Optional[str] = None  # Merge node waiting for this fanout
    created_at: float = field(default_factory=time.time)
    
    def all_branches_complete(self) -> bool:
        """Check if all branches have completed"""
        return len(self.branches) > 0 and all(branch.completed for branch in self.branches.values())
    
    def get_completed_items(self) -> List[str]:
        """Get list of items whose branches have completed"""
        return [item for item, branch in self.branches.items() if branch.completed]
    
    def get_pending_items(self) -> List[str]:
        """Get list of items whose branches are still pending"""
        return [item for item, branch in self.branches.items() if not branch.completed]


@dataclass
class LoopTracker:
    """Tracks a loop execution"""
    loop_node_id: str
    body_node_id: Optional[str] = None
    current_iteration: int = 0
    max_iters: int = 0
    should_continue: bool = True
    started_at: float = field(default_factory=time.time)
    last_iteration_at: Optional[float] = None


class ExecutionTracker:
    """Tracks fanout branches and loop iterations for coordination"""
    
    def __init__(self):
        self.fanouts: Dict[str, FanoutTracker] = {}  # fanout_node_id -> FanoutTracker
        self.loops: Dict[str, LoopTracker] = {}  # loop_node_id -> LoopTracker
        self.branch_to_fanout: Dict[str, str] = {}  # branch_id -> fanout_node_id
    
    def register_fanout(
        self,
        fanout_node_id: str,
        items: List[str],
        branch_node_ids: List[str],
        merge_node_id: Optional[str] = None
    ) -> FanoutTracker:
        """Register a fanout operation"""
        tracker = FanoutTracker(
            fanout_node_id=fanout_node_id,
            items=items,
            branch_node_ids=branch_node_ids,
            merge_node_id=merge_node_id
        )
        self.fanouts[fanout_node_id] = tracker
        logger.info(f"Registered fanout {fanout_node_id} with {len(items)} items")
        return tracker
    
    def register_branch(
        self,
        fanout_node_id: str,
        item: str,
        branch_id: str,
        branch_node_id: str
    ) -> BranchTracker:
        """Register a branch in a fanout"""
        if fanout_node_id not in self.fanouts:
            raise ValueError(f"Fanout {fanout_node_id} not registered")
        
        tracker = self.fanouts[fanout_node_id]
        branch_tracker = BranchTracker(
            branch_id=branch_id,
            fanout_node_id=fanout_node_id,
            item=item,
            branch_node_id=branch_node_id
        )
        tracker.branches[item] = branch_tracker
        self.branch_to_fanout[branch_id] = fanout_node_id
        logger.debug(f"Registered branch {branch_id} for item {item} in fanout {fanout_node_id}")
        return branch_tracker
    
    def mark_branch_complete(self, branch_id: str, result: Any = None) -> None:
        """Mark a branch as complete"""
        fanout_node_id = self.branch_to_fanout.get(branch_id)
        if not fanout_node_id:
            logger.warning(f"Branch {branch_id} not found in tracker")
            return
        
        tracker = self.fanouts[fanout_node_id]
        for branch in tracker.branches.values():
            if branch.branch_id == branch_id:
                branch.completed = True
                branch.result = result
                logger.debug(f"Branch {branch_id} marked complete for fanout {fanout_node_id}")
                break
    
    def get_fanout_tracker(self, fanout_node_id: str) -> Optional[FanoutTracker]:
        """Get fanout tracker"""
        return self.fanouts.get(fanout_node_id)
    
    def register_loop(
        self,
        loop_node_id: str,
        max_iters: int,
        body_node_id: Optional[str] = None
    ) -> LoopTracker:
        """Register a loop"""
        tracker = LoopTracker(
            loop_node_id=loop_node_id,
            body_node_id=body_node_id,
            max_iters=max_iters
        )
        self.loops[loop_node_id] = tracker
        logger.info(f"Registered loop {loop_node_id} with max_iters={max_iters}")
        return tracker
    
    def get_loop_tracker(self, loop_node_id: str) -> Optional[LoopTracker]:
        """Get loop tracker"""
        return self.loops.get(loop_node_id)
    
    def increment_loop_iteration(self, loop_node_id: str) -> int:
        """Increment loop iteration and return new iteration number"""
        tracker = self.loops.get(loop_node_id)
        if not tracker:
            raise ValueError(f"Loop {loop_node_id} not registered")
        
        tracker.current_iteration += 1
        tracker.last_iteration_at = time.time()
        return tracker.current_iteration
    
    def set_loop_should_continue(self, loop_node_id: str, should_continue: bool) -> None:
        """Set whether loop should continue"""
        tracker = self.loops.get(loop_node_id)
        if tracker:
            tracker.should_continue = should_continue
