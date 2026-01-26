"""Fanout Node - Parallel execution of multiple branches"""

from __future__ import annotations
from typing import Dict, Any, List
import logging

from .base import BaseNode
from ...models.graph_schema import NodeConfig
from ..workflow_state import WorkflowState
from ..agent_runner import AgentRunner
from ..node_result import NodeResult

logger = logging.getLogger(__name__)


class FanoutNode(BaseNode):
    """Node that executes multiple branches in parallel"""
    
    async def execute(self) -> NodeResult:
        """Prepare fanout and return branch information"""
        if not self.config.branches:
            return NodeResult.failed(f"Fanout node {self.config.id} missing branches")
        
        # Get list of items to fan out over (e.g., reporter_ids)
        fanout_items = self.get_input("items", [])
        if not fanout_items:
            # Default: use reporter_ids from settings
            fanout_items = getattr(self.settings, 'reporter_ids', [])
        
        if not fanout_items:
            logger.warning(f"Fanout node {self.config.id}: No items to fan out over")
            return NodeResult.success(
                state_updates={"fanout_items": []},
                artifacts={"fanout_items": [], "branch_count": 0}
            )
        
        logger.info(f"FanoutNode {self.config.id}: Fanning out over {len(fanout_items)} items")
        
        # Store fanout items in state for branch nodes to access
        # Results will be collected by the executor when branches complete
        return NodeResult.success(
            state_updates={"fanout_items": fanout_items},
            artifacts={
                "fanout_items": fanout_items,
                "branch_count": len(fanout_items),
                "branches": self.config.branches
            },
            next_nodes=self.config.branches  # Execute all branches
        )
