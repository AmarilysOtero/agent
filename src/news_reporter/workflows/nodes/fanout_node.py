"""Fanout Node - Parallel execution of multiple branches"""

from __future__ import annotations
from typing import Dict, Any, List
import asyncio
import logging

from .base import BaseNode
from ..graph_schema import NodeConfig
from ..workflow_state import WorkflowState
from ..agent_runner import AgentRunner

logger = logging.getLogger(__name__)


class FanoutNode(BaseNode):
    """Node that executes multiple branches in parallel"""
    
    async def execute(self) -> Dict[str, Any]:
        """Execute all branches in parallel and collect results"""
        if not self.config.branches:
            raise ValueError(f"Fanout node {self.config.id} missing branches")
        
        # Get list of items to fan out over (e.g., reporter_ids)
        fanout_items = self.get_input("items", [])
        if not fanout_items:
            # Default: use reporter_ids from settings
            fanout_items = getattr(self.settings, 'reporter_ids', [])
        
        if not fanout_items:
            logger.warning(f"Fanout node {self.config.id}: No items to fan out over")
            return {"results": []}
        
        logger.info(f"FanoutNode {self.config.id}: Fanning out over {len(fanout_items)} items")
        
        # Store fanout items in state for branch nodes to access
        self.state.set("fanout_items", fanout_items)
        
        # Results will be collected by the executor when branches complete
        return {
            "fanout_items": fanout_items,
            "branch_count": len(fanout_items)
        }
