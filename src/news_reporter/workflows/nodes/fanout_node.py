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
        """Prepare fanout and forward parent output
        
        Fanout is a control node - branches are derived from graph topology.
        This node forwards the parent's output to all branches.
        """
        # Get parent output to broadcast (from parent_result)
        parent_output = None
        if hasattr(self, 'parent_result') and self.parent_result:
            parent_output = self.parent_result.state_updates.get('latest')
        
        # Fallback to state.latest or goal
        if parent_output is None:
            parent_output = self.state.latest or self.state.goal
        
        # Forward parent output so branches receive it
        # Executor will handle branch scheduling from graph edges
        return NodeResult.success(
            state_updates={"latest": parent_output},
            artifacts={"forwarded_output": parent_output}
        )
