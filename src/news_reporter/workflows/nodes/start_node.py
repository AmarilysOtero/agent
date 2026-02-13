"""Start Node - Entry point pass-through node"""

from __future__ import annotations
from typing import Dict, Any
import logging

from .base import BaseNode
from ...models.graph_schema import NodeConfig
from ..workflow_state import WorkflowState
from ..agent_runner import AgentRunner
from ..node_result import NodeResult, NodeStatus

logger = logging.getLogger(__name__)


class StartNode(BaseNode):
    """Node that serves as the workflow entry point
    
    The Start node is a pass-through node that:
    - Accepts the workflow goal as input
    - Returns the goal unchanged as output (state.latest)
    - Has no side effects
    - Cannot be deleted or have incoming edges
    """
    
    async def execute(self) -> NodeResult:
        """Execute start node - simply pass through the goal"""
        logger.info(f"StartNode {self.config.id}: Pass-through entry point")
        
        # Start node passes the workflow goal to downstream nodes
        # Set state.latest to goal so downstream nodes can use it
        state_updates = {
            "latest": self.state.goal
        }
        
        return NodeResult.success(
            state_updates=state_updates,
            artifacts={"start_goal": self.state.goal}
        )
