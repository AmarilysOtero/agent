"""Loop Node - Iterative execution with max_iters"""

from __future__ import annotations
from typing import Dict, Any
import logging

from .base import BaseNode
from ..graph_schema import NodeConfig
from ..workflow_state import WorkflowState
from ..agent_runner import AgentRunner
from ..condition_evaluator import ConditionEvaluator

logger = logging.getLogger(__name__)


class LoopNode(BaseNode):
    """Node that executes a loop with max iterations and condition checking"""
    
    async def execute(self) -> Dict[str, Any]:
        """Execute loop logic (iteration tracking handled by executor)"""
        max_iters = self.config.max_iters or 3
        current_iter = self.get_input("current_iter", 1)
        
        # Check loop condition if specified
        should_continue = True
        if self.config.loop_condition:
            should_continue = ConditionEvaluator.evaluate(
                self.config.loop_condition,
                self.state
            )
        
        # Check if we've exceeded max iterations
        if current_iter > max_iters:
            logger.info(f"LoopNode {self.config.id}: Max iterations ({max_iters}) reached")
            return {
                "continue": False,
                "reason": "max_iters_reached"
            }
        
        # Check condition
        if not should_continue:
            logger.info(f"LoopNode {self.config.id}: Loop condition not met, stopping")
            return {
                "continue": False,
                "reason": "condition_not_met"
            }
        
        logger.info(f"LoopNode {self.config.id}: Continuing iteration {current_iter}/{max_iters}")
        return {
            "continue": True,
            "current_iter": current_iter
        }
