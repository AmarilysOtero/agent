"""Conditional Node - Routes based on conditions"""

from __future__ import annotations
from typing import Dict, Any
import logging

from .base import BaseNode
from ..graph_schema import NodeConfig
from ..workflow_state import WorkflowState
from ..agent_runner import AgentRunner
from ..condition_evaluator import ConditionEvaluator

logger = logging.getLogger(__name__)


class ConditionalNode(BaseNode):
    """Node that evaluates a condition and routes accordingly"""
    
    async def execute(self) -> Dict[str, Any]:
        """Evaluate condition and return routing decision"""
        if not self.config.condition:
            raise ValueError(f"Conditional node {self.config.id} missing condition")
        
        result = ConditionEvaluator.evaluate(self.config.condition, self.state)
        
        logger.info(f"ConditionalNode {self.config.id}: Condition '{self.config.condition}' = {result}")
        
        return {
            "condition_result": result,
            "route": "true" if result else "false"
        }
