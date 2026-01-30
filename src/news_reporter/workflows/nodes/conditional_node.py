"""Conditional Node - Routes based on conditions"""

from __future__ import annotations
from typing import Dict, Any, List
import logging

from .base import BaseNode
from ...models.graph_schema import NodeConfig
from ..workflow_state import WorkflowState
from ..agent_runner import AgentRunner
from ..condition_evaluator import ConditionEvaluator
from ..node_result import NodeResult

logger = logging.getLogger(__name__)


class ConditionalNode(BaseNode):
    """Node that evaluates a condition and routes accordingly"""
    
    async def execute(self) -> NodeResult:
        """Evaluate condition and return routing decision"""
        if not self.config.condition:
            return NodeResult.failed(f"Conditional node {self.config.id} missing condition")
        
        try:
            result = ConditionEvaluator.evaluate(self.config.condition, self.state)
            
            logger.info(f"ConditionalNode {self.config.id}: Condition '{self.config.condition}' = {result}")
            
            # Determine next nodes based on condition
            # This will be handled by GraphExecutor using edge conditions
            # But we can also specify explicit next_nodes in params
            next_nodes: List[str] = []
            if result:
                next_nodes = self.config.params.get("true_nodes", [])
            else:
                next_nodes = self.config.params.get("false_nodes", [])
            
            return NodeResult.success(
                state_updates={
                    f"conditional.{self.config.id}.result": result
                },
                artifacts={
                    "condition_result": result,
                    "route": "true" if result else "false"
                },
                next_nodes=next_nodes
            )
        except Exception as e:
            logger.error(f"ConditionalNode {self.config.id} evaluation failed: {e}")
            return NodeResult.failed(str(e))
