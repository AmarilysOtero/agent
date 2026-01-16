"""Loop Node - Iterative execution with explicit termination contract"""

from __future__ import annotations
from typing import Dict, Any, Optional
import logging

from .base import BaseNode
from ..graph_schema import NodeConfig
from ..workflow_state import WorkflowState
from ..agent_runner import AgentRunner
from ..condition_evaluator import ConditionEvaluator
from ..node_result import NodeResult, NodeStatus

logger = logging.getLogger(__name__)


class LoopNode(BaseNode):
    """
    Node for iterative execution with explicit termination contract.
    
    Termination Contract:
    - max_iters: Maximum number of iterations (required)
    - continue_condition: Condition to continue looping (optional)
    - body_node_id: Node to execute in loop body (optional, uses outgoing edges if not set)
    - on_iteration_state_patch: State updates to apply at start of each iteration (optional)
    """
    
    async def execute(self) -> NodeResult:
        """
        Check loop termination contract and determine if loop should continue.
        
        Returns NodeResult with:
        - state_updates: Updates for current iteration
        - artifacts: Loop metadata (iteration, should_continue, etc.)
        - next_nodes: Body node to execute if continuing
        """
        # Get loop configuration
        max_iters = self.config.max_iters
        if max_iters is None:
            return NodeResult.failed(f"LoopNode {self.config.id} missing required max_iters")
        
        continue_condition = self.config.loop_condition
        body_node_id = self.config.params.get("body_node_id")
        on_iteration_state_patch = self.config.params.get("on_iteration_state_patch", {})
        
        # Get current iteration
        loop_state_key = f"loop_state.{self.config.id}"
        loop_state = self.state.get(loop_state_key, {"iteration": 0})
        current_iter = loop_state.get("iteration", 0) + 1
        
        # Check termination conditions
        should_continue = True
        termination_reason = None
        
        # Check max iterations
        if current_iter > max_iters:
            should_continue = False
            termination_reason = f"max_iters ({max_iters}) reached"
        
        # Check continue condition
        elif continue_condition:
            try:
                should_continue = ConditionEvaluator.evaluate(continue_condition, self.state)
                if not should_continue:
                    termination_reason = f"continue_condition not met: {continue_condition}"
            except Exception as e:
                logger.error(f"Error evaluating loop condition: {e}")
                should_continue = False
                termination_reason = f"condition evaluation error: {e}"
        
        # Update loop state
        loop_state["iteration"] = current_iter
        loop_state["should_continue"] = should_continue
        if termination_reason:
            loop_state["termination_reason"] = termination_reason
        
        # Apply iteration state patch
        state_updates = {
            loop_state_key: loop_state,
            "current_iter": current_iter
        }
        state_updates.update(on_iteration_state_patch)
        
        # Determine next nodes
        next_nodes = []
        if should_continue and body_node_id:
            next_nodes = [body_node_id]
        elif should_continue:
            # Use outgoing edges to find body nodes
            # This will be handled by GraphExecutor
            pass
        
        logger.info(
            f"LoopNode {self.config.id}: Iteration {current_iter}/{max_iters}, "
            f"continue={should_continue}, reason={termination_reason}"
        )
        
        return NodeResult.success(
            state_updates=state_updates,
            artifacts={
                "iteration": current_iter,
                "max_iters": max_iters,
                "should_continue": should_continue,
                "termination_reason": termination_reason,
                "body_node_id": body_node_id
            },
            next_nodes=next_nodes
        )
