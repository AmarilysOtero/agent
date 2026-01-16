"""Agent Node - Executes a single agent"""

from __future__ import annotations
from typing import Dict, Any, Optional
import logging

from .base import BaseNode
from ..graph_schema import NodeConfig
from ..workflow_state import WorkflowState
from ..agent_runner import AgentRunner
from ..node_result import NodeResult, NodeStatus
from ..agent_adapter import AgentAdapterRegistry

logger = logging.getLogger(__name__)


class AgentNode(BaseNode):
    """Node that executes a single agent via AgentRunner with AgentAdapter"""
    
    async def execute(self) -> NodeResult:
        """Execute the agent and return NodeResult"""
        if not self.config.agent_id:
            return NodeResult.failed(f"Agent node {self.config.id} missing agent_id")
        
        # Get adapter for this agent
        adapter = AgentAdapterRegistry.get_adapter_by_agent_id(self.config.agent_id)
        if not adapter:
            logger.warning(f"No adapter found for agent_id {self.config.agent_id}, using default behavior")
            # Fallback to old behavior
            return await self._execute_without_adapter()
        
        # Check skip condition (if any)
        skip_condition = self.config.params.get("skip_condition")
        if skip_condition:
            from ..condition_evaluator import ConditionEvaluator
            if ConditionEvaluator.evaluate(skip_condition, self.state):
                logger.info(f"AgentNode {self.config.id}: Skipped due to condition: {skip_condition}")
                return NodeResult.skipped(f"Skip condition met: {skip_condition}")
        
        # Build input using adapter
        params = self.config.params.copy()
        try:
            input_data = adapter.build_input(self.state, params)
            
            # Execute agent
            logger.info(f"AgentNode {self.config.id}: Executing agent {self.config.agent_id}")
            result = await self.runner.run(
                agent_id=self.config.agent_id,
                input_data=input_data,
                context=self.state,
                **params
            )
            
            # Parse output using adapter
            state_updates = adapter.parse_output(result, self.state)
            
            # Create NodeResult
            node_result = NodeResult.success(
                state_updates=state_updates,
                artifacts={"agent_output": result}
            )
            node_result.add_metric("agent_id", self.config.agent_id)
            
            return node_result
            
        except Exception as e:
            logger.error(f"AgentNode {self.config.id} failed: {e}", exc_info=True)
            return NodeResult.failed(str(e), {"agent_id": self.config.agent_id})
    
    async def _execute_without_adapter(self) -> NodeResult:
        """Fallback execution without adapter (old behavior)"""
        # Prepare input from state
        input_data = {}
        for input_key, state_path in self.config.inputs.items():
            value = self.state.get(state_path)
            input_data[input_key] = value
        
        # If no structured inputs, use goal as default
        if not input_data:
            input_data = {"goal": self.state.goal}
        
        # Get additional params
        params = self.config.params.copy()
        
        # Execute agent
        result = await self.runner.run(
            agent_id=self.config.agent_id,
            input_data=input_data,
            context=self.state,
            **params
        )
        
        # Map outputs to state (default behavior)
        state_updates = {}
        if self.config.outputs:
            for output_key, state_path in self.config.outputs.items():
                if output_key == "result":
                    state_updates[state_path] = result
                elif isinstance(result, dict) and output_key in result:
                    state_updates[state_path] = result[output_key]
        else:
            state_updates["latest"] = str(result)
        
        return NodeResult.success(state_updates=state_updates, artifacts={"agent_output": result})
