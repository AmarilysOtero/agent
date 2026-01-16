"""Agent Node - Executes a single agent"""

from __future__ import annotations
from typing import Dict, Any
import logging

from .base import BaseNode
from ..graph_schema import NodeConfig
from ..workflow_state import WorkflowState
from ..agent_runner import AgentRunner

logger = logging.getLogger(__name__)


class AgentNode(BaseNode):
    """Node that executes a single agent via AgentRunner"""
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the agent and return outputs"""
        if not self.config.agent_id:
            raise ValueError(f"Agent node {self.config.id} missing agent_id")
        
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
        logger.info(f"AgentNode {self.config.id}: Executing agent {self.config.agent_id}")
        result = await self.runner.run(
            agent_id=self.config.agent_id,
            input_data=input_data,
            context=self.state,
            **params
        )
        
        # Map outputs to state
        outputs = {}
        if self.config.outputs:
            # If outputs are mapped, use them
            for output_key, state_path in self.config.outputs.items():
                # For single output, use result directly
                if output_key == "result":
                    self.state.set(state_path, result)
                    outputs[output_key] = result
                else:
                    # For multiple outputs, result should be a dict
                    if isinstance(result, dict) and output_key in result:
                        value = result[output_key]
                        self.state.set(state_path, value)
                        outputs[output_key] = value
        else:
            # Default: store result in state.latest
            self.state.latest = str(result)
            outputs["result"] = result
        
        return outputs
