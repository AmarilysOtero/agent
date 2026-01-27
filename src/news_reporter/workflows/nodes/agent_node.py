"""Agent Node - Executes a single agent"""

from __future__ import annotations
from typing import Dict, Any, Optional
import logging

from .base import BaseNode
from ...models.graph_schema import NodeConfig
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
                config=self.runner.config,
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
        
        # If no structured inputs, infer input for automatic chaining
        if not input_data:
            # Check if we have a parent result (for chaining)
            parent_result = getattr(self, 'parent_result', None)
            
            if parent_result:
                # Use parent's output for chaining
                # Prefer latest, fallback to agent_output artifact
                if 'latest' in parent_result.state_updates:
                    input_text = parent_result.state_updates['latest']
                elif 'agent_output' in parent_result.artifacts:
                    input_text = str(parent_result.artifacts['agent_output'])
                else:
                    # Fallback to first state update value
                    input_text = next(iter(parent_result.state_updates.values()), self.state.goal)
                
                # Downstream nodes receive ONLY parent output (no goal)
                input_data = {"input": str(input_text)}
                logger.debug(f"AgentNode {self.config.id}: Chaining - input='{str(input_text)[:120]}...'")
            else:
                # Entry node - use workflow goal only
                input_data = {"goal": self.state.goal}
                logger.debug(f"AgentNode {self.config.id}: Entry node - goal='{self.state.goal[:100]}...')")
        
        # Get additional params
        params = self.config.params.copy()
        
        # Log resolved inputs for traceability
        input_keys = list(input_data.keys())
        input_preview = " ".join([f"{k}='{str(v)[:120]}...'" if len(str(v)) > 120 else f"{k}='{v}'" 
                                   for k, v in input_data.items()])
        logger.info(f"[{self.config.id}] agent_id={self.config.agent_id} input_keys={input_keys} {input_preview}")
        
        # Execute agent
        result = await self.runner.run(
            agent_id=self.config.agent_id,
            input_data=input_data,
            context=self.state,
            config=self.runner.config,
            **params
        )
        
        # Map outputs to state (custom mappings)
        state_updates = {}
        if self.config.outputs:
            for output_key, state_path in self.config.outputs.items():
                if output_key == "result":
                    state_updates[state_path] = result
                elif isinstance(result, dict) and output_key in result:
                    state_updates[state_path] = result[output_key]
        
        # ALWAYS write terminal outputs (for arbitrary agent graphs)
        # This enables:
        # 1. Terminal output resolution via outputs.<node_id>
        # 2. Agent chaining via latest
        state_updates["latest"] = str(result)
        state_updates[f"outputs.{self.config.id}"] = str(result)
        
        return NodeResult.success(state_updates=state_updates, artifacts={"agent_output": result})
