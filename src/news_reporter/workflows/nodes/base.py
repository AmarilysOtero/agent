"""Base Node class for graph execution"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

from ...models.graph_schema import NodeConfig
from ..workflow_state import WorkflowState
from ..agent_runner import AgentRunner
from ..node_result import NodeResult

logger = logging.getLogger(__name__)


class BaseNode(ABC):
    """Abstract base class for all graph nodes"""
    
    def __init__(
        self,
        config: NodeConfig,
        state: WorkflowState,
        runner: AgentRunner,
        settings: Any  # Settings from config
    ):
        self.config = config
        self.state = state
        self.runner = runner
        self.settings = settings
    
    @abstractmethod
    async def execute(self) -> NodeResult:
        """
        Execute the node's logic.
        
        Returns:
            NodeResult with state_updates, artifacts, next_nodes, status, metrics
        """
        pass
    
    def get_input(self, input_key: str, default: Any = None) -> Any:
        """Get input value from state based on input mapping"""
        if input_key in self.config.inputs:
            path = self.config.inputs[input_key]
            return self.state.get(path, default)
        return default
    
    def set_output(self, output_key: str, value: Any) -> None:
        """Set output value to state based on output mapping"""
        if output_key in self.config.outputs:
            state_path = self.config.outputs[output_key]
            self.state.set(state_path, value)
        else:
            logger.warning(f"Output key '{output_key}' not found in node {self.config.id} outputs mapping")
