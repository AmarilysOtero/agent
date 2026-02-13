"""Node registry and factory"""

from __future__ import annotations
from typing import Dict, Type

from .base import BaseNode
from .agent_node import AgentNode
from .fanout_node import FanoutNode
from .loop_node import LoopNode
from .conditional_node import ConditionalNode
from .merge_node import MergeNode
from .start_node import StartNode

# Node type registry
NODE_TYPES: Dict[str, Type[BaseNode]] = {
    "agent": AgentNode,
    "fanout": FanoutNode,
    "loop": LoopNode,
    "conditional": ConditionalNode,
    "merge": MergeNode,
    "start": StartNode,
}


def create_node(
    node_type: str,
    config,
    state,
    runner,
    settings
) -> BaseNode:
    """
    Factory function to create nodes by type.
    
    Args:
        node_type: Node type string ("agent", "fanout", etc.)
        config: NodeConfig instance
        state: WorkflowState instance
        runner: AgentRunner instance
        settings: Settings instance
    
    Returns:
        BaseNode instance
    """
    node_class = NODE_TYPES.get(node_type)
    if not node_class:
        raise ValueError(f"Unknown node type: {node_type}. Available: {list(NODE_TYPES.keys())}")
    
    return node_class(config, state, runner, settings)


__all__ = [
    "BaseNode",
    "AgentNode",
    "FanoutNode",
    "LoopNode",
    "ConditionalNode",
    "MergeNode",
    "NODE_TYPES",
    "create_node",
    "StartNode",
]