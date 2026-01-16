"""Merge Node - Combines multiple inputs"""

from __future__ import annotations
from typing import Dict, Any
import logging

from .base import BaseNode
from ..graph_schema import NodeConfig
from ..workflow_state import WorkflowState
from ..agent_runner import AgentRunner

logger = logging.getLogger(__name__)


class MergeNode(BaseNode):
    """Node that merges multiple inputs into a single output"""
    
    async def execute(self) -> Dict[str, Any]:
        """Merge inputs from state and return combined result"""
        # Get items to merge (e.g., final outputs from multiple reporters)
        merge_key = self.config.params.get("merge_key", "final")  # "final", "drafts", etc.
        items = self.state.get(merge_key, {})
        
        if not items:
            logger.warning(f"MergeNode {self.config.id}: No items to merge from '{merge_key}'")
            return {"merged": ""}
        
        # Merge strategy
        strategy = self.config.params.get("strategy", "stitch")  # "stitch", "join", "list"
        
        if strategy == "stitch":
            # Stitch with headers (for reporter outputs)
            stitched = []
            for key, value in items.items():
                stitched.append(f"### {key}\n{value}")
            merged = "\n\n---\n\n".join(stitched)
        elif strategy == "join":
            # Simple join with separator
            separator = self.config.params.get("separator", "\n\n")
            merged = separator.join(str(v) for v in items.values())
        elif strategy == "list":
            # Return as list
            merged = list(items.values())
        else:
            # Default: just join
            merged = "\n\n".join(str(v) for v in items.values())
        
        logger.info(f"MergeNode {self.config.id}: Merged {len(items)} items using strategy '{strategy}'")
        
        # Set output
        output_path = self.config.outputs.get("merged", "latest")
        self.state.set(output_path, merged)
        
        return {"merged": merged}
