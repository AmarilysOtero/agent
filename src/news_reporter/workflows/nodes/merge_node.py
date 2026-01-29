"""Merge Node - Combines multiple inputs with join barrier semantics"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import logging

from .base import BaseNode
from ...models.graph_schema import NodeConfig
from ..workflow_state import WorkflowState
from ..agent_runner import AgentRunner
from ..node_result import NodeResult, NodeStatus

logger = logging.getLogger(__name__)


class MergeNode(BaseNode):
    """
    Node that merges multiple inputs with explicit merge strategies and join barriers.
    
    Merge Strategies:
    - concat_text: Concatenate text values with separator
    - collect_list: Collect values into a list
    - merge_dict: Merge dictionaries (deep merge)
    - stitch: Stitch with headers (for reporter outputs)
    - custom_template: Use custom template from params
    """
    
    async def execute(self) -> NodeResult:
        """Merge inputs from parent_result or state with join barrier semantics"""
        # Get merge configuration
        merge_key = self.config.params.get("merge_key", "final")  # "final", "drafts", etc.
        strategy = self.config.params.get("strategy", "concat_text")
        expected_keys = self.config.params.get("expected_keys")  # Join barrier: wait for these keys
        timeout = self.config.params.get("timeout", 60.0)  # Timeout for join barrier
        
        # Get items to merge - prioritize parent_result (from fanout branches)
        items = None
        if hasattr(self, 'parent_result') and self.parent_result:
            # Check for branch outputs from fanout (primary path)
            if "branches" in self.parent_result.state_updates:
                items = self.parent_result.state_updates["branches"]
                logger.debug(f"MergeNode {self.config.id}: Using branch outputs from parent_result: {list(items.keys())}")
            # Check for legacy 'final' in parent_result (compatibility)
            elif merge_key in self.parent_result.state_updates:
                items = self.parent_result.state_updates[merge_key]
                logger.debug(f"MergeNode {self.config.id}: Using '{merge_key}' from parent_result")
        
        # Fallback to state if not provided via parent_result
        if items is None:
            items = self.state.get(merge_key, {})
            if items:
                logger.debug(f"MergeNode {self.config.id}: Using '{merge_key}' from state: {list(items.keys())}")
        
        # Ensure items is a dict
        if not isinstance(items, dict):
            logger.warning(f"MergeNode {self.config.id}: Items is not a dict, converting: {type(items)}")
            items = {}
        
        # Join barrier: wait for expected keys if specified
        if expected_keys:
            if not isinstance(expected_keys, list):
                expected_keys = [expected_keys]
            
            missing_keys = set(expected_keys) - set(items.keys())
            if missing_keys:
                logger.warning(
                    f"MergeNode {self.config.id}: Waiting for keys: {missing_keys}. "
                    f"Found: {list(items.keys())}"
                )
                # Phase 3: Join barrier - wait for expected keys
                # The executor will handle async waiting by re-queuing this node
                # until all expected keys are present or timeout is reached
                # For now, we'll return a result indicating we need to wait
                # The executor's _handle_merge_node will handle the actual waiting
                logger.info(
                    f"MergeNode {self.config.id}: Join barrier active. "
                    f"Waiting for keys: {missing_keys}. Found: {list(items.keys())}"
                )
                # Return result that will trigger executor to wait
                return NodeResult.success(
                    state_updates={},
                    artifacts={
                        "waiting_for_keys": list(missing_keys),
                        "found_keys": list(items.keys()),
                        "strategy": strategy,
                        "item_count": len(items)
                    }
                )
        
        if not items:
            logger.warning(f"MergeNode {self.config.id}: No items to merge (checked parent_result.branches, parent_result.{merge_key}, state.{merge_key})")
            # return NodeResult.success(
            #     state_updates={"latest": ""},
            #     artifacts={"merged": "", "strategy": strategy, "item_count": 0}
            # )
            error_msg = f"MergeNode {self.config.id}: No items to merge..."
            return NodeResult.failed(error_msg)
        
        # Apply merge strategy
        merged = self._apply_strategy(strategy, items, self.config.params)
        
        logger.info(
            f"MergeNode {self.config.id}: Merged {len(items)} items using strategy '{strategy}'. "
            f"Keys: {list(items.keys())}"
        )
        
        # Set output - always write to 'latest' for standard chaining, and also outputs.<node_id>
        state_updates = {
            "latest": merged,
            f"outputs.{self.config.id}": merged
        }
        
        return NodeResult.success(
            state_updates=state_updates,
            artifacts={
                "merged": merged,
                "strategy": strategy,
                "item_count": len(items),
                "expected_keys": expected_keys,
                "actual_keys": list(items.keys())
            }
        )
    
    def _apply_strategy(self, strategy: str, items: Dict[str, Any], params: Dict[str, Any]) -> Any:
        """Apply merge strategy to items"""
        if strategy == "concat_text":
            separator = params.get("separator", " | ")
            return separator.join(str(v) for v in items.values())
        
        elif strategy == "collect_list":
            return list(items.values())
        
        elif strategy == "merge_dict":
            # Deep merge dictionaries
            merged = {}
            for key, value in items.items():
                if isinstance(value, dict) and isinstance(merged.get(key), dict):
                    # Recursive merge
                    merged[key] = self._deep_merge(merged[key], value)
                else:
                    merged[key] = value
            return merged
        
        elif strategy == "stitch":
            # Stitch with headers (for reporter outputs)
            stitched = []
            for key, value in items.items():
                header = params.get("header_template", "### {key}").format(key=key)
                stitched.append(f"{header}\n{value}")
            separator = params.get("separator", "\n\n---\n\n")
            return separator.join(stitched)
        
        elif strategy == "custom_template":
            template = params.get("template", "{items}")
            # Simple template substitution
            # For more complex templates, use a template engine
            return template.format(items=items, count=len(items))
        
        else:
            # Default: concat_text
            logger.warning(f"Unknown merge strategy '{strategy}', using concat_text")
            separator = params.get("separator", "\n\n")
            return separator.join(str(v) for v in items.values())
    
    @staticmethod
    def _deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = MergeNode._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
