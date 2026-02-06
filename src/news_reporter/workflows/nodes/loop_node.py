"""Loop Node - Foundry-style for_each iteration with explicit termination contract (Phase 1)"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import logging

from .base import BaseNode
from ...models.graph_schema import NodeConfig
from ..workflow_state import WorkflowState
from ..agent_runner import AgentRunner
from ..condition_evaluator import ConditionEvaluator
from ..node_result import NodeResult, NodeStatus

logger = logging.getLogger(__name__)


class LoopNode(BaseNode):
    """
    Foundry-style for_each Loop Node (Phase 1).
    
    Supports:
    - TurnCount: items = [1..max_iters]
    - System.LastMessage.Text: items from system_vars (set once at run start)
    - Feedback iteration: final output of iteration N becomes input to iteration N+1
    - Deterministic empty-items exit (no state.latest update)
    - Routing via loop_continue / loop_exit edge tags
    
    Phase 1 constraints:
    - Cannot be graph entry (validation enforced)
    - No FanOut/Merge composition
    - State stored in state.loop_state[loop_id] (no dot paths)
    """
    
    async def execute(self) -> NodeResult:
        """
        Execute loop node: resolve items, manage iterations, inject feedback.
        
        Returns NodeResult with:
        - state_updates: {latest: inject_value, ...} when continuing
        - artifacts: {should_continue, iter_no, current_item, termination_reason, ...}
        """
        loop_id = self.config.id
        
        # Get configuration
        max_iters = self.config.max_iters
        item_source = self.config.params.get("item_source", "TurnCount")
        body_node_id = self.config.params.get("body_node_id")
        
        # Validate TurnCount mode requires max_iters
        if item_source == "TurnCount" and max_iters is None:
            return NodeResult.failed(
                f"LoopNode {loop_id}: TurnCount mode requires max_iters parameter"
            )
        
        # Get loop context (None on first entry, exists on re-entry)
        ctx = self.state.loop_state.get(loop_id)
        logger.info(f"[DEBUG] LoopNode {loop_id}: state_id={id(self.state)}, ctx={'FOUND' if ctx else 'NONE'}, all_loop_states={list(self.state.loop_state.keys())}")
        if ctx:
            logger.info(f"[DEBUG] LoopNode {loop_id}: Re-entry - iter_no={ctx.get('iter_no')}/{len(ctx.get('items', []))}")
        else:
            logger.info(f"[DEBUG] LoopNode {loop_id}: First entry - will initialize context")
        
        # First entry: resolve items, initialize context
        if ctx is None:
            items = self._resolve_items(item_source, max_iters)
            
            # Empty items → exit immediately (deterministic)
            if not items:
                logger.info(
                    f"LoopNode {loop_id}: No items resolved from '{item_source}', "
                    f"exiting immediately"
                )
                return NodeResult.success(
                    artifacts={
                        "should_continue": False,
                        "termination_reason": "no_items",
                        "item_source": item_source,
                        "iter_no": 0,
                        "current_item": None,
                        **({"max_iters": max_iters} if item_source == "TurnCount" else {})
                    }
                    # NOTE: No state_updates["latest"] - preserve upstream seed
                )
            
            # Initialize context
            seed = self.state.latest  # Upstream payload
            iter_no = 1
            current_item = items[0]
            inject_value = seed
            
            ctx = {
                "items": items,
                "item_source": item_source,
                "iter_no": iter_no,
                "seed": seed,
                "last_value": seed,
                "current_item": current_item
            }
            
            # Store context directly in state.loop_state[loop_id]
            self.state.loop_state[loop_id] = ctx
            logger.info(f"[DEBUG] LoopNode {loop_id}: Stored initial context, state_id={id(self.state)}, loop_state_keys={list(self.state.loop_state.keys())}")
            
            logger.info(
                f"LoopNode {loop_id}: First entry - iter_no={iter_no}/{len(items)}, "
                f"item_source={item_source}, current_item={current_item}"
            )
            
            # Inject seed into state.latest for body node
            state_updates = {
                "latest": inject_value,
                "current_iter": iter_no
            }
            
            return NodeResult.success(
                state_updates=state_updates,
                artifacts={
                    "should_continue": True,
                    "iter_no": iter_no,
                    "current_item": current_item,
                    "item_source": item_source,
                    "body_node_id": body_node_id,
                    **({"max_iters": max_iters} if item_source == "TurnCount" else {})
                }
            )
        
        # Re-entry from back-edge: body node completed iteration
        items = ctx["items"]
        completed_value = self.state.latest  # Body node final output
        
        # Check if all iterations complete
        if ctx["iter_no"] >= len(items):
            logger.info(
                f"LoopNode {loop_id}: All iterations complete ({ctx['iter_no']}/{len(items)}), "
                f"exiting with final value"
            )
            return NodeResult.success(
                state_updates={
                    "latest": completed_value  # Pass final loop output to downstream nodes
                },
                artifacts={
                    "should_continue": False,
                    "termination_reason": "iterations_complete",
                    "iter_no": ctx["iter_no"],
                    "current_item": ctx["current_item"],
                    "item_source": ctx["item_source"],
                    **({"max_iters": max_iters} if item_source == "TurnCount" else {})
                }
            )
        
        # Continue to next iteration
        iter_no = ctx["iter_no"] + 1
        current_item = items[iter_no - 1]
        inject_value = completed_value  # Feedback from previous iteration
        
        # Update context
        ctx["iter_no"] = iter_no
        ctx["current_item"] = current_item
        ctx["last_value"] = completed_value
        self.state.loop_state[loop_id] = ctx
        
        logger.info(
            f"LoopNode {loop_id}: Iteration {iter_no}/{len(items)}, "
            f"current_item={current_item}, injecting feedback from previous iteration"
        )
        
        # Inject completed value as input for next iteration
        state_updates = {
            "latest": inject_value,
            "current_iter": iter_no
        }
        
        return NodeResult.success(
            state_updates=state_updates,
            artifacts={
                "should_continue": True,
                "iter_no": iter_no,
                "current_item": current_item,
                "item_source": ctx["item_source"],
                "body_node_id": body_node_id,
                **({"max_iters": max_iters} if item_source == "TurnCount" else {})
            }
        )
    
    def _resolve_items(self, item_source: str, max_iters: Optional[int]) -> List[Any]:
        """
        Resolve items based on item_source.
        
        Supported sources:
        - "TurnCount": returns [1, 2, ..., max_iters]
        - "System.LastMessage.Text": returns [text] or [] if missing/empty
        
        Returns:
            List of items (empty list if no items)
        """
        if item_source == "TurnCount":
            if max_iters is None:
                logger.error(f"LoopNode {self.config.id}: TurnCount requires max_iters")
                return []
            return list(range(1, max_iters + 1))
        
        elif item_source == "System.LastMessage.Text":
            text = self.state.system_vars.get("System.LastMessage.Text")
            
            # Strict empty handling: None, empty string, or whitespace-only → no items
            if text is None or (isinstance(text, str) and not text.strip()):
                logger.debug(
                    f"LoopNode {self.config.id}: System.LastMessage.Text is missing/empty, "
                    f"returning no items"
                )
                return []
            
            # Return single-item list containing the text
            return [text]
        
        else:
            logger.error(
                f"LoopNode {self.config.id}: Unsupported item_source '{item_source}', "
                f"returning no items"
            )
            return []
