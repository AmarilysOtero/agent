"""Graph Executor - Queue-based execution with NodeResult and ExecutionContext (Phase 5)"""

from __future__ import annotations
from typing import Dict, List, Set, Optional, Any, Tuple, TYPE_CHECKING
from collections import deque, defaultdict
import asyncio
import time
import logging

from ..models.graph_schema import GraphDefinition, NodeConfig, EdgeConfig, GraphLimits
from .workflow_state import WorkflowState
from .agent_runner import AgentRunner
from .condition_evaluator import ConditionEvaluator
from .execution_context import ExecutionContext
from .node_result import NodeResult, NodeStatus
from .execution_tracker import ExecutionTracker, FanoutTracker, LoopTracker
from .state_checkpoint import StateCheckpoint
from .nodes import create_node, AgentNode, FanoutNode, LoopNode, ConditionalNode, MergeNode
from .agent_adapter import AgentAdapterRegistry
from .performance_metrics import get_metrics_collector
from .cache_manager import get_cache_manager
from .retry_handler import RetryConfig, RetryHandler
from .execution_monitor import get_execution_monitor
from .code_execution_tracer import CodeExecutionTracer
from ..config import Settings
from pathlib import Path

logger = logging.getLogger(__name__)


class ExecutionToken:
    """Token representing a node execution task in the queue"""
    
    def __init__(
        self,
        node_id: str,
        context: ExecutionContext,
        parent_result: Optional[NodeResult] = None
    ):
        self.node_id = node_id
        self.context = context
        self.parent_result = parent_result
        self.created_at = time.time()
    
    def __repr__(self) -> str:
        return f"ExecutionToken(node_id={self.node_id}, branch_id={self.context.branch_id[:8]}, iter={self.context.iteration})"


class GraphExecutor:
    """
    Queue-based graph executor with support for:
    - Dynamic graph execution (cycles, loops)
    - Branch tracking with ExecutionContext
    - Structured outputs with NodeResult
    - Join barriers for fanout/merge
    - Skip semantics
    - Performance metrics (Phase 4)
    - Retry mechanisms (Phase 4)
    - Caching (Phase 4)
    - Real-time monitoring (Phase 5)
    """
    
    def __init__(self, graph_def: GraphDefinition, config: Settings):
        self.graph_def = graph_def
        self.config = config
        self.runner = AgentRunner(config)
        
        # Initialize agent adapters
        AgentAdapterRegistry.initialize_defaults(config)
        
        # Validate graph
        errors = graph_def.validate()
        if errors:
            # Filter out triage-specific validation error - entry nodes will be inferred
            # Only structural errors (cycles, missing nodes) should block execution
            non_triage_errors = [e for e in errors if "Entry node 'triage'" not in e]
            if non_triage_errors:
                raise ValueError(f"Invalid graph definition: {', '.join(non_triage_errors)}")
        
        # Build execution graph
        self._build_execution_graph()
        
        # Get execution limits from graph definition
        self.limits = graph_def.limits or GraphLimits()
        self.max_steps = self.limits.max_steps or 1000
        self.timeout_ms = self.limits.timeout_ms
        self.max_parallel = self.limits.max_parallel
        
        # Phase 3: State checkpointing (optional)
        checkpoint_dir = getattr(config, 'checkpoint_dir', None)
        self.checkpoint_manager = StateCheckpoint(checkpoint_dir) if checkpoint_dir else None
        
        # Phase 4: Performance metrics, retry, and caching
        self.metrics_collector = get_metrics_collector()
        self.cache_manager = get_cache_manager()
        
        # Retry configuration
        retry_config = RetryConfig(
            max_retries=getattr(config, 'max_retries', 3),
            initial_delay_ms=getattr(config, 'retry_delay_ms', 1000.0)
        )
        self.retry_handler = RetryHandler(retry_config)
        
        # Phase 5: Execution monitoring
        self.monitor = get_execution_monitor()
    
    def _build_execution_graph(self) -> None:
        """Build internal data structures for efficient execution"""
        # Node lookup
        self.nodes: Dict[str, NodeConfig] = {node.id: node for node in self.graph_def.nodes}
        
        # Adjacency lists
        self.outgoing_edges: Dict[str, List[EdgeConfig]] = {}
        self.incoming_edges: Dict[str, List[EdgeConfig]] = {}
        
        for edge in self.graph_def.edges:
            if edge.from_node not in self.outgoing_edges:
                self.outgoing_edges[edge.from_node] = []
            self.outgoing_edges[edge.from_node].append(edge)
            
            if edge.to_node not in self.incoming_edges:
                self.incoming_edges[edge.to_node] = []
            self.incoming_edges[edge.to_node].append(edge)
    
    async def execute(self, goal: str) -> str:
        """
        Execute the graph with given goal using queue-based system.
        
        Args:
            goal: User goal/query
        
        Returns:
            Final output string
        
        Raises:
            TimeoutError: If execution exceeds timeout
            RuntimeError: If execution fails or exceeds max steps
        """
        # NOTE: `goal` is the inbound user text for this run (System.LastMessage.Text source of truth).
        user_text = goal
        
        # Initialize state
        state = WorkflowState(goal=goal)
        
        # Set System.LastMessage.Text once at run start (never derived from state.latest)
        state.system_vars["System.LastMessage.Text"] = user_text
        
        logger.info("=" * 100)
        logger.info("=" * 100)
        logger.info(f"🚀 GRAPH EXECUTOR STARTING - Goal: {goal}")
        logger.info("=" * 100)
        logger.info("=" * 100)
        state.append_log("INFO", f"Starting graph execution with goal: {goal}")
        
        # Initialize code execution tracer
        tracer = CodeExecutionTracer()
        
        # Find entry nodes
        entry_nodes = self.graph_def.get_entry_nodes()
        if not entry_nodes:
            raise ValueError("Graph has no entry nodes")
        
        # Initialize execution context
        root_context = ExecutionContext(node_id=entry_nodes[0])
        run_id = root_context.run_id
        
        # Phase 3: Try to restore from checkpoint if available
        if self.checkpoint_manager:
            restored_state = self.checkpoint_manager.restore_state(run_id)
            if restored_state:
                state = restored_state
                state.append_log("INFO", f"Resumed execution from checkpoint for run {run_id}")
                logger.info(f"Resumed execution from checkpoint for run {run_id}")
        
        # Execute graph with timeout
        start_time = time.time()
        
        # Phase 3: Save initial checkpoint
        if self.checkpoint_manager:
            try:
                self.checkpoint_manager.save_checkpoint(
                    run_id=run_id,
                    state=state,
                    metadata={"goal": goal, "start_time": start_time}
                )
            except Exception as e:
                logger.warning(f"Failed to save initial checkpoint: {e}")
        try:
            if self.timeout_ms:
                await asyncio.wait_for(
                    self._execute_queue_based(state, entry_nodes, root_context, tracer),
                    timeout=self.timeout_ms / 1000.0
                )
            else:
                await self._execute_queue_based(state, entry_nodes, root_context, tracer)
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            error_msg = f"Graph execution timed out after {duration:.2f}s (limit: {self.timeout_ms}ms)"
            state.append_log("ERROR", error_msg)
            logger.error(error_msg)
            raise TimeoutError(error_msg)
        except Exception as e:
            state.append_log("ERROR", f"Graph execution failed: {str(e)}")
            logger.error(f"Graph execution failed: {e}", exc_info=True)
            # Phase 4: End metrics collection on error
            self.metrics_collector.end_workflow()
            # Phase 5: Emit workflow failed event
            self.monitor.workflow_failed(run_id, str(e))
            raise
        finally:
            # Phase 4: End metrics collection
            self.metrics_collector.end_workflow()
        
        # Get final output
        final_output = self._get_final_output(state)
        
        # Phase 5: Emit workflow completed event
        duration_ms = (time.time() - start_time) * 1000
        self.monitor.workflow_completed(run_id, final_output, duration_ms)
        
        # Write code execution script
        try:
            output_dir = Path(__file__).parent
            state_snapshot = {
                "goal": state.goal,
                "triage": state.get("triage"),
                "latest": state.latest,
                "outputs": state.outputs,
                "conditional": state.get("conditional", {}),
                "loop_state": state.get("loop_state", {})
            }
            tracer.write_execution_script(
                output_dir=output_dir,
                run_id=run_id,
                goal=goal,
                start_time=start_time,
                duration_ms=duration_ms,
                final_output=final_output,
                state_snapshot=state_snapshot
            )
        except Exception as e:
            logger.warning(f"Failed to write code execution script: {e}")
        
        return final_output
    
    async def _execute_queue_based(
        self,
        state: WorkflowState,
        entry_nodes: List[str],
        root_context: ExecutionContext,
        tracer: CodeExecutionTracer
    ) -> None:
        """Execute graph using queue-based system with Phase 3 enhancements"""
        # Execution queue
        queue: deque[ExecutionToken] = deque()
        
        # Track execution state
        executed: Set[Tuple[str, int]] = set()  # (node_id, iteration) tuples
        executing: Set[str] = set()  # Nodes currently executing
        node_results: Dict[str, NodeResult] = {}  # Results by node_id
        
        # Branch tracking
        branch_contexts: Dict[str, ExecutionContext] = {}  # branch_id -> context
        branch_results: Dict[str, List[NodeResult]] = {}  # branch_id -> results
        
        # Phase 3: Execution tracker for fanout/loop coordination
        tracker = ExecutionTracker()
        
        # Step counter for limits
        step_count = 0
        run_id = root_context.run_id  # Store for checkpointing
        
        # Add entry nodes to queue
        for node_id in entry_nodes:
            context = root_context.create_child_branch(node_id)
            context.node_id = node_id
            branch_contexts[context.branch_id] = context
            queue.append(ExecutionToken(node_id=node_id, context=context))
        
        # Process queue
        while queue and step_count < self.max_steps:
            step_count += 1
            logger.info(f"[DEBUG] Step {step_count}: Queue size={len(queue)}")
            
            # Check for parallel execution limit
            if self.max_parallel and len(executing) >= self.max_parallel:
                # Wait a bit if we're at parallel limit
                await asyncio.sleep(0.01)
                continue
            
            token = queue.popleft()
            node_id = token.node_id
            context = token.context
            
            # Create execution key (node_id, iteration)
            exec_key = (node_id, context.iteration)
            
            # Skip if already executed (unless it's a loop iteration or loop node re-entry)
            node_config = self.nodes.get(node_id)
            is_loop_node = node_config and node_config.type == "loop"
            
            # Loop nodes can be re-executed (they manage iterations internally)
            # Other nodes are skipped if already executed at this iteration
            if exec_key in executed and not is_loop_node:
                logger.debug(f"[DEBUG] Skipping already executed node {node_id} at iteration {context.iteration}")
                continue
            
            logger.info(f"[DEBUG] Processing node {node_id}, is_loop={is_loop_node}, exec_key={exec_key}")
            
            # Check if node is already executing (prevent duplicate execution)
            if node_id in executing:
                # Re-queue for later
                queue.append(token)
                await asyncio.sleep(0.01)  # Small delay
                continue
            
            # Execute node
            executing.add(node_id)
            try:
                # Record node execution
                tracer.record_execution(
                    method_name="_execute_node",
                    args={"node_id": node_id},
                    kwargs={"state": state, "context": context, "parent_result": token.parent_result}
                )
                
                result = await self._execute_node(node_id, state, context, token.parent_result)
                node_results[node_id] = result
                
                # Record result
                tracer.record_execution(
                    method_name="_execute_node",
                    result=result,
                    result_type="NodeResult",
                    args={"node_id": node_id}
                )
               # Track execution
                executed.add(exec_key)
                node_results[node_id] = result
                
                # Apply state updates
                self._apply_state_updates(state, result.state_updates)
                
                # Update branch results
                if context.branch_id in branch_results:
                    branch_results[context.branch_id].append(result)
                else:
                    branch_results[context.branch_id] = [result]
                
                # Remove from executing set
                executing.discard(node_id)
                
                # ============================================================================
                # PHASE 5: Loop Feedback Routing
                # ============================================================================
                # If this node was reached via loop_continue (has parent_loop_id), 
                # check if it has outgoing edges. If not, route back to loop.
                # If yes, follow edges normally (for sequential body chains like B→C→Loop)
                if context.parent_loop_id:
                    outgoing = self.outgoing_edges.get(node_id, [])
                    has_outgoing = len(outgoing) > 0
                    
                    if not has_outgoing:
                        # Terminal node in loop body - route back to loop
                        logger.info(
                            f"LoopNode {context.parent_loop_id}: Body node {node_id} completed (terminal), "
                            f"routing back to loop for re-entry"
                        )
                        
                        # Create context for loop re-entry
                        loop_context = context.create_child_branch(context.parent_loop_id)
                        loop_context.parent_loop_id = None  # CRITICAL: Clear parent_loop_id - loop is not its own body!
                        
                        # Queue loop node for re-entry
                        queue.append(ExecutionToken(
                            node_id=context.parent_loop_id,
                            context=loop_context,
                            parent_result=result
                        ))
                        logger.info(f"[DEBUG] Queued loop re-entry, queue size now: {len(queue)}")
                        
                        # Skip normal edge routing - loop re-entry takes precedence
                        continue
                    else:
                        # Non-terminal node in loop body - continue to next node in body chain
                        logger.info(
                            f"LoopNode {context.parent_loop_id}: Body node {node_id} has outgoing edges, "
                            f"continuing to next body node(s)"
                        )
                        # Fall through to normal edge routing, parent_loop_id will propagate
                
                # ============================================================================
                # Normal Edge Routing (non-loop nodes)
                # ============================================================================
                # Phase 3: Mark branch as complete if it's part of a fanout
                # Only mark complete if this branch is actually tracked (part of a fanout)
                if context.branch_id in tracker.branch_to_fanout:
                    tracker.mark_branch_complete(context.branch_id, result)
                
                # Phase 3: Check if fanout is complete and trigger merge
                await self._check_fanout_completion(node_id, state, tracker, queue, context)
                
                # Phase 3: Handle special node types
                next_nodes = await self._handle_special_nodes(
                    node_id, result, state, context, tracker, queue
                )
                
                # If next_nodes not set by special handler, determine normally
                if next_nodes is None:
                    next_nodes = self._determine_next_nodes(node_id, result, state, tracker)
                
                # Phase 3: Check for loop back (body node completing, needs to loop back to loop node)
                # First, check if this node is part of a loop by checking if any loop node has an outgoing edge to this node
                # IMPORTANT: Only consider "loop_body" edges, not exit edges (edges without "loop_body" condition are exit paths)
                loop_node_id = None
                for loop_id, loop_tracker in tracker.loops.items():
                    # Check if this node is the explicit body node
                    if loop_tracker.body_node_id == node_id:
                        loop_node_id = loop_id
                        break
                    # Check if this node is reachable from the loop node via a "loop_body" edge (is a body node)
                    loop_outgoing = self.outgoing_edges.get(loop_id, [])
                    for edge in loop_outgoing:
                        # Only consider edges with "loop_body" condition as loop body edges
                        # Exit edges (no condition or other conditions) should not make a node part of the loop
                        if edge.to_node == node_id and edge.condition == "loop_body":
                            loop_node_id = loop_id
                            break
                    if loop_node_id:
                        break
                
                # If this node is part of a loop and no next nodes were determined, loop back to loop node
                if loop_node_id and not next_nodes:
                    logger.info(f"Node {node_id} is part of loop {loop_node_id} and has no valid edges, looping back to check exit condition")
                    next_nodes = [loop_node_id]
                
                # Add next nodes to queue
                for next_node_id in next_nodes:
                    # Create child branch (loop iteration handling is done in _handle_loop_node)
                    next_context = context.create_child_branch(next_node_id)
                    next_context.node_id = next_node_id
                    
                    # Phase 5: Check if this edge is a loop edge and set parent_loop_id
                    outgoing = self.outgoing_edges.get(node_id, [])
                    for edge in outgoing:
                        if edge.to_node == next_node_id:
                            if edge.condition == "loop_continue":
                                # Body node - track parent loop for feedback routing
                                next_context.parent_loop_id = node_id
                                # CRITICAL: Use loop's internal iter_no for body execution
                                loop_iter = result.artifacts.get("iter_no", 1)
                                next_context.iteration = loop_iter
                                logger.debug(f"Setting parent_loop_id={node_id} for body node {next_node_id}, iter={loop_iter}")
                            elif edge.condition == "loop_exit":
                                # Exit node - clear loop membership
                                next_context.parent_loop_id = None
                                logger.debug(f"Clearing parent_loop_id for exit node {next_node_id}")
                            else:
                                # Normal edge - inherit parent_loop_id if current node is in a loop
                                if context.parent_loop_id:
                                    next_context.parent_loop_id = context.parent_loop_id
                                    next_context.iteration = context.iteration  # Also inherit iteration
                                    logger.debug(f"Propagating parent_loop_id={context.parent_loop_id} to {next_node_id} (loop body chain)")
                            break
                    
                    branch_contexts[next_context.branch_id] = next_context
                    
                    # Regression guard: Log parent_result presence for debugging
                    has_parent = result is not None
                    logger.debug(
                        f"Enqueueing {next_node_id}: parent_result={'present' if has_parent else 'None'} "
                        f"(parent={node_id})"
                    )
                    
                    queue.append(ExecutionToken(
                        node_id=next_node_id,
                        context=next_context,
                        parent_result=result
                    ))
                
                executed.add(exec_key)
                logger.info(f"[DEBUG] Added {exec_key} to executed set, total executed={len(executed)}")
                
                # Phase 3: Periodic checkpointing (every 10 steps)
                if self.checkpoint_manager and step_count % 10 == 0:
                    self.checkpoint_manager.save_checkpoint(
                        run_id=run_id,  # Use stored run_id
                        state=state,
                        metadata={"step_count": step_count, "executed_nodes": len(executed)}
                    )
                
            except Exception as e:
                logger.error(f"Node {node_id} execution error: {e}", exc_info=True)
                result = NodeResult.failed(str(e))
                node_results[node_id] = result
                state.append_log("ERROR", f"Node {node_id} failed: {str(e)}", node_id=node_id)
                
                # Phase 3: Error recovery - continue or stop based on config
                # Note: config is Settings, not a dict, so we check graph_def limits
                error_strategy = getattr(self.graph_def, 'error_strategy', 'continue')
                if error_strategy == "stop":
                    raise RuntimeError(f"Node {node_id} failed and error_strategy is 'stop': {e}")
            finally:
                executing.discard(node_id)
        
        if step_count >= self.max_steps:
            logger.warning(f"Graph execution reached max steps ({self.max_steps})")
            state.append_log("WARNING", f"Execution stopped after {self.max_steps} steps")
            raise RuntimeError(f"Execution exceeded max steps limit: {self.max_steps}")
        
        # Check for stuck execution
        if executing:
            logger.warning(f"Some nodes are still executing: {executing}")
        
        # Check for incomplete fanouts
        incomplete_fanouts = [
            fanout_id for fanout_id, fanout in tracker.fanouts.items()
            if not fanout.all_branches_complete()
        ]
        if incomplete_fanouts:
            logger.warning(f"Incomplete fanouts: {incomplete_fanouts}")
    
    async def _handle_special_nodes(
        self,
        node_id: str,
        result: NodeResult,
        state: WorkflowState,
        context: ExecutionContext,
        tracker: ExecutionTracker,
        queue: deque[ExecutionToken]
    ) -> Optional[List[str]]:
        """Handle special node types (fanout, loop, merge) - Phase 3"""
        node_config = self.nodes[node_id]
        
        # Handle fanout nodes
        if node_config.type == "fanout":
            return await self._handle_fanout_node(node_id, result, state, context, tracker, queue)
        
        # Handle loop nodes
        elif node_config.type == "loop":
            return await self._handle_loop_node(node_id, result, state, context, tracker, queue)
        
        # Handle merge nodes
        elif node_config.type == "merge":
            return await self._handle_merge_node(node_id, result, state, context, tracker, queue)
        
        return None
    
    def _find_merge_node(
        self,
        fanout_node_id: str,
        branch_node_ids: List[str]
    ) -> Optional[str]:
        """Find merge node that receives outputs from all branches
        
        Looks for a merge-type node that has incoming edges from all branch nodes.
        Falls back to fanout params if specified.
        """
        # Find candidate merge nodes (type == "merge")
        merge_candidates = [
            node_id for node_id, config in self.nodes.items()
            if config.type == "merge"
        ]
        
        for merge_id in merge_candidates:
            # Check if merge has incoming edges from ALL branch nodes
            incoming = self.incoming_edges.get(merge_id, [])
            incoming_sources = {edge.from_node for edge in incoming}
            
            if set(branch_node_ids).issubset(incoming_sources):
                logger.debug(f"Found merge node {merge_id} for fanout {fanout_node_id}")
                return merge_id
        
        # Fallback: check fanout params
        node_config = self.nodes[fanout_node_id]
        if node_config.params and "merge_node_id" in node_config.params:
            merge_id = node_config.params["merge_node_id"]
            logger.debug(f"Using merge node {merge_id} from fanout params")
            return merge_id
        
        logger.warning(f"No merge node found for fanout {fanout_node_id}")
        return None
    
    async def _handle_fanout_node(
        self,
        node_id: str,
        result: NodeResult,
        state: WorkflowState,
        context: ExecutionContext,
        tracker: ExecutionTracker,
        queue: deque[ExecutionToken]
    ) -> List[str]:
        """Handle fanout node - create branches for each item
        
        Derives branch nodes from graph edges instead of requiring artifacts.
        Broadcasts parent output to all branches.
        """
        node_config = self.nodes[node_id]
        
        # Derive branch_node_ids from outgoing edges
        outgoing = self.outgoing_edges.get(node_id, [])
        branch_node_ids = [edge.to_node for edge in outgoing]
        
        # Override with node config if specified
        if node_config.branches:
            branch_node_ids = node_config.branches
        
        # Validate: fanout must have at least one branch
        if not branch_node_ids:
            logger.warning(f"Fanout node {node_id} has no outgoing edges")
            return []
        
        # Get parent output to broadcast (from parent_result, not state.latest)
        parent_output = result.state_updates.get('latest')
        if parent_output is None:
            # Fallback to artifacts or goal
            parent_output = result.artifacts.get('output') or state.goal
        
        # For simple parallel workflows: single-item broadcast
        fanout_items = [parent_output]
        
        # Find correct merge node using helper
        merge_node_id = self._find_merge_node(node_id, branch_node_ids)
        
        # TASK 1: Use branch_node_ids as items to prevent tracking collision
        # For simple broadcast fanout, each branch IS the item (not item × branches)
        # This ensures tracker.branches has one entry per branch instead of overwriting
        fanout_items_for_tracker = branch_node_ids  # Each branch is uniquely tracked
        
        # Register fanout in tracker with CORRECT merge node
        fanout_tracker = tracker.register_fanout(
            fanout_node_id=node_id,
            items=fanout_items_for_tracker,  # Use branch_node_ids as items
            branch_node_ids=branch_node_ids,
            merge_node_id=merge_node_id
        )
        
        logger.info(
            f"\n\nFanout {node_id}: broadcasting to {len(branch_node_ids)} branches, "
            f"merge_node={merge_node_id}"
        )
        
        # Create one branch execution per branch node (broadcast same parent_result to all)
        for branch_node_id in branch_node_ids:
            # Create branch context
            branch_context = context.create_child_branch(branch_node_id)
            branch_context.node_id = branch_node_id
            
            # Register branch in tracker (item = branch_node_id for unique tracking)
            tracker.register_branch(
                fanout_node_id=node_id,
                item=branch_node_id,  # Use branch_node_id as the unique item key
                branch_id=branch_context.branch_id,
                branch_node_id=branch_node_id
            )
                
            # Add to queue with parent_result for broadcast
            # Regression guard: Fanout branches MUST have parent_result to avoid {goal} seeding
            # This would  break goal/input semantics
            if result is None:
                logger.error(
                    f"SANITY CHECK FAILED: Fanout branch {branch_node_id} enqueued with parent_result=None! "
                    f"This would cause it to use {{goal}} instead of {{input}}. "
                    f"Fanout:{node_id}, Item:{item}"
                )
                raise RuntimeError(
                    f"Fanout branch {branch_node_id} missing parent_result" 
                )
            
            logger.debug(
                f"Enqueueing fanout branch {branch_node_id}: parent_result=present (fanout={node_id})"
            )
            
            queue.append(ExecutionToken(
                node_id=branch_node_id,
                context=branch_context,
                parent_result=result  # Broadcasts parent output to branches
            ))
        
        # Don't continue to merge node yet - wait for all branches
        # Merge node will be triggered when all branches complete
        return []
    
    async def _check_fanout_completion(
        self,
        node_id: str,
        state: WorkflowState,
        tracker: ExecutionTracker,
        queue: deque[ExecutionToken],
        context: ExecutionContext
    ) -> None:
        """Check if a fanout is complete and trigger merge node if so"""
        # TASK 2: Find which fanout this branch belongs to using branch_id mapping
        # This is more correct than node_id scanning in multi-fanout graphs
        fanout_node_id = tracker.branch_to_fanout.get(context.branch_id)
        if not fanout_node_id:
            # This branch is not part of any tracked fanout
            return
        
        fanout_tracker = tracker.fanouts.get(fanout_node_id)
        if not fanout_tracker:
            # Fanout tracker not found (shouldn't happen)
            logger.warning(f"Branch {context.branch_id} maps to fanout {fanout_node_id} but tracker not found")
            return
        
        # Check if all branches are complete
        if fanout_tracker.all_branches_complete():
            merge_node_id = fanout_tracker.merge_node_id
            if merge_node_id:
                # All branches complete, trigger merge node
                logger.info(
                    f"All branches complete for fanout {fanout_tracker.fanout_node_id}, "
                    f"triggering merge node {merge_node_id}"
                )
                
                # TASK 2: Check if merge already triggered (idempotence)
                if fanout_tracker.merge_triggered:
                    logger.debug(f"Merge {merge_node_id} already triggered for fanout {fanout_tracker.fanout_node_id}, skipping")
                    return
                
                # TASK 5: Collect branch outputs in deterministic order
                branch_outputs = {}
                # Order by branch_node_ids to ensure deterministic merge ordering
                for branch_node_id in fanout_tracker.branch_node_ids:
                    # Find the branch tracker for this node
                    for branch in fanout_tracker.branches.values():
                        if branch.branch_node_id == branch_node_id:
                            # Branch has .result attribute (NodeResult), not .output
                            if branch.result and hasattr(branch.result, 'state_updates'):
                                # Get latest output from branch result
                                branch_output = branch.result.state_updates.get('latest')
                                if branch_output is not None:
                                    branch_outputs[branch_node_id] = branch_output
                            break
                
                logger.debug(f"Collected {len(branch_outputs)} branch outputs for merge (ordered): {list(branch_outputs.keys())}")
                
                # TASK 3: Create proper merge context with valid branch_id
                import uuid
                merge_context = ExecutionContext(
                    run_id=context.run_id,  # Reuse run_id
                    node_id=merge_node_id,
                    branch_id=str(uuid.uuid4()),  # Generate unique branch_id for merge
                    iteration=0
                )
                
                # Create NodeResult with branch outputs as state_updates
                # This allows merge node to execute and consume the branch results
                merge_input_result = NodeResult.success(
                    state_updates={"branches": branch_outputs}
                )
                
                # TASK 2: Mark merge as triggered
                fanout_tracker.merge_triggered = True
                
                # Enqueue merge node with branch outputs
                queue.append(ExecutionToken(
                    node_id=merge_node_id,
                    context=merge_context,
                    parent_result=merge_input_result  # Pass branch outputs
                ))
    
    async def _handle_loop_node(
        self,
        node_id: str,
        result: NodeResult,
        state: WorkflowState,
        context: ExecutionContext,
        tracker: ExecutionTracker,
        queue: deque[ExecutionToken]
    ) -> List[str]:
        """
        Handle loop node - Phase 1 for_each routing with loop_continue/loop_exit.
        
        Routing is controlled exclusively by loop node artifacts:
        - should_continue=True → route via loop_continue edges or explicit body_node_id
        - should_continue=False → route via loop_exit edges
        
        NEVER evaluate loop_continue/loop_exit as conditions (they are deterministic tags).
        """
        node_config = self.nodes[node_id]
        artifacts = result.artifacts
        
        should_continue = artifacts.get("should_continue", False)
        body_node_id = artifacts.get("body_node_id")
        
        logger.info(
            f"LoopNode {node_id}: should_continue={should_continue}, "
            f"body_node_id={body_node_id}"
        )
        
        if not should_continue:
            # Loop exit: route via loop_exit edges
            outgoing = self.outgoing_edges.get(node_id, [])
            exit_nodes = []
            
            for edge in outgoing:
                if edge.condition == "loop_exit":
                    exit_nodes.append(edge.to_node)
                    logger.debug(f"LoopNode {node_id}: Exit edge to {edge.to_node}")
            
            # NO FALLBACK: Require explicit loop_exit tags
            if not exit_nodes:
                error_msg = (
                    f"LoopNode {node_id}: No loop_exit edges found. "
                    f"Loop nodes require explicit 'loop_exit' edge tags for exit routing."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            logger.info(f"LoopNode {node_id}: Exiting loop, routing to {exit_nodes}")
            return exit_nodes
        
        # Loop continue: route via loop_continue edges or explicit body_node_id
        if body_node_id:
            # Explicit body node specified in config
            logger.info(f"LoopNode {node_id}: Continuing to explicit body node {body_node_id}")
            return [body_node_id]
        
        # Find body via loop_continue edges (STRICT - no condition=None fallback)
        outgoing = self.outgoing_edges.get(node_id, [])
        continue_nodes = []
        
        for edge in outgoing:
            if edge.condition == "loop_continue":
                continue_nodes.append(edge.to_node)
                logger.debug(f"LoopNode {node_id}: Continue edge to {edge.to_node}")
        
        # NO FALLBACK: Require explicit loop_continue tags
        if not continue_nodes:
            error_msg = (
                f"LoopNode {node_id}: No loop_continue edges found. "
                f"Loop nodes require explicit 'loop_continue' edge tags for body routing."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info(
            f"LoopNode {node_id}: Continuing loop, routing to body nodes {continue_nodes}"
        )
        return continue_nodes
    
    
    async def _handle_merge_node(
        self,
        node_id: str,
        result: NodeResult,
        state: WorkflowState,
        context: ExecutionContext,
        tracker: ExecutionTracker,
        queue: deque[ExecutionToken]
    ) -> List[str]:
        """Handle merge node - check join barrier"""
        node_config = self.nodes[node_id]
        expected_keys = node_config.params.get("expected_keys")
        
        if not expected_keys:
            # No join barrier, proceed normally
            return self._determine_next_nodes(node_id, result, state)
        
        # Check if all expected keys are present
        merge_key = node_config.params.get("merge_key", "final")
        items = state.get(merge_key, {})
        
        if not isinstance(expected_keys, list):
            expected_keys = [expected_keys]
        
        missing_keys = set(expected_keys) - set(items.keys())
        
        if missing_keys:
            # Join barrier not met - wait for branches to complete
            timeout = node_config.params.get("timeout", 60.0)
            start_wait = time.time()
            max_wait_time = timeout / 1000.0  # Convert ms to seconds
            
            # Check if we're waiting for a fanout
            fanout_tracker = None
            for fanout in tracker.fanouts.values():
                if fanout.merge_node_id == node_id:
                    fanout_tracker = fanout
                    break
            
            if fanout_tracker:
                # Wait for fanout branches to complete
                wait_count = 0
                max_waits = int(max_wait_time * 10)  # Check every 0.1s
                
                while not fanout_tracker.all_branches_complete() and wait_count < max_waits:
                    await asyncio.sleep(0.1)  # Wait a bit
                    wait_count += 1
                    # Re-check items
                    items = state.get(merge_key, {})
                    missing_keys = set(expected_keys) - set(items.keys())
                    if not missing_keys:
                        break
                
                if missing_keys and wait_count >= max_waits:
                    logger.warning(
                        f"MergeNode {node_id}: Join barrier timeout after {max_wait_time}s. "
                        f"Missing keys: {missing_keys}. Proceeding with available keys."
                    )
            else:
                # No fanout tracker, check if we should wait or proceed
                # Re-queue merge node to check again later (executor will handle retries)
                logger.info(
                    f"MergeNode {node_id}: Join barrier not met. Missing: {missing_keys}. "
                    f"Will retry later."
                )
                # Re-queue with a small delay
                await asyncio.sleep(0.5)
                queue.append(ExecutionToken(
                    node_id=node_id,
                    context=context,
                    parent_result=result
                ))
                return []  # Don't proceed yet
        
        # All keys present (or timeout), proceed
        return self._determine_next_nodes(node_id, result, state)
    
    async def _execute_node(
        self,
        node_id: str,
        state: WorkflowState,
        context: ExecutionContext,
        parent_result: Optional[NodeResult]
    ) -> NodeResult:
        """Execute a single node with Phase 4 enhancements (caching, retry, metrics)"""
        node_config = self.nodes[node_id]
        start_time = time.time()
        
        # Phase 4: Check cache (but skip cache for loop nodes AND loop body nodes)
        # Loop nodes need to re-evaluate, body nodes need fresh execution each iteration
        # Always get node inputs (needed for caching even if we skip cache check for loops)
        node_inputs = self._get_node_inputs(node_config, state)
        
        cache_hit = False
        is_loop_body = context.parent_loop_id is not None
        has_parent_result = parent_result is not None
        is_agent_node = node_config.type == "agent"
        
        # Debug logging
        logger.info(f"[CACHE DEBUG] {node_id}: type={node_config.type}, parent_result={'YES' if has_parent_result else 'NO'}, is_loop_body={is_loop_body}")
        
        # Skip cache for: loop nodes, loop body nodes, agent nodes (dynamic parent_result inputs), 
        # and nodes with parent_result (chained nodes)
        # Agent nodes removed from cacheing because cache key doesn't include parent_result data
        if not is_agent_node and node_config.type != "loop" and not is_loop_body and not has_parent_result:
            cached_result = self.cache_manager.get(node_id, node_inputs)
            cache_hit = cached_result is not None
            if cache_hit:
                logger.info(f"[CACHE DEBUG] {node_id}: Cache HIT - will use cached result")
        else:
            skip_reasons = []
            if is_agent_node:
                skip_reasons.append("agent_node")
            if node_config.type == "loop":
                skip_reasons.append("loop_node")
            if is_loop_body:
                skip_reasons.append("loop_body")
            if has_parent_result:
                skip_reasons.append("has_parent_result")
            logger.info(f"[CACHE DEBUG] {node_id}: Cache SKIP - reasons: {', '.join(skip_reasons)}")
        
        if cache_hit:
            logger.info(f"Cache hit for node {node_id}")
            # Convert cached result to NodeResult if needed
            if isinstance(cached_result, dict):
                result = NodeResult(
                    state_updates=cached_result.get("state_updates", {}),
                    artifacts=cached_result.get("artifacts", {}),
                    status=NodeStatus.SUCCESS
                )
            else:
                result = NodeResult.success(
                    state_updates={},
                    artifacts={"cached_result": cached_result}
                )
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            # Record metrics
            self.metrics_collector.record_node_execution(
                node_id=node_id,
                node_type=node_config.type,
                status="success",
                duration_ms=duration_ms,
                start_time=start_time,
                end_time=end_time,
                cache_hit=True
            )
            
            state.append_log("INFO", f"Node {node_id} completed from cache", node_id=node_id)
            return result
        
        # Create node instance
        node = create_node(
            node_type=node_config.type,
            config=node_config,
            state=state,
            runner=self.runner,
            settings=self.config
        )
        
        # Pass parent_result to node for automatic chaining
        node.parent_result = parent_result
        
        state.append_log("INFO", f"Executing node: {node_id} (type: {node_config.type})", node_id=node_id)
        
        # Phase 5: Emit node started event
        self.monitor.node_started(context.run_id, node_id, node_config.type, context)
        
        # Phase 4: Execute with retry
        async def execute_node():
            return await node.execute()
        
        result, retry_count = await self.retry_handler.execute_with_retry(
            node_id=node_id,
            execute_fn=execute_node
        )
        
        # Phase 4: Cache successful results (but skip caching for loop nodes - they need to re-evaluate)
        if result.status == NodeStatus.SUCCESS and node_config.type != "loop":
            self.cache_manager.set(
                node_id=node_id,
                inputs=node_inputs,
                value=result.to_dict()
            )
        
        # Add trace
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        state.add_trace(
            node_id=node_id,
            start_time=start_time,
            end_time=end_time,
            inputs=node_inputs,
            outputs=result.to_dict()
        )
        
        # Phase 4: Record metrics
        self.metrics_collector.record_node_execution(
            node_id=node_id,
            node_type=node_config.type,
            status=result.status.value,
            duration_ms=duration_ms,
            start_time=start_time,
            end_time=end_time,
            retry_count=retry_count,
            error=result.error,
            cache_hit=False
        )
        
        # Phase 5: Emit node completion event
        if result.status == NodeStatus.SUCCESS:
            self.monitor.node_completed(context.run_id, node_id, result, context)
        elif result.status == NodeStatus.FAILED:
            self.monitor.node_failed(context.run_id, node_id, result.error or "Unknown error", context)
        elif result.status == NodeStatus.SKIPPED:
            self.monitor.node_completed(context.run_id, node_id, result, context)  # Use completed for skipped
        
        state.append_log("INFO", f"Node {node_id} completed with status: {result.status.value}", node_id=node_id)
        
        return result
    
    def _apply_state_updates(self, state: WorkflowState, updates: Dict[str, Any]) -> None:
        """Apply state updates from NodeResult
        
        Ensures parent paths are set before nested paths to prevent overwriting.
        For example, "triage" should be set before "triage.intents" to ensure
        the nested structure is preserved correctly.
        """
        # Sort updates: parent paths (fewer dots) before nested paths (more dots)
        # This ensures that when we set "triage", it happens before "triage.intents"
        sorted_updates = sorted(updates.items(), key=lambda x: x[0].count('.'))
        
        for path, value in sorted_updates:
            state.set(path, value)
            
            # Log key state updates for traceability
            if path == "latest":
                value_preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                logger.info(f"State update: latest='{value_preview}'")
            elif path.startswith("outputs."):
                node_id = path.split(".", 1)[1]
                value_preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                logger.info(f"State update: outputs.{node_id}='{value_preview}' (total outputs: {len(state.outputs)})")
    
    def _determine_next_nodes(
        self,
        node_id: str,
        result: NodeResult,
        state: WorkflowState,
        tracker: Optional['ExecutionTracker'] = None
    ) -> List[str]:
        """Determine next nodes to execute"""
        # If NodeResult specifies next_nodes, use those
        if result.next_nodes:
            return result.next_nodes
        
        # Check if this is a loop node that has exited
        node_config = self.nodes.get(node_id)
        loop_exited = False
        if node_config and node_config.type == "loop":
            # Check if loop has exited (should_continue = False)
            should_continue = result.artifacts.get("should_continue", True)
            loop_exited = not should_continue
        
        # Otherwise, use graph edges
        outgoing = self.outgoing_edges.get(node_id, [])
        next_nodes = []
        
        for edge in outgoing:
            # If loop has exited, skip "loop_body" edges (they're only for entering the loop)
            if loop_exited and edge.condition == "loop_body":
                continue
            
            # Skip merge nodes that are join targets of an active fanout
            # If this node is a fanout branch and edge points to the fanout's merge node,
            # skip it - the merge will be triggered by _check_fanout_completion()
            if tracker:
                skip_merge = False
                for fanout_id, fanout_tracker in tracker.fanouts.items():
                    # Check if current node is a branch of this fanout
                    if node_id in fanout_tracker.branch_node_ids:
                        # Check if edge points to the fanout's merge node
                        if fanout_tracker.merge_node_id and edge.to_node == fanout_tracker.merge_node_id:
                            logger.debug(
                                f"Skipping branch→merge edge {node_id}→{edge.to_node} "
                                f"(merge will be triggered by fanout completion handler)"
                            )
                            skip_merge = True
                            break
                if skip_merge:
                    continue
            
            # Check edge condition
            if edge.condition:
                # "loop_body" is a special condition handled in _handle_loop_node
                # Skip it here to avoid evaluation errors
                if edge.condition == "loop_body":
                    continue
                
                condition_met = ConditionEvaluator.evaluate(edge.condition, state)
                if not condition_met:
                    continue  # Skip this edge
            
            next_nodes.append(edge.to_node)
        
        return next_nodes
    
    def _get_node_inputs(self, node_config: NodeConfig, state: WorkflowState) -> Dict[str, Any]:
        """Get node inputs for tracing"""
        inputs = {}
        for input_key, state_path in node_config.inputs.items():
            inputs[input_key] = state.get(state_path)
        return inputs
    
    def _get_final_output(self, state: WorkflowState) -> str:
        """Get final output from state"""
        # Try terminal nodes first
        terminal_nodes = self.graph_def.get_terminal_nodes()
        if terminal_nodes:
            for node_id in terminal_nodes:
                output = state.get(f"outputs.{node_id}")
                if output:
                    return str(output)
        
        # Fallback: use state.latest or merged output
        if state.latest:
            return state.latest
        
        # Check for merged output
        merged = state.get("merged")
        if merged:
            return str(merged)
        
        return "No output generated"
