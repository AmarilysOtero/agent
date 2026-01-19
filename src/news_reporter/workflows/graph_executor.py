"""Graph Executor - Queue-based execution with NodeResult and ExecutionContext (Phase 5)"""

from __future__ import annotations
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import deque
import asyncio
import time
import logging

from .graph_schema import GraphDefinition, NodeConfig, EdgeConfig, GraphLimits
from .workflow_state import WorkflowState
from .agent_runner import AgentRunner
from .condition_evaluator import ConditionEvaluator
from .execution_context import ExecutionContext
from .node_result import NodeResult, NodeStatus
from .execution_tracker import ExecutionTracker, FanoutTracker, LoopTracker
from .state_checkpoint import StateCheckpoint
from .nodes import create_node
from .agent_adapter import AgentAdapterRegistry
from .performance_metrics import get_metrics_collector
from .cache_manager import get_cache_manager
from .retry_handler import RetryConfig, RetryHandler
from .execution_monitor import get_execution_monitor
from ..config import Settings

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
            raise ValueError(f"Invalid graph definition: {', '.join(errors)}")
        
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
        # Initialize state
        state = WorkflowState(goal=goal)
        logger.info("=" * 100)
        logger.info("=" * 100)
        logger.info(f"🚀 GRAPH EXECUTOR STARTING - Goal: {goal}")
        logger.info("=" * 100)
        logger.info("=" * 100)
        state.append_log("INFO", f"Starting graph execution with goal: {goal}")
        
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
                    self._execute_queue_based(state, entry_nodes, root_context),
                    timeout=self.timeout_ms / 1000.0
                )
            else:
                await self._execute_queue_based(state, entry_nodes, root_context)
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
        
        return final_output
    
    async def _execute_queue_based(
        self,
        state: WorkflowState,
        entry_nodes: List[str],
        root_context: ExecutionContext
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
            
            # Skip if already executed (unless it's a loop iteration)
            if exec_key in executed and context.iteration == 0:
                continue
            
            # Check if node is already executing (prevent duplicate execution)
            if node_id in executing:
                # Re-queue for later
                queue.append(token)
                await asyncio.sleep(0.01)  # Small delay
                continue
            
            # Execute node
            executing.add(node_id)
            try:
                result = await self._execute_node(node_id, state, context, token.parent_result)
                node_results[node_id] = result
                
                # Apply state updates
                self._apply_state_updates(state, result.state_updates)
                
                # Track branch results
                if context.branch_id not in branch_results:
                    branch_results[context.branch_id] = []
                branch_results[context.branch_id].append(result)
                
                # Phase 3: Mark branch as complete if it's part of a fanout
                # Only mark complete if this branch is actually tracked (part of a fanout)
                if context.branch_id in tracker.branch_to_fanout:
                    tracker.mark_branch_complete(context.branch_id, result)
                
                # Phase 3: Check if fanout is complete and trigger merge
                await self._check_fanout_completion(node_id, state, tracker, queue)
                
                # Phase 3: Handle special node types
                next_nodes = await self._handle_special_nodes(
                    node_id, result, state, context, tracker, queue
                )
                
                # If next_nodes not set by special handler, determine normally
                if next_nodes is None:
                    next_nodes = self._determine_next_nodes(node_id, result, state)
                
                # Phase 3: Check for loop back (body node completing, needs to loop back to loop node)
                # Find if any outgoing edge goes to a loop node
                outgoing = self.outgoing_edges.get(node_id, [])
                for edge in outgoing:
                    next_node_config = self.nodes.get(edge.to_node)
                    if next_node_config and next_node_config.type == "loop":
                        # This is a loop back - check if loop should continue
                        loop_tracker = tracker.get_loop_tracker(edge.to_node)
                        if loop_tracker and loop_tracker.should_continue:
                            # Loop back to loop node for next iteration check
                            next_nodes = [edge.to_node]
                            break
                
                # Add next nodes to queue
                for next_node_id in next_nodes:
                    # Create child branch (loop iteration handling is done in _handle_loop_node)
                    next_context = context.create_child_branch(next_node_id)
                    next_context.node_id = next_node_id
                    branch_contexts[next_context.branch_id] = next_context
                    queue.append(ExecutionToken(
                        node_id=next_node_id,
                        context=next_context,
                        parent_result=result
                    ))
                
                executed.add(exec_key)
                
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
    
    async def _handle_fanout_node(
        self,
        node_id: str,
        result: NodeResult,
        state: WorkflowState,
        context: ExecutionContext,
        tracker: ExecutionTracker,
        queue: deque[ExecutionToken]
    ) -> List[str]:
        """Handle fanout node - create branches for each item"""
        node_config = self.nodes[node_id]
        fanout_items = result.artifacts.get("fanout_items", [])
        branch_node_ids = result.artifacts.get("branches", [])
        
        if not fanout_items or not branch_node_ids:
            logger.warning(f"Fanout node {node_id} has no items or branches")
            return []
        
        # Find next node after fanout (could be merge, loop, or any other node)
        next_node_id = None
        outgoing = self.outgoing_edges.get(node_id, [])
        for edge in outgoing:
            # Check edge condition if present
            if edge.condition:
                from .condition_evaluator import ConditionEvaluator
                if not ConditionEvaluator.evaluate(edge.condition, state):
                    continue  # Skip this edge
            next_node_id = edge.to_node
            break  # Use first valid outgoing edge
        
        # Register fanout in tracker (use next_node_id instead of just merge_node_id)
        fanout_tracker = tracker.register_fanout(
            fanout_node_id=node_id,
            items=fanout_items,
            branch_node_ids=branch_node_ids,
            merge_node_id=next_node_id  # Store next node (may not be merge)
        )
        
        # Create branches for each item
        for item in fanout_items:
            # Set current item in state for branch nodes
            state.set("current_fanout_item", item)
            
            # For each branch node, create a branch
            for branch_node_id in branch_node_ids:
                branch_context = context.create_child_branch(branch_node_id)
                branch_context.node_id = branch_node_id
                
                # Register branch in tracker
                tracker.register_branch(
                    fanout_node_id=node_id,
                    item=item,
                    branch_id=branch_context.branch_id,
                    branch_node_id=branch_node_id
                )
                
                # Add to queue
                queue.append(ExecutionToken(
                    node_id=branch_node_id,
                    context=branch_context,
                    parent_result=result
                ))
        
        # Don't continue to merge node yet - wait for all branches
        # Merge node will be triggered when all branches complete
        return []
    
    async def _check_fanout_completion(
        self,
        node_id: str,
        state: WorkflowState,
        tracker: ExecutionTracker,
        queue: deque[ExecutionToken]
    ) -> None:
        """Check if a fanout is complete and trigger merge node if so"""
        # Find which fanout this node belongs to (if any)
        fanout_tracker = None
        for fanout in tracker.fanouts.values():
            # Check if this node is one of the branch nodes in the fanout
            if node_id in fanout.branch_node_ids:
                fanout_tracker = fanout
                break
        
        if not fanout_tracker:
            # This node is not part of a fanout, nothing to check
            return
        
        # Check if all branches are complete
        if fanout_tracker.all_branches_complete():
            next_node_id = fanout_tracker.merge_node_id  # Actually the next node (may not be merge)
            if next_node_id:
                # All branches complete, trigger next node
                logger.info(
                    f"All branches complete for fanout {fanout_tracker.fanout_node_id}, "
                    f"triggering next node {next_node_id}"
                )
                # Create context for next node (use root context or create new one)
                next_context = ExecutionContext(
                    node_id=next_node_id,
                    branch_id="",  # Next node is not part of a branch
                    iteration=0
                )
                queue.append(ExecutionToken(
                    node_id=next_node_id,
                    context=next_context,
                    parent_result=None
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
        """Handle loop node - loop back to body or continue"""
        node_config = self.nodes[node_id]
        artifacts = result.artifacts
        
        should_continue = artifacts.get("should_continue", False)
        body_node_id = artifacts.get("body_node_id")
        
        if not should_continue:
            # Loop is done, continue to next nodes
            return self._determine_next_nodes(node_id, result, state)
        
        # Loop should continue
        if body_node_id:
            # Use explicit body node
            loop_tracker = tracker.get_loop_tracker(node_id)
            if not loop_tracker:
                # Register loop
                loop_tracker = tracker.register_loop(
                    loop_node_id=node_id,
                    max_iters=node_config.max_iters or 10,
                    body_node_id=body_node_id
                )
            
            # Increment iteration
            new_iter = tracker.increment_loop_iteration(node_id)
            tracker.set_loop_should_continue(node_id, should_continue)
            
            # Create iteration context and loop back to body
            body_context = context.create_iteration(body_node_id)
            body_context.node_id = body_node_id
            
            queue.append(ExecutionToken(
                node_id=body_node_id,
                context=body_context,
                parent_result=result
            ))
            
            # After body completes, it should loop back to loop node
            # This is handled by the executor checking if body's next node is the loop node
            return []
        else:
            # Use outgoing edges to find body
            outgoing = self.outgoing_edges.get(node_id, [])
            body_nodes = [edge.to_node for edge in outgoing]
            if body_nodes:
                # Loop back to first body node
                body_context = context.create_iteration(body_nodes[0])
                body_context.node_id = body_nodes[0]
                
                queue.append(ExecutionToken(
                    node_id=body_nodes[0],
                    context=body_context,
                    parent_result=result
                ))
                return []
        
        # No body found, just continue
        return self._determine_next_nodes(node_id, result, state)
    
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
        
        # Phase 4: Check cache
        node_inputs = self._get_node_inputs(node_config, state)
        cached_result = self.cache_manager.get(node_id, node_inputs)
        cache_hit = cached_result is not None
        
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
        
        # Phase 4: Cache successful results
        if result.status == NodeStatus.SUCCESS:
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
        """Apply state updates from NodeResult"""
        for path, value in updates.items():
            state.set(path, value)
    
    def _determine_next_nodes(
        self,
        node_id: str,
        result: NodeResult,
        state: WorkflowState
    ) -> List[str]:
        """Determine next nodes to execute"""
        # If NodeResult specifies next_nodes, use those
        if result.next_nodes:
            return result.next_nodes
        
        # Otherwise, use graph edges
        outgoing = self.outgoing_edges.get(node_id, [])
        next_nodes = []
        
        for edge in outgoing:
            # Check edge condition
            if edge.condition:
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
