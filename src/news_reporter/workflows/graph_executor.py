"""Graph Executor - Queue-based execution with NodeResult and ExecutionContext (Phase 2)"""

from __future__ import annotations
from typing import Dict, List, Set, Optional, Any
from collections import deque
import asyncio
import time
import logging

from .graph_schema import GraphDefinition, NodeConfig, EdgeConfig
from .workflow_state import WorkflowState
from .agent_runner import AgentRunner
from .condition_evaluator import ConditionEvaluator
from .execution_context import ExecutionContext
from .node_result import NodeResult, NodeStatus
from .nodes import create_node
from .agent_adapter import AgentAdapterRegistry
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
        """
        # Initialize state
        state = WorkflowState(goal=goal)
        state.append_log("INFO", f"Starting graph execution with goal: {goal}")
        
        # Find entry nodes
        entry_nodes = self.graph_def.get_entry_nodes()
        if not entry_nodes:
            raise ValueError("Graph has no entry nodes")
        
        # Initialize execution context
        root_context = ExecutionContext(node_id=entry_nodes[0])
        
        # Execute graph
        try:
            await self._execute_queue_based(state, entry_nodes, root_context)
        except Exception as e:
            state.append_log("ERROR", f"Graph execution failed: {str(e)}")
            logger.error(f"Graph execution failed: {e}", exc_info=True)
            raise
        
        # Get final output
        return self._get_final_output(state)
    
    async def _execute_queue_based(
        self,
        state: WorkflowState,
        entry_nodes: List[str],
        root_context: ExecutionContext
    ) -> None:
        """Execute graph using queue-based system"""
        # Execution queue
        queue: deque[ExecutionToken] = deque()
        
        # Track execution state
        executed: Set[str] = set()  # Nodes that have completed
        executing: Set[str] = set()  # Nodes currently executing
        node_results: Dict[str, NodeResult] = {}  # Results by node_id
        
        # Branch tracking
        branch_contexts: Dict[str, ExecutionContext] = {}  # branch_id -> context
        branch_results: Dict[str, List[NodeResult]] = {}  # branch_id -> results
        
        # Add entry nodes to queue
        for node_id in entry_nodes:
            context = root_context.create_child_branch(node_id)
            context.node_id = node_id
            branch_contexts[context.branch_id] = context
            queue.append(ExecutionToken(node_id=node_id, context=context))
        
        # Process queue
        max_iterations = 1000  # Safety limit
        iteration = 0
        
        while queue and iteration < max_iterations:
            iteration += 1
            token = queue.popleft()
            node_id = token.node_id
            context = token.context
            
            # Skip if already executed (unless it's a loop iteration)
            if node_id in executed and context.iteration == 0:
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
                
                # Handle different statuses
                if result.status == NodeStatus.SKIPPED:
                    logger.info(f"Node {node_id} was skipped: {result.metrics.get('skip_reason')}")
                    # Continue to next nodes anyway
                elif result.status == NodeStatus.FAILED:
                    logger.error(f"Node {node_id} failed: {result.error}")
                    # Decide whether to continue or stop
                    # For now, we'll continue (could be configurable)
                
                # Determine next nodes
                next_nodes = self._determine_next_nodes(node_id, result, state)
                
                # Add next nodes to queue
                for next_node_id in next_nodes:
                    next_context = context.create_child_branch(next_node_id)
                    next_context.node_id = next_node_id
                    branch_contexts[next_context.branch_id] = next_context
                    queue.append(ExecutionToken(
                        node_id=next_node_id,
                        context=next_context,
                        parent_result=result
                    ))
                
                executed.add(node_id)
                
            except Exception as e:
                logger.error(f"Node {node_id} execution error: {e}", exc_info=True)
                result = NodeResult.failed(str(e))
                node_results[node_id] = result
                state.append_log("ERROR", f"Node {node_id} failed: {str(e)}", node_id=node_id)
            finally:
                executing.discard(node_id)
        
        if iteration >= max_iterations:
            logger.warning(f"Graph execution reached max iterations ({max_iterations})")
            state.append_log("WARNING", f"Execution stopped after {max_iterations} iterations")
        
        # Check for stuck execution
        if executing:
            logger.warning(f"Some nodes are still executing: {executing}")
    
    async def _execute_node(
        self,
        node_id: str,
        state: WorkflowState,
        context: ExecutionContext,
        parent_result: Optional[NodeResult]
    ) -> NodeResult:
        """Execute a single node"""
        node_config = self.nodes[node_id]
        start_time = time.time()
        
        # Create node instance
        node = create_node(
            node_type=node_config.type,
            config=node_config,
            state=state,
            runner=self.runner,
            settings=self.config
        )
        
        state.append_log("INFO", f"Executing node: {node_id} (type: {node_config.type})", node_id=node_id)
        
        # Execute node
        result = await node.execute()
        
        # Add trace
        end_time = time.time()
        state.add_trace(
            node_id=node_id,
            start_time=start_time,
            end_time=end_time,
            inputs=self._get_node_inputs(node_config, state),
            outputs=result.to_dict()
        )
        
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
