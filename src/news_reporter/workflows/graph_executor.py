"""Graph Executor - Orchestrates graph execution"""

from __future__ import annotations
from typing import Dict, List, Set, Optional, Any
import asyncio
import time
import logging
import json

from .graph_schema import GraphDefinition, NodeConfig, EdgeConfig
from .workflow_state import WorkflowState
from .agent_runner import AgentRunner
from .condition_evaluator import ConditionEvaluator
from .nodes import create_node
from ..config import Settings

logger = logging.getLogger(__name__)


class GraphExecutor:
    """
    Executes a graph definition with support for:
    - Topological ordering
    - Conditional routing
    - Fan-out (parallel execution)
    - Loops with max_iters
    - Instrumentation and tracing
    """
    
    def __init__(self, graph_def: GraphDefinition, config: Settings):
        self.graph_def = graph_def
        self.config = config
        self.runner = AgentRunner(config)
        
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
        Execute the graph with given goal.
        
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
            raise ValueError("Graph has no entry nodes (nodes with no incoming edges)")
        
        # Execute graph
        try:
            await self._execute_from_nodes(entry_nodes, state)
        except Exception as e:
            state.append_log("ERROR", f"Graph execution failed: {str(e)}")
            logger.error(f"Graph execution failed: {e}", exc_info=True)
            raise
        
        # Get final output
        terminal_nodes = self.graph_def.get_terminal_nodes()
        if terminal_nodes:
            # Try to get output from terminal nodes
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
    
    async def _execute_from_nodes(self, node_ids: List[str], state: WorkflowState) -> None:
        """Execute graph starting from given nodes"""
        executed: Set[str] = set()
        executing: Set[str] = set()
        loop_iterations: Dict[str, int] = {}  # Track iterations for loop nodes
        
        async def execute_node(node_id: str) -> Dict[str, Any]:
            """Execute a single node"""
            if node_id in executed or node_id in executing:
                return {}
            
            # Check dependencies
            incoming = self.incoming_edges.get(node_id, [])
            for edge in incoming:
                if edge.from_node not in executed:
                    # Dependency not executed yet - will be retried later
                    return {}
            
            executing.add(node_id)
            node_config = self.nodes[node_id]
            start_time = time.time()
            
            try:
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
                outputs = await node.execute()
                
                end_time = time.time()
                state.add_trace(
                    node_id=node_id,
                    start_time=start_time,
                    end_time=end_time,
                    inputs={k: state.get(v) for k, v in node_config.inputs.items()},
                    outputs=outputs
                )
                
                executed.add(node_id)
                executing.remove(node_id)
                
                state.append_log("INFO", f"Node {node_id} completed", node_id=node_id)
                
                # Handle special node types
                if node_config.type == "fanout":
                    await self._handle_fanout(node_id, node_config, state, executed, executing, loop_iterations)
                elif node_config.type == "loop":
                    await self._handle_loop(node_id, node_config, state, executed, executing, loop_iterations)
                else:
                    # Regular node - continue to next nodes
                    await self._continue_execution(node_id, state, executed, executing, loop_iterations)
                
            except Exception as e:
                end_time = time.time()
                state.add_trace(
                    node_id=node_id,
                    start_time=start_time,
                    end_time=end_time,
                    error=str(e)
                )
                executing.remove(node_id)
                state.append_log("ERROR", f"Node {node_id} failed: {str(e)}", node_id=node_id)
                raise
        
        # Start execution from entry nodes
        while True:
            progress_made = False
            # Find nodes ready to execute
            ready_nodes = [
                node_id for node_id in self.nodes.keys()
                if node_id not in executed and node_id not in executing
                and all(
                    edge.from_node in executed
                    for edge in self.incoming_edges.get(node_id, [])
                )
            ]
            
            if not ready_nodes:
                # Check if we're done or stuck
                if len(executed) == len(self.nodes):
                    break  # All nodes executed
                elif not executing and not progress_made:
                    # Stuck - some nodes can't execute
                    remaining = set(self.nodes.keys()) - executed - executing
                    raise RuntimeError(f"Graph execution stuck. Remaining nodes: {remaining}")
                # Wait for executing nodes or retry
                await asyncio.sleep(0.1)
                continue
            
            # Execute ready nodes in parallel (if not in a loop)
            await asyncio.gather(*[execute_node(node_id) for node_id in ready_nodes])
    
    async def _continue_execution(
        self,
        from_node_id: str,
        state: WorkflowState,
        executed: Set[str],
        executing: Set[str],
        loop_iterations: Dict[str, int]
    ) -> None:
        """Continue execution from a completed node"""
        outgoing = self.outgoing_edges.get(from_node_id, [])
        
        for edge in outgoing:
            # Check edge condition
            if edge.condition:
                condition_met = ConditionEvaluator.evaluate(edge.condition, state)
                if not condition_met:
                    continue  # Skip this edge
            
            # Execute target node
            await self._execute_from_nodes([edge.to_node], state)
    
    async def _handle_fanout(
        self,
        node_id: str,
        node_config: NodeConfig,
        state: WorkflowState,
        executed: Set[str],
        executing: Set[str],
        loop_iterations: Dict[str, int]
    ) -> None:
        """Handle fanout node - execute branches in parallel"""
        if not node_config.branches:
            return
        
        # Get items to fan out over
        fanout_items = state.get("fanout_items", [])
        if not fanout_items:
            fanout_items = getattr(self.config, 'reporter_ids', [])
        
        # Execute each branch for each item
        branch_tasks = []
        for item in fanout_items:
            # Set current item in state for branch nodes
            state.set("current_fanout_item", item)
            
            # Execute all branches in parallel for this item
            for branch_id in node_config.branches:
                branch_tasks.append(
                    self._execute_from_nodes([branch_id], state)
                )
        
        # Wait for all branches to complete
        if branch_tasks:
            await asyncio.gather(*branch_tasks)
        
        # Continue to next nodes
        await self._continue_execution(node_id, state, executed, executing, loop_iterations)
    
    async def _handle_loop(
        self,
        node_id: str,
        node_config: NodeConfig,
        state: WorkflowState,
        executed: Set[str],
        executing: Set[str],
        loop_iterations: Dict[str, int]
    ) -> None:
        """Handle loop node - iterate until condition or max_iters"""
        max_iters = node_config.max_iters or 3
        current_iter = loop_iterations.get(node_id, 0) + 1
        loop_iterations[node_id] = current_iter
        
        # Set current iteration in state
        state.set("current_iter", current_iter)
        
        # Check if we should continue
        should_continue = True
        if node_config.loop_condition:
            should_continue = ConditionEvaluator.evaluate(node_config.loop_condition, state)
        
        if current_iter > max_iters:
            state.append_log("INFO", f"Loop {node_id} reached max iterations ({max_iters})")
            # Continue to next nodes
            await self._continue_execution(node_id, state, executed, executing, loop_iterations)
            return
        
        if not should_continue:
            state.append_log("INFO", f"Loop {node_id} condition not met, exiting")
            # Continue to next nodes
            await self._continue_execution(node_id, state, executed, executing, loop_iterations)
            return
        
        # Continue looping - execute loop body nodes
        outgoing = self.outgoing_edges.get(node_id, [])
        for edge in outgoing:
            if edge.condition == "loop_body":  # Special edge for loop body
                # Execute loop body
                await self._execute_from_nodes([edge.to_node], state)
                # Loop back
                await self._handle_loop(node_id, node_config, state, executed, executing, loop_iterations)
                return
        
        # No loop body edge found, just continue
        await self._continue_execution(node_id, state, executed, executing, loop_iterations)
