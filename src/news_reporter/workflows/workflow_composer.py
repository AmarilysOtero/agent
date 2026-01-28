"""Workflow Composer - Compose and nest workflows"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import logging

from ..models.graph_schema import GraphDefinition, NodeConfig, EdgeConfig

logger = logging.getLogger(__name__)


class WorkflowComposer:
    """Composes and nests workflows"""
    
    @staticmethod
    def compose(
        workflows: List[GraphDefinition],
        composition_strategy: str = "sequential"
    ) -> GraphDefinition:
        """
        Compose multiple workflows into a single workflow.
        
        Args:
            workflows: List of workflows to compose
            composition_strategy: "sequential", "parallel", or "conditional"
        
        Returns:
            Composed GraphDefinition
        """
        if not workflows:
            raise ValueError("No workflows to compose")
        
        if composition_strategy == "sequential":
            return WorkflowComposer._compose_sequential(workflows)
        elif composition_strategy == "parallel":
            return WorkflowComposer._compose_parallel(workflows)
        elif composition_strategy == "conditional":
            return WorkflowComposer._compose_conditional(workflows)
        else:
            raise ValueError(f"Unknown composition strategy: {composition_strategy}")
    
    @staticmethod
    def _compose_sequential(workflows: List[GraphDefinition]) -> GraphDefinition:
        """Compose workflows sequentially"""
        all_nodes = []
        all_edges = []
        
        # Prefix node IDs to avoid conflicts
        for i, workflow in enumerate(workflows):
            prefix = f"wf{i}_"
            
            # Add nodes with prefix
            for node in workflow.nodes:
                prefixed_node = NodeConfig(
                    id=f"{prefix}{node.id}",
                    type=node.type,
                    agent_id=getattr(node, 'agent_id', None),
                    inputs=node.inputs,
                    outputs=node.outputs,
                    params=node.params,
                    max_iters=getattr(node, 'max_iters', None),
                    branches=getattr(node, 'branches', None)
                )
                all_nodes.append(prefixed_node)
            
            # Add edges with prefix
            for edge in workflow.edges:
                all_edges.append(EdgeConfig(
                    from_node=f"{prefix}{edge.from_node}",
                    to_node=f"{prefix}{edge.to_node}",
                    condition=edge.condition
                ))
            
            # Connect workflows sequentially
            if i > 0:
                # Connect last node of previous workflow to first node of current
                prev_entry = workflows[i-1].entry_node_id
                curr_entry = workflow.entry_node_id
                
                # Find terminal nodes of previous workflow
                prev_terminal = WorkflowComposer._find_terminal_nodes(workflows[i-1])
                if prev_terminal:
                    for terminal in prev_terminal:
                        all_edges.append(EdgeConfig(
                            from_node=f"wf{i-1}_{terminal}",
                            to_node=f"{prefix}{curr_entry}"
                        ))
        
        # Use first workflow's entry node
        entry_node = f"wf0_{workflows[0].entry_node_id}"
        
        return GraphDefinition(
            nodes=all_nodes,
            edges=all_edges,
            entry_node_id=entry_node
        )
    
    @staticmethod
    def _compose_parallel(workflows: List[GraphDefinition]) -> GraphDefinition:
        """Compose workflows in parallel"""
        all_nodes = []
        all_edges = []
        
        # Create a fanout node
        fanout_node = NodeConfig(
            id="compose_fanout",
            type="fanout",
            branches=[f"wf{i}" for i in range(len(workflows))]
        )
        all_nodes.append(fanout_node)
        
        # Add all workflow nodes with prefix
        for i, workflow in enumerate(workflows):
            prefix = f"wf{i}_"
            
            for node in workflow.nodes:
                prefixed_node = NodeConfig(
                    id=f"{prefix}{node.id}",
                    type=node.type,
                    agent_id=getattr(node, 'agent_id', None),
                    inputs=node.inputs,
                    outputs=node.outputs,
                    params=node.params
                )
                all_nodes.append(prefixed_node)
            
            # Connect fanout to workflow entry
            all_edges.append(EdgeConfig(
                from_node="compose_fanout",
                to_node=f"{prefix}{workflow.entry_node_id}"
            ))
            
            # Add workflow edges
            for edge in workflow.edges:
                all_edges.append(EdgeConfig(
                    from_node=f"{prefix}{edge.from_node}",
                    to_node=f"{prefix}{edge.to_node}",
                    condition=edge.condition
                ))
        
        # Create merge node
        merge_node = NodeConfig(
            id="compose_merge",
            type="merge",
            params={"strategy": "collect_list"}
        )
        all_nodes.append(merge_node)
        
        # Connect terminal nodes to merge
        for i, workflow in enumerate(workflows):
            terminal_nodes = WorkflowComposer._find_terminal_nodes(workflow)
            for terminal in terminal_nodes:
                all_edges.append(EdgeConfig(
                    from_node=f"wf{i}_{terminal}",
                    to_node="compose_merge"
                ))
        
        return GraphDefinition(
            nodes=all_nodes,
            edges=all_edges,
            entry_node_id="compose_fanout"
        )
    
    @staticmethod
    def _compose_conditional(workflows: List[GraphDefinition]) -> GraphDefinition:
        """Compose workflows conditionally (if-else style)"""
        if len(workflows) != 2:
            raise ValueError("Conditional composition requires exactly 2 workflows")
        
        all_nodes = []
        all_edges = []
        
        # Create conditional node
        conditional_node = NodeConfig(
            id="compose_conditional",
            type="conditional",
            params={"condition": "true"}  # Would be configurable
        )
        all_nodes.append(conditional_node)
        
        # Add both workflows
        for i, workflow in enumerate(workflows):
            prefix = f"wf{i}_"
            
            for node in workflow.nodes:
                prefixed_node = NodeConfig(
                    id=f"{prefix}{node.id}",
                    type=node.type,
                    agent_id=getattr(node, 'agent_id', None),
                    inputs=node.inputs,
                    outputs=node.outputs,
                    params=node.params
                )
                all_nodes.append(prefixed_node)
            
            # Connect conditional to workflow
            all_edges.append(EdgeConfig(
                from_node="compose_conditional",
                to_node=f"{prefix}{workflow.entry_node_id}",
                condition="true" if i == 0 else "false"
            ))
            
            # Add workflow edges
            for edge in workflow.edges:
                all_edges.append(EdgeConfig(
                    from_node=f"{prefix}{edge.from_node}",
                    to_node=f"{prefix}{edge.to_node}",
                    condition=edge.condition
                ))
        
        # Create merge node
        merge_node = NodeConfig(
            id="compose_merge",
            type="merge",
            params={"strategy": "concat_text"}
        )
        all_nodes.append(merge_node)
        
        # Connect terminal nodes to merge
        for i, workflow in enumerate(workflows):
            terminal_nodes = WorkflowComposer._find_terminal_nodes(workflow)
            for terminal in terminal_nodes:
                all_edges.append(EdgeConfig(
                    from_node=f"wf{i}_{terminal}",
                    to_node="compose_merge"
                ))
        
        return GraphDefinition(
            nodes=all_nodes,
            edges=all_edges,
            entry_node_id="compose_conditional"
        )
    
    @staticmethod
    def _find_terminal_nodes(workflow: GraphDefinition) -> List[str]:
        """Find terminal nodes (nodes with no outgoing edges)"""
        outgoing = set()
        for edge in workflow.edges:
            outgoing.add(edge.from_node)
        
        terminal = []
        for node in workflow.nodes:
            if node.id not in outgoing:
                terminal.append(node.id)
        
        return terminal if terminal else [workflow.entry_node_id]
    
    @staticmethod
    def nest_workflow(
        parent_workflow: GraphDefinition,
        nested_workflow: GraphDefinition,
        at_node_id: str
    ) -> GraphDefinition:
        """
        Nest a workflow inside another workflow at a specific node.
        
        Args:
            parent_workflow: Parent workflow
            nested_workflow: Workflow to nest
            at_node_id: Node ID where to nest the workflow
        
        Returns:
            Modified GraphDefinition with nested workflow
        """
        all_nodes = list(parent_workflow.nodes)
        all_edges = list(parent_workflow.edges)
        
        # Prefix nested workflow nodes
        prefix = f"nested_{at_node_id}_"
        
        for node in nested_workflow.nodes:
            prefixed_node = NodeConfig(
                id=f"{prefix}{node.id}",
                type=node.type,
                agent_id=getattr(node, 'agent_id', None),
                inputs=node.inputs,
                outputs=node.outputs,
                params=node.params
            )
            all_nodes.append(prefixed_node)
        
        # Replace the target node's outgoing edges
        # Connect target node to nested workflow entry
        all_edges.append(EdgeConfig(
            from_node=at_node_id,
            to_node=f"{prefix}{nested_workflow.entry_node_id}"
        ))
        
        # Add nested workflow edges
        for edge in nested_workflow.edges:
            all_edges.append(EdgeConfig(
                from_node=f"{prefix}{edge.from_node}",
                to_node=f"{prefix}{edge.to_node}",
                condition=edge.condition
            ))
        
        # Connect nested workflow terminal to original target's children
        nested_terminal = WorkflowComposer._find_terminal_nodes(nested_workflow)
        original_outgoing = [e for e in all_edges if e.from_node == at_node_id]
        
        for terminal in nested_terminal:
            for edge in original_outgoing:
                all_edges.append(EdgeConfig(
                    from_node=f"{prefix}{terminal}",
                    to_node=edge.to_node,
                    condition=edge.condition
                ))
        
        # Remove original edges from target node
        all_edges = [e for e in all_edges if not (e.from_node == at_node_id and e not in original_outgoing)]
        
        return GraphDefinition(
            nodes=all_nodes,
            edges=all_edges,
            entry_node_id=parent_workflow.entry_node_id
        )
