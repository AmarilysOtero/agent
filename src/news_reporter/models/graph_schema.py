"""Graph Definition Schema - JSON contract for workflow graphs"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class NodeConfig(BaseModel):
    """Configuration for a single node in the graph"""
    
    id: str
    type: str  # "agent", "fanout", "loop", "conditional", "merge"
    agent_id: Optional[str] = None  # For agent nodes
    inputs: Dict[str, Any] = Field(default_factory=dict)  # Input mappings
    outputs: Dict[str, str] = Field(default_factory=dict)  # Output mappings to state paths
    params: Dict[str, Any] = Field(default_factory=dict)  # Node-specific parameters
    
    # For fanout nodes
    branches: Optional[List[str]] = None  # List of branch node IDs
    
    # For loop nodes
    max_iters: Optional[int] = None
    loop_condition: Optional[str] = None  # Condition to continue looping
    
    # For conditional nodes
    condition: Optional[str] = None  # Condition expression


class EdgeConfig(BaseModel):
    """Configuration for an edge (connection) between nodes"""
    
    from_node: str
    to_node: str
    condition: Optional[str] = None  # Optional condition for routing


class GraphLimits(BaseModel):
    """Execution limits for the graph"""
    
    max_steps: Optional[int] = None
    timeout_ms: Optional[int] = None
    max_iters: Optional[int] = None
    max_parallel: Optional[int] = None


class GraphDefinition(BaseModel):
    """Complete graph definition"""
    
    nodes: List[NodeConfig]
    edges: List[EdgeConfig]
    
    # Explicit entry point (instead of inferring from "no incoming edges")
    entry_node_id: str = "triage"  # Default entry point
    
    # Future-proof fields (not used yet, but defined for future tool support)
    toolsets: List[str] = Field(default_factory=list)
    policy_profile: str = "read_only"
    limits: Optional[GraphLimits] = None
    
    # Metadata
    name: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = None
    
    def get_node(self, node_id: str) -> Optional[NodeConfig]:
        """Get node by ID"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_edges_from(self, node_id: str) -> List[EdgeConfig]:
        """Get all edges starting from a node"""
        return [edge for edge in self.edges if edge.from_node == node_id]
    
    def get_edges_to(self, node_id: str) -> List[EdgeConfig]:
        """Get all edges ending at a node"""
        return [edge for edge in self.edges if edge.to_node == node_id]
    
    def get_entry_nodes(self) -> List[str]:
        """Get entry nodes - uses explicit entry_node_id if set, otherwise falls back to nodes with no incoming edges"""
        # Use explicit entry_node_id if it exists in nodes
        if self.entry_node_id:
            entry_node = self.get_node(self.entry_node_id)
            if entry_node:
                return [self.entry_node_id]
            # If entry_node_id doesn't exist, log warning and fall back
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Entry node '{self.entry_node_id}' not found in graph nodes. Falling back to nodes with no incoming edges.")
        
        # Fallback: nodes with no incoming edges
        nodes_with_incoming = {edge.to_node for edge in self.edges}
        all_node_ids = {node.id for node in self.nodes}
        return list(all_node_ids - nodes_with_incoming)
    
    def get_terminal_nodes(self) -> List[str]:
        """Get nodes with no outgoing edges (terminal nodes)"""
        nodes_with_outgoing = {edge.from_node for edge in self.edges}
        all_node_ids = {node.id for node in self.nodes}
        return list(all_node_ids - nodes_with_outgoing)
    
    def validate(self) -> List[str]:
        """Validate graph structure and return list of errors"""
        errors = []
        
        # Check all nodes referenced in edges exist
        node_ids = {node.id for node in self.nodes}
        for edge in self.edges:
            if edge.from_node not in node_ids:
                errors.append(f"Edge references unknown node: {edge.from_node}")
            if edge.to_node not in node_ids:
                errors.append(f"Edge references unknown node: {edge.to_node}")
        
        # Check for duplicate node IDs
        seen_ids = set()
        for node in self.nodes:
            if node.id in seen_ids:
                errors.append(f"Duplicate node ID: {node.id}")
            seen_ids.add(node.id)
        
        # # Check entry_node_id exists
        # if self.entry_node_id:
        #     if self.entry_node_id not in node_ids:
        #         errors.append(f"Entry node '{self.entry_node_id}' not found in graph nodes")
        
        # Check node type-specific requirements
        for node in self.nodes:
            if node.type == "agent" and not node.agent_id:
                errors.append(f"Agent node {node.id} missing agent_id")
            elif node.type == "fanout" and not node.branches:
                errors.append(f"Fanout node {node.id} missing branches")
            elif node.type == "loop" and node.max_iters is None:
                errors.append(f"Loop node {node.id} missing max_iters")
            elif node.type == "conditional" and not node.condition:
                errors.append(f"Conditional node {node.id} missing condition")
        
        return errors
