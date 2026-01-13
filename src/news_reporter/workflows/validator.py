# src/news_reporter/workflows/validator.py
"""Workflow graph validation (PR 2)"""
from __future__ import annotations
import logging
from typing import List, Dict, Set, Tuple
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ValidationError:
    """Validation error with structured information"""
    def __init__(self, code: str, message: str, nodeId: str = None, edgeId: str = None):
        self.code = code
        self.message = message
        self.nodeId = nodeId
        self.edgeId = edgeId
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        result = {"code": self.code, "message": self.message}
        if self.nodeId:
            result["nodeId"] = self.nodeId
        if self.edgeId:
            result["edgeId"] = self.edgeId
        return result


class ValidationResult:
    """Validation result with valid flag and errors"""
    def __init__(self, valid: bool, errors: List[ValidationError] = None):
        self.valid = valid
        self.errors = errors or []
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "valid": self.valid,
            "errors": [e.to_dict() for e in self.errors]
        }


def validate_workflow(graph: dict) -> ValidationResult:
    """
    Validate workflow graph according to canonical contracts.
    
    Args:
        graph: Workflow graph with 'nodes' and 'edges' lists
        
    Returns:
        ValidationResult with valid flag and any errors
    """
    errors: List[ValidationError] = []
    
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    
    # Basic graph existence check
    if not nodes:
        errors.append(ValidationError(
            code="EMPTY_GRAPH",
            message="Workflow must contain at least one node"
        ))
        return ValidationResult(valid=False, errors=errors)
    
    # === PRE-VALIDATION: Node field requirements ===
    
    # 1. Validate each node has required fields (id, type)
    for i, node in enumerate(nodes):
        node_id = node.get("id")
        node_type = node.get("type")
        
        # Check for missing or empty id
        if not node_id or not str(node_id).strip():
            errors.append(ValidationError(
                code="MISSING_NODE_ID",
                message=f"Node at index {i} is missing required 'id' field"
            ))
        
        # Check for missing or empty type
        if not node_type or not str(node_type).strip():
            errors.append(ValidationError(
                code="MISSING_NODE_TYPE",
                message=f"Node at index {i} (id: {node_id if node_id else 'unknown'}) is missing required 'type' field"
            ))
    
    # If critical node fields are missing, stop validation early
    if errors:
        return ValidationResult(valid=False, errors=errors)
    
    # 2. Check for duplicate node IDs
    seen_ids = set()
    for node in nodes:
        node_id = node["id"]
        if node_id in seen_ids:
            errors.append(ValidationError(
                code="DUPLICATE_NODE_ID",
                message=f"Duplicate node id '{node_id}'"
            ))
        seen_ids.add(node_id)
    
    # If duplicates exist, stop validation (node_map would be corrupted)
    if any(e.code == "DUPLICATE_NODE_ID" for e in errors):
        return ValidationResult(valid=False, errors=errors)
    
    # Build node lookup (safe now, all IDs are unique and present)
    node_map = {n["id"]: n for n in nodes}
    
    # 3. Detect and deduplicate edges
    seen_edges = set()
    unique_edges = []
    for edge in edges:
        edge_key = (edge.get("source"), edge.get("target"))
        if edge_key in seen_edges:
            edge_id = edge.get("id", f"{edge.get('source')}->{edge.get('target')}")
            errors.append(ValidationError(
                code="DUPLICATE_EDGE",
                message=f"Duplicate edge from '{edge.get('source')}' to '{edge.get('target')}'",
                edgeId=edge_id
            ))
        else:
            seen_edges.add(edge_key)
            unique_edges.append(edge)
    
    # Use deduplicated edges for further validation
    edges = unique_edges
    
    # === GRAPH STRUCTURE VALIDATION ===
    
    # 1. Validate edges reference existing nodes
    for edge in edges:
        edge_id = edge.get("id", f"{edge.get('source')}->{edge.get('target')}")
        source = edge.get("source")
        target = edge.get("target")
        
        if not source or source not in node_map:
            errors.append(ValidationError(
                code="INVALID_EDGE_SOURCE",
                message=f"Edge source '{source}' does not exist",
                edgeId=edge_id
            ))
        
        if not target or target not in node_map:
            errors.append(ValidationError(
                code="INVALID_EDGE_TARGET",
                message=f"Edge target '{target}' does not exist",
                edgeId=edge_id
            ))
        
        # No self-loops
        if source == target:
            errors.append(ValidationError(
                code="SELF_LOOP",
                message=f"Self-loop edges are not allowed",
                edgeId=edge_id,
                nodeId=source
            ))
    
    # 2. Find root nodes (in-degree == 0)
    in_degree = {n["id"]: 0 for n in nodes}
    for edge in edges:
        target = edge.get("target")
        if target in in_degree:
            in_degree[target] += 1
    
    roots = [nid for nid, deg in in_degree.items() if deg == 0]
    
    if len(roots) == 0:
        errors.append(ValidationError(
            code="NO_ROOT",
            message="Workflow must have at least one root node (no incoming edges)"
        ))
    elif len(roots) > 1:
        errors.append(ValidationError(
            code="MULTIPLE_ROOTS",
            message=f"Workflow must have exactly one root node (found {len(roots)} nodes with no incoming edges)"
        ))
    
    # 3. Root must be StartNode
    if len(roots) == 1:
        root_id = roots[0]
        root_node = node_map[root_id]
        if root_node.get("type") != "StartNode":
            errors.append(ValidationError(
                code="INVALID_ROOT_TYPE",
                message=f"Root node must be of type 'StartNode' (found '{root_node.get('type')}')",
                nodeId=root_id
            ))
    
    # 4. DAG check (cycle detection using Kahn's algorithm)
    if not has_cycle(nodes, edges):
        # No cycle, continue
        pass
    else:
        errors.append(ValidationError(
            code="CYCLE_DETECTED",
            message="Workflow graph contains a cycle (only DAGs are supported)"
        ))
    
    # 5. Reachability check (all nodes must be reachable from root)
    if len(roots) == 1:
        reachable = get_reachable_nodes(roots[0], nodes, edges, node_map)
        unreachable = set(node_map.keys()) - reachable
        if unreachable:
            for nid in unreachable:
                errors.append(ValidationError(
                    code="ORPHAN_NODE",
                    message=f"Node is not reachable from the root StartNode",
                    nodeId=nid
                ))
    
    # === NODE SCHEMA VALIDATION ===
    
    for node in nodes:
        node_id = node.get("id")
        node_type = node.get("type")
        # Check both 'config' and 'data' fields (frontend sends 'data', backend uses 'config')
        config = node.get("config") or node.get("data") or {}
        
        # Allowed node types
        if node_type not in ["StartNode", "SendMessage", "InvokeAgent"]:
            errors.append(ValidationError(
                code="UNKNOWN_NODE_TYPE",
                message=f"Unknown node type '{node_type}' (allowed: StartNode, SendMessage, InvokeAgent)",
                nodeId=node_id
            ))
        
        # SendMessage validation
        if node_type == "SendMessage":
            message = config.get("message", "")
            if not message or not message.strip():
                errors.append(ValidationError(
                    code="MISSING_MESSAGE",
                    message="SendMessage node requires a non-empty 'message' field",
                    nodeId=node_id
                ))
        
        # InvokeAgent validation
        if node_type == "InvokeAgent":
            agent_id = config.get("agentId", "") or config.get("selectedAgent", "")
            if not agent_id or not agent_id.strip():
                errors.append(ValidationError(
                    code="MISSING_AGENT_ID",
                    message="InvokeAgent node requires a non-empty 'agentId' field",
                    nodeId=node_id
                ))
            # Note: config.input is optional, no validation needed
    
    # Return result
    return ValidationResult(valid=len(errors) == 0, errors=errors)


def has_cycle(nodes: List[dict], edges: List[dict]) -> bool:
    """
    Detect if graph has a cycle using Kahn's algorithm.
    Returns True if cycle exists, False if DAG.
    """
    # Build adjacency list and in-degree
    graph = defaultdict(list)
    in_degree = {n["id"]: 0 for n in nodes}
    
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source and target and source in in_degree and target in in_degree:
            graph[source].append(target)
            in_degree[target] += 1
    
    # Kahn's algorithm
    queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
    visited_count = 0
    
    while queue:
        current = queue.popleft()
        visited_count += 1
        
        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # If we didn't visit all nodes, there's a cycle
    return visited_count != len(nodes)


def get_reachable_nodes(start_id: str, nodes: List[dict], edges: List[dict], node_map: Dict[str, dict]) -> Set[str]:
    """
    Get all nodes reachable from start_id using BFS.
    Only traverses edges to nodes that exist in node_map.
    
    Args:
        start_id: Starting node ID
        nodes: List of all nodes
        edges: List of all edges
        node_map: Dictionary mapping node IDs to node objects
        
    Returns:
        Set of reachable node IDs
    """
    # Build adjacency list
    graph = defaultdict(list)
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source and target:
            graph[source].append(target)
    
    # BFS - only traverse to known nodes
    reachable = {start_id}
    queue = deque([start_id])
    
    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            # Only enqueue if neighbor exists in node_map (safety check)
            if neighbor in node_map and neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)
    
    return reachable
