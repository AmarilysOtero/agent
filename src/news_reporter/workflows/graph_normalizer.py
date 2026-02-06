"""Graph Normalizer - Defensive normalization of workflow graphs"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import logging

from ..models.graph_schema import GraphDefinition, NodeConfig, EdgeConfig

logger = logging.getLogger(__name__)

# UI cluster condition markers (both variations for defensive compatibility)
UI_CLUSTER_CONDITIONS = {"__ui_cluster__", "ui_cluster"}


def _normalize_condition(cond: Optional[str]) -> str:
    """Normalize a condition string for comparison by removing special chars and whitespace"""
    if not cond:
        return ""
    # Remove asterisks, underscores, and whitespace for comparison
    return cond.strip().replace("*", "").replace("_", "").lower()


def _is_ui_cluster_condition(condition: Optional[str]) -> bool:
    """Check if a condition is a UI cluster marker (handles variations like **ui_cluster**)"""
    if not condition:
        return False
    
    # Exact match check (fast path)
    if condition in UI_CLUSTER_CONDITIONS:
        return True
    
    # Normalized check for variations like "**ui_cluster**", " ui_cluster ", etc.
    normalized = _normalize_condition(condition)
    return "uicluster" in normalized


def normalize_workflow_graph(definition: GraphDefinition) -> GraphDefinition:
    """
    Defensively normalize workflow graph to remove UI artifacts and ensure Loop integrity.
    
    This function:
    1. Strips UI helper nodes (loop_body, loop_exit)
    2. Strips UI cluster edges (__ui_cluster__, ui_cluster, **ui_cluster**, etc.)
    3. Validates helper nodes have required connections
    4. Converts helper edges to proper loop_continue/loop_exit edges
    5. Validates Loop nodes have required routing
    
    Args:
        definition: Raw workflow definition (potentially with UI artifacts)
    
    Returns:
        Normalized GraphDefinition safe for execution
    
    Raises:
        ValueError: If helper nodes aren't properly wired or validation fails
    """
    logger.info(f"[Normalizer] Starting normalization for: {definition.name}")
    logger.info(f"[Normalizer] Input: {len(definition.nodes)} nodes, {len(definition.edges)} edges, entry_node_id={definition.entry_node_id}")
    logger.info(f"[Normalizer] Input nodes: {[(n.id, n.type) for n in definition.nodes]}")
    logger.info(f"[Normalizer] Input edges: {[(e.from_node, e.to_node, e.condition or 'NO_COND') for e in definition.edges]}")
    
    print(f"\n[NORMALIZER] Starting: entry_node={definition.entry_node_id}, nodes={len(definition.nodes)}, edges={len(definition.edges)}")
    print(f"[NORMALIZER] Nodes: {[(n.id, n.type) for n in definition.nodes]}")
    
    # Track helper node mappings (helper_id -> loop_id)
    helper_to_loop: Dict[str, str] = {}
    loop_body_helpers: Dict[str, str] = {}  # loop_id -> body_helper_id
    loop_exit_helpers: Dict[str, str] = {}  # loop_id -> exit_helper_id
    
    # Identify helper nodes and their parent loops
    for node in definition.nodes:
        if node.type == "loop_body":
            # Extract loop ID from helper node ID (assumes pattern: {loop_id}_body)
            if node.id.endswith("_body"):
                loop_id = node.id.rsplit("_body", 1)[0]
                helper_to_loop[node.id] = loop_id
                loop_body_helpers[loop_id] = node.id
                logger.info(f"[Normalizer] Found loop_body helper: {node.id} -> loop {loop_id}")
        elif node.type == "loop_exit":
            # Extract loop ID from helper node ID (assumes pattern: {loop_id}_exit)
            if node.id.endswith("_exit"):
                loop_id = node.id.rsplit("_exit", 1)[0]
                helper_to_loop[node.id] = loop_id
                loop_exit_helpers[loop_id] = node.id
                logger.info(f"[Normalizer] Found loop_exit helper: {node.id} -> loop {loop_id}")
    
    # CRITICAL: Validate helper nodes have required connections BEFORE conversion
    for loop_id, body_helper_id in loop_body_helpers.items():
        # Find real outgoing edges from loop_body (excluding UI cluster edges)
        body_outgoing = [
            e for e in definition.edges 
            if e.from_node == body_helper_id and not _is_ui_cluster_condition(e.condition)
        ]
        
        if len(body_outgoing) == 0:
            raise ValueError(
                f"Loop Body ({body_helper_id}) must be connected to a body entry node. "
                f"Please wire Loop Body → <first node in loop>."
            )
        elif len(body_outgoing) > 1:
            targets = [e.to_node for e in body_outgoing]
            raise ValueError(
                f"Loop Body ({body_helper_id}) has {len(body_outgoing)} outgoing connections to {targets}. "
                f"Loop Body can only connect to ONE body entry node."
            )
    
    for loop_id, exit_helper_id in loop_exit_helpers.items():
        # Find real outgoing edges from loop_exit (excluding UI cluster edges)
        exit_outgoing = [
            e for e in definition.edges 
            if e.from_node == exit_helper_id and not _is_ui_cluster_condition(e.condition)
        ]
        
        if len(exit_outgoing) == 0:
            raise ValueError(
                f"Exit Loop ({exit_helper_id}) must be connected to an exit successor node. "
                f"Please wire Exit Loop → <next node after loop>."
            )
        elif len(exit_outgoing) > 1:
            targets = [e.to_node for e in exit_outgoing]
            raise ValueError(
                f"Exit Loop ({exit_helper_id}) has {len(exit_outgoing)} outgoing connections to {targets}. "
                f"Exit Loop can only connect to ONE exit successor node."
            )
    
    # Filter out helper nodes
    backend_nodes = [n for n in definition.nodes if n.type not in ("loop_body", "loop_exit")]
    logger.info(f"[Normalizer] Filtered {len(definition.nodes) - len(backend_nodes)} helper nodes")
    
    # Process edges: strip UI edges, convert helper edges to loop edges
    backend_edges: List[EdgeConfig] = []
    
    logger.info(f"[Normalizer] Processing {len(definition.edges)} edges...")
    for edge in definition.edges:
        edge_desc = f"{edge.from_node}→{edge.to_node} [{edge.condition or 'NO_CONDITION'}]"
        
        # Skip UI cluster edges (robust checking for variations)
        if _is_ui_cluster_condition(edge.condition):
            logger.info(f"[Normalizer] ✓ Filtering UI cluster edge: {edge_desc}")
            continue
        
        source_node_type = _get_node_type(definition.nodes, edge.from_node)
        
        # Convert loop_body → X edges to loop → X [loop_continue]
        if source_node_type == "loop_body":
            loop_id = helper_to_loop.get(edge.from_node)
            if loop_id:
                converted_edge = EdgeConfig(
                    from_node=loop_id,
                    to_node=edge.to_node,
                    condition="loop_continue"
                )
                backend_edges.append(converted_edge)
                logger.info(f"[Normalizer] ✓ Converted body edge: {edge_desc} => {loop_id}→{edge.to_node} [loop_continue]")
            else:
                logger.warning(f"[Normalizer] Could not find loop for helper {edge.from_node}, skipping edge")
            continue
        
        # Convert loop_exit → Y edges to loop → Y [loop_exit]
        if source_node_type == "loop_exit":
            loop_id = helper_to_loop.get(edge.from_node)
            if loop_id:
                converted_edge = EdgeConfig(
                    from_node=loop_id,
                    to_node=edge.to_node,
                    condition="loop_exit"
                )
                backend_edges.append(converted_edge)
                logger.info(f"[Normalizer] ✓ Converted exit edge: {edge_desc} => {loop_id}→{edge.to_node} [loop_exit]")
            else:
                logger.warning(f"[Normalizer] Could not find loop for helper {edge.from_node}, skipping edge")
            continue
        
        # Filter out any direct loop → X edges (should not exist if UI is correct)
        if source_node_type == "loop" and not edge.condition:
            logger.warning(f"[Normalizer] Stripping untagged loop edge: {edge_desc}")
            continue
        
        # Pass through normal edges
        logger.debug(f"[Normalizer] Pass-through edge: {edge_desc}")
        backend_edges.append(edge)
    
    logger.info(f"[Normalizer] Output: {len(backend_nodes)} nodes, {len(backend_edges)} edges")
    logger.info(f"[Normalizer] Final nodes: {[(n.id, n.type) for n in backend_nodes]}")
    logger.info(f"[Normalizer] Final edges: {[(e.from_node, e.to_node, e.condition or 'NO_COND') for e in backend_edges]}")
    
    # Fix entry_node_id: must exist in backend AND have no incoming edges
    backend_node_ids = {n.id for n in backend_nodes}
    nodes_with_incoming = {e.to_node for e in backend_edges}
    entry_node_id = definition.entry_node_id
    
    if entry_node_id:
        if entry_node_id not in backend_node_ids:
            logger.warning(
                f"[Normalizer] Entry node '{entry_node_id}' not in backend nodes. "
                f"Will auto-detect entry node."
            )
            entry_node_id = None
        elif entry_node_id in nodes_with_incoming:
            logger.warning(
                f"[Normalizer] Entry node '{entry_node_id}' has incoming edges (not a true entry). "
                f"Will auto-detect entry node from nodes with no incoming edges."
            )
            print(f"[NORMALIZER] ⚠️ Invalid entry_node '{entry_node_id}' has incoming edges. Auto-detecting...")
            entry_node_id = None
    
    # Create normalized graph
    normalized = GraphDefinition(
        nodes=backend_nodes,
        edges=backend_edges,
        entry_node_id=entry_node_id,
        name=definition.name,
        description=definition.description,
        version=definition.version,
        limits=definition.limits,
        toolsets=definition.toolsets,
        policy_profile=definition.policy_profile
    )
    
    # Validate normalized graph
    _validate_normalized_graph(normalized)
    
    # Log the final entry nodes detected
    logger.info(f"[Normalizer] Created GraphDefinition with entry_node_id={normalized.entry_node_id}")
    detected_entry_nodes = normalized.get_entry_nodes()
    logger.info(f"[Normalizer] ✅ Normalization complete. Entry nodes detected: {detected_entry_nodes}")
    print(f"[NORMALIZER] ✅ Complete: entry_node_id={normalized.entry_node_id}, Entry nodes={detected_entry_nodes}, nodes={len(normalized.nodes)}, edges={len(normalized.edges)}\n")
    
    if not detected_entry_nodes:
        logger.error(f"[Normalizer] ❌ No entry nodes detected! This will cause execution to fail.")
        print(f"[NORMALIZER] ❌ ERROR: No entry nodes detected!")
        print(f"[NORMALIZER] Debug: nodes_with_incoming={{{', '.join([e.to_node for e in normalized.edges])}}}")
        print(f"[NORMALIZER] Debug: all_node_ids={{{', '.join([n.id for n in normalized.nodes])}}}")
    
    return normalized


def _get_node_type(nodes: List[NodeConfig], node_id: str) -> Optional[str]:
    """Helper to get node type by ID"""
    for node in nodes:
        if node.id == node_id:
            return node.type
    return None


def _validate_normalized_graph(graph: GraphDefinition) -> None:
    """Validate that normalized graph has no UI artifacts"""
    
    # Check for UI helper nodes
    helper_nodes = [n for n in graph.nodes if n.type in ("loop_body", "loop_exit")]
    if helper_nodes:
        raise ValueError(f"Normalized graph contains {len(helper_nodes)} helper nodes: {[n.id for n in helper_nodes]}")
    
    # Check for UI cluster edges
    ui_edges = [e for e in graph.edges if e.condition in UI_CLUSTER_CONDITIONS]
    if ui_edges:
        raise ValueError(f"Normalized graph contains {len(ui_edges)} UI cluster edges")
    
    # Check loop nodes have required edges
    loop_nodes = [n for n in graph.nodes if n.type == "loop"]
    for loop_node in loop_nodes:
        outgoing = [e for e in graph.edges if e.from_node == loop_node.id]
        
        has_continue = any(e.condition == "loop_continue" for e in outgoing)
        has_exit = any(e.condition == "loop_exit" for e in outgoing)
        
        if not has_continue:
            logger.error(f"[Normalizer] Loop {loop_node.id} missing loop_continue edge!")
        if not has_exit:
            logger.error(f"[Normalizer] Loop {loop_node.id} missing loop_exit edge!")
        
        # Check for untagged edges
        untagged = [e for e in outgoing if not e.condition or e.condition == ""]
        if untagged:
            raise ValueError(f"Loop {loop_node.id} has {len(untagged)} untagged outgoing edges: {[e.to_node for e in untagged]}")
    
    logger.info("[Normalizer] ✓ Graph validation passed")
