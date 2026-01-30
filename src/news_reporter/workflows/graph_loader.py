"""Graph loader - Loads graph definitions from JSON"""

from __future__ import annotations
from typing import Optional
import json
import os
from pathlib import Path

from ..models.graph_schema import GraphDefinition, NodeConfig, EdgeConfig
from ..config import Settings


def load_graph_definition(
    graph_path: Optional[str] = None,
    config: Optional[Settings] = None
) -> GraphDefinition:
    """
    Load graph definition from JSON file.
    
    Args:
        graph_path: Path to JSON file (default: default_workflow.json)
        config: Settings instance for variable substitution
    
    Returns:
        GraphDefinition instance
    """
    if graph_path is None:
        # Default to default_workflow.json in same directory
        graph_path = Path(__file__).parent / "default_workflow.json"
    
    graph_path = Path(graph_path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph definition not found: {graph_path}")
    
    # Load JSON
    with open(graph_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    # Substitute environment variables in agent_id fields
    if config:
        graph_data = _substitute_agent_ids(graph_data, config)
    
    # Parse nodes
    nodes = [NodeConfig(**node_data) for node_data in graph_data.get("nodes", [])]
    
    # Parse edges
    edges = [EdgeConfig(**edge_data) for edge_data in graph_data.get("edges", [])]
    
    # Create graph definition
    graph_def = GraphDefinition(
        nodes=nodes,
        edges=edges,
        toolsets=graph_data.get("toolsets", []),
        policy_profile=graph_data.get("policy_profile", "read_only"),
        limits=graph_data.get("limits"),
        name=graph_data.get("name"),
        description=graph_data.get("description"),
        version=graph_data.get("version")
    )
    
    return graph_def


def _substitute_agent_ids(graph_data: dict, config: Settings) -> dict:
    """Substitute ${VAR} placeholders in agent_id fields with actual values"""
    # Create mapping of env vars to config values
    substitutions = {
        "AGENT_ID_TRIAGE": config.agent_id_triage,
        "AGENT_ID_AISEARCH": config.agent_id_aisearch,
        "AGENT_ID_AISEARCHSQL": getattr(config, 'agent_id_aisearch_sql', None),
        "AGENT_ID_NEO4J_SEARCH": getattr(config, 'agent_id_neo4j_search', None),
        "AGENT_ID_REVIEWER": config.agent_id_reviewer,
        "AGENT_ID_REPORTER": config.reporter_ids[0] if config.reporter_ids else None,
    }
    
    # Substitute in nodes
    for node in graph_data.get("nodes", []):
        if "agent_id" in node and isinstance(node["agent_id"], str):
            agent_id = node["agent_id"]
            if agent_id.startswith("${") and agent_id.endswith("}"):
                var_name = agent_id[2:-1]
                if var_name in substitutions:
                    node["agent_id"] = substitutions[var_name]
                else:
                    # Try environment variable
                    node["agent_id"] = os.getenv(var_name, agent_id)
    
    return graph_data
