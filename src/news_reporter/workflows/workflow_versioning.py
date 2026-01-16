"""Workflow Versioning - Manage workflow versions and history"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import json
import time
import logging
from pathlib import Path
from datetime import datetime

from .graph_schema import GraphDefinition, NodeConfig, EdgeConfig

logger = logging.getLogger(__name__)


class WorkflowVersionManager:
    """Manages workflow versions and history"""
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize version manager.
        
        Args:
            storage_dir: Directory to store workflow versions (default: ./workflow_versions)
        """
        if storage_dir:
            self.storage_dir = Path(storage_dir)
        else:
            self.storage_dir = Path("./workflow_versions")
        
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def save_version(
        self,
        workflow_id: str,
        graph_def: GraphDefinition,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a workflow version.
        
        Args:
            workflow_id: Unique workflow identifier
            graph_def: Graph definition to save
            version: Version string (auto-generated if None)
            metadata: Additional metadata
        
        Returns:
            Version string
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        version_data = {
            "workflow_id": workflow_id,
            "version": version,
            "timestamp": time.time(),
            "created_at": datetime.now().isoformat(),
            "graph": graph_def.model_dump(),
            "metadata": metadata or {}
        }
        
        version_file = self.storage_dir / f"{workflow_id}_{version}.json"
        
        with open(version_file, 'w') as f:
            json.dump(version_data, f, indent=2, default=str)
        
        logger.info(f"Saved workflow {workflow_id} version {version}")
        return version
    
    def list_versions(self, workflow_id: str) -> List[Dict[str, Any]]:
        """List all versions of a workflow"""
        versions = []
        for version_file in self.storage_dir.glob(f"{workflow_id}_*.json"):
            try:
                with open(version_file, 'r') as f:
                    version_data = json.load(f)
                versions.append({
                    "version": version_data["version"],
                    "timestamp": version_data["timestamp"],
                    "created_at": version_data["created_at"],
                    "metadata": version_data.get("metadata", {})
                })
            except Exception as e:
                logger.warning(f"Failed to load version file {version_file}: {e}")
        
        return sorted(versions, key=lambda x: x["timestamp"], reverse=True)
    
    def load_version(self, workflow_id: str, version: str) -> Optional[GraphDefinition]:
        """Load a specific workflow version"""
        version_file = self.storage_dir / f"{workflow_id}_{version}.json"
        
        if not version_file.exists():
            return None
        
        try:
            with open(version_file, 'r') as f:
                version_data = json.load(f)
            
            graph_data = version_data["graph"]
            nodes = [NodeConfig(**node_data) for node_data in graph_data.get("nodes", [])]
            edges = [EdgeConfig(**edge_data) for edge_data in graph_data.get("edges", [])]
            return GraphDefinition(
                nodes=nodes,
                edges=edges,
                entry_node_id=graph_data.get("entry_node_id"),
                toolsets=graph_data.get("toolsets", []),
                policy_profile=graph_data.get("policy_profile", "read_only"),
                limits=graph_data.get("limits"),
                name=graph_data.get("name"),
                description=graph_data.get("description"),
                version=graph_data.get("version")
            )
        except Exception as e:
            logger.error(f"Failed to load version {version} for workflow {workflow_id}: {e}")
            return None
    
    def compare_versions(
        self,
        workflow_id: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Compare two workflow versions.
        
        Returns:
            Dictionary with differences
        """
        graph1 = self.load_version(workflow_id, version1)
        graph2 = self.load_version(workflow_id, version2)
        
        if not graph1 or not graph2:
            return {"error": "One or both versions not found"}
        
        differences = {
            "nodes_added": [],
            "nodes_removed": [],
            "nodes_modified": [],
            "edges_added": [],
            "edges_removed": [],
            "edges_modified": []
        }
        
        # Compare nodes
        nodes1 = {node.id: node for node in graph1.nodes}
        nodes2 = {node.id: node for node in graph2.nodes}
        
        for node_id in set(nodes1.keys()) | set(nodes2.keys()):
            if node_id not in nodes1:
                differences["nodes_added"].append(node_id)
            elif node_id not in nodes2:
                differences["nodes_removed"].append(node_id)
            else:
                # Check if node was modified
                if nodes1[node_id].model_dump() != nodes2[node_id].model_dump():
                    differences["nodes_modified"].append(node_id)
        
        # Compare edges
        edges1 = {(e.from_node, e.to_node): e for e in graph1.edges}
        edges2 = {(e.from_node, e.to_node): e for e in graph2.edges}
        
        for edge_key in set(edges1.keys()) | set(edges2.keys()):
            if edge_key not in edges1:
                differences["edges_added"].append(edge_key)
            elif edge_key not in edges2:
                differences["edges_removed"].append(edge_key)
            else:
                # Check if edge was modified
                if edges1[edge_key].model_dump() != edges2[edge_key].model_dump():
                    differences["edges_modified"].append(edge_key)
        
        return differences
