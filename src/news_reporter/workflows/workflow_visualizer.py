"""Workflow Visualizer - Generate visual representations of workflows"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import json
import logging

from .graph_schema import GraphDefinition, NodeConfig, EdgeConfig

logger = logging.getLogger(__name__)


class WorkflowVisualizer:
    """Generates visual representations of workflows"""
    
    def __init__(self, graph_def: GraphDefinition):
        self.graph_def = graph_def
    
    def to_dot(self) -> str:
        """
        Generate Graphviz DOT format representation.
        
        Returns:
            DOT format string
        """
        lines = ["digraph workflow {"]
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=box, style=rounded];")
        
        # Add nodes
        for node in self.graph_def.nodes:
            node_label = self._format_node_label(node)
            node_color = self._get_node_color(node.type)
            lines.append(
                f'  "{node.id}" [label="{node_label}", fillcolor={node_color}, style="rounded,filled"];'
            )
        
        # Add edges
        for edge in self.graph_def.edges:
            edge_label = ""
            if edge.condition:
                edge_label = f' [label="{edge.condition[:30]}..."]'
            lines.append(f'  "{edge.from_node}" -> "{edge.to_node}"{edge_label};')
        
        lines.append("}")
        return "\n".join(lines)
    
    def to_mermaid(self) -> str:
        """
        Generate Mermaid diagram format.
        
        Returns:
            Mermaid format string
        """
        lines = ["graph LR"]
        
        # Add nodes with styling
        for node in self.graph_def.nodes:
            node_label = self._format_node_label(node)
            node_shape = self._get_mermaid_shape(node.type)
            lines.append(f'    {node.id}{node_shape}["{node_label}"]')
        
        # Add edges
        for edge in self.graph_def.edges:
            edge_label = ""
            if edge.condition:
                edge_label = f'|"{edge.condition[:20]}..."|'
            lines.append(f'    {edge.from_node} -->{edge_label} {edge.to_node}')
        
        return "\n".join(lines)
    
    def to_json_graph(self) -> Dict[str, Any]:
        """
        Generate JSON format for graph visualization libraries (e.g., vis.js, cytoscape).
        
        Returns:
            Dictionary with nodes and edges
        """
        nodes = []
        for node in self.graph_def.nodes:
            nodes.append({
                "id": node.id,
                "label": node.id,
                "type": node.type,
                "title": self._format_node_label(node),
                "color": self._get_node_color(node.type),
                "agent_id": getattr(node, 'agent_id', None),
                "shape": self._get_vis_shape(node.type)
            })
        
        edges = []
        for edge in self.graph_def.edges:
            edges.append({
                "from": edge.from_node,
                "to": edge.to_node,
                "label": edge.condition[:30] if edge.condition else "",
                "arrows": "to",
                "dashes": bool(edge.condition)  # Dashed for conditional edges
            })
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    def to_summary(self) -> Dict[str, Any]:
        """
        Generate a human-readable summary of the workflow.
        
        Returns:
            Summary dictionary
        """
        node_counts = {}
        for node in self.graph_def.nodes:
            node_counts[node.type] = node_counts.get(node.type, 0) + 1
        
        return {
            "name": getattr(self.graph_def, 'name', 'Unnamed Workflow'),
            "description": getattr(self.graph_def, 'description', ''),
            "version": getattr(self.graph_def, 'version', '1.0.0'),
            "entry_node": self.graph_def.entry_node_id,
            "total_nodes": len(self.graph_def.nodes),
            "total_edges": len(self.graph_def.edges),
            "node_types": node_counts,
            "has_loops": any(node.type == "loop" for node in self.graph_def.nodes),
            "has_fanout": any(node.type == "fanout" for node in self.graph_def.nodes),
            "has_merge": any(node.type == "merge" for node in self.graph_def.nodes),
            "limits": self.graph_def.limits.model_dump() if self.graph_def.limits else None
        }
    
    def _format_node_label(self, node: NodeConfig) -> str:
        """Format node label for visualization"""
        label = node.id
        if node.type == "agent" and hasattr(node, 'agent_id'):
            agent_id = getattr(node, 'agent_id', '')
            # Shorten agent ID if too long
            if len(agent_id) > 20:
                agent_id = agent_id[:17] + "..."
            label = f"{node.id}\n({agent_id})"
        elif node.type == "loop":
            max_iters = getattr(node, 'max_iters', '?')
            label = f"{node.id}\n(max: {max_iters})"
        return label
    
    def _get_node_color(self, node_type: str) -> str:
        """Get color for node type"""
        colors = {
            "agent": "lightblue",
            "conditional": "lightyellow",
            "fanout": "lightgreen",
            "merge": "lightcoral",
            "loop": "lightpink"
        }
        return colors.get(node_type, "lightgray")
    
    def _get_mermaid_shape(self, node_type: str) -> str:
        """Get Mermaid shape for node type"""
        shapes = {
            "agent": "(( ))",
            "conditional": "{ }",
            "fanout": "[ ]",
            "merge": "[ ]",
            "loop": "([ ])"
        }
        return shapes.get(node_type, "[ ]")
    
    def _get_vis_shape(self, node_type: str) -> str:
        """Get vis.js shape for node type"""
        shapes = {
            "agent": "box",
            "conditional": "diamond",
            "fanout": "ellipse",
            "merge": "ellipse",
            "loop": "hexagon"
        }
        return shapes.get(node_type, "box")


class WorkflowVersionManager:
    """Manages workflow versions and history"""
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize version manager.
        
        Args:
            storage_dir: Directory to store workflow versions (default: ./workflow_versions)
        """
        from pathlib import Path
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
        import time
        from datetime import datetime
        
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
            
            from .graph_schema import GraphDefinition, NodeConfig, EdgeConfig
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
