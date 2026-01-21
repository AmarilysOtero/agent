"""Workflow Optimizer - Analyze and optimize workflow performance"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Set
import logging
from dataclasses import dataclass, field

from .graph_schema import GraphDefinition, NodeConfig, EdgeConfig
from .performance_metrics import WorkflowMetrics, NodeMetrics

logger = logging.getLogger(__name__)


@dataclass
class OptimizationSuggestion:
    """A suggestion for workflow optimization"""
    type: str  # "parallelize", "cache", "remove_redundant", "merge_nodes", etc.
    severity: str  # "high", "medium", "low"
    description: str
    affected_nodes: List[str] = field(default_factory=list)
    estimated_improvement: Optional[float] = None  # Percentage improvement
    implementation_hint: Optional[str] = None


@dataclass
class WorkflowAnalysis:
    """Analysis results for a workflow"""
    workflow_id: str
    total_nodes: int
    total_edges: int
    critical_path_length: int
    estimated_duration_ms: float
    parallelization_opportunities: int
    cache_opportunities: int
    redundant_nodes: List[str] = field(default_factory=list)
    bottlenecks: List[str] = field(default_factory=list)
    suggestions: List[OptimizationSuggestion] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "workflow_id": self.workflow_id,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "critical_path_length": self.critical_path_length,
            "estimated_duration_ms": self.estimated_duration_ms,
            "parallelization_opportunities": self.parallelization_opportunities,
            "cache_opportunities": self.cache_opportunities,
            "redundant_nodes": self.redundant_nodes,
            "bottlenecks": self.bottlenecks,
            "suggestions": [
                {
                    "type": s.type,
                    "severity": s.severity,
                    "description": s.description,
                    "affected_nodes": s.affected_nodes,
                    "estimated_improvement": s.estimated_improvement
                }
                for s in self.suggestions
            ]
        }


class WorkflowOptimizer:
    """Analyzes and optimizes workflows"""
    
    def __init__(self, graph_def: GraphDefinition):
        self.graph_def = graph_def
        self._build_graph_structure()
    
    def _build_graph_structure(self) -> None:
        """Build internal graph structures for analysis"""
        self.nodes: Dict[str, NodeConfig] = {node.id: node for node in self.graph_def.nodes}
        self.outgoing_edges: Dict[str, List[EdgeConfig]] = {}
        self.incoming_edges: Dict[str, List[EdgeConfig]] = {}
        
        for edge in self.graph_def.edges:
            if edge.from_node not in self.outgoing_edges:
                self.outgoing_edges[edge.from_node] = []
            self.outgoing_edges[edge.from_node].append(edge)
            
            if edge.to_node not in self.incoming_edges:
                self.incoming_edges[edge.to_node] = []
            self.incoming_edges[edge.to_node].append(edge)
    
    def analyze(self, metrics: Optional[WorkflowMetrics] = None) -> WorkflowAnalysis:
        """
        Analyze workflow for optimization opportunities.
        
        Args:
            metrics: Optional historical metrics for analysis
        
        Returns:
            WorkflowAnalysis with suggestions
        """
        suggestions = []
        
        # Find parallelization opportunities
        parallel_opps = self._find_parallelization_opportunities()
        if parallel_opps:
            suggestions.append(OptimizationSuggestion(
                type="parallelize",
                severity="high",
                description=f"Found {len(parallel_opps)} opportunities to parallelize nodes",
                affected_nodes=parallel_opps,
                estimated_improvement=30.0,  # Rough estimate
                implementation_hint="Consider using fanout nodes for independent branches"
            ))
        
        # Find cache opportunities
        cache_opps = self._find_cache_opportunities()
        if cache_opps:
            suggestions.append(OptimizationSuggestion(
                type="cache",
                severity="medium",
                description=f"Found {len(cache_opps)} nodes that could benefit from caching",
                affected_nodes=cache_opps,
                estimated_improvement=50.0,
                implementation_hint="Enable caching for nodes with repeated inputs"
            ))
        
        # Find redundant nodes
        redundant = self._find_redundant_nodes()
        if redundant:
            suggestions.append(OptimizationSuggestion(
                type="remove_redundant",
                severity="medium",
                description=f"Found {len(redundant)} potentially redundant nodes",
                affected_nodes=redundant,
                estimated_improvement=10.0,
                implementation_hint="Review if these nodes are necessary"
            ))
        
        # Find bottlenecks
        bottlenecks = self._find_bottlenecks(metrics)
        if bottlenecks:
            suggestions.append(OptimizationSuggestion(
                type="optimize_bottleneck",
                severity="high",
                description=f"Found {len(bottlenecks)} bottleneck nodes",
                affected_nodes=bottlenecks,
                estimated_improvement=20.0,
                implementation_hint="Consider optimizing or parallelizing these nodes"
            ))
        
        # Calculate critical path
        critical_path = self._calculate_critical_path()
        
        return WorkflowAnalysis(
            workflow_id=getattr(self.graph_def, 'name', 'unknown'),
            total_nodes=len(self.graph_def.nodes),
            total_edges=len(self.graph_def.edges),
            critical_path_length=len(critical_path),
            estimated_duration_ms=0.0,  # Would use metrics if available
            parallelization_opportunities=len(parallel_opps),
            cache_opportunities=len(cache_opps),
            redundant_nodes=redundant,
            bottlenecks=bottlenecks,
            suggestions=suggestions
        )
    
    def _find_parallelization_opportunities(self) -> List[str]:
        """Find nodes that could be parallelized"""
        opportunities = []
        
        # Find nodes with multiple independent incoming edges
        for node_id, incoming in self.incoming_edges.items():
            if len(incoming) > 1:
                # Check if incoming nodes are independent (no dependencies between them)
                from_nodes = [e.from_node for e in incoming]
                if self._are_independent(from_nodes):
                    opportunities.append(node_id)
        
        return opportunities
    
    def _are_independent(self, node_ids: List[str]) -> bool:
        """Check if nodes are independent (no dependencies)"""
        # Simple check: if any node is reachable from another, they're not independent
        for i, node1 in enumerate(node_ids):
            for node2 in node_ids[i+1:]:
                if self._is_reachable(node1, node2) or self._is_reachable(node2, node1):
                    return False
        return True
    
    def _is_reachable(self, from_node: str, to_node: str, visited: Optional[Set[str]] = None) -> bool:
        """Check if to_node is reachable from from_node"""
        if visited is None:
            visited = set()
        
        if from_node == to_node:
            return True
        
        if from_node in visited:
            return False
        
        visited.add(from_node)
        
        for edge in self.outgoing_edges.get(from_node, []):
            if self._is_reachable(edge.to_node, to_node, visited):
                return True
        
        return False
    
    def _find_cache_opportunities(self) -> List[str]:
        """Find nodes that could benefit from caching"""
        opportunities = []
        
        # Agent nodes are good candidates for caching
        for node in self.graph_def.nodes:
            if node.type == "agent":
                opportunities.append(node.id)
        
        return opportunities
    
    def _find_redundant_nodes(self) -> List[str]:
        """Find potentially redundant nodes"""
        redundant = []
        
        # Find nodes with no outputs or that don't affect final output
        for node in self.graph_def.nodes:
            if node.id not in self.outgoing_edges:
                # Terminal node - check if it's actually used
                if not self._leads_to_output(node.id):
                    redundant.append(node.id)
        
        return redundant
    
    def _leads_to_output(self, node_id: str) -> bool:
        """Check if node leads to a terminal/output node"""
        visited = set()
        queue = [node_id]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            # If this is a terminal node (no outgoing edges), it's an output
            if current not in self.outgoing_edges or not self.outgoing_edges[current]:
                return True
            
            for edge in self.outgoing_edges[current]:
                if edge.to_node not in visited:
                    queue.append(edge.to_node)
        
        return False
    
    def _find_bottlenecks(self, metrics: Optional[WorkflowMetrics]) -> List[str]:
        """Find bottleneck nodes based on metrics"""
        bottlenecks = []
        
        if not metrics:
            return bottlenecks
        
        # Find nodes with longest execution time
        node_durations = {}
        for node_metric in metrics.node_metrics:
            node_durations[node_metric.node_id] = node_metric.duration_ms
        
        if node_durations:
            avg_duration = sum(node_durations.values()) / len(node_durations)
            # Nodes taking more than 2x average are bottlenecks
            for node_id, duration in node_durations.items():
                if duration > avg_duration * 2:
                    bottlenecks.append(node_id)
        
        return bottlenecks
    
    def _calculate_critical_path(self) -> List[str]:
        """Calculate the critical path (longest path) through the workflow"""
        # Simplified: find longest path from entry to any terminal node
        entry_nodes = self.graph_def.get_entry_nodes()
        if not entry_nodes:
            return []
        
        longest_path = []
        visited = set()
        
        def dfs(node_id: str, path: List[str]) -> None:
            nonlocal longest_path
            if node_id in visited:
                return
            
            visited.add(node_id)
            current_path = path + [node_id]
            
            if len(current_path) > len(longest_path):
                longest_path = current_path
            
            for edge in self.outgoing_edges.get(node_id, []):
                dfs(edge.to_node, current_path)
            
            visited.remove(node_id)
        
        for entry_node in entry_nodes:
            dfs(entry_node, [])
        
        return longest_path
    
    def optimize(self, suggestions: Optional[List[OptimizationSuggestion]] = None) -> GraphDefinition:
        """
        Generate an optimized version of the workflow.
        
        Args:
            suggestions: Optional list of suggestions to apply
        
        Returns:
            Optimized GraphDefinition
        """
        # For now, return the original graph
        # In the future, this could apply optimizations automatically
        logger.info("Workflow optimization not yet fully implemented")
        return self.graph_def
