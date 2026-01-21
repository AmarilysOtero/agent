"""Workflow Templates - Predefined workflow templates and presets"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

from .graph_schema import GraphDefinition, NodeConfig, EdgeConfig, GraphLimits


class WorkflowTemplate:
    """A workflow template"""
    
    def __init__(
        self,
        template_id: str,
        name: str,
        description: str,
        graph_def: GraphDefinition,
        parameters: Optional[Dict[str, Any]] = None
    ):
        self.template_id = template_id
        self.name = name
        self.description = description
        self.graph_def = graph_def
        self.parameters = parameters or {}
    
    def instantiate(self, **kwargs) -> GraphDefinition:
        """
        Instantiate template with provided parameters.
        
        Args:
            **kwargs: Template parameters
        
        Returns:
            Instantiated GraphDefinition
        """
        # For now, return a copy of the graph definition
        # In the future, this could do parameter substitution
        return self.graph_def


class WorkflowTemplateRegistry:
    """Registry of workflow templates"""
    
    def __init__(self):
        self.templates: Dict[str, WorkflowTemplate] = {}
        self._load_default_templates()
    
    def register(self, template: WorkflowTemplate) -> None:
        """Register a template"""
        self.templates[template.template_id] = template
    
    def get(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get a template by ID"""
        return self.templates.get(template_id)
    
    def list_all(self) -> List[Dict[str, Any]]:
        """List all templates"""
        return [
            {
                "template_id": t.template_id,
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters
            }
            for t in self.templates.values()
        ]
    
    def _load_default_templates(self) -> None:
        """Load default templates"""
        # Simple linear workflow template
        simple_template = self._create_simple_template()
        self.register(simple_template)
        
        # Parallel processing template
        parallel_template = self._create_parallel_template()
        self.register(parallel_template)
        
        # Review loop template
        review_template = self._create_review_loop_template()
        self.register(review_template)
    
    def _create_simple_template(self) -> WorkflowTemplate:
        """Create a simple linear workflow template"""
        nodes = [
            NodeConfig(id="start", type="agent", agent_id="${AGENT_ID_START}"),
            NodeConfig(id="process", type="agent", agent_id="${AGENT_ID_PROCESS}"),
            NodeConfig(id="end", type="agent", agent_id="${AGENT_ID_END}")
        ]
        edges = [
            EdgeConfig(from_node="start", to_node="process"),
            EdgeConfig(from_node="process", to_node="end")
        ]
        
        graph_def = GraphDefinition(
            nodes=nodes,
            edges=edges,
            entry_node_id="start",
            limits=GraphLimits(max_steps=50, timeout_ms=60000)
        )
        
        return WorkflowTemplate(
            template_id="simple_linear",
            name="Simple Linear Workflow",
            description="A simple three-step linear workflow",
            graph_def=graph_def,
            parameters={
                "AGENT_ID_START": "Agent ID for start node",
                "AGENT_ID_PROCESS": "Agent ID for process node",
                "AGENT_ID_END": "Agent ID for end node"
            }
        )
    
    def _create_parallel_template(self) -> WorkflowTemplate:
        """Create a parallel processing template"""
        nodes = [
            NodeConfig(id="start", type="agent", agent_id="${AGENT_ID_START}"),
            NodeConfig(id="fanout", type="fanout", branches=["worker1", "worker2"]),
            NodeConfig(id="worker1", type="agent", agent_id="${AGENT_ID_WORKER}"),
            NodeConfig(id="worker2", type="agent", agent_id="${AGENT_ID_WORKER}"),
            NodeConfig(id="merge", type="merge", params={"strategy": "collect_list"}),
            NodeConfig(id="end", type="agent", agent_id="${AGENT_ID_END}")
        ]
        edges = [
            EdgeConfig(from_node="start", to_node="fanout"),
            EdgeConfig(from_node="fanout", to_node="worker1"),
            EdgeConfig(from_node="fanout", to_node="worker2"),
            EdgeConfig(from_node="worker1", to_node="merge"),
            EdgeConfig(from_node="worker2", to_node="merge"),
            EdgeConfig(from_node="merge", to_node="end")
        ]
        
        graph_def = GraphDefinition(
            nodes=nodes,
            edges=edges,
            entry_node_id="start",
            limits=GraphLimits(max_steps=100, timeout_ms=120000, max_parallel=5)
        )
        
        return WorkflowTemplate(
            template_id="parallel_processing",
            name="Parallel Processing Workflow",
            description="Fanout to multiple workers, then merge results",
            graph_def=graph_def,
            parameters={
                "AGENT_ID_START": "Agent ID for start node",
                "AGENT_ID_WORKER": "Agent ID for worker nodes",
                "AGENT_ID_END": "Agent ID for end node"
            }
        )
    
    def _create_review_loop_template(self) -> WorkflowTemplate:
        """Create a review loop template"""
        nodes = [
            NodeConfig(id="generate", type="agent", agent_id="${AGENT_ID_GENERATOR}"),
            NodeConfig(
                id="review_loop",
                type="loop",
                max_iters=3,
                params={"body_node_id": "review"}
            ),
            NodeConfig(id="review", type="agent", agent_id="${AGENT_ID_REVIEWER}"),
            NodeConfig(id="improve", type="agent", agent_id="${AGENT_ID_IMPROVER}"),
            NodeConfig(id="finalize", type="agent", agent_id="${AGENT_ID_FINALIZER}")
        ]
        edges = [
            EdgeConfig(from_node="generate", to_node="review_loop"),
            EdgeConfig(from_node="review_loop", to_node="review"),
            EdgeConfig(
                from_node="review",
                to_node="improve",
                condition="review.decision != 'accept'"
            ),
            EdgeConfig(from_node="improve", to_node="review_loop"),
            EdgeConfig(from_node="review_loop", to_node="finalize")
        ]
        
        graph_def = GraphDefinition(
            nodes=nodes,
            edges=edges,
            entry_node_id="generate",
            limits=GraphLimits(max_steps=50, timeout_ms=180000, max_iters=3)
        )
        
        return WorkflowTemplate(
            template_id="review_loop",
            name="Review Loop Workflow",
            description="Generate, review, and improve in a loop until accepted",
            graph_def=graph_def,
            parameters={
                "AGENT_ID_GENERATOR": "Agent ID for content generation",
                "AGENT_ID_REVIEWER": "Agent ID for review",
                "AGENT_ID_IMPROVER": "Agent ID for improvement",
                "AGENT_ID_FINALIZER": "Agent ID for finalization"
            }
        )


# Global template registry
_global_registry = WorkflowTemplateRegistry()


def get_template_registry() -> WorkflowTemplateRegistry:
    """Get the global template registry"""
    return _global_registry
