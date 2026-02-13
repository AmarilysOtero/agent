"""Workflow Documentation - Documentation and knowledge base"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..models.graph_schema import GraphDefinition

logger = logging.getLogger(__name__)


class DocumentationType(str, Enum):
    """Types of documentation"""
    WORKFLOW = "workflow"  # Workflow documentation
    NODE = "node"  # Node documentation
    API = "api"  # API documentation
    TUTORIAL = "tutorial"  # Tutorial/guide
    FAQ = "faq"  # FAQ


@dataclass
class Documentation:
    """A documentation entry"""
    doc_id: str
    type: DocumentationType
    title: str
    content: str
    workflow_id: Optional[str] = None
    node_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    created_by: Optional[str] = None
    views: int = 0


@dataclass
class KnowledgeBaseEntry:
    """A knowledge base entry"""
    entry_id: str
    title: str
    content: str
    category: str
    tags: List[str] = field(default_factory=list)
    related_workflows: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    helpful_count: int = 0


class WorkflowDocumentation:
    """Manages workflow documentation and knowledge base"""
    
    def __init__(self):
        self.documentation: Dict[str, Documentation] = {}
        self.knowledge_base: Dict[str, KnowledgeBaseEntry] = {}
        self._doc_counter = 0
        self._kb_counter = 0
    
    def add_documentation(
        self,
        doc_id: str,
        type: DocumentationType,
        title: str,
        content: str,
        workflow_id: Optional[str] = None,
        node_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        created_by: Optional[str] = None
    ) -> Documentation:
        """Add documentation"""
        doc = Documentation(
            doc_id=doc_id,
            type=type,
            title=title,
            content=content,
            workflow_id=workflow_id,
            node_id=node_id,
            tags=tags or [],
            created_by=created_by,
            created_at=datetime.now()
        )
        
        self.documentation[doc_id] = doc
        logger.info(f"Added documentation: {doc_id}")
        return doc
    
    def get_documentation(
        self,
        doc_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        node_id: Optional[str] = None,
        type: Optional[DocumentationType] = None
    ) -> List[Documentation]:
        """Get documentation with optional filtering"""
        docs = list(self.documentation.values())
        
        if doc_id:
            doc = self.documentation.get(doc_id)
            return [doc] if doc else []
        
        if workflow_id:
            docs = [d for d in docs if d.workflow_id == workflow_id]
        
        if node_id:
            docs = [d for d in docs if d.node_id == node_id]
        
        if type:
            docs = [d for d in docs if d.type == type]
        
        return docs
    
    def generate_workflow_docs(
        self,
        workflow_id: str,
        workflow: GraphDefinition
    ) -> Documentation:
        """Auto-generate documentation for a workflow"""
        # Generate markdown documentation
        # Exclude start node from count (start is just entry point, not executable)
        node_count = len([n for n in workflow.nodes if n.type != 'start'])
        content = f"""# {workflow_id}

## Overview
This workflow contains {node_count} executable nodes and {len(workflow.edges)} edges.

## Entry Point
Entry node: {workflow.entry_node_id}

## Nodes
"""
        for node in workflow.nodes:
            content += f"\n### {node.id}\n"
            content += f"- Type: {node.type}\n"
            if hasattr(node, 'agent_id') and node.agent_id:
                content += f"- Agent: {node.agent_id}\n"
            if node.params:
                content += f"- Parameters: {len(node.params)} configured\n"
        
        content += "\n## Edges\n"
        for edge in workflow.edges:
            content += f"- {edge.from_node} → {edge.to_node}"
            if edge.condition:
                content += f" (condition: {edge.condition})"
            content += "\n"
        
        doc = self.add_documentation(
            doc_id=f"doc_{workflow_id}",
            type=DocumentationType.WORKFLOW,
            title=f"Documentation for {workflow_id}",
            content=content,
            workflow_id=workflow_id
        )
        
        return doc
    
    def add_knowledge_base_entry(
        self,
        entry_id: str,
        title: str,
        content: str,
        category: str,
        tags: Optional[List[str]] = None,
        related_workflows: Optional[List[str]] = None
    ) -> KnowledgeBaseEntry:
        """Add a knowledge base entry"""
        entry = KnowledgeBaseEntry(
            entry_id=entry_id,
            title=title,
            content=content,
            category=category,
            tags=tags or [],
            related_workflows=related_workflows or [],
            created_at=datetime.now()
        )
        
        self.knowledge_base[entry_id] = entry
        logger.info(f"Added knowledge base entry: {entry_id}")
        return entry
    
    def search_knowledge_base(
        self,
        query: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[KnowledgeBaseEntry]:
        """Search knowledge base"""
        entries = list(self.knowledge_base.values())
        
        # Simple text search
        query_lower = query.lower()
        matching = [
            e for e in entries
            if query_lower in e.title.lower() or query_lower in e.content.lower()
        ]
        
        if category:
            matching = [e for e in matching if e.category == category]
        
        if tags:
            matching = [
                e for e in matching
                if any(tag in e.tags for tag in tags)
            ]
        
        return matching[:limit]
    
    def get_knowledge_base_entry(self, entry_id: str) -> Optional[KnowledgeBaseEntry]:
        """Get a knowledge base entry by ID"""
        return self.knowledge_base.get(entry_id)
    
    def increment_views(self, doc_id: str) -> None:
        """Increment view count for documentation"""
        doc = self.documentation.get(doc_id)
        if doc:
            doc.views += 1
            logger.debug(f"Incremented views for {doc_id}: {doc.views}")


# Global documentation instance
_global_documentation = WorkflowDocumentation()


def get_workflow_documentation() -> WorkflowDocumentation:
    """Get the global workflow documentation instance"""
    return _global_documentation
