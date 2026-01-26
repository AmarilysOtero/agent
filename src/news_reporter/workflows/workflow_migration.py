"""Workflow Migration - Migration and transformation tools"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Callable
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..models.graph_schema import GraphDefinition, NodeConfig, EdgeConfig

logger = logging.getLogger(__name__)


class MigrationType(str, Enum):
    """Types of migrations"""
    VERSION_UPGRADE = "version_upgrade"
    FORMAT_CONVERSION = "format_conversion"
    STRUCTURE_REFACTOR = "structure_refactor"
    NODE_REPLACEMENT = "node_replacement"
    OPTIMIZATION = "optimization"


@dataclass
class MigrationRule:
    """A migration rule"""
    rule_id: str
    name: str
    migration_type: MigrationType
    source_version: str
    target_version: str
    transform: Callable[[GraphDefinition], GraphDefinition]
    enabled: bool = True


@dataclass
class MigrationResult:
    """Result of a migration"""
    migration_id: str
    workflow_id: str
    source_version: str
    target_version: str
    success: bool
    transformed_workflow: Optional[GraphDefinition] = None
    changes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    timestamp: Optional[datetime] = None


class WorkflowMigration:
    """Manages workflow migration and transformation"""
    
    def __init__(self):
        self.migration_rules: Dict[str, MigrationRule] = {}
        self.migration_history: List[MigrationResult] = []
        self._migration_counter = 0
    
    def add_migration_rule(
        self,
        rule_id: str,
        name: str,
        migration_type: MigrationType,
        source_version: str,
        target_version: str,
        transform: Callable[[GraphDefinition], GraphDefinition]
    ) -> MigrationRule:
        """Add a migration rule"""
        rule = MigrationRule(
            rule_id=rule_id,
            name=name,
            migration_type=migration_type,
            source_version=source_version,
            target_version=target_version,
            transform=transform
        )
        self.migration_rules[rule_id] = rule
        logger.info(f"Added migration rule: {rule_id}")
        return rule
    
    def migrate_workflow(
        self,
        workflow_id: str,
        workflow: GraphDefinition,
        target_version: str
    ) -> MigrationResult:
        """Migrate a workflow to a target version"""
        # Find applicable migration rules
        applicable_rules = [
            r for r in self.migration_rules.values()
            if r.enabled and r.target_version == target_version
        ]
        
        if not applicable_rules:
            return MigrationResult(
                migration_id=f"mig_{self._migration_counter}",
                workflow_id=workflow_id,
                source_version="unknown",
                target_version=target_version,
                success=False,
                errors=[f"No migration rules found for version {target_version}"]
            )
        
        # Apply migrations in order
        current_workflow = workflow
        changes = []
        errors = []
        
        for rule in applicable_rules:
            try:
                transformed = rule.transform(current_workflow)
                changes.append(f"Applied {rule.name}: {rule.migration_type.value}")
                current_workflow = transformed
            except Exception as e:
                errors.append(f"Error applying {rule.name}: {e}")
                logger.error(f"Migration error: {e}")
        
        success = len(errors) == 0
        
        result = MigrationResult(
            migration_id=f"mig_{self._migration_counter}",
            workflow_id=workflow_id,
            source_version=applicable_rules[0].source_version if applicable_rules else "unknown",
            target_version=target_version,
            success=success,
            transformed_workflow=current_workflow if success else None,
            changes=changes,
            errors=errors,
            timestamp=datetime.now()
        )
        
        self._migration_counter += 1
        self.migration_history.append(result)
        
        return result
    
    def transform_node_type(
        self,
        workflow: GraphDefinition,
        old_type: str,
        new_type: str,
        node_id_filter: Optional[Callable[[str], bool]] = None
    ) -> GraphDefinition:
        """Transform nodes of one type to another"""
        new_nodes = []
        changes = []
        
        for node in workflow.nodes:
            if node.type == old_type:
                if node_id_filter is None or node_id_filter(node.id):
                    new_node = NodeConfig(
                        id=node.id,
                        type=new_type,
                        agent_id=getattr(node, 'agent_id', None),
                        inputs=node.inputs,
                        outputs=node.outputs,
                        params=node.params
                    )
                    new_nodes.append(new_node)
                    changes.append(f"Transformed node {node.id} from {old_type} to {new_type}")
                else:
                    new_nodes.append(node)
            else:
                new_nodes.append(node)
        
        return GraphDefinition(
            nodes=new_nodes,
            edges=workflow.edges,
            entry_node_id=workflow.entry_node_id
        )
    
    def refactor_structure(
        self,
        workflow: GraphDefinition,
        refactor_strategy: str = "extract_subgraph"
    ) -> GraphDefinition:
        """Refactor workflow structure"""
        # Simplified refactoring - would implement actual strategies
        if refactor_strategy == "extract_subgraph":
            # Extract a subgraph into a nested workflow
            # For now, just return the original
            return workflow
        
        return workflow
    
    def get_migration_history(
        self,
        workflow_id: Optional[str] = None,
        limit: int = 50
    ) -> List[MigrationResult]:
        """Get migration history"""
        history = list(self.migration_history)
        
        if workflow_id:
            history = [m for m in history if m.workflow_id == workflow_id]
        
        history.sort(key=lambda m: m.timestamp or datetime.min, reverse=True)
        return history[:limit]


# Global migration instance
_global_migration = WorkflowMigration()


def get_workflow_migration() -> WorkflowMigration:
    """Get the global workflow migration instance"""
    return _global_migration
