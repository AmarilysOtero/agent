"""Workflow Deployment - CI/CD integration, migration, and deployment tools"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..models.graph_schema import GraphDefinition
from .workflow_persistence import WorkflowPersistence, WorkflowRecord

logger = logging.getLogger(__name__)


class DeploymentStatus(str, Enum):
    """Deployment status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class Deployment:
    """Deployment record"""
    deployment_id: str
    workflow_id: str
    version: str
    source_environment: str
    target_environment: str
    status: DeploymentStatus
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    rollback_version: Optional[str] = None


@dataclass
class Migration:
    """Workflow migration record"""
    migration_id: str
    from_version: str
    to_version: str
    workflow_id: str
    migration_script: Optional[str] = None
    applied_at: Optional[datetime] = None
    status: str = "pending"


class WorkflowDeployment:
    """Manages workflow deployment and migration"""
    
    def __init__(self, persistence: Optional[WorkflowPersistence] = None):
        self.persistence = persistence or WorkflowPersistence()
        self.deployments: Dict[str, Deployment] = {}
        self.migrations: Dict[str, Migration] = {}
        self.environments: List[str] = ["dev", "staging", "production"]
    
    def deploy_workflow(
        self,
        workflow_id: str,
        target_environment: str,
        source_environment: str = "dev",
        version: Optional[str] = None
    ) -> Deployment:
        """
        Deploy a workflow to a target environment.
        
        Args:
            workflow_id: Workflow to deploy
            target_environment: Target environment
            source_environment: Source environment
            version: Optional version override
        
        Returns:
            Deployment record
        """
        workflow = self.persistence.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        deployment_version = version or workflow.version
        
        deployment = Deployment(
            deployment_id=f"deploy_{datetime.now().timestamp()}",
            workflow_id=workflow_id,
            version=deployment_version,
            source_environment=source_environment,
            target_environment=target_environment,
            status=DeploymentStatus.IN_PROGRESS,
            created_at=datetime.now()
        )
        
        self.deployments[deployment.deployment_id] = deployment
        
        try:
            # Validate workflow
            graph_def = GraphDefinition(**workflow.graph_definition)
            errors = graph_def.validate()
            if errors:
                raise ValueError(f"Invalid workflow: {', '.join(errors)}")
            
            # Perform deployment (simplified - would actually deploy to target)
            logger.info(f"Deploying workflow {workflow_id} v{deployment_version} to {target_environment}")
            
            # Simulate deployment steps
            # 1. Export workflow
            workflow_data = self.persistence.export_workflow(workflow_id)
            
            # 2. Transform for target environment (update configs, endpoints, etc.)
            transformed_workflow = self._transform_for_environment(workflow_data, target_environment)
            
            # 3. Import to target (would actually import to target system)
            # For now, just mark as success
            deployment.status = DeploymentStatus.SUCCESS
            deployment.completed_at = datetime.now()
            
            logger.info(f"Successfully deployed workflow {workflow_id} to {target_environment}")
        
        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            deployment.error = str(e)
            deployment.completed_at = datetime.now()
            logger.error(f"Failed to deploy workflow {workflow_id}: {e}")
        
        return deployment
    
    def _transform_for_environment(
        self,
        workflow_data: Dict[str, Any],
        environment: str
    ) -> Dict[str, Any]:
        """Transform workflow configuration for target environment"""
        # This would update endpoints, credentials, resource limits, etc.
        # based on the target environment
        transformed = workflow_data.copy()
        
        # Example: Update agent IDs based on environment
        if "graph_definition" in transformed:
            graph = transformed["graph_definition"]
            # Would transform node configurations here
        
        return transformed
    
    def rollback_deployment(
        self,
        deployment_id: str,
        rollback_version: Optional[str] = None
    ) -> bool:
        """Rollback a deployment"""
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return False
        
        try:
            rollback_ver = rollback_version or deployment.rollback_version
            if not rollback_ver:
                # Find previous version
                workflow = self.persistence.get_workflow(deployment.workflow_id)
                # Simplified - would find actual previous version
                rollback_ver = workflow.version
            
            logger.info(f"Rolling back deployment {deployment_id} to version {rollback_ver}")
            
            # Perform rollback
            deployment.status = DeploymentStatus.ROLLED_BACK
            deployment.rollback_version = rollback_ver
            deployment.completed_at = datetime.now()
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to rollback deployment {deployment_id}: {e}")
            return False
    
    def create_migration(
        self,
        workflow_id: str,
        from_version: str,
        to_version: str,
        migration_script: Optional[str] = None
    ) -> Migration:
        """Create a workflow migration"""
        migration = Migration(
            migration_id=f"migrate_{datetime.now().timestamp()}",
            workflow_id=workflow_id,
            from_version=from_version,
            to_version=to_version,
            migration_script=migration_script
        )
        
        self.migrations[migration.migration_id] = migration
        logger.info(f"Created migration from {from_version} to {to_version} for workflow {workflow_id}")
        
        return migration
    
    def apply_migration(self, migration_id: str) -> bool:
        """Apply a migration"""
        migration = self.migrations.get(migration_id)
        if not migration:
            return False
        
        try:
            workflow = self.persistence.get_workflow(migration.workflow_id)
            if not workflow:
                raise ValueError(f"Workflow {migration.workflow_id} not found")
            
            # Apply migration script if provided
            if migration.migration_script:
                # Would execute migration script here
                logger.info(f"Executing migration script for {migration_id}")
            
            # Update workflow version
            workflow.version = migration.to_version
            workflow.updated_at = datetime.now()
            self.persistence.save_workflow(workflow)
            
            migration.status = "applied"
            migration.applied_at = datetime.now()
            
            logger.info(f"Applied migration {migration_id}")
            return True
        
        except Exception as e:
            migration.status = "failed"
            logger.error(f"Failed to apply migration {migration_id}: {e}")
            return False
    
    def export_for_deployment(
        self,
        workflow_id: str,
        environment: str
    ) -> Dict[str, Any]:
        """Export workflow in deployment-ready format"""
        workflow = self.persistence.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow_data = self.persistence.export_workflow(workflow_id)
        transformed = self._transform_for_environment(workflow_data, environment)
        
        return {
            "workflow": transformed,
            "environment": environment,
            "exported_at": datetime.now().isoformat(),
            "version": workflow.version
        }
    
    def get_deployment_history(
        self,
        workflow_id: Optional[str] = None,
        environment: Optional[str] = None
    ) -> List[Deployment]:
        """Get deployment history"""
        deployments = list(self.deployments.values())
        
        if workflow_id:
            deployments = [d for d in deployments if d.workflow_id == workflow_id]
        
        if environment:
            deployments = [d for d in deployments if d.target_environment == environment]
        
        # Sort by created_at descending
        deployments.sort(key=lambda d: d.created_at or datetime.min, reverse=True)
        
        return deployments


# Global deployment instance
_global_deployment = WorkflowDeployment()


def get_workflow_deployment() -> WorkflowDeployment:
    """Get the global workflow deployment instance"""
    return _global_deployment
