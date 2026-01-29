"""Workflow Governance - Compliance, policies, and governance features"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Callable
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..models.graph_schema import GraphDefinition

logger = logging.getLogger(__name__)


class PolicyType(str, Enum):
    """Types of governance policies"""
    NAMING = "naming"  # Naming conventions
    SECURITY = "security"  # Security requirements
    PERFORMANCE = "performance"  # Performance requirements
    COST = "cost"  # Cost limits
    COMPLIANCE = "compliance"  # Compliance requirements
    RESOURCE = "resource"  # Resource limits


class PolicySeverity(str, Enum):
    """Policy violation severity"""
    ERROR = "error"  # Must be fixed
    WARNING = "warning"  # Should be fixed
    INFO = "info"  # Informational


@dataclass
class Policy:
    """A governance policy"""
    policy_id: str
    name: str
    type: PolicyType
    description: str
    rule: Callable[[GraphDefinition], bool]  # Returns True if compliant
    severity: PolicySeverity = PolicySeverity.WARNING
    enabled: bool = True


@dataclass
class PolicyViolation:
    """A policy violation"""
    policy_id: str
    policy_name: str
    severity: PolicySeverity
    message: str
    workflow_id: Optional[str] = None
    node_id: Optional[str] = None
    timestamp: Optional[datetime] = None


class WorkflowGovernance:
    """Manages workflow governance, compliance, and policies"""
    
    def __init__(self):
        self.policies: Dict[str, Policy] = {}
        self.violations: List[PolicyViolation] = []
        self._initialize_default_policies()
    
    def _initialize_default_policies(self) -> None:
        """Initialize default governance policies"""
        # Naming policy
        self.add_policy(
            policy_id="naming_convention",
            name="Naming Convention",
            type=PolicyType.NAMING,
            description="Workflow and node names must follow conventions",
            rule=lambda g: all(
                len(node.id) > 0 and node.id.replace("_", "").replace("-", "").isalnum()
                for node in g.nodes
            ),
            severity=PolicySeverity.WARNING
        )
        
        # Security policy - no hardcoded secrets
        self.add_policy(
            policy_id="no_hardcoded_secrets",
            name="No Hardcoded Secrets",
            type=PolicyType.SECURITY,
            description="Workflows must not contain hardcoded secrets",
            rule=lambda g: not any(
                "password" in str(node.params).lower() or
                "secret" in str(node.params).lower() or
                "key" in str(node.params).lower()
                for node in g.nodes
            ),
            severity=PolicySeverity.ERROR
        )
        
        # Performance policy - max nodes
        self.add_policy(
            policy_id="max_nodes",
            name="Maximum Nodes",
            type=PolicyType.PERFORMANCE,
            description="Workflows should not exceed 100 nodes",
            rule=lambda g: len(g.nodes) <= 100,
            severity=PolicySeverity.WARNING
        )
    
    def add_policy(
        self,
        policy_id: str,
        name: str,
        type: PolicyType,
        description: str,
        rule: Callable[[GraphDefinition], bool],
        severity: PolicySeverity = PolicySeverity.WARNING
    ) -> Policy:
        """Add a governance policy"""
        policy = Policy(
            policy_id=policy_id,
            name=name,
            type=type,
            description=description,
            rule=rule,
            severity=severity
        )
        self.policies[policy_id] = policy
        logger.info(f"Added policy: {policy_id}")
        return policy
    
    def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy"""
        if policy_id in self.policies:
            del self.policies[policy_id]
            logger.info(f"Removed policy: {policy_id}")
            return True
        return False
    
    def validate_workflow(
        self,
        workflow: GraphDefinition,
        workflow_id: Optional[str] = None
    ) -> List[PolicyViolation]:
        """Validate a workflow against all policies"""
        violations = []
        
        for policy in self.policies.values():
            if not policy.enabled:
                continue
            
            try:
                is_compliant = policy.rule(workflow)
                if not is_compliant:
                    violation = PolicyViolation(
                        policy_id=policy.policy_id,
                        policy_name=policy.name,
                        severity=policy.severity,
                        message=f"Policy violation: {policy.description}",
                        workflow_id=workflow_id,
                        timestamp=datetime.now()
                    )
                    violations.append(violation)
                    self.violations.append(violation)
            except Exception as e:
                logger.error(f"Error evaluating policy {policy.policy_id}: {e}")
                violation = PolicyViolation(
                    policy_id=policy.policy_id,
                    policy_name=policy.name,
                    severity=PolicySeverity.ERROR,
                    message=f"Error evaluating policy: {e}",
                    workflow_id=workflow_id,
                    timestamp=datetime.now()
                )
                violations.append(violation)
        
        return violations
    
    def get_violations(
        self,
        workflow_id: Optional[str] = None,
        severity: Optional[PolicySeverity] = None,
        limit: int = 100
    ) -> List[PolicyViolation]:
        """Get policy violations with optional filtering"""
        violations = list(self.violations)
        
        if workflow_id:
            violations = [v for v in violations if v.workflow_id == workflow_id]
        
        if severity:
            violations = [v for v in violations if v.severity == severity]
        
        # Sort by timestamp descending
        violations.sort(key=lambda v: v.timestamp or datetime.min, reverse=True)
        
        return violations[:limit]
    
    def get_compliance_report(
        self,
        workflow_id: str,
        workflow: GraphDefinition
    ) -> Dict[str, Any]:
        """Get a compliance report for a workflow"""
        violations = self.validate_workflow(workflow, workflow_id)
        
        error_count = sum(1 for v in violations if v.severity == PolicySeverity.ERROR)
        warning_count = sum(1 for v in violations if v.severity == PolicySeverity.WARNING)
        info_count = sum(1 for v in violations if v.severity == PolicySeverity.INFO)
        
        total_policies = len([p for p in self.policies.values() if p.enabled])
        compliant_policies = total_policies - len(violations)
        compliance_rate = compliant_policies / total_policies if total_policies > 0 else 0.0
        
        return {
            "workflow_id": workflow_id,
            "total_policies": total_policies,
            "compliant_policies": compliant_policies,
            "compliance_rate": compliance_rate,
            "violations": {
                "errors": error_count,
                "warnings": warning_count,
                "info": info_count,
                "total": len(violations)
            },
            "violation_details": [
                {
                    "policy_id": v.policy_id,
                    "policy_name": v.policy_name,
                    "severity": v.severity.value,
                    "message": v.message
                }
                for v in violations
            ]
        }


# Global governance instance
_global_governance = WorkflowGovernance()


def get_workflow_governance() -> WorkflowGovernance:
    """Get the global workflow governance instance"""
    return _global_governance
