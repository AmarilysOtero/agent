"""Workflow Multi-Tenant - Multi-tenant support and isolation"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TenantTier(str, Enum):
    """Tenant subscription tiers"""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class Tenant:
    """A tenant"""
    tenant_id: str
    name: str
    tier: TenantTier = TenantTier.FREE
    limits: Dict[str, Any] = field(default_factory=dict)  # Resource limits
    created_at: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TenantQuota:
    """Quota limits for a tenant"""
    max_workflows: int = 10
    max_executions_per_day: int = 100
    max_storage_mb: int = 100
    max_concurrent_executions: int = 5
    features: List[str] = field(default_factory=list)  # Enabled features


class WorkflowMultiTenant:
    """Manages multi-tenant support and isolation"""
    
    def __init__(self):
        self.tenants: Dict[str, Tenant] = {}
        self.tenant_quotas: Dict[str, TenantQuota] = {}
        self.workflow_tenants: Dict[str, str] = {}  # workflow_id -> tenant_id
        self.execution_counts: Dict[str, Dict[str, int]] = {}  # tenant_id -> {date: count}
        self._initialize_tier_quotas()
    
    def _initialize_tier_quotas(self) -> None:
        """Initialize default quotas for each tier"""
        self.tier_quotas = {
            TenantTier.FREE: TenantQuota(
                max_workflows=5,
                max_executions_per_day=50,
                max_storage_mb=50,
                max_concurrent_executions=2,
                features=[]
            ),
            TenantTier.BASIC: TenantQuota(
                max_workflows=20,
                max_executions_per_day=500,
                max_storage_mb=500,
                max_concurrent_executions=10,
                features=["analytics", "scheduling"]
            ),
            TenantTier.PROFESSIONAL: TenantQuota(
                max_workflows=100,
                max_executions_per_day=5000,
                max_storage_mb=5000,
                max_concurrent_executions=50,
                features=["analytics", "scheduling", "collaboration", "ai"]
            ),
            TenantTier.ENTERPRISE: TenantQuota(
                max_workflows=-1,  # Unlimited
                max_executions_per_day=-1,  # Unlimited
                max_storage_mb=-1,  # Unlimited
                max_concurrent_executions=200,
                features=["analytics", "scheduling", "collaboration", "ai", "governance", "custom"]
            )
        }
    
    def create_tenant(
        self,
        tenant_id: str,
        name: str,
        tier: TenantTier = TenantTier.FREE
    ) -> Tenant:
        """Create a new tenant"""
        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            tier=tier,
            created_at=datetime.now()
        )
        
        self.tenants[tenant_id] = tenant
        self.tenant_quotas[tenant_id] = self.tier_quotas[tier].__dict__.copy()
        self.execution_counts[tenant_id] = {}
        
        logger.info(f"Created tenant: {tenant_id} (tier: {tier.value})")
        return tenant
    
    def assign_workflow_to_tenant(
        self,
        workflow_id: str,
        tenant_id: str
    ) -> bool:
        """Assign a workflow to a tenant"""
        if tenant_id not in self.tenants:
            return False
        
        self.workflow_tenants[workflow_id] = tenant_id
        logger.info(f"Assigned workflow {workflow_id} to tenant {tenant_id}")
        return True
    
    def get_tenant_for_workflow(self, workflow_id: str) -> Optional[str]:
        """Get tenant ID for a workflow"""
        return self.workflow_tenants.get(workflow_id)
    
    def check_quota(
        self,
        tenant_id: str,
        resource: str,
        amount: int = 1
    ) -> bool:
        """Check if tenant has quota available"""
        tenant = self.tenants.get(tenant_id)
        if not tenant or not tenant.is_active:
            return False
        
        quota = self.tenant_quotas.get(tenant_id, {})
        
        if resource == "workflows":
            current_count = sum(1 for wf_id, t_id in self.workflow_tenants.items() if t_id == tenant_id)
            max_count = quota.get("max_workflows", 0)
            if max_count == -1:  # Unlimited
                return True
            return (current_count + amount) <= max_count
        
        elif resource == "executions_per_day":
            today = datetime.now().date().isoformat()
            today_count = self.execution_counts.get(tenant_id, {}).get(today, 0)
            max_count = quota.get("max_executions_per_day", 0)
            if max_count == -1:  # Unlimited
                return True
            return (today_count + amount) <= max_count
        
        elif resource == "concurrent_executions":
            # Would need to track active executions
            max_count = quota.get("max_concurrent_executions", 0)
            if max_count == -1:  # Unlimited
                return True
            # Simplified - would check actual concurrent executions
            return True
        
        return True
    
    def record_execution(self, tenant_id: str) -> None:
        """Record an execution for quota tracking"""
        today = datetime.now().date().isoformat()
        if tenant_id not in self.execution_counts:
            self.execution_counts[tenant_id] = {}
        self.execution_counts[tenant_id][today] = self.execution_counts[tenant_id].get(today, 0) + 1
    
    def get_tenant_quota_status(self, tenant_id: str) -> Dict[str, Any]:
        """Get quota status for a tenant"""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return {"error": "Tenant not found"}
        
        quota = self.tenant_quotas.get(tenant_id, {})
        today = datetime.now().date().isoformat()
        today_executions = self.execution_counts.get(tenant_id, {}).get(today, 0)
        
        workflow_count = sum(1 for wf_id, t_id in self.workflow_tenants.items() if t_id == tenant_id)
        
        return {
            "tenant_id": tenant_id,
            "tier": tenant.tier.value,
            "quotas": {
                "workflows": {
                    "current": workflow_count,
                    "max": quota.get("max_workflows", 0)
                },
                "executions_today": {
                    "current": today_executions,
                    "max": quota.get("max_executions_per_day", 0)
                }
            },
            "features": quota.get("features", [])
        }
    
    def upgrade_tenant(self, tenant_id: str, new_tier: TenantTier) -> bool:
        """Upgrade tenant to a new tier"""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False
        
        tenant.tier = new_tier
        self.tenant_quotas[tenant_id] = self.tier_quotas[new_tier].__dict__.copy()
        logger.info(f"Upgraded tenant {tenant_id} to tier {new_tier.value}")
        return True


# Global multi-tenant instance
_global_multitenant = WorkflowMultiTenant()


def get_workflow_multitenant() -> WorkflowMultiTenant:
    """Get the global workflow multi-tenant instance"""
    return _global_multitenant
