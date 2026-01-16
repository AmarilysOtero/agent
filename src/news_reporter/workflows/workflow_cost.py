"""Workflow Cost Management - Cost tracking, billing, and optimization"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class CostType(str, Enum):
    """Types of costs"""
    API_CALL = "api_call"  # LLM API calls
    COMPUTE = "compute"  # Compute resources
    STORAGE = "storage"  # Storage costs
    NETWORK = "network"  # Network/bandwidth
    CUSTOM = "custom"  # Custom costs


@dataclass
class CostEntry:
    """A cost entry"""
    entry_id: str
    workflow_id: str
    cost_type: CostType = CostType.API_CALL
    amount: float = 0.0
    execution_id: Optional[str] = None
    node_id: Optional[str] = None
    currency: str = "USD"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None


@dataclass
class CostBudget:
    """A cost budget"""
    budget_id: str
    amount: float
    workflow_id: Optional[str] = None  # None = global budget
    currency: str = "USD"
    period: str = "monthly"  # daily, weekly, monthly, yearly
    alert_threshold: float = 0.8  # Alert at 80% of budget
    enabled: bool = True
    created_at: Optional[datetime] = None


@dataclass
class CostReport:
    """A cost report"""
    period_start: datetime
    period_end: datetime
    total_cost: float
    workflow_id: Optional[str] = None
    cost_by_type: Dict[str, float] = field(default_factory=dict)
    cost_by_node: Dict[str, float] = field(default_factory=dict)
    execution_count: int = 0
    avg_cost_per_execution: float = 0.0


class WorkflowCostManager:
    """Manages workflow costs, billing, and budgets"""
    
    def __init__(self):
        self.cost_entries: List[CostEntry] = []
        self.budgets: Dict[str, CostBudget] = {}
        self._entry_counter = 0
        
        # Default pricing (per unit)
        self.pricing = {
            CostType.API_CALL: 0.002,  # $0.002 per API call
            CostType.COMPUTE: 0.0001,  # $0.0001 per second
            CostType.STORAGE: 0.00001,  # $0.00001 per MB per day
            CostType.NETWORK: 0.000001  # $0.000001 per MB
        }
    
    def record_cost(
        self,
        workflow_id: str,
        cost_type: CostType,
        amount: Optional[float] = None,
        execution_id: Optional[str] = None,
        node_id: Optional[str] = None,
        units: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CostEntry:
        """Record a cost entry"""
        # Calculate amount if not provided
        if amount is None:
            base_price = self.pricing.get(cost_type, 0.0)
            amount = base_price * units
        
        entry = CostEntry(
            entry_id=f"cost_{self._entry_counter}",
            workflow_id=workflow_id,
            execution_id=execution_id,
            node_id=node_id,
            cost_type=cost_type,
            amount=amount,
            metadata=metadata or {},
            timestamp=datetime.now()
        )
        
        self._entry_counter += 1
        self.cost_entries.append(entry)
        
        # Check budgets
        self._check_budgets(workflow_id, amount)
        
        logger.debug(f"Recorded cost: {amount} {entry.currency} for workflow {workflow_id}")
        return entry
    
    def _check_budgets(self, workflow_id: str, amount: float) -> None:
        """Check if cost exceeds budgets"""
        # Check workflow-specific budgets
        for budget in self.budgets.values():
            if not budget.enabled:
                continue
            
            if budget.workflow_id and budget.workflow_id != workflow_id:
                continue
            
            # Calculate current period cost
            current_cost = self.get_period_cost(
                workflow_id=budget.workflow_id,
                period=budget.period
            )
            
            # Check threshold
            threshold = budget.amount * budget.alert_threshold
            if current_cost >= threshold:
                logger.warning(
                    f"Budget {budget.budget_id} threshold reached: "
                    f"${current_cost:.2f} / ${budget.amount:.2f}"
                )
    
    def add_budget(
        self,
        budget_id: str,
        amount: float,
        workflow_id: Optional[str] = None,
        period: str = "monthly",
        alert_threshold: float = 0.8
    ) -> CostBudget:
        """Add a cost budget"""
        budget = CostBudget(
            budget_id=budget_id,
            workflow_id=workflow_id,
            amount=amount,
            period=period,
            alert_threshold=alert_threshold,
            created_at=datetime.now()
        )
        self.budgets[budget_id] = budget
        logger.info(f"Added budget: {budget_id}")
        return budget
    
    def get_period_cost(
        self,
        workflow_id: Optional[str] = None,
        period: str = "monthly"
    ) -> float:
        """Get total cost for a period"""
        now = datetime.now()
        
        if period == "daily":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "weekly":
            start = now - timedelta(days=7)
        elif period == "monthly":
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif period == "yearly":
            start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            start = now - timedelta(days=30)  # Default to 30 days
        
        entries = self.cost_entries
        
        if workflow_id:
            entries = [e for e in entries if e.workflow_id == workflow_id]
        
        entries = [e for e in entries if e.timestamp and e.timestamp >= start]
        
        return sum(e.amount for e in entries)
    
    def generate_cost_report(
        self,
        workflow_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> CostReport:
        """Generate a cost report"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        entries = self.cost_entries
        
        if workflow_id:
            entries = [e for e in entries if e.workflow_id == workflow_id]
        
        entries = [
            e for e in entries
            if e.timestamp and start_date <= e.timestamp <= end_date
        ]
        
        total_cost = sum(e.amount for e in entries)
        
        # Cost by type
        cost_by_type = {}
        for entry in entries:
            cost_type = entry.cost_type.value
            cost_by_type[cost_type] = cost_by_type.get(cost_type, 0.0) + entry.amount
        
        # Cost by node
        cost_by_node = {}
        for entry in entries:
            if entry.node_id:
                cost_by_node[entry.node_id] = cost_by_node.get(entry.node_id, 0.0) + entry.amount
        
        # Execution count
        execution_ids = set(e.execution_id for e in entries if e.execution_id)
        execution_count = len(execution_ids)
        
        avg_cost = total_cost / execution_count if execution_count > 0 else 0.0
        
        return CostReport(
            workflow_id=workflow_id,
            period_start=start_date,
            period_end=end_date,
            total_cost=total_cost,
            cost_by_type=cost_by_type,
            cost_by_node=cost_by_node,
            execution_count=execution_count,
            avg_cost_per_execution=avg_cost
        )
    
    def get_budget_status(self, budget_id: str) -> Dict[str, Any]:
        """Get status of a budget"""
        budget = self.budgets.get(budget_id)
        if not budget:
            return {"error": "Budget not found"}
        
        current_cost = self.get_period_cost(
            workflow_id=budget.workflow_id,
            period=budget.period
        )
        
        remaining = budget.amount - current_cost
        percentage_used = (current_cost / budget.amount * 100) if budget.amount > 0 else 0.0
        
        return {
            "budget_id": budget_id,
            "amount": budget.amount,
            "current_cost": current_cost,
            "remaining": remaining,
            "percentage_used": percentage_used,
            "alert_threshold": budget.alert_threshold,
            "threshold_reached": current_cost >= (budget.amount * budget.alert_threshold)
        }


# Global cost manager instance
_global_cost_manager = WorkflowCostManager()


def get_workflow_cost_manager() -> WorkflowCostManager:
    """Get the global workflow cost manager"""
    return _global_cost_manager
