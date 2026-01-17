"""Workflow Alerting - Advanced monitoring and alerting system"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Callable
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .performance_metrics import WorkflowMetrics

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertType(str, Enum):
    """Types of alerts"""
    PERFORMANCE = "performance"  # Slow execution, high latency
    ERROR = "error"  # High error rate, failures
    COST = "cost"  # Budget exceeded, cost spike
    AVAILABILITY = "availability"  # Service unavailable
    CUSTOM = "custom"  # Custom alert conditions


@dataclass
class AlertRule:
    """An alert rule"""
    rule_id: str
    name: str
    alert_type: AlertType
    severity: AlertSeverity
    condition: Callable[[WorkflowMetrics], bool]  # Returns True if alert should trigger
    threshold: Optional[float] = None
    window_minutes: int = 60  # Time window for evaluation
    enabled: bool = True
    cooldown_minutes: int = 15  # Cooldown period before re-alerting


@dataclass
class Alert:
    """An alert"""
    alert_id: str
    rule_id: str
    workflow_id: Optional[str] = None
    severity: AlertSeverity = AlertSeverity.MEDIUM
    message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    triggered_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False


class WorkflowAlerting:
    """Advanced monitoring and alerting system"""
    
    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alerts: List[Alert] = []
        self._alert_counter = 0
        self.metrics_history: List[WorkflowMetrics] = []
        self._initialize_default_rules()
    
    def _initialize_default_rules(self) -> None:
        """Initialize default alert rules"""
        # Performance alert
        self.add_alert_rule(
            rule_id="high_latency",
            name="High Execution Latency",
            alert_type=AlertType.PERFORMANCE,
            severity=AlertSeverity.HIGH,
            condition=lambda m: m.total_duration_ms > 60000,  # > 1 minute
            threshold=60000
        )
        
        # Error rate alert
        self.add_alert_rule(
            rule_id="high_error_rate",
            name="High Error Rate",
            alert_type=AlertType.ERROR,
            severity=AlertSeverity.CRITICAL,
            condition=lambda m: m.failed_nodes > 0 and (m.failed_nodes / max(m.total_nodes, 1)) > 0.2,  # > 20% failure
            threshold=0.2
        )
    
    def add_alert_rule(
        self,
        rule_id: str,
        name: str,
        alert_type: AlertType,
        severity: AlertSeverity,
        condition: Callable[[WorkflowMetrics], bool],
        threshold: Optional[float] = None,
        window_minutes: int = 60,
        cooldown_minutes: int = 15
    ) -> AlertRule:
        """Add an alert rule"""
        rule = AlertRule(
            rule_id=rule_id,
            name=name,
            alert_type=alert_type,
            severity=severity,
            condition=condition,
            threshold=threshold,
            window_minutes=window_minutes,
            cooldown_minutes=cooldown_minutes
        )
        self.alert_rules[rule_id] = rule
        logger.info(f"Added alert rule: {rule_id}")
        return rule
    
    def evaluate_metrics(
        self,
        metrics: WorkflowMetrics,
        workflow_id: Optional[str] = None
    ) -> List[Alert]:
        """Evaluate metrics and trigger alerts"""
        self.metrics_history.append(metrics)
        
        # Keep only recent history (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        self.metrics_history = [
            m for m in self.metrics_history
            if m.start_time and m.start_time >= cutoff.timestamp()
        ]
        
        triggered_alerts = []
        
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            # Check cooldown
            recent_alerts = [
                a for a in self.alerts
                if a.rule_id == rule.rule_id and
                   a.workflow_id == workflow_id and
                   a.triggered_at and
                   (datetime.now() - a.triggered_at).total_seconds() < (rule.cooldown_minutes * 60)
            ]
            if recent_alerts:
                continue  # Still in cooldown
            
            # Evaluate condition
            try:
                should_alert = rule.condition(metrics)
                if should_alert:
                    alert = Alert(
                        alert_id=f"alert_{self._alert_counter}",
                        rule_id=rule.rule_id,
                        workflow_id=workflow_id,
                        severity=rule.severity,
                        message=f"{rule.name}: Threshold exceeded",
                        metrics={
                            "duration_ms": metrics.total_duration_ms,
                            "failed_nodes": metrics.failed_nodes,
                            "total_nodes": metrics.total_nodes
                        },
                        triggered_at=datetime.now()
                    )
                    
                    self._alert_counter += 1
                    self.alerts.append(alert)
                    triggered_alerts.append(alert)
                    
                    logger.warning(f"Alert triggered: {rule.name} for workflow {workflow_id}")
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule.rule_id}: {e}")
        
        return triggered_alerts
    
    def get_alerts(
        self,
        workflow_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        resolved: Optional[bool] = None,
        limit: int = 100
    ) -> List[Alert]:
        """Get alerts with optional filtering"""
        alerts = list(self.alerts)
        
        if workflow_id:
            alerts = [a for a in alerts if a.workflow_id == workflow_id]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if resolved is not None:
            if resolved:
                alerts = [a for a in alerts if a.resolved_at is not None]
            else:
                alerts = [a for a in alerts if a.resolved_at is None]
        
        alerts.sort(key=lambda a: a.triggered_at or datetime.min, reverse=True)
        return alerts[:limit]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        alert = next((a for a in self.alerts if a.alert_id == alert_id), None)
        if alert:
            alert.acknowledged = True
            logger.info(f"Acknowledged alert: {alert_id}")
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        alert = next((a for a in self.alerts if a.alert_id == alert_id), None)
        if alert:
            alert.resolved_at = datetime.now()
            logger.info(f"Resolved alert: {alert_id}")
            return True
        return False
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alerts"""
        total = len(self.alerts)
        unresolved = sum(1 for a in self.alerts if a.resolved_at is None)
        by_severity = {}
        
        for severity in AlertSeverity:
            count = sum(1 for a in self.alerts if a.severity == severity and a.resolved_at is None)
            by_severity[severity.value] = count
        
        return {
            "total_alerts": total,
            "unresolved": unresolved,
            "resolved": total - unresolved,
            "by_severity": by_severity
        }


# Global alerting instance
_global_alerting = WorkflowAlerting()


def get_workflow_alerting() -> WorkflowAlerting:
    """Get the global workflow alerting instance"""
    return _global_alerting
