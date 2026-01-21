"""Workflow Analytics - Generate insights and analytics from execution data"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .performance_metrics import WorkflowMetrics, NodeMetrics

logger = logging.getLogger(__name__)


@dataclass
class WorkflowInsight:
    """An insight about workflow performance"""
    type: str  # "performance", "reliability", "cost", "usage"
    severity: str  # "critical", "warning", "info"
    title: str
    description: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class WorkflowAnalytics:
    """Analytics for a workflow"""
    workflow_id: str
    total_runs: int
    success_rate: float
    avg_duration_ms: float
    p50_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    total_cost_estimate: Optional[float] = None
    most_used_nodes: List[Dict[str, Any]] = field(default_factory=list)
    slowest_nodes: List[Dict[str, Any]] = field(default_factory=list)
    error_prone_nodes: List[Dict[str, Any]] = field(default_factory=list)
    insights: List[WorkflowInsight] = field(default_factory=list)
    trends: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "workflow_id": self.workflow_id,
            "total_runs": self.total_runs,
            "success_rate": self.success_rate,
            "avg_duration_ms": self.avg_duration_ms,
            "p50_duration_ms": self.p50_duration_ms,
            "p95_duration_ms": self.p95_duration_ms,
            "p99_duration_ms": self.p99_duration_ms,
            "total_cost_estimate": self.total_cost_estimate,
            "most_used_nodes": self.most_used_nodes,
            "slowest_nodes": self.slowest_nodes,
            "error_prone_nodes": self.error_prone_nodes,
            "insights": [
                {
                    "type": i.type,
                    "severity": i.severity,
                    "title": i.title,
                    "description": i.description,
                    "recommendations": i.recommendations
                }
                for i in self.insights
            ],
            "trends": self.trends
        }


class WorkflowAnalyticsEngine:
    """Generates analytics and insights from workflow execution data"""
    
    def __init__(self):
        self.metrics_history: List[WorkflowMetrics] = []
    
    def add_metrics(self, metrics: WorkflowMetrics) -> None:
        """Add metrics to history"""
        self.metrics_history.append(metrics)
    
    def analyze_workflow(
        self,
        workflow_id: str,
        time_range_days: Optional[int] = None
    ) -> WorkflowAnalytics:
        """
        Analyze workflow performance.
        
        Args:
            workflow_id: Workflow identifier
            time_range_days: Optional time range in days
        
        Returns:
            WorkflowAnalytics
        """
        # Filter metrics by workflow and time range
        relevant_metrics = self._filter_metrics(workflow_id, time_range_days)
        
        if not relevant_metrics:
            return WorkflowAnalytics(
                workflow_id=workflow_id,
                total_runs=0,
                success_rate=0.0,
                avg_duration_ms=0.0,
                p50_duration_ms=0.0,
                p95_duration_ms=0.0,
                p99_duration_ms=0.0
            )
        
        # Calculate statistics
        total_runs = len(relevant_metrics)
        successful_runs = sum(1 for m in relevant_metrics if m.failed_nodes == 0)
        success_rate = successful_runs / total_runs if total_runs > 0 else 0.0
        
        durations = [m.total_duration_ms for m in relevant_metrics]
        durations.sort()
        
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        p50 = durations[len(durations) // 2] if durations else 0.0
        p95 = durations[int(len(durations) * 0.95)] if durations else 0.0
        p99 = durations[int(len(durations) * 0.99)] if durations else 0.0
        
        # Analyze nodes
        most_used = self._analyze_node_usage(relevant_metrics)
        slowest = self._analyze_slowest_nodes(relevant_metrics)
        error_prone = self._analyze_error_prone_nodes(relevant_metrics)
        
        # Generate insights
        insights = self._generate_insights(relevant_metrics, avg_duration, success_rate)
        
        # Calculate trends
        trends = self._calculate_trends(relevant_metrics)
        
        return WorkflowAnalytics(
            workflow_id=workflow_id,
            total_runs=total_runs,
            success_rate=success_rate,
            avg_duration_ms=avg_duration,
            p50_duration_ms=p50,
            p95_duration_ms=p95,
            p99_duration_ms=p99,
            most_used_nodes=most_used,
            slowest_nodes=slowest,
            error_prone_nodes=error_prone,
            insights=insights,
            trends=trends
        )
    
    def _filter_metrics(
        self,
        workflow_id: str,
        time_range_days: Optional[int]
    ) -> List[WorkflowMetrics]:
        """Filter metrics by workflow ID and time range"""
        filtered = []
        cutoff_time = None
        
        if time_range_days:
            cutoff_time = time.time() - (time_range_days * 24 * 60 * 60)
        
        for metrics in self.metrics_history:
            # Match workflow by goal or run_id pattern (simplified)
            if cutoff_time and metrics.start_time < cutoff_time:
                continue
            filtered.append(metrics)
        
        return filtered
    
    def _analyze_node_usage(self, metrics_list: List[WorkflowMetrics]) -> List[Dict[str, Any]]:
        """Analyze which nodes are used most frequently"""
        node_counts = {}
        
        for metrics in metrics_list:
            for node_metric in metrics.node_metrics:
                node_id = node_metric.node_id
                node_counts[node_id] = node_counts.get(node_id, 0) + 1
        
        sorted_nodes = sorted(node_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"node_id": node_id, "usage_count": count}
            for node_id, count in sorted_nodes[:10]
        ]
    
    def _analyze_slowest_nodes(self, metrics_list: List[WorkflowMetrics]) -> List[Dict[str, Any]]:
        """Analyze slowest nodes"""
        node_durations = {}
        
        for metrics in metrics_list:
            for node_metric in metrics.node_metrics:
                node_id = node_metric.node_id
                if node_id not in node_durations:
                    node_durations[node_id] = []
                node_durations[node_id].append(node_metric.duration_ms)
        
        avg_durations = {
            node_id: sum(durations) / len(durations)
            for node_id, durations in node_durations.items()
        }
        
        sorted_nodes = sorted(avg_durations.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"node_id": node_id, "avg_duration_ms": duration}
            for node_id, duration in sorted_nodes[:10]
        ]
    
    def _analyze_error_prone_nodes(self, metrics_list: List[WorkflowMetrics]) -> List[Dict[str, Any]]:
        """Analyze nodes that fail most often"""
        node_errors = {}
        
        for metrics in metrics_list:
            for node_metric in metrics.node_metrics:
                if node_metric.status == "failed":
                    node_id = node_metric.node_id
                    node_errors[node_id] = node_errors.get(node_id, 0) + 1
        
        sorted_nodes = sorted(node_errors.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"node_id": node_id, "error_count": count}
            for node_id, count in sorted_nodes[:10]
        ]
    
    def _generate_insights(
        self,
        metrics_list: List[WorkflowMetrics],
        avg_duration: float,
        success_rate: float
    ) -> List[WorkflowInsight]:
        """Generate insights from metrics"""
        insights = []
        
        # Performance insight
        if avg_duration > 60000:  # > 1 minute
            insights.append(WorkflowInsight(
                type="performance",
                severity="warning",
                title="Long Execution Time",
                description=f"Average execution time is {avg_duration/1000:.1f}s",
                recommendations=[
                    "Consider optimizing slow nodes",
                    "Enable caching for repeated operations",
                    "Review parallelization opportunities"
                ]
            ))
        
        # Reliability insight
        if success_rate < 0.9:
            insights.append(WorkflowInsight(
                type="reliability",
                severity="critical" if success_rate < 0.7 else "warning",
                title="Low Success Rate",
                description=f"Success rate is {success_rate*100:.1f}%",
                recommendations=[
                    "Review error logs",
                    "Check node configurations",
                    "Enable retry mechanisms"
                ]
            ))
        
        # Cache efficiency insight
        total_cache_hits = sum(m.cache_hits for m in metrics_list)
        total_cache_misses = sum(m.cache_misses for m in metrics_list)
        if total_cache_hits + total_cache_misses > 0:
            cache_hit_rate = total_cache_hits / (total_cache_hits + total_cache_misses)
            if cache_hit_rate < 0.3:
                insights.append(WorkflowInsight(
                    type="performance",
                    severity="info",
                    title="Low Cache Hit Rate",
                    description=f"Cache hit rate is {cache_hit_rate*100:.1f}%",
                    recommendations=[
                        "Review cache configuration",
                        "Check if inputs are too variable for caching"
                    ]
                ))
        
        return insights
    
    def _calculate_trends(self, metrics_list: List[WorkflowMetrics]) -> Dict[str, Any]:
        """Calculate trends over time"""
        if len(metrics_list) < 2:
            return {}
        
        # Sort by time
        sorted_metrics = sorted(metrics_list, key=lambda m: m.start_time)
        
        # Calculate duration trend
        recent_avg = sum(m.total_duration_ms for m in sorted_metrics[-5:]) / min(5, len(sorted_metrics))
        older_avg = sum(m.total_duration_ms for m in sorted_metrics[:-5]) / max(1, len(sorted_metrics) - 5)
        
        duration_trend = "improving" if recent_avg < older_avg else "degrading"
        
        # Calculate success rate trend
        recent_success = sum(1 for m in sorted_metrics[-5:] if m.failed_nodes == 0) / min(5, len(sorted_metrics))
        older_success = sum(1 for m in sorted_metrics[:-5] if m.failed_nodes == 0) / max(1, len(sorted_metrics) - 5)
        
        success_trend = "improving" if recent_success > older_success else "degrading"
        
        return {
            "duration_trend": duration_trend,
            "success_rate_trend": success_trend,
            "recent_avg_duration_ms": recent_avg,
            "older_avg_duration_ms": older_avg
        }


# Global analytics engine
_global_analytics = WorkflowAnalyticsEngine()


def get_analytics_engine() -> WorkflowAnalyticsEngine:
    """Get the global analytics engine"""
    return _global_analytics
