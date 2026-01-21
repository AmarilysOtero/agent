"""Performance Metrics - Collect and report execution metrics"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class NodeMetrics:
    """Metrics for a single node execution"""
    node_id: str
    node_type: str
    status: str
    duration_ms: float
    start_time: float
    end_time: float
    retry_count: int = 0
    error: Optional[str] = None
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    cache_hit: bool = False


@dataclass
class WorkflowMetrics:
    """Overall workflow execution metrics"""
    run_id: str
    goal: str
    total_duration_ms: float
    total_nodes_executed: int
    successful_nodes: int
    failed_nodes: int
    skipped_nodes: int
    total_retries: int
    cache_hits: int
    cache_misses: int
    start_time: float
    end_time: float
    node_metrics: List[NodeMetrics] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "total_duration_ms": self.total_duration_ms,
            "total_nodes_executed": self.total_nodes_executed,
            "successful_nodes": self.successful_nodes,
            "failed_nodes": self.failed_nodes,
            "skipped_nodes": self.skipped_nodes,
            "total_retries": self.total_retries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "node_metrics": [
                {
                    "node_id": m.node_id,
                    "node_type": m.node_type,
                    "status": m.status,
                    "duration_ms": m.duration_ms,
                    "retry_count": m.retry_count,
                    "cache_hit": m.cache_hit
                }
                for m in self.node_metrics
            ]
        }


class PerformanceCollector:
    """Collects performance metrics during workflow execution"""
    
    def __init__(self):
        self.metrics: Dict[str, WorkflowMetrics] = {}  # run_id -> metrics
        self.current_run_id: Optional[str] = None
        self.current_metrics: Optional[WorkflowMetrics] = None
    
    def start_workflow(self, run_id: str, goal: str) -> None:
        """Start tracking a workflow execution"""
        self.current_run_id = run_id
        self.current_metrics = WorkflowMetrics(
            run_id=run_id,
            goal=goal,
            total_duration_ms=0.0,
            total_nodes_executed=0,
            successful_nodes=0,
            failed_nodes=0,
            skipped_nodes=0,
            total_retries=0,
            cache_hits=0,
            cache_misses=0,
            start_time=time.time(),
            end_time=0.0
        )
        self.metrics[run_id] = self.current_metrics
        logger.info(f"Started metrics collection for run {run_id}")
    
    def record_node_execution(
        self,
        node_id: str,
        node_type: str,
        status: str,
        duration_ms: float,
        start_time: float,
        end_time: float,
        retry_count: int = 0,
        error: Optional[str] = None,
        cache_hit: bool = False,
        input_size: Optional[int] = None,
        output_size: Optional[int] = None
    ) -> None:
        """Record metrics for a node execution"""
        if not self.current_metrics:
            return
        
        node_metric = NodeMetrics(
            node_id=node_id,
            node_type=node_type,
            status=status,
            duration_ms=duration_ms,
            start_time=start_time,
            end_time=end_time,
            retry_count=retry_count,
            error=error,
            cache_hit=cache_hit,
            input_size=input_size,
            output_size=output_size
        )
        
        self.current_metrics.node_metrics.append(node_metric)
        self.current_metrics.total_nodes_executed += 1
        
        if status == "success":
            self.current_metrics.successful_nodes += 1
        elif status == "failed":
            self.current_metrics.failed_nodes += 1
        elif status == "skipped":
            self.current_metrics.skipped_nodes += 1
        
        if cache_hit:
            self.current_metrics.cache_hits += 1
        else:
            self.current_metrics.cache_misses += 1
        
        self.current_metrics.total_retries += retry_count
    
    def end_workflow(self) -> Optional[WorkflowMetrics]:
        """End tracking and return final metrics"""
        if not self.current_metrics:
            return None
        
        self.current_metrics.end_time = time.time()
        self.current_metrics.total_duration_ms = (
            (self.current_metrics.end_time - self.current_metrics.start_time) * 1000
        )
        
        logger.info(
            f"Workflow {self.current_run_id} completed: "
            f"{self.current_metrics.total_nodes_executed} nodes, "
            f"{self.current_metrics.total_duration_ms:.2f}ms"
        )
        
        metrics = self.current_metrics
        self.current_run_id = None
        self.current_metrics = None
        
        return metrics
    
    def get_metrics(self, run_id: str) -> Optional[WorkflowMetrics]:
        """Get metrics for a specific run"""
        return self.metrics.get(run_id)
    
    def get_all_metrics(self) -> List[WorkflowMetrics]:
        """Get all collected metrics"""
        return list(self.metrics.values())
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all runs"""
        if not self.metrics:
            return {}
        
        all_metrics = list(self.metrics.values())
        total_runs = len(all_metrics)
        
        avg_duration = sum(m.total_duration_ms for m in all_metrics) / total_runs
        avg_nodes = sum(m.total_nodes_executed for m in all_metrics) / total_runs
        total_retries = sum(m.total_retries for m in all_metrics)
        total_cache_hits = sum(m.cache_hits for m in all_metrics)
        total_cache_misses = sum(m.cache_misses for m in all_metrics)
        
        cache_hit_rate = (
            total_cache_hits / (total_cache_hits + total_cache_misses)
            if (total_cache_hits + total_cache_misses) > 0 else 0
        )
        
        return {
            "total_runs": total_runs,
            "avg_duration_ms": avg_duration,
            "avg_nodes_per_run": avg_nodes,
            "total_retries": total_retries,
            "cache_hit_rate": cache_hit_rate,
            "total_cache_hits": total_cache_hits,
            "total_cache_misses": total_cache_misses
        }


# Global metrics collector instance
_global_collector = PerformanceCollector()


def get_metrics_collector() -> PerformanceCollector:
    """Get the global metrics collector"""
    return _global_collector
