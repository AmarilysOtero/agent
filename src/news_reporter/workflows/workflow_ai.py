"""Workflow AI - AI/ML features for auto-optimization and intelligent features"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..models.graph_schema import GraphDefinition
from .performance_metrics import WorkflowMetrics

logger = logging.getLogger(__name__)


class AITaskType(str, Enum):
    """Types of AI tasks"""
    OPTIMIZATION = "optimization"  # Auto-optimize workflow
    PREDICTION = "prediction"  # Predict execution time/cost
    RECOMMENDATION = "recommendation"  # Recommend improvements
    ANOMALY_DETECTION = "anomaly_detection"  # Detect anomalies


@dataclass
class AIPrediction:
    """An AI prediction"""
    prediction_id: str
    task_type: AITaskType
    workflow_id: str
    prediction: Dict[str, Any]  # Prediction results
    confidence: float = 0.0  # 0.0 to 1.0
    model_version: str = "1.0"
    created_at: Optional[datetime] = None


@dataclass
class AIRecommendation:
    """An AI recommendation"""
    recommendation_id: str
    workflow_id: str
    type: str  # "optimize", "cache", "parallelize", etc.
    description: str
    expected_improvement: float  # Percentage improvement
    confidence: float = 0.0
    created_at: Optional[datetime] = None


class WorkflowAI:
    """AI/ML features for workflow optimization and intelligence"""
    
    def __init__(self):
        self.predictions: List[AIPrediction] = []
        self.recommendations: List[AIRecommendation] = []
        self._prediction_counter = 0
        self._recommendation_counter = 0
    
    def predict_execution_time(
        self,
        workflow_id: str,
        workflow: GraphDefinition,
        historical_metrics: Optional[List[WorkflowMetrics]] = None
    ) -> AIPrediction:
        """Predict execution time for a workflow"""
        # Simplified prediction based on node count and historical data
        # Exclude start node from count (start is just entry point, not executable)
        node_count = len([n for n in workflow.nodes if n.type != 'start'])
        base_time = node_count * 2.0  # 2 seconds per node (simplified)
        
        # Adjust based on historical data if available
        if historical_metrics:
            avg_time = sum(m.total_duration_ms for m in historical_metrics) / len(historical_metrics) / 1000
            predicted_time = (base_time + avg_time) / 2
            confidence = 0.8
        else:
            predicted_time = base_time
            confidence = 0.5
        
        prediction = AIPrediction(
            prediction_id=f"pred_{self._prediction_counter}",
            task_type=AITaskType.PREDICTION,
            workflow_id=workflow_id,
            prediction={
                "predicted_duration_seconds": predicted_time,
                "node_count": node_count,
                "complexity_score": self._calculate_complexity(workflow)
            },
            confidence=confidence,
            created_at=datetime.now()
        )
        
        self._prediction_counter += 1
        self.predictions.append(prediction)
        
        return prediction
    
    def predict_cost(
        self,
        workflow_id: str,
        workflow: GraphDefinition,
        historical_metrics: Optional[List[WorkflowMetrics]] = None
    ) -> AIPrediction:
        """Predict cost for a workflow execution"""
        # Simplified cost prediction
        # Exclude start node from count (start is just entry point, not executable)
        node_count = len([n for n in workflow.nodes if n.type != 'start'])
        base_cost = node_count * 0.01  # $0.01 per node (simplified)
        
        if historical_metrics:
            # Would use actual cost data if available
            predicted_cost = base_cost
            confidence = 0.7
        else:
            predicted_cost = base_cost
            confidence = 0.5
        
        prediction = AIPrediction(
            prediction_id=f"pred_{self._prediction_counter}",
            task_type=AITaskType.PREDICTION,
            workflow_id=workflow_id,
            prediction={
                "predicted_cost_usd": predicted_cost,
                "node_count": node_count
            },
            confidence=confidence,
            created_at=datetime.now()
        )
        
        self._prediction_counter += 1
        self.predictions.append(prediction)
        
        return prediction
    
    def generate_recommendations(
        self,
        workflow_id: str,
        workflow: GraphDefinition,
        metrics: Optional[WorkflowMetrics] = None
    ) -> List[AIRecommendation]:
        """Generate AI recommendations for workflow improvement"""
        recommendations = []
        
        # Check for optimization opportunities
        # Exclude start node from count (start is just entry point, not executable)
        node_count = len([n for n in workflow.nodes if n.type != 'start'])
        if node_count > 20:
            recommendations.append(AIRecommendation(
                recommendation_id=f"rec_{self._recommendation_counter}",
                workflow_id=workflow_id,
                type="optimize",
                description=f"Workflow has {node_count} nodes. Consider breaking into smaller workflows.",
                expected_improvement=15.0,
                confidence=0.7,
                created_at=datetime.now()
            ))
            self._recommendation_counter += 1
        
        # Check for parallelization opportunities
        # Simplified - would analyze graph structure
        if node_count > 5:
            recommendations.append(AIRecommendation(
                recommendation_id=f"rec_{self._recommendation_counter}",
                workflow_id=workflow_id,
                type="parallelize",
                description="Consider parallelizing independent nodes to reduce execution time.",
                expected_improvement=30.0,
                confidence=0.6,
                created_at=datetime.now()
            ))
            self._recommendation_counter += 1
        
        # Check for caching opportunities
        if metrics and metrics.cache_hits == 0 and metrics.cache_misses > 0:
            recommendations.append(AIRecommendation(
                recommendation_id=f"rec_{self._recommendation_counter}",
                workflow_id=workflow_id,
                type="cache",
                description="Enable caching for repeated operations to improve performance.",
                expected_improvement=40.0,
                confidence=0.8,
                created_at=datetime.now()
            ))
            self._recommendation_counter += 1
        
        self.recommendations.extend(recommendations)
        return recommendations
    
    def detect_anomalies(
        self,
        workflow_id: str,
        metrics: WorkflowMetrics,
        historical_metrics: List[WorkflowMetrics]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in workflow execution"""
        anomalies = []
        
        if not historical_metrics:
            return anomalies
        
        # Calculate averages
        avg_duration = sum(m.total_duration_ms for m in historical_metrics) / len(historical_metrics)
        avg_failed_nodes = sum(m.failed_nodes for m in historical_metrics) / len(historical_metrics)
        
        # Check for duration anomaly
        if metrics.total_duration_ms > avg_duration * 2:
            anomalies.append({
                "type": "duration_anomaly",
                "message": f"Execution time ({metrics.total_duration_ms}ms) is significantly higher than average ({avg_duration}ms)",
                "severity": "high"
            })
        
        # Check for failure anomaly
        if metrics.failed_nodes > avg_failed_nodes * 2:
            anomalies.append({
                "type": "failure_anomaly",
                "message": f"Failed nodes ({metrics.failed_nodes}) is significantly higher than average ({avg_failed_nodes})",
                "severity": "high"
            })
        
        return anomalies
    
    def _calculate_complexity(self, workflow: GraphDefinition) -> float:
        """Calculate workflow complexity score"""
        # Exclude start node from complexity calculation
        node_count = len([n for n in workflow.nodes if n.type != 'start'])
        edge_count = len(workflow.edges)
        
        # Simple complexity metric
        complexity = (node_count * 1.0) + (edge_count * 0.5)
        return complexity
    
    def get_recommendations(
        self,
        workflow_id: Optional[str] = None,
        limit: int = 10
    ) -> List[AIRecommendation]:
        """Get recommendations with optional filtering"""
        recommendations = list(self.recommendations)
        
        if workflow_id:
            recommendations = [r for r in recommendations if r.workflow_id == workflow_id]
        
        # Sort by expected improvement descending
        recommendations.sort(key=lambda r: r.expected_improvement, reverse=True)
        
        return recommendations[:limit]


# Global AI instance
_global_ai = WorkflowAI()


def get_workflow_ai() -> WorkflowAI:
    """Get the global workflow AI instance"""
    return _global_ai
