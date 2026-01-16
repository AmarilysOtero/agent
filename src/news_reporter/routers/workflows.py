"""Workflow API Router - Endpoints for graph workflow execution"""

from __future__ import annotations
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field

from ..config import Settings
from ..workflows.workflow_factory import run_graph_workflow, run_sequential_goal
from ..workflows.performance_metrics import get_metrics_collector
from ..workflows.state_checkpoint import StateCheckpoint
from ..workflows.workflow_visualizer import WorkflowVisualizer
from ..workflows.workflow_versioning import WorkflowVersionManager
from ..workflows.execution_monitor import get_execution_monitor, EventStream
from ..workflows.workflow_templates import get_template_registry
from ..workflows.graph_loader import load_graph_definition

router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"])


class WorkflowRequest(BaseModel):
    """Request model for workflow execution"""
    goal: str = Field(..., description="User goal/query")
    graph_path: Optional[str] = Field(None, description="Optional path to graph JSON file")
    use_graph: bool = Field(True, description="Use graph executor (True) or sequential (False)")
    checkpoint_dir: Optional[str] = Field(None, description="Directory for state checkpoints")


class WorkflowResponse(BaseModel):
    """Response model for workflow execution"""
    run_id: str
    result: str
    metrics: Optional[Dict[str, Any]] = None
    execution_time_ms: float


@router.post("/execute", response_model=WorkflowResponse)
async def execute_workflow(
    request: WorkflowRequest,
    config: Settings = None
) -> WorkflowResponse:
    """
    Execute a workflow with the given goal.
    
    Args:
        request: Workflow execution request
    
    Returns:
        Workflow execution result with metrics
    """
    import time
    import uuid
    
    config = Settings.load()
    
    run_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Setup checkpointing if requested
    if request.checkpoint_dir:
        config.checkpoint_dir = request.checkpoint_dir
    
    # Start metrics collection
    metrics_collector = get_metrics_collector()
    metrics_collector.start_workflow(run_id, request.goal)
    
    try:
        # Execute workflow
        if request.use_graph:
            result = await run_graph_workflow(
                cfg=config,
                goal=request.goal,
                graph_path=request.graph_path
            )
        else:
            result = await run_sequential_goal(cfg=config, goal=request.goal)
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Get final metrics
        workflow_metrics = metrics_collector.end_workflow()
        metrics_dict = workflow_metrics.to_dict() if workflow_metrics else None
        
        return WorkflowResponse(
            run_id=run_id,
            result=result,
            metrics=metrics_dict,
            execution_time_ms=execution_time_ms
        )
    
    except Exception as e:
        metrics_collector.end_workflow()
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")


@router.get("/metrics/{run_id}")
async def get_workflow_metrics(run_id: str) -> Dict[str, Any]:
    """Get metrics for a specific workflow run"""
    metrics_collector = get_metrics_collector()
    metrics = metrics_collector.get_metrics(run_id)
    
    if not metrics:
        raise HTTPException(status_code=404, detail=f"Metrics not found for run {run_id}")
    
    return metrics.to_dict()


@router.get("/metrics")
async def get_all_metrics() -> Dict[str, Any]:
    """Get summary statistics for all workflow runs"""
    metrics_collector = get_metrics_collector()
    return {
        "summary": metrics_collector.get_summary_stats(),
        "runs": [m.to_dict() for m in metrics_collector.get_all_metrics()]
    }


@router.get("/checkpoints")
async def list_checkpoints(checkpoint_dir: Optional[str] = Query(None)) -> Dict[str, Any]:
    """List available checkpoints"""
    if not checkpoint_dir:
        raise HTTPException(status_code=400, detail="checkpoint_dir query parameter required")
    
    checkpoint_manager = StateCheckpoint(checkpoint_dir)
    checkpoints = checkpoint_manager.list_checkpoints()
    
    return {
        "checkpoint_dir": checkpoint_dir,
        "checkpoints": checkpoints,
        "count": len(checkpoints)
    }


@router.post("/checkpoints/{run_id}/restore")
async def restore_checkpoint(
    run_id: str,
    checkpoint_dir: Optional[str] = Body(None)
) -> Dict[str, Any]:
    """Restore workflow state from checkpoint"""
    if not checkpoint_dir:
        raise HTTPException(status_code=400, detail="checkpoint_dir required in request body")
    
    checkpoint_manager = StateCheckpoint(checkpoint_dir)
    state = checkpoint_manager.restore_state(run_id)
    
    if not state:
        raise HTTPException(status_code=404, detail=f"Checkpoint not found for run {run_id}")
    
    return {
        "run_id": run_id,
        "state": state.model_dump(),
        "restored": True
    }


@router.get("/cache/stats")
async def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    from ..workflows.cache_manager import get_cache_manager
    
    cache_manager = get_cache_manager()
    return cache_manager.get_stats()


@router.post("/cache/clear")
async def clear_cache() -> Dict[str, str]:
    """Clear the cache"""
    from ..workflows.cache_manager import get_cache_manager
    
    cache_manager = get_cache_manager()
    cache_manager.clear()
    
    return {"status": "cache_cleared"}


# Phase 5: Workflow visualization and management endpoints

@router.get("/visualize/{workflow_id}")
async def visualize_workflow(workflow_id: str, format: str = Query("mermaid", regex="^(dot|mermaid|json)$")) -> Dict[str, Any]:
    """Visualize a workflow in various formats"""
    # Load workflow (in real implementation, would load from storage)
    # For now, use default workflow
    config = Settings.load()
    graph_def = load_graph_definition(None, config)
    
    visualizer = WorkflowVisualizer(graph_def)
    
    if format == "dot":
        return {"format": "dot", "content": visualizer.to_dot()}
    elif format == "mermaid":
        return {"format": "mermaid", "content": visualizer.to_mermaid()}
    else:  # json
        return {"format": "json", "graph": visualizer.to_json_graph()}


@router.get("/summary/{workflow_id}")
async def get_workflow_summary(workflow_id: str) -> Dict[str, Any]:
    """Get workflow summary"""
    config = Settings.load()
    graph_def = load_graph_definition(None, config)
    
    visualizer = WorkflowVisualizer(graph_def)
    return visualizer.to_summary()


@router.get("/templates")
async def list_templates() -> Dict[str, Any]:
    """List all available workflow templates"""
    registry = get_template_registry()
    return {"templates": registry.list_all()}


@router.post("/templates/{template_id}/instantiate")
async def instantiate_template(
    template_id: str,
    parameters: Dict[str, Any] = Body(...)
) -> Dict[str, Any]:
    """Instantiate a workflow template"""
    registry = get_template_registry()
    template = registry.get(template_id)
    
    if not template:
        raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
    
    graph_def = template.instantiate(**parameters)
    return {
        "template_id": template_id,
        "graph": graph_def.model_dump()
    }


@router.get("/monitor/stream/{run_id}")
async def stream_execution_events(run_id: str):
    """Stream execution events for a workflow run (Server-Sent Events)"""
    from fastapi.responses import StreamingResponse
    import json
    
    monitor = get_execution_monitor()
    
    async def event_generator():
        async with EventStream(monitor, run_id) as stream:
            async for event in stream.stream():
                if event:
                    yield f"data: {json.dumps(event.to_dict())}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@router.get("/monitor/runs")
async def list_active_runs() -> Dict[str, Any]:
    """List all active workflow runs"""
    monitor = get_execution_monitor()
    return {"active_runs": monitor.get_active_runs()}


@router.get("/monitor/runs/{run_id}/events")
async def get_run_events(run_id: str) -> Dict[str, Any]:
    """Get event history for a workflow run"""
    monitor = get_execution_monitor()
    events = monitor.get_event_history(run_id)
    return {
        "run_id": run_id,
        "events": [e.to_dict() for e in events],
        "count": len(events)
    }


@router.get("/versions/{workflow_id}")
async def list_workflow_versions(workflow_id: str) -> Dict[str, Any]:
    """List all versions of a workflow"""
    version_manager = WorkflowVersionManager()
    versions = version_manager.list_versions(workflow_id)
    return {"workflow_id": workflow_id, "versions": versions}


@router.get("/versions/{workflow_id}/{version}")
async def get_workflow_version(workflow_id: str, version: str) -> Dict[str, Any]:
    """Get a specific workflow version"""
    version_manager = WorkflowVersionManager()
    graph_def = version_manager.load_version(workflow_id, version)
    
    if not graph_def:
        raise HTTPException(status_code=404, detail=f"Version {version} not found for workflow {workflow_id}")
    
    return {"workflow_id": workflow_id, "version": version, "graph": graph_def.model_dump()}


@router.get("/versions/{workflow_id}/compare")
async def compare_workflow_versions(
    workflow_id: str,
    version1: str = Query(...),
    version2: str = Query(...)
) -> Dict[str, Any]:
    """Compare two workflow versions"""
    version_manager = WorkflowVersionManager()
    differences = version_manager.compare_versions(workflow_id, version1, version2)
    return {
        "workflow_id": workflow_id,
        "version1": version1,
        "version2": version2,
        "differences": differences
    }
