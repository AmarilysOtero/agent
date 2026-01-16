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


# Phase 6: Workflow optimization, scheduling, analytics, testing, composition

@router.get("/analyze/{workflow_id}")
async def analyze_workflow(workflow_id: str) -> Dict[str, Any]:
    """Analyze workflow for optimization opportunities"""
    config = Settings.load()
    graph_def = load_graph_definition(None, config)
    
    optimizer = WorkflowOptimizer(graph_def)
    analysis = optimizer.analyze()
    
    return analysis.to_dict()


@router.get("/schedules")
async def list_schedules() -> Dict[str, Any]:
    """List all scheduled workflows"""
    scheduler = get_workflow_scheduler()
    return {"schedules": scheduler.list_schedules()}


@router.post("/schedules")
async def create_schedule(
    schedule_id: str = Body(...),
    workflow_id: str = Body(...),
    schedule_type: str = Body(...),
    interval_seconds: Optional[float] = Body(None),
    time_of_day: Optional[str] = Body(None),
    days_of_week: Optional[List[int]] = Body(None)
) -> Dict[str, Any]:
    """Create a new workflow schedule"""
    scheduler = get_workflow_scheduler()
    
    schedule_type_enum = ScheduleType(schedule_type)
    config = scheduler.add_schedule(
        schedule_id=schedule_id,
        workflow_id=workflow_id,
        schedule_type=schedule_type_enum,
        interval_seconds=interval_seconds,
        time_of_day=time_of_day,
        days_of_week=days_of_week
    )
    
    return {
        "schedule_id": config.schedule_id,
        "workflow_id": config.workflow_id,
        "schedule_type": config.schedule_type.value,
        "next_run": config.next_run
    }


@router.post("/schedules/{schedule_id}/enable")
async def enable_schedule(schedule_id: str) -> Dict[str, str]:
    """Enable a schedule"""
    scheduler = get_workflow_scheduler()
    success = scheduler.enable_schedule(schedule_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Schedule {schedule_id} not found")
    
    return {"status": "enabled"}


@router.post("/schedules/{schedule_id}/disable")
async def disable_schedule(schedule_id: str) -> Dict[str, str]:
    """Disable a schedule"""
    scheduler = get_workflow_scheduler()
    success = scheduler.disable_schedule(schedule_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Schedule {schedule_id} not found")
    
    return {"status": "disabled"}


@router.get("/analytics/{workflow_id}")
async def get_workflow_analytics(
    workflow_id: str,
    time_range_days: Optional[int] = Query(30)
) -> Dict[str, Any]:
    """Get analytics for a workflow"""
    analytics_engine = get_analytics_engine()
    analytics = analytics_engine.analyze_workflow(workflow_id, time_range_days)
    
    return analytics.to_dict()


@router.post("/tests/{workflow_id}/add")
async def add_test_case(
    workflow_id: str,
    test_id: str = Body(...),
    name: str = Body(...),
    goal: str = Body(...),
    expected_output: Optional[str] = Body(None)
) -> Dict[str, Any]:
    """Add a test case for a workflow"""
    config = Settings.load()
    graph_def = load_graph_definition(None, config)
    
    tester = WorkflowTester(graph_def)
    test_case = TestCase(
        test_id=test_id,
        name=name,
        description=name,
        goal=goal,
        expected_output=expected_output
    )
    tester.add_test_case(test_case)
    
    return {"test_id": test_id, "status": "added"}


@router.post("/tests/{workflow_id}/run")
async def run_workflow_tests(workflow_id: str) -> Dict[str, Any]:
    """Run all tests for a workflow"""
    config = Settings.load()
    graph_def = load_graph_definition(None, config)
    
    tester = WorkflowTester(graph_def)
    
    async def execute_workflow(goal: str):
        return await run_graph_workflow(config, goal)
    
    results = await tester.run_all_tests(execute_workflow)
    summary = tester.get_test_summary(results)
    
    return {
        "summary": summary,
        "results": {test_id: r.to_dict() for test_id, r in results.items()}
    }


@router.post("/compose")
async def compose_workflows(
    workflow_ids: List[str] = Body(...),
    strategy: str = Body("sequential")
) -> Dict[str, Any]:
    """Compose multiple workflows into one"""
    config = Settings.load()
    
    # Load all workflows (simplified - would load from storage in production)
    workflows = []
    for _ in workflow_ids:
        graph_def = load_graph_definition(None, config)
        workflows.append(graph_def)
    
    composed = WorkflowComposer.compose(workflows, strategy)
    
    return {
        "composed_workflow": composed.model_dump(),
        "strategy": strategy,
        "source_workflows": workflow_ids
    }
