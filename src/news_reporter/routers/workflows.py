# routers\workflows.py
"""Workflow API Router - Endpoints for graph workflow execution"""

from __future__ import annotations
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Query, Body, Depends
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
from ..workflows.workflow_persistence import get_workflow_persistence, WorkflowRecord, ExecutionRecord, WorkflowStatus
from ..workflows.workflow_security import get_workflow_security, Permission, Role
from ..workflows.workflow_collaboration import get_workflow_collaboration, ShareLevel
from ..workflows.workflow_notifications import get_notification_manager, NotificationType, NotificationChannel
from ..workflows.workflow_integrations import get_workflow_integrations, WebhookConfig, EventSubscription
from ..workflows.workflow_deployment import get_workflow_deployment, DeploymentStatus
from ..workflows.workflow_cost import get_workflow_cost_manager, CostType
from ..workflows.workflow_backup import get_workflow_backup_manager, BackupType
from ..workflows.workflow_debugger import get_workflow_debugger, BreakpointType
from ..workflows.workflow_governance import get_workflow_governance, PolicyType, PolicySeverity
from ..workflows.workflow_ai import get_workflow_ai
from ..workflows.workflow_documentation import get_workflow_documentation, DocumentationType
from ..workflows.workflow_marketplace import get_workflow_marketplace, MarketplaceCategory, ListingStatus
from ..workflows.workflow_patterns import get_workflow_patterns, PatternType
from ..workflows.workflow_migration import get_workflow_migration, MigrationType
from ..workflows.workflow_alerting import get_workflow_alerting, AlertSeverity, AlertType
from ..workflows.workflow_multitenant import get_workflow_multitenant, TenantTier
from ..workflows.workflow_gateway import get_workflow_gateway, RateLimitStrategy

router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"])


class WorkflowRequest(BaseModel):
    """Request model for workflow execution"""
    goal: str = Field(..., description="User goal/query")
    graph_path: Optional[str] = Field(None, description="Optional path to graph JSON file")
    workflow_definition: Optional[Dict[str, Any]] = Field(None, description="Optional workflow definition (JSON)")
    workflow_id: Optional[str] = Field(None, description="Optional workflow ID to load from persistence")
    use_graph: bool = Field(True, description="Use graph executor (True) or sequential (False)")
    checkpoint_dir: Optional[str] = Field(None, description="Directory for state checkpoints")


class WorkflowResponse(BaseModel):
    """Response model for workflow execution"""
    run_id: str
    result: str
    metrics: Optional[Dict[str, Any]] = None
    execution_time_ms: float

@router.get("/agents")
async def list_foundry_agents():
    """
    List all available workflow agents from Azure AI Foundry (API only, no hardcoded config).
    Uses the same implementation as /api/agents/all.
    """
    try:
        from ..agents.agents import list_agents_from_foundry
        try:
            agents = list_agents_from_foundry()
        except Exception as foundry_error:
            print(f"Foundry queries failed: {foundry_error}")
            # Return empty list rather than crashing
            return []
        
        print(f"Returned {len(agents)} workflow agents: {[a.get('name') for a in agents]}")
        return agents
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list workflow agents: {str(e)}")

@router.post("/execute", response_model=WorkflowResponse)
async def execute_workflow(
    request: WorkflowRequest
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
    from ..models.graph_schema import GraphDefinition
    from ..workflows.graph_normalizer import normalize_workflow_graph
    
    config = Settings.load()
    run_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Start metrics collection
    metrics_collector = get_metrics_collector()
    metrics_collector.start_workflow(run_id, request.goal)
    
    try:
        # Load workflow definition
        if request.workflow_id:
            # Load from persistence
            persistence = get_workflow_persistence()
            workflow_record = persistence.get_workflow(request.workflow_id)
            if not workflow_record:
                raise HTTPException(status_code=404, detail=f"Workflow {request.workflow_id} not found")
            workflow_def = workflow_record.graph_definition
        elif request.workflow_definition:
            workflow_def = request.workflow_definition
        elif request.graph_path:
            # Load from file
            workflow_def = load_graph_definition(request.graph_path)
        else:
            raise HTTPException(status_code=400, detail="Must provide workflow_id, workflow_definition, or graph_path")
        
        # CRITICAL: Normalize workflow to strip UI artifacts before execution
        raw_graph = GraphDefinition.model_validate(workflow_def)
        normalized_graph = normalize_workflow_graph(raw_graph)
        
        # Validate normalized graph
        errors = normalized_graph.validate()
        if errors:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid graph definition: {', '.join(errors)}"
            )
        
        # Execute workflow
        if request.use_graph:
            result = await run_graph_workflow(
                cfg=config,
                goal=request.goal,
                workflow_definition=normalized_graph.model_dump()
            )
        else:
            result = await run_sequential_goal(cfg=config, goal=request.goal)
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Get final metrics
        workflow_metrics = metrics_collector.end_workflow()
        metrics_dict = workflow_metrics.to_dict() if workflow_metrics else None
        
        # Record execution
        persistence = get_workflow_persistence()
        execution_record = ExecutionRecord(
            execution_id=run_id,  # Use run_id as unique execution identifier
            run_id=run_id,
            workflow_id=request.workflow_id or "adhoc",
            goal=request.goal,
            result=result,
            # execution_time_ms=execution_time_ms,
            status=WorkflowStatus.COMPLETED
        )
        persistence.save_execution(execution_record)
        
        return WorkflowResponse(
            run_id=run_id,
            result=result,
            metrics=metrics_dict,
            execution_time_ms=execution_time_ms
        )
        
    except HTTPException:
        metrics_collector.end_workflow() # Ensure metrics are ended even on HTTPException
        raise
    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        metrics_collector.end_workflow() # Ensure metrics are ended on general exception
        
        # Record failed execution
        persistence = get_workflow_persistence()
        execution_record = ExecutionRecord(
            execution_id=run_id,  # Use run_id as unique execution identifier
            run_id=run_id,
            workflow_id=request.workflow_id or "adhoc",
            goal=request.goal,
            result=str(e),
            # execution_time_ms=execution_time_ms,
            status=WorkflowStatus.FAILED
        )
        persistence.save_execution(execution_record)
        
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


# ========== Phase 7: Persistence, Security, Collaboration, Notifications, Integrations, Deployment ==========

@router.post("/persist")
async def save_workflow(
    workflow_id: str = Body(...),
    name: str = Body(...),
    graph_definition: Dict[str, Any] = Body(...),
    description: Optional[str] = Body(None)
) -> Dict[str, Any]:
    """Save a workflow definition"""
    persistence = get_workflow_persistence()
    workflow = WorkflowRecord(
        workflow_id=workflow_id,
        name=name,
        description=description,
        graph_definition=graph_definition
    )
    persistence.save_workflow(workflow)
    return {"workflow_id": workflow_id, "status": "saved"}


# Frontend-compatible endpoints for workflow definitions
@router.get("/definitions")
async def get_workflow_definition(
    graph_path: Optional[str] = Query(None),
    workflow_id: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """Get workflow definition by path or ID"""
    config = Settings.load()
    
    # If workflow_id is provided, try to load from persistence
    if workflow_id:
        persistence = get_workflow_persistence()
        workflow = persistence.get_workflow(workflow_id)
        if workflow:
            return workflow.graph_definition
    
    # Otherwise, load from graph_path or default
    try:
        graph_def = load_graph_definition(graph_path, config)
        return graph_def.model_dump()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Workflow definition not found: {graph_path or 'default'}")


@router.post("/definitions", status_code=201)
async def save_workflow_definition(
    request: dict
) -> dict:
    """
    Save a workflow definition.
    
    Request body:
        {
            "definition": GraphDefinition dict,
            "name": str (optional)
        }
    
    Returns: {
        "success": true,
        "path": "path/to/saved/file.json",
        "workflow_id": "unique_identifier"
    }
    """
    from ..workflows.graph_normalizer import normalize_workflow_graph
    from ..models.graph_schema import GraphDefinition
    import uuid
    
    # Parse request to extract definition and name
    if "definition" in request:
        definition_data = request["definition"]
        name = request.get("name")
    else:
        # Request is the definition itself (flat structure)
        definition_data = request
        name = request.get("name")
    
    # Parse the graph definition WITHOUT validation to allow incomplete workflows
    # Using model_construct bypasses Pydantic validation
    try:
        raw_definition = GraphDefinition.model_construct(**definition_data)
    except Exception as e:
        # If construction fails entirely, try using model_validate to get better error messages
        # but wrap it to allow saving anyway
        try:
            raw_definition = GraphDefinition.model_validate(definition_data)
        except Exception:
            # Even if validation fails, we still want to save the raw data
            raise HTTPException(status_code=400, detail=f"Invalid workflow structure: {str(e)}")
    
    # CRITICAL: Defensively normalize graph to strip UI helpers and convert to loop_continue/loop_exit
    # This may fail for incomplete graphs, so wrap in try-except
    try:
        normalized_definition = normalize_workflow_graph(raw_definition)
        graph_to_save = normalized_definition.model_dump()
    except Exception as normalize_error:
        # If normalization fails (e.g., incomplete loop clusters), save the raw definition
        print(f"Warning: Graph normalization failed, saving raw definition: {normalize_error}")
        graph_to_save = definition_data
    
    # REMOVED: Validation step - workflows can now be saved incomplete
    # errors = normalized_definition.validate()
    # if errors:
    #     raise HTTPException(status_code=400, detail=f"Validation errors: {', '.join(errors)}")
    
    # Generate workflow_id if not provided
    workflow_id = definition_data.get("workflow_id") or str(uuid.uuid4())
    
    # Save using persistence
    persistence = get_workflow_persistence()
    workflow = WorkflowRecord(
        workflow_id=workflow_id,
        name=name or definition_data.get("name") or "Untitled Workflow",
        description=definition_data.get("description"),
        graph_definition=graph_to_save
    )
    persistence.save_workflow(workflow)
    
    print(f"Saved workflow {workflow_id} ({name}) without validation")
    
    return {
        "success": True,
        "workflow_id": workflow_id,
        "path": f"workflows/{workflow_id}.json"  # Informational
    }


@router.post("/definitions/validate")
async def validate_workflow_definition(
    definition: Dict[str, Any] = Body(...)
) -> Dict[str, Any]:
    """Validate a workflow definition"""
    from ..models.graph_schema import GraphDefinition
    from ..workflows.graph_normalizer import normalize_workflow_graph

    try:
        # Parse the raw definition
        raw_definition = GraphDefinition.model_validate(definition)
        
        # CRITICAL: Normalize to strip UI helpers before validation
        normalized_definition = normalize_workflow_graph(raw_definition)
        
        # Validate the normalized graph
        errors = normalized_definition.validate()
        
        return {
            "valid": len(errors) == 0,
            "errors": errors if errors else None
        }
    except ValueError as e:
        # Normalization error (e.g., helper nodes not wired)
        return {
            "valid": False,
            "errors": [str(e)]
        }
    except Exception as e:
        return {
            "valid": False,
            "errors": [f"Invalid workflow definition: {str(e)}"]
        }


@router.get("/persist/{workflow_id}")
async def get_persisted_workflow(workflow_id: str) -> Dict[str, Any]:
    """Get a persisted workflow"""
    persistence = get_workflow_persistence()
    workflow = persistence.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow.to_dict()


@router.get("/persist")
async def list_persisted_workflows(
    tags: Optional[List[str]] = Query(None),
    is_active: Optional[bool] = Query(None, description="Filter by active status. Defaults to True (only active workflows)")
) -> List[Dict[str, Any]]:
    """List persisted workflows. By default, only returns active workflows."""
    persistence = get_workflow_persistence()
    # Default to only active workflows if not specified
    if is_active is None:
        is_active = True
    workflows = persistence.list_workflows(tags=tags, is_active=is_active)
    return [w.to_dict() for w in workflows]


@router.delete("/persist/{workflow_id}")
async def delete_persisted_workflow(workflow_id: str) -> Dict[str, Any]:
    """Delete a persisted workflow (soft delete by setting is_active=False)"""
    persistence = get_workflow_persistence()
    
    # Check if workflow exists first
    workflow = persistence.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    # Delete the workflow
    success = persistence.delete_workflow(workflow_id)
    
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to delete workflow {workflow_id}")
    
    return {"success": True, "message": f"Workflow {workflow_id} deleted successfully"}


@router.get("/active")
async def get_active_workflow() -> Dict[str, Any]:
    """Get the currently active workflow"""
    persistence = get_workflow_persistence()
    workflow = persistence.get_active_workflow()
    
    if not workflow:
        raise HTTPException(status_code=404, detail="No active workflow found")
    
    return workflow.to_dict()


@router.post("/{workflow_id}/set-active")
async def set_active_workflow(workflow_id: str) -> Dict[str, Any]:
    """Set a workflow as active (deactivates all other workflows)"""
    persistence = get_workflow_persistence()
    success = persistence.set_active_workflow(workflow_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    return {
        "workflow_id": workflow_id,
        "status": "active",
        "message": f"Workflow {workflow_id} is now active"
    }


@router.get("/executions/{execution_id}")
async def get_execution(execution_id: str) -> Dict[str, Any]:
    """Get execution record"""
    persistence = get_workflow_persistence()
    execution = persistence.get_execution(execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    return execution.to_dict()


@router.get("/executions")
async def list_executions(
    workflow_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(100)
) -> List[Dict[str, Any]]:
    """List execution records"""
    persistence = get_workflow_persistence()
    exec_status = WorkflowStatus(status) if status else None
    executions = persistence.list_executions(workflow_id=workflow_id, status=exec_status, limit=limit)
    return [e.to_dict() for e in executions]


@router.get("/executions/{run_id}/status")
async def get_execution_status(run_id: str) -> Dict[str, Any]:
    """Get execution status for a specific run"""
    # Try to get from execution monitor first (for active runs)
    monitor = get_execution_monitor()
    run_status = monitor.get_run_status(run_id)
    
    if run_status:
        # Active run - return status from monitor
        return {
            "run_id": run_id,
            "status": run_status.get("status", "running"),
            "current_node": run_status.get("current_node"),
            "progress": run_status.get("progress", 0),
            "state": run_status.get("state"),
            "goal": run_status.get("goal"),
            "duration_ms": run_status.get("duration_ms"),
            "error": run_status.get("error"),
        }
    
    # Check persistence for completed/failed runs
    persistence = get_workflow_persistence()
    execution = persistence.get_execution(run_id)
    
    if execution:
        return {
            "run_id": execution.run_id,
            "status": execution.status.value if hasattr(execution.status, 'value') else str(execution.status),
            "result": execution.result,
            "created_at": execution.created_at.isoformat() if execution.created_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
        }
    
    raise HTTPException(status_code=404, detail=f"Execution {run_id} not found")


@router.post("/security/users")
async def create_user(
    user_id: str = Body(...),
    username: str = Body(...),
    email: Optional[str] = Body(None),
    roles: Optional[List[str]] = Body(None)
) -> Dict[str, Any]:
    """Create a user"""
    security = get_workflow_security()
    role_objs = [Role(r) for r in (roles or [])]
    user = security.create_user(user_id, username, email, role_objs)
    return {"user_id": user.user_id, "username": user.username}


@router.post("/security/tokens")
async def create_token(
    user_id: str = Body(...),
    expires_in_hours: int = Body(24)
) -> Dict[str, Any]:
    """Create an access token"""
    security = get_workflow_security()
    token = security.create_token(user_id, expires_in_hours)
    return {"token": token.token, "expires_at": token.expires_at.isoformat()}


@router.post("/security/permissions/{workflow_id}")
async def grant_permission(
    workflow_id: str,
    user_id: str = Body(...),
    permission: str = Body(...)
) -> Dict[str, Any]:
    """Grant permission to a user"""
    security = get_workflow_security()
    perm = Permission(permission)
    success = security.grant_permission(user_id, workflow_id, perm)
    return {"success": success}


@router.post("/collaboration/teams")
async def create_team(
    team_id: str = Body(...),
    name: str = Body(...),
    description: Optional[str] = Body(None)
) -> Dict[str, Any]:
    """Create a team"""
    collaboration = get_workflow_collaboration()
    team = collaboration.create_team(team_id, name, description)
    return {"team_id": team.team_id, "name": team.name}


@router.post("/collaboration/share/{workflow_id}")
async def share_workflow(
    workflow_id: str,
    owner_id: str = Body(...),
    share_level: str = Body("private"),
    shared_with_users: Optional[List[str]] = Body(None)
) -> Dict[str, Any]:
    """Share a workflow"""
    collaboration = get_workflow_collaboration()
    share = collaboration.share_workflow(
        workflow_id,
        owner_id,
        ShareLevel(share_level),
        shared_with_users=shared_with_users or []
    )
    return {"workflow_id": workflow_id, "share_level": share_level}


@router.post("/notifications/rules")
async def add_notification_rule(
    rule_id: str = Body(...),
    name: str = Body(...),
    event_type: str = Body(...),
    workflow_id: Optional[str] = Body(None),
    channels: Optional[List[str]] = Body(None),
    recipients: Optional[List[str]] = Body(None)
) -> Dict[str, Any]:
    """Add a notification rule"""
    from ..workflows.workflow_notifications import NotificationRule
    manager = get_notification_manager()
    rule = NotificationRule(
        rule_id=rule_id,
        name=name,
        workflow_id=workflow_id,
        event_type=event_type,
        channels=[NotificationChannel(c) for c in (channels or [])],
        recipients=recipients or []
    )
    manager.add_rule(rule)
    return {"rule_id": rule_id, "status": "added"}


@router.post("/integrations/webhooks")
async def register_webhook(
    webhook_id: str = Body(...),
    url: str = Body(...),
    method: str = Body("POST")
) -> Dict[str, Any]:
    """Register a webhook"""
    integrations = get_workflow_integrations()
    config = WebhookConfig(webhook_id=webhook_id, url=url, method=method)
    integrations.register_webhook(config)
    return {"webhook_id": webhook_id, "status": "registered"}


@router.post("/integrations/events")
async def subscribe_event(
    subscription_id: str = Body(...),
    event_type: str = Body(...),
    workflow_id: Optional[str] = Body(None),
    webhook_url: Optional[str] = Body(None)
) -> Dict[str, Any]:
    """Subscribe to workflow events"""
    integrations = get_workflow_integrations()
    subscription = EventSubscription(
        subscription_id=subscription_id,
        event_type=event_type,
        workflow_id=workflow_id,
        webhook_url=webhook_url
    )
    integrations.subscribe_event(subscription)
    return {"subscription_id": subscription_id, "status": "subscribed"}


@router.post("/deploy/{workflow_id}")
async def deploy_workflow(
    workflow_id: str,
    target_environment: str = Body(...),
    source_environment: str = Body("dev")
) -> Dict[str, Any]:
    """Deploy a workflow to an environment"""
    deployment = get_workflow_deployment()
    deploy = deployment.deploy_workflow(workflow_id, target_environment, source_environment)
    return {
        "deployment_id": deploy.deployment_id,
        "status": deploy.status.value,
        "workflow_id": workflow_id
    }


@router.get("/deploy/history")
async def get_deployment_history(
    workflow_id: Optional[str] = Query(None),
    environment: Optional[str] = Query(None)
) -> List[Dict[str, Any]]:
    """Get deployment history"""
    deployment = get_workflow_deployment()
    deployments = deployment.get_deployment_history(workflow_id, environment)
    return [
        {
            "deployment_id": d.deployment_id,
            "workflow_id": d.workflow_id,
            "version": d.version,
            "status": d.status.value,
            "target_environment": d.target_environment,
            "created_at": d.created_at.isoformat() if d.created_at else None
        }
        for d in deployments
    ]


# ========== Phase 8: Cost Management, Backup, Debugger, Governance, AI, Documentation ==========

@router.post("/cost/record")
async def record_cost(
    workflow_id: str = Body(...),
    cost_type: str = Body(...),
    amount: Optional[float] = Body(None),
    units: float = Body(1.0)
) -> Dict[str, Any]:
    """Record a cost entry"""
    cost_manager = get_workflow_cost_manager()
    entry = cost_manager.record_cost(workflow_id, CostType(cost_type), amount=amount, units=units)
    return {"entry_id": entry.entry_id, "amount": entry.amount}


@router.post("/cost/budgets")
async def add_budget(
    budget_id: str = Body(...),
    amount: float = Body(...),
    workflow_id: Optional[str] = Body(None),
    period: str = Body("monthly")
) -> Dict[str, Any]:
    """Add a cost budget"""
    cost_manager = get_workflow_cost_manager()
    budget = cost_manager.add_budget(budget_id, amount, workflow_id, period)
    return {"budget_id": budget_id, "amount": amount}


@router.get("/cost/report")
async def get_cost_report(
    workflow_id: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """Generate a cost report"""
    from datetime import datetime
    cost_manager = get_workflow_cost_manager()
    start = datetime.fromisoformat(start_date) if start_date else None
    end = datetime.fromisoformat(end_date) if end_date else None
    report = cost_manager.generate_cost_report(workflow_id, start, end)
    return {
        "total_cost": report.total_cost,
        "cost_by_type": report.cost_by_type,
        "execution_count": report.execution_count,
        "avg_cost_per_execution": report.avg_cost_per_execution
    }


@router.post("/backup/create")
async def create_backup(
    backup_type: str = Body("full"),
    workflow_ids: Optional[List[str]] = Body(None)
) -> Dict[str, Any]:
    """Create a workflow backup"""
    from ..workflows.workflow_persistence import get_workflow_persistence
    backup_manager = get_workflow_backup_manager()
    backup_manager.set_persistence(get_workflow_persistence())
    backup = backup_manager.create_backup(backup_type=BackupType(backup_type), workflow_ids=workflow_ids)
    return {
        "backup_id": backup.backup_id,
        "backup_type": backup.backup_type.value,
        "workflow_count": len(backup.workflow_ids)
    }


@router.post("/backup/{backup_id}/restore")
async def restore_backup(
    backup_id: str,
    overwrite: bool = Body(False)
) -> Dict[str, Any]:
    """Restore a backup"""
    backup_manager = get_workflow_backup_manager()
    result = backup_manager.restore_backup(backup_id, overwrite)
    return result


@router.post("/debugger/breakpoints")
async def add_breakpoint(
    breakpoint_id: str = Body(...),
    type: str = Body(...),
    node_id: Optional[str] = Body(None),
    condition: Optional[str] = Body(None)
) -> Dict[str, Any]:
    """Add a debugger breakpoint"""
    debugger = get_workflow_debugger()
    bp = debugger.add_breakpoint(breakpoint_id, BreakpointType(type), node_id, condition)
    return {"breakpoint_id": bp.breakpoint_id, "type": bp.type.value}


@router.get("/debugger/traces")
async def get_traces(
    node_id: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    limit: int = Query(100)
) -> List[Dict[str, Any]]:
    """Get debug traces"""
    debugger = get_workflow_debugger()
    traces = debugger.get_traces(node_id, event_type, limit)
    return [
        {
            "trace_id": t.trace_id,
            "node_id": t.node_id,
            "event_type": t.event_type,
            "timestamp": t.timestamp.isoformat()
        }
        for t in traces
    ]


@router.post("/governance/validate/{workflow_id}")
async def validate_workflow_governance(workflow_id: str) -> Dict[str, Any]:
    """Validate a workflow against governance policies"""
    from ..workflows.graph_loader import load_graph_definition
    from ..config import Settings
    config = Settings.load()
    graph_def = load_graph_definition(None, config)
    governance = get_workflow_governance()
    violations = governance.validate_workflow(graph_def, workflow_id)
    return {
        "workflow_id": workflow_id,
        "violation_count": len(violations),
        "violations": [
            {
                "policy_id": v.policy_id,
                "severity": v.severity.value,
                "message": v.message
            }
            for v in violations
        ]
    }


@router.get("/governance/compliance/{workflow_id}")
async def get_compliance_report(workflow_id: str) -> Dict[str, Any]:
    """Get compliance report for a workflow"""
    from ..workflows.graph_loader import load_graph_definition
    from ..config import Settings
    config = Settings.load()
    graph_def = load_graph_definition(None, config)
    governance = get_workflow_governance()
    report = governance.get_compliance_report(workflow_id, graph_def)
    return report


@router.post("/ai/predict/{workflow_id}")
async def predict_execution(
    workflow_id: str,
    prediction_type: str = Body("execution_time")
) -> Dict[str, Any]:
    """Get AI predictions for a workflow"""
    from ..workflows.graph_loader import load_graph_definition
    from ..config import Settings
    config = Settings.load()
    graph_def = load_graph_definition(None, config)
    ai = get_workflow_ai()
    
    if prediction_type == "execution_time":
        prediction = ai.predict_execution_time(workflow_id, graph_def)
    elif prediction_type == "cost":
        prediction = ai.predict_cost(workflow_id, graph_def)
    else:
        raise HTTPException(status_code=400, detail="Invalid prediction type")
    
    return {
        "prediction_id": prediction.prediction_id,
        "prediction": prediction.prediction,
        "confidence": prediction.confidence
    }


@router.get("/ai/recommendations/{workflow_id}")
async def get_ai_recommendations(workflow_id: str) -> List[Dict[str, Any]]:
    """Get AI recommendations for a workflow"""
    from ..workflows.graph_loader import load_graph_definition
    from ..config import Settings
    config = Settings.load()
    graph_def = load_graph_definition(None, config)
    ai = get_workflow_ai()
    recommendations = ai.generate_recommendations(workflow_id, graph_def)
    return [
        {
            "recommendation_id": r.recommendation_id,
            "type": r.type,
            "description": r.description,
            "expected_improvement": r.expected_improvement,
            "confidence": r.confidence
        }
        for r in recommendations
    ]


@router.post("/documentation")
async def add_documentation(
    doc_id: str = Body(...),
    type: str = Body(...),
    title: str = Body(...),
    content: str = Body(...),
    workflow_id: Optional[str] = Body(None)
) -> Dict[str, Any]:
    """Add documentation"""
    docs = get_workflow_documentation()
    doc = docs.add_documentation(
        doc_id,
        DocumentationType(type),
        title,
        content,
        workflow_id=workflow_id
    )
    return {"doc_id": doc.doc_id, "title": doc.title}


@router.get("/documentation/{workflow_id}")
async def get_workflow_documentation(workflow_id: str) -> List[Dict[str, Any]]:
    """Get documentation for a workflow"""
    docs = get_workflow_documentation()
    documentation = docs.get_documentation(workflow_id=workflow_id)
    return [
        {
            "doc_id": d.doc_id,
            "title": d.title,
            "type": d.type.value,
            "content": d.content[:200]  # First 200 chars
        }
        for d in documentation
    ]


@router.post("/documentation/{workflow_id}/generate")
async def generate_workflow_docs(workflow_id: str) -> Dict[str, Any]:
    """Auto-generate documentation for a workflow"""
    from ..workflows.graph_loader import load_graph_definition
    from ..config import Settings
    config = Settings.load()
    graph_def = load_graph_definition(None, config)
    docs = get_workflow_documentation()
    doc = docs.generate_workflow_docs(workflow_id, graph_def)
    return {"doc_id": doc.doc_id, "title": doc.title}


# ========== Phase 9: Marketplace, Patterns, Migration, Alerting, Multi-Tenant, Gateway ==========

@router.post("/marketplace/listings")
async def create_marketplace_listing(
    listing_id: str = Body(...),
    workflow_id: str = Body(...),
    title: str = Body(...),
    description: str = Body(...),
    category: str = Body(...),
    author_id: str = Body(...),
    tags: Optional[List[str]] = Body(None),
    price: float = Body(0.0)
) -> Dict[str, Any]:
    """Create a marketplace listing"""
    marketplace = get_workflow_marketplace()
    listing = marketplace.create_listing(
        listing_id,
        workflow_id,
        title,
        description,
        MarketplaceCategory(category),
        author_id,
        tags=tags,
        price=price
    )
    return {"listing_id": listing.listing_id, "status": listing.status.value}


@router.get("/marketplace/search")
async def search_marketplace(
    query: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    min_rating: float = Query(0.0),
    limit: int = Query(20)
) -> List[Dict[str, Any]]:
    """Search marketplace listings"""
    marketplace = get_workflow_marketplace()
    cat = MarketplaceCategory(category) if category else None
    listings = marketplace.search_listings(query=query, category=cat, min_rating=min_rating, limit=limit)
    return [
        {
            "listing_id": l.listing_id,
            "title": l.title,
            "category": l.category.value,
            "rating": l.rating,
            "download_count": l.download_count,
            "price": l.price
        }
        for l in listings
    ]


@router.post("/patterns/state-machines")
async def create_state_machine(
    machine_id: str = Body(...),
    initial_state: str = Body(...)
) -> Dict[str, Any]:
    """Create a state machine"""
    patterns = get_workflow_patterns()
    from src.news_reporter.workflows.workflow_patterns import StateMachineState
    states = {
        initial_state: StateMachineState(state_id=initial_state, name=initial_state)
    }
    machine = patterns.create_state_machine(machine_id, initial_state, states)
    return {"machine_id": machine.machine_id, "current_state": machine.current_state}


@router.post("/patterns/events")
async def emit_event(
    event_type: str = Body(...),
    source: str = Body(...),
    payload: Optional[Dict[str, Any]] = Body(None)
) -> Dict[str, Any]:
    """Emit an event"""
    patterns = get_workflow_patterns()
    event = patterns.emit_event(event_type, source, payload)
    return {"event_id": event.event_id, "event_type": event.event_type}


@router.post("/migration/migrate/{workflow_id}")
async def migrate_workflow(
    workflow_id: str,
    target_version: str = Body(...)
) -> Dict[str, Any]:
    """Migrate a workflow to a target version"""
    from ..workflows.graph_loader import load_graph_definition
    from ..config import Settings
    config = Settings.load()
    graph_def = load_graph_definition(None, config)
    migration = get_workflow_migration()
    result = migration.migrate_workflow(workflow_id, graph_def, target_version)
    return {
        "migration_id": result.migration_id,
        "success": result.success,
        "changes": result.changes,
        "errors": result.errors
    }


@router.post("/alerting/rules")
async def add_alert_rule(
    rule_id: str = Body(...),
    name: str = Body(...),
    alert_type: str = Body(...),
    severity: str = Body(...),
    threshold: Optional[float] = Body(None)
) -> Dict[str, Any]:
    """Add an alert rule"""
    alerting = get_workflow_alerting()
    # Simplified - would need actual condition function
    rule = alerting.add_alert_rule(
        rule_id,
        name,
        AlertType(alert_type),
        AlertSeverity(severity),
        lambda m: m.total_duration_ms > (threshold or 0),
        threshold=threshold
    )
    return {"rule_id": rule.rule_id, "name": rule.name}


@router.get("/alerting/alerts")
async def get_alerts(
    workflow_id: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    resolved: Optional[bool] = Query(None)
) -> List[Dict[str, Any]]:
    """Get alerts"""
    alerting = get_workflow_alerting()
    sev = AlertSeverity(severity) if severity else None
    alerts = alerting.get_alerts(workflow_id=workflow_id, severity=sev, resolved=resolved)
    return [
        {
            "alert_id": a.alert_id,
            "rule_id": a.rule_id,
            "severity": a.severity.value,
            "message": a.message,
            "triggered_at": a.triggered_at.isoformat() if a.triggered_at else None
        }
        for a in alerts
    ]


@router.post("/multitenant/tenants")
async def create_tenant(
    tenant_id: str = Body(...),
    name: str = Body(...),
    tier: str = Body("free")
) -> Dict[str, Any]:
    """Create a tenant"""
    multitenant = get_workflow_multitenant()
    tenant = multitenant.create_tenant(tenant_id, name, TenantTier(tier))
    return {"tenant_id": tenant.tenant_id, "tier": tenant.tier.value}


@router.get("/multitenant/tenants/{tenant_id}/quota")
async def get_tenant_quota(tenant_id: str) -> Dict[str, Any]:
    """Get tenant quota status"""
    multitenant = get_workflow_multitenant()
    status = multitenant.get_tenant_quota_status(tenant_id)
    return status


@router.post("/gateway/api-keys")
async def create_api_key(
    key_id: str = Body(...),
    user_id: Optional[str] = Body(None),
    tenant_id: Optional[str] = Body(None)
) -> Dict[str, Any]:
    """Create an API key"""
    gateway = get_workflow_gateway()
    api_key = gateway.create_api_key(key_id, user_id=user_id, tenant_id=tenant_id)
    return {"key_id": api_key.key_id, "key_value": api_key.key_value}


@router.get("/gateway/stats")
async def get_gateway_stats(
    endpoint: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    hours: int = Query(24)
) -> Dict[str, Any]:
    """Get API gateway request statistics"""
    gateway = get_workflow_gateway()
    stats = gateway.get_request_stats(endpoint=endpoint, user_id=user_id, hours=hours)
    return stats
