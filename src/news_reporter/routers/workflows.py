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
    is_active: Optional[bool] = Query(None)
) -> List[Dict[str, Any]]:
    """List persisted workflows"""
    persistence = get_workflow_persistence()
    workflows = persistence.list_workflows(tags=tags, is_active=is_active)
    return [w.to_dict() for w in workflows]


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
