#!/usr/bin/env python3
"""
Executor Smoke Tests: Test actual executor graph traversal and node execution
"""
import sys
from pathlib import Path

# Add parent directory to path (pytest compatibility)
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock

from src.news_reporter.workflows.executor import WorkflowExecutor
import src.news_reporter.workflows.executor as executor_mod  # FIX B: For patch.object
from src.news_reporter.models.workflow import (
    Workflow, WorkflowRun, WorkflowGraph, NodeResult
)
from src.news_reporter.config import Settings


# ===== Helper Functions for Robust Mock Call Extraction =====

def get_node_id(call):
    """Extract nodeId from persist_node_result call (supports args and kwargs)"""
    if len(call.args) >= 2:
        return call.args[1]  # node_id is second positional arg
    return call.kwargs.get('node_id')


def get_node_result(call):
    """Extract NodeResult from persist_node_result call (supports args and kwargs)"""
    if len(call.args) >= 3:
        return call.args[2]  # NodeResult is third arg
    return call.kwargs.get('node_result')


# ===== Fix 3: Executor Smoke Tests =====

@pytest.mark.asyncio
async def test_sequential_traverse():
    """Test that simple sequential graph executes both nodes (Fix 2 verification)"""
    # Create simple workflow: start → send
    workflow = Workflow(
        id="test-wf",
        userId="test-user",
        name="Sequential Test",
        validationStatus="valid",
        graph=WorkflowGraph(
            nodes=[
                {"id": "start", "type": "StartNode"},
                {"id": "send", "type": "SendMessage", "data": {"message": "Hello"}},
            ],
            edges=[
                {"source": "start", "target": "send"}
            ]
        )
    )
    
    run = WorkflowRun(
        id="test-run",
        workflowId="test-wf",
        userId="test-user",
        status="queued"
    )
    
    # Mock repository with stateful update_run_status
    mock_repo = AsyncMock()
    
    # FIX 3: Make update_run_status mutate the run object (synchronous for reliability)
    def update_status(run_id, status, **kwargs):
        run.status = status
        if "error" in kwargs:
            run.error = kwargs["error"]
        if "startedAt" in kwargs:
            run.startedAt = kwargs["startedAt"]
        if "completedAt" in kwargs:
            run.completedAt = kwargs["completedAt"]
        if "heartbeatAt" in kwargs:
            run.heartbeatAt = kwargs["heartbeatAt"]
    
    mock_repo.update_run_status = AsyncMock(side_effect=update_status)
    mock_repo.update_run_heartbeat = AsyncMock()
    mock_repo.persist_node_result = AsyncMock()
    
    # get_run should return updated run object
    async def mock_get_run(run_id, user_id):
        return run
    mock_repo.get_run = AsyncMock(side_effect=mock_get_run)
    
    # Mock config
    mock_cfg = MagicMock(spec=Settings)
    
    # Execute (FIX 1: method renamed from run() to execute())
    executor = WorkflowExecutor(workflow, run, mock_repo, mock_cfg)
    result_run = await executor.execute()
    
    # Assertions
    # Both nodes should have results persisted
    assert mock_repo.persist_node_result.call_count == 2
    
    # FIX 2: Check that both nodes were executed (using helper)
    persisted_node_ids = {get_node_id(call) for call in mock_repo.persist_node_result.call_args_list}
    assert "start" in persisted_node_ids
    assert "send" in persisted_node_ids
    
    # Run should succeed
    assert result_run.status == "succeeded"


@pytest.mark.asyncio
async def test_fanin_enqueue():
    """Test that fan-in graph respects gating and executes all nodes (Fix 2 verification)"""
    # Create diamond graph: start→b_send, start→c_send, b_send→d_invoke, c_send→d_invoke
    workflow = Workflow(
        id="test-wf",
        userId="test-user",
        name="Fan-in Test",
        validationStatus="valid",
        graph=WorkflowGraph(
            nodes=[
                {"id": "start", "type": "StartNode"},
                {"id": "b_send", "type": "SendMessage", "data": {"message": "B"}},
                {"id": "c_send", "type": "SendMessage", "data": {"message": "C"}},
                {"id": "d_invoke", "type": "InvokeAgent", "data": {"agentId": "TRIAGE"}},
            ],
            edges=[
                {"source": "start", "target": "b_send"},
                {"source": "start", "target": "c_send"},
                {"source": "b_send", "target": "d_invoke"},
                {"source": "c_send", "target": "d_invoke"},
            ]
        )
    )
    
    run = WorkflowRun(
        id="test-run",
        workflowId="test-wf",
        userId="test-user",
        status="queued"
    )
    
    # Mock repository with stateful update_run_status
    mock_repo = AsyncMock()
    
    # FIX 3: Make update_run_status mutate the run object (synchronous for reliability)
    def update_status(run_id, status, **kwargs):
        run.status = status
        if "error" in kwargs:
            run.error = kwargs["error"]
        if "startedAt" in kwargs:
            run.startedAt = kwargs["startedAt"]
        if "completedAt" in kwargs:
            run.completedAt = kwargs["completedAt"]
        if "heartbeatAt" in kwargs:
            run.heartbeatAt = kwargs["heartbeatAt"]
    
    mock_repo.update_run_status = AsyncMock(side_effect=update_status)
    mock_repo.update_run_heartbeat = AsyncMock()
    mock_repo.persist_node_result = AsyncMock()
    
    # get_run should return updated run object
    async def mock_get_run(run_id, user_id):
        return run
    mock_repo.get_run = AsyncMock(side_effect=mock_get_run)
    
    # Mock config
    mock_cfg = MagicMock(spec=Settings)
    
    # FIX B: Mock invoke_agent using patch.object
    with patch.object(executor_mod, "invoke_agent", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = "Agent output"
        
        # Execute (FIX 1: method renamed from run() to execute())
        executor = WorkflowExecutor(workflow, run, mock_repo, mock_cfg)
        result_run = await executor.execute()
        
        # Assertions
        # All 4 nodes should have results persisted
        assert mock_repo.persist_node_result.call_count == 4
        
        # FIX 2: Check that all nodes were executed (using helper)
        persisted_node_ids = {get_node_id(call) for call in mock_repo.persist_node_result.call_args_list}
        assert "start" in persisted_node_ids
        assert "b_send" in persisted_node_ids
        assert "c_send" in persisted_node_ids
        assert "d_invoke" in persisted_node_ids
        
        # Invoke agent should have been called
        assert mock_invoke.called
        
        # Run should succeed
        assert result_run.status == "succeeded"


@pytest.mark.asyncio
async def test_node_failure_propagation():
    """Test that node failures are properly propagated to run (Fix 1 verification)"""
    # Create simple workflow with InvokeAgent that will fail
    workflow = Workflow(
        id="test-wf",
        userId="test-user",
        name="Failure Test",
        validationStatus="valid",
        graph=WorkflowGraph(
            nodes=[
                {"id": "start", "type": "StartNode"},
                {"id": "invoke_fail", "type": "InvokeAgent", "data": {"agentId": "TRIAGE"}},
            ],
            edges=[
                {"source": "start", "target": "invoke_fail"}
            ]
        )
    )
    
    run = WorkflowRun(
        id="test-run",
        workflowId="test-wf",
        userId="test-user",
        status="queued"
    )
    
    # Mock repository with stateful update_run_status
    mock_repo = AsyncMock()
    
    # FIX 3: Make update_run_status mutate the run object (synchronous for reliability)
    def update_status(run_id, status, **kwargs):
        run.status = status
        if "error" in kwargs:
            run.error = kwargs["error"]
        if "startedAt" in kwargs:
            run.startedAt = kwargs["startedAt"]
        if "completedAt" in kwargs:
            run.completedAt = kwargs["completedAt"]
        if "heartbeatAt" in kwargs:
            run.heartbeatAt = kwargs["heartbeatAt"]
    
    mock_repo.update_run_status = AsyncMock(side_effect=update_status)
    mock_repo.update_run_heartbeat = AsyncMock()
    mock_repo.persist_node_result = AsyncMock()
    
    # get_run should return updated run object
    async def mock_get_run(run_id, user_id):
        return run
    mock_repo.get_run = AsyncMock(side_effect=mock_get_run)
    
    # Mock config
    mock_cfg = MagicMock(spec=Settings)
    
    # FIX B: Mock invoke_agent to raise exception using patch.object
    with patch.object(executor_mod, "invoke_agent", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.side_effect = RuntimeError("Agent crashed!")
        
        # Execute (FIX 1: method renamed from run() to execute())
        executor = WorkflowExecutor(workflow, run, mock_repo, mock_cfg)
        result_run = await executor.execute()
        
        # Assertions
        # Failed node result should be persisted
        assert mock_repo.persist_node_result.call_count >= 1
        
        # FIX 2: Find the invoke_fail node result (using helper)
        invoke_fail_call = None
        for call in mock_repo.persist_node_result.call_args_list:
            if get_node_id(call) == "invoke_fail":
                invoke_fail_call = call
                break
        
        assert invoke_fail_call is not None, "invoke_fail node result should be persisted"
        
        # FIX 2: Check that node result has error (using helper)
        node_result = get_node_result(invoke_fail_call)
        assert node_result is not None, "NodeResult should be in call args/kwargs"
        assert node_result.status == "failed"
        assert node_result.error is not None
        assert "Agent crashed!" in node_result.error.message
        
        # Run should be marked failed
        assert result_run.status == "failed"
        
        # FIX 1: Run error should include original exception message (not NameError)
        assert result_run.error is not None
        assert "Agent crashed!" in result_run.error or "invoke_fail" in result_run.error


@pytest.mark.asyncio
async def test_missing_agentid_fails_cleanly():
    """Test that missing agentId fails with clear error (Fix 2 from hardening patch)"""
    workflow = Workflow(
        id="test-wf",
        userId="test-user",
        name="Missing AgentId Test",
        validationStatus="valid",
        graph=WorkflowGraph(
            nodes=[
                {"id": "start", "type": "StartNode"},
                {"id": "invoke_no_agent", "type": "InvokeAgent", "data": {}},  # No agentId!
            ],
            edges=[
                {"source": "start", "target": "invoke_no_agent"}
            ]
        )
    )
    
    run = WorkflowRun(
        id="test-run",
        workflowId="test-wf",
        userId="test-user",
        status="queued"
    )
    
    # Mock repository with stateful update_run_status
    mock_repo = AsyncMock()
    
    # FIX 3: Make update_run_status mutate the run object (synchronous for reliability)
    def update_status(run_id, status, **kwargs):
        run.status = status
        if "error" in kwargs:
            run.error = kwargs["error"]
        if "startedAt" in kwargs:
            run.startedAt = kwargs["startedAt"]
        if "completedAt" in kwargs:
            run.completedAt = kwargs["completedAt"]
        if "heartbeatAt" in kwargs:
            run.heartbeatAt = kwargs["heartbeatAt"]
    
    mock_repo.update_run_status = AsyncMock(side_effect=update_status)
    mock_repo.update_run_heartbeat = AsyncMock()
    mock_repo.persist_node_result = AsyncMock()
    
    # get_run should return updated run object
    async def mock_get_run(run_id, user_id):
        return run
    mock_repo.get_run = AsyncMock(side_effect=mock_get_run)
    
    # Mock config
    mock_cfg = MagicMock(spec=Settings)
    
    # Execute (FIX 1: method renamed from run() to execute())
    executor = WorkflowExecutor(workflow, run, mock_repo, mock_cfg)
    result_run = await executor.execute()
    
    # Assertions
    assert result_run.status == "failed"
    # FIX C: Strengthen assertion to match canonical message
    assert result_run.error is not None
    assert "missing required config.agentid" in result_run.error.lower()
