#!/usr/bin/env python3
"""
PR 3 Unit Tests: Workflow Executor Core
Tests sequential execution, fan-in gating, and deterministic ordering.
"""
import sys
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.news_reporter.workflows.executor import WorkflowExecutor, truncate_output
from src.news_reporter.models.workflow import Workflow, WorkflowRun, WorkflowGraph, NodeResult
from src.news_reporter.config import Settings


class MockRepository:
    """Mock repository for testing without real Mongo"""
    def __init__(self):
        self.status_updates = []
        self.heartbeat_updates = []
        self.node_results = {}
    
    async def update_run_status(self, run_id, status, **extra):
        self.status_updates.append({"run_id": run_id, "status": status, **extra})
    
    async def update_run_heartbeat(self, run_id):
        self.heartbeat_updates.append(run_id)
    
    async def persist_node_result(self, run_id, node_id, result):
        if run_id not in self.node_results:
            self.node_results[run_id] = {}
        self.node_results[run_id][node_id] = result
    
    async def get_run(self, run_id, user_id):
        # Return a minimal run object for reload
        return WorkflowRun(
            id=run_id,
            workflowId="test_workflow",
            userId=user_id,
            status="succeeded"
        )


def test_truncate_output_string():
    """Test truncate_output with plain string"""
    output, truncated, preview = truncate_output("Hello World")
    assert output == "Hello World"
    assert truncated == False
    assert preview is None


def test_truncate_output_dict():
    """Test truncate_output with dict (JSON serialization)"""
    data = {"key": "value", "number": 42}
    output, truncated, preview = truncate_output(data)
    assert output == '{"key": "value", "number": 42}'
    assert truncated == False


def test_truncate_output_large():
    """Test truncate_output with large string"""
    large_str = "x" * (20 * 1024)  # 20KB
    output, truncated, preview = truncate_output(large_str)
    assert truncated == True
    assert len(preview) == 500
    assert len(output) < len(large_str)


async def test_deterministic_order():
    """Test that nodes are executed in deterministic lexicographic order"""
    # Diamond graph: start -> b, start -> c, b & c -> d
    # B and C should execute in alphabetical order (b then c)
    workflow = Workflow(
        id="test_workflow",
        userId="test_user",
        name="Diamond Test",
        validationStatus="valid",  # PR4: required for execution
        graph=WorkflowGraph(
            nodes=[
                {"id": "start", "type": "StartNode"},
                {"id": "c_node", "type": "SendMessage", "config": {"message": "C"}},
                {"id": "b_node", "type": "SendMessage", "config": {"message": "B"}},
                {"id": "d_node", "type": "InvokeAgent", "config": {"agentId": "test"}}
            ],
            edges=[
                {"source": "start", "target": "b_node"},
                {"source": "start", "target": "c_node"},
                {"source": "b_node", "target": "d_node"},
                {"source": "c_node", "target": "d_node"}
            ]
        )
    )
    
    run = WorkflowRun(
        id="test_run",
        workflowId="test_workflow",
        userId="test_user"
    )
    
    repo = MockRepository()
    mock_cfg = MagicMock(spec=Settings)  # PR4: required cfg parameter
    executor = WorkflowExecutor(workflow, run, repo, mock_cfg)
    
    # Execute
    await executor.execute()  # PR4: renamed from run() to execute()
    
    # Check execution order by looking at when nodes were persisted
    persisted_order = list(repo.node_results["test_run"].keys())
    
    # Should be: start, b_node (before c_node due to lexicographic sort), c_node, d_node
    assert persisted_order[0] == "start"
    assert persisted_order[1] == "b_node"  # b comes before c lexicographically
    assert persisted_order[2] == "c_node"
    assert persisted_order[3] == "d_node"


async def test_fan_in_gating():
    """Test that fan-in node waits for all parents to complete"""
    # Diamond graph: start -> b, start -> c, b & c -> d
    workflow = Workflow(
        id="test_workflow",
        userId="test_user",
        name="Fan-in Test",
        validationStatus="valid",  # PR4: required for execution
        graph=WorkflowGraph(
            nodes=[
                {"id": "start", "type": "StartNode"},
                {"id": "b", "type": "SendMessage", "config": {"message": "B"}},
                {"id": "c", "type": "SendMessage", "config": {"message": "C"}},
                {"id": "d", "type": "SendMessage", "config": {"message": "D"}}
            ],
            edges=[
                {"source": "start", "target": "b"},
                {"source": "start", "target": "c"},
                {"source": "b", "target": "d"},
                {"source": "c", "target": "d"}
            ]
        )
    )
    
    run = WorkflowRun(
        id="test_run",
        workflowId="test_workflow",
        userId="test_user"
    )
    
    repo = MockRepository()
    mock_cfg = MagicMock(spec=Settings)  # PR4: required cfg parameter
    executor = WorkflowExecutor(workflow, run, repo, mock_cfg)
    
    # Execute
    await executor.execute()  # PR4: renamed from run() to execute()
    
    # Check that d's inputs include both b and c
    d_result = repo.node_results["test_run"]["d"]
    assert "b" in d_result.inputs
    assert "c" in d_result.inputs
    assert d_result.inputs["b"] == "B"
    assert d_result.inputs["c"] == "C"


async def test_persistence_calls():
    """Test that repository methods are called correctly"""
    workflow = Workflow(
        id="test_workflow",
        userId="test_user",
        name="Persistence Test",
        validationStatus="valid",  # PR4: required for execution
        graph=WorkflowGraph(
            nodes=[
                {"id": "start", "type": "StartNode"},
                {"id": "msg", "type": "SendMessage", "config": {"message": "Hello"}}
            ],
            edges=[
                {"source": "start", "target": "msg"}
            ]
        )
    )
    
    run = WorkflowRun(
        id="test_run",
        workflowId="test_workflow",
        userId="test_user"
    )
    
    repo = MockRepository()
    mock_cfg = MagicMock(spec=Settings)  # PR4: required cfg parameter
    executor = WorkflowExecutor(workflow, run, repo, mock_cfg)
    
    # Execute
    await executor.execute()  # PR4: renamed from run() to execute()
    
    # Check status transitions: queued -> running -> succeeded
    assert len(repo.status_updates) >= 2
    assert repo.status_updates[0]["status"] == "running"
    assert "startedAt" in repo.status_updates[0]
    assert repo.status_updates[-1]["status"] == "succeeded"
    assert "completedAt" in repo.status_updates[-1]
    
    # Check that both nodes were persisted
    assert "start" in repo.node_results["test_run"]
    assert "msg" in repo.node_results["test_run"]
    
    # Check heartbeat was updated
    assert len(repo.heartbeat_updates) >= 2


async def test_startnode_exclusion():
    """Test that StartNode output is excluded from InvokeAgent inputs"""
    # Import for patching
    import src.news_reporter.workflows.executor as executor_mod
    from unittest.mock import patch, AsyncMock
    
    workflow = Workflow(
        id="test_workflow",
        userId="test_user",
        name="StartNode Exclusion Test",
        validationStatus="valid",  # PR4: required for execution
        graph=WorkflowGraph(
            nodes=[
                {"id": "start", "type": "StartNode"},
                {"id": "agent", "type": "InvokeAgent", "config": {"agentId": "test"}}
            ],
            edges=[
                {"source": "start", "target": "agent"}
            ]
        )
    )
    
    run = WorkflowRun(
        id="test_run",
        workflowId="test_workflow",
        userId="test_user"
    )
    
    repo = MockRepository()
    mock_cfg = MagicMock(spec=Settings)  # PR4: required cfg parameter
    
    # PR4: Mock invoke_agent
    with patch.object(executor_mod, "invoke_agent", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = "INVOKE_AGENT_STUBBED"
        
        executor = WorkflowExecutor(workflow, run, repo, mock_cfg)
        
        # Execute
        await executor.execute()  # PR4: renamed from run() to execute()
    
    # Check that agent's inputs do NOT include start (StartNode excluded)
    agent_result = repo.node_results["test_run"]["agent"]
    assert "start" not in agent_result.inputs
    assert len(agent_result.inputs) == 0


async def test_sendmessage_and_invokeagent_outputs():
    """Test that SendMessage and InvokeAgent produce expected outputs"""
    # Import for patching
    import src.news_reporter.workflows.executor as executor_mod
    from unittest.mock import patch, AsyncMock
    
    workflow = Workflow(
        id="test_workflow",
        userId="test_user",
        name="Output Test",
        validationStatus="valid",  # PR4: required for execution
        graph=WorkflowGraph(
            nodes=[
                {"id": "start", "type": "StartNode"},
                {"id": "msg", "type": "SendMessage", "config": {"message": "Test Message"}},
                {"id": "agent", "type": "InvokeAgent", "config": {"agentId": "test"}}
            ],
            edges=[
                {"source": "start", "target": "msg"},
                {"source": "msg", "target": "agent"}
            ]
        )
    )
    
    run = WorkflowRun(
        id="test_run",
        workflowId="test_workflow",
        userId="test_user"
    )
    
    repo = MockRepository()
    mock_cfg = MagicMock(spec=Settings)  # PR4: required cfg parameter
    
    # PR4: Mock invoke_agent to return stubbed output
    with patch.object(executor_mod, "invoke_agent", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = "INVOKE_AGENT_STUBBED"
        
        executor = WorkflowExecutor(workflow, run, repo, mock_cfg)
        
        # Execute
        await executor.execute()  # PR4: renamed from run() to execute()
    
    # Check SendMessage output
    msg_result = repo.node_results["test_run"]["msg"]
    assert msg_result.output == "Test Message"
    assert msg_result.status == "succeeded"
    
    # Check InvokeAgent output (stubbed)
    agent_result = repo.node_results["test_run"]["agent"]
    assert agent_result.output == "INVOKE_AGENT_STUBBED"
    assert agent_result.status == "succeeded"
    
    # Check that agent's inputs include msg (but not start)
    assert "msg" in agent_result.inputs
    assert agent_result.inputs["msg"] == "Test Message"


if __name__ == "__main__":
    print("Running PR3 executor tests...")
    
    test_truncate_output_string()
    print("[PASS] test_truncate_output_string")
    
    test_truncate_output_dict()
    print("[PASS] test_truncate_output_dict")
    
    test_truncate_output_large()
    print("[PASS] test_truncate_output_large")
    
    asyncio.run(test_deterministic_order())
    print("[PASS] test_deterministic_order")
    
    asyncio.run(test_fan_in_gating())
    print("[PASS] test_fan_in_gating")
    
    asyncio.run(test_persistence_calls())
    print("[PASS] test_persistence_calls")
    
    asyncio.run(test_startnode_exclusion())
    print("[PASS] test_startnode_exclusion")
    
    asyncio.run(test_sendmessage_and_invokeagent_outputs())
    print("[PASS] test_sendmessage_and_invokeagent_outputs")
    
    print("\n=== All PR3 executor tests PASSED ===")
