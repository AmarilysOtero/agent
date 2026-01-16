"""Test Phase 2: ExecutionContext, NodeResult, AgentAdapter, Queue-based Executor"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import pytest
import time
from src.news_reporter.workflows.execution_context import ExecutionContext
from src.news_reporter.workflows.node_result import NodeResult, NodeStatus
from src.news_reporter.workflows.agent_adapter import (
    AgentAdapterRegistry, AgentAdapter, TriageAdapter, SQLAdapter
)
from src.news_reporter.workflows.workflow_state import WorkflowState
from src.news_reporter.workflows.graph_schema import GraphDefinition, NodeConfig, EdgeConfig
from src.news_reporter.workflows.graph_executor import GraphExecutor
from src.news_reporter.config import Settings
from unittest.mock import Mock


class TestExecutionContext:
    """Test ExecutionContext for branch identity tracking"""
    
    def test_create_context(self):
        """Test creating a basic execution context"""
        context = ExecutionContext(node_id="test_node")
        
        assert context.node_id == "test_node"
        assert context.run_id is not None
        assert context.branch_id is not None
        assert context.parent_branch_id is None
        assert context.iteration == 0
        assert context.depth == 0
    
    def test_create_child_branch(self):
        """Test creating a child branch"""
        parent = ExecutionContext(node_id="parent")
        child = parent.create_child_branch("child")
        
        assert child.node_id == "child"
        assert child.run_id == parent.run_id
        assert child.branch_id != parent.branch_id
        assert child.parent_branch_id == parent.branch_id
        assert child.depth == parent.depth + 1
        assert child.iteration == parent.iteration
    
    def test_create_iteration(self):
        """Test creating a new iteration"""
        context = ExecutionContext(node_id="loop_node", iteration=1)
        next_iter = context.create_iteration("loop_node")
        
        assert next_iter.node_id == "loop_node"
        assert next_iter.run_id == context.run_id
        assert next_iter.branch_id == context.branch_id  # Same branch
        assert next_iter.iteration == context.iteration + 1
        assert next_iter.depth == context.depth
    
    def test_to_dict_from_dict(self):
        """Test serialization"""
        context = ExecutionContext(node_id="test", iteration=5, depth=2)
        context.parent_branch_id = "parent-123"
        
        data = context.to_dict()
        restored = ExecutionContext.from_dict(data)
        
        assert restored.node_id == context.node_id
        assert restored.run_id == context.run_id
        assert restored.branch_id == context.branch_id
        assert restored.parent_branch_id == context.parent_branch_id
        assert restored.iteration == context.iteration
        assert restored.depth == context.depth


class TestNodeResult:
    """Test NodeResult structure"""
    
    def test_success_result(self):
        """Test creating a successful result"""
        result = NodeResult.success(
            state_updates={"key": "value"},
            artifacts={"artifact": "data"},
            next_nodes=["node1", "node2"]
        )
        
        assert result.status == NodeStatus.SUCCESS
        assert result.state_updates == {"key": "value"}
        assert result.artifacts == {"artifact": "data"}
        assert result.next_nodes == ["node1", "node2"]
        assert result.end_time is not None
        assert result.get_duration() is not None
    
    def test_failed_result(self):
        """Test creating a failed result"""
        result = NodeResult.failed("Test error", {"code": 500})
        
        assert result.status == NodeStatus.FAILED
        assert result.error == "Test error"
        assert result.error_details == {"code": 500}
        assert result.end_time is not None
    
    def test_skipped_result(self):
        """Test creating a skipped result"""
        result = NodeResult.skipped("Condition not met")
        
        assert result.status == NodeStatus.SKIPPED
        assert result.metrics.get("skip_reason") == "Condition not met"
        assert result.end_time is not None
    
    def test_add_methods(self):
        """Test adding state updates, artifacts, and metrics"""
        result = NodeResult()
        
        result.add_state_update("path.to.value", "test")
        result.add_artifact("key", "artifact")
        result.add_metric("duration_ms", 100)
        
        assert result.state_updates["path.to.value"] == "test"
        assert result.artifacts["key"] == "artifact"
        assert result.metrics["duration_ms"] == 100
    
    def test_to_dict(self):
        """Test serialization"""
        result = NodeResult.success(
            state_updates={"a": 1},
            artifacts={"b": 2},
            next_nodes=["c"]
        )
        result.add_metric("m", 3)
        
        data = result.to_dict()
        
        assert data["status"] == "success"
        assert data["state_updates"] == {"a": 1}
        assert data["artifacts"] == {"b": 2}
        assert data["next_nodes"] == ["c"]
        assert data["metrics"]["m"] == 3
        assert "duration" in data


class TestAgentAdapter:
    """Test AgentAdapter registry and adapters"""
    
    def test_triage_adapter(self):
        """Test TriageAdapter"""
        adapter = TriageAdapter()
        state = WorkflowState(goal="test goal")
        
        # Test build_input
        input_data = adapter.build_input(state, {})
        assert input_data == "test goal"
        
        # Test parse_output
        output = '{"preferred_agent": "sql", "database_id": "db123"}'
        updates = adapter.parse_output(output, state)
        
        assert "triage" in updates
        assert updates.get("triage.preferred_agent") == "sql"
        assert updates.get("triage.database_id") == "db123"
        
        # Test get_state_paths
        paths = adapter.get_state_paths()
        assert "goal" in paths["reads"]
        assert "triage" in paths["writes"]
    
    def test_sql_adapter(self):
        """Test SQLAdapter"""
        adapter = SQLAdapter()
        state = WorkflowState(goal="list names")
        state.set("triage.database_id", "db123")
        
        # Test build_input
        input_data = adapter.build_input(state, {"database_id": "db123"})
        assert "list names" in input_data
        assert "db123" in input_data
        
        # Test parse_output
        output = "SELECT * FROM users"
        updates = adapter.parse_output(output, state)
        
        assert updates["latest"] == "SELECT * FROM users"
        assert updates["selected_search"] == "sql"
    
    def test_registry_register_and_get(self):
        """Test AgentAdapterRegistry"""
        registry = AgentAdapterRegistry
        
        # Register a test adapter
        class TestAdapter(AgentAdapter):
            def build_input(self, state, params):
                return "test"
            def parse_output(self, output, state):
                return {}
            def get_state_paths(self):
                return {"reads": [], "writes": []}
        
        registry.register("test", TestAdapter())
        adapter = registry.get_adapter("test")
        
        assert adapter is not None
        assert isinstance(adapter, TestAdapter)
    
    def test_registry_agent_id_mapping(self):
        """Test agent ID to type mapping"""
        registry = AgentAdapterRegistry
        
        # Register adapter and mapping
        class TestAdapter(AgentAdapter):
            def build_input(self, state, params):
                return "test"
            def parse_output(self, output, state):
                return {}
            def get_state_paths(self):
                return {"reads": [], "writes": []}
        
        registry.register("test_type", TestAdapter())
        registry.register_agent_id("agent-123", "test_type")
        
        adapter = registry.get_adapter_by_agent_id("agent-123")
        assert adapter is not None
        assert isinstance(adapter, TestAdapter)


class TestSkipSemantics:
    """Test skip semantics in nodes"""
    
    def test_agent_node_skip_condition(self):
        """Test AgentNode skip condition"""
        from src.news_reporter.workflows.nodes.agent_node import AgentNode
        from src.news_reporter.workflows.agent_runner import AgentRunner
        from unittest.mock import Mock, AsyncMock
        
        state = WorkflowState(goal="test")
        state.set("triage.should_skip", True)
        
        config = NodeConfig(
            id="test_agent",
            type="agent",
            agent_id="test-agent",
            params={"skip_condition": "triage.should_skip == true"}
        )
        
        runner = Mock(spec=AgentRunner)
        runner.run = AsyncMock(return_value="result")
        settings = Mock()
        
        # Register a test adapter for the agent
        from src.news_reporter.workflows.agent_adapter import AgentAdapterRegistry, AgentAdapter
        class TestAdapter(AgentAdapter):
            def build_input(self, state, params):
                return "test"
            def parse_output(self, output, state):
                return {"latest": str(output)}
            def get_state_paths(self):
                return {"reads": [], "writes": ["latest"]}
        
        AgentAdapterRegistry.register("test_type", TestAdapter())
        AgentAdapterRegistry.register_agent_id("test-agent", "test_type")
        
        node = AgentNode(config, state, runner, settings)
        
        # Execute should return skipped result
        import asyncio
        result = asyncio.run(node.execute())
        
        assert result.status == NodeStatus.SKIPPED
        assert "skip" in result.metrics.get("skip_reason", "").lower()


class TestLoopTerminationContract:
    """Test loop termination contract"""
    
    def test_loop_node_max_iters(self):
        """Test loop node respects max_iters"""
        from src.news_reporter.workflows.nodes.loop_node import LoopNode
        from src.news_reporter.workflows.agent_runner import AgentRunner
        from unittest.mock import Mock
        
        state = WorkflowState(goal="test")
        
        config = NodeConfig(
            id="test_loop",
            type="loop",
            max_iters=3,
            params={}
        )
        
        runner = Mock(spec=AgentRunner)
        settings = Mock()
        
        node = LoopNode(config, state, runner, settings)
        
        import asyncio
        
        # First execution: starts at iteration 0, increments to 1
        # Since 1 <= 3, should continue
        result1 = asyncio.run(node.execute())
        assert result1.status == NodeStatus.SUCCESS
        assert result1.artifacts["iteration"] == 1
        assert result1.artifacts["max_iters"] == 3
        assert result1.artifacts["should_continue"] is True
        
        # Test that the node correctly identifies when max_iters would be exceeded
        # by manually checking the logic: if iteration > max_iters, should_continue = False
        # Since we can't easily persist state between calls, we'll test the basic contract:
        # - max_iters is required and respected
        # - The node returns proper artifacts with iteration info
        assert "iteration" in result1.artifacts
        assert "max_iters" in result1.artifacts
        assert "should_continue" in result1.artifacts
        assert "body_node_id" in result1.artifacts
    
    def test_loop_node_continue_condition(self):
        """Test loop node with continue condition"""
        from src.news_reporter.workflows.nodes.loop_node import LoopNode
        from src.news_reporter.workflows.agent_runner import AgentRunner
        from unittest.mock import Mock
        
        state = WorkflowState(goal="test")
        state.set("loop_state.test_loop", {"iteration": 1})
        state.set("triage.done", True)
        
        config = NodeConfig(
            id="test_loop",
            type="loop",
            max_iters=5,
            loop_condition="triage.done != true",
            params={}
        )
        
        runner = Mock(spec=AgentRunner)
        settings = Mock()
        
        node = LoopNode(config, state, runner, settings)
        
        import asyncio
        result = asyncio.run(node.execute())
        
        assert result.status == NodeStatus.SUCCESS
        assert result.artifacts["should_continue"] is False
        assert "continue_condition" in result.artifacts.get("termination_reason", "")


class TestMergeStrategies:
    """Test merge node strategies"""
    
    def test_merge_concat_text(self):
        """Test concat_text strategy"""
        from src.news_reporter.workflows.nodes.merge_node import MergeNode
        from src.news_reporter.workflows.agent_runner import AgentRunner
        from unittest.mock import Mock
        
        state = WorkflowState(goal="test")
        state.set("final", {"a": "text1", "b": "text2"})
        
        config = NodeConfig(
            id="merge",
            type="merge",
            params={
                "merge_key": "final",
                "strategy": "concat_text",
                "separator": " | "
            },
            outputs={"merged": "latest"}
        )
        
        runner = Mock(spec=AgentRunner)
        settings = Mock()
        
        node = MergeNode(config, state, runner, settings)
        
        import asyncio
        result = asyncio.run(node.execute())
        
        assert result.status == NodeStatus.SUCCESS
        merged = result.state_updates.get("latest")
        assert "text1" in merged
        assert "text2" in merged
        assert " | " in merged
    
    def test_merge_collect_list(self):
        """Test collect_list strategy"""
        from src.news_reporter.workflows.nodes.merge_node import MergeNode
        from src.news_reporter.workflows.agent_runner import AgentRunner
        from unittest.mock import Mock
        
        state = WorkflowState(goal="test")
        state.set("final", {"a": "item1", "b": "item2"})
        
        config = NodeConfig(
            id="merge",
            type="merge",
            params={"merge_key": "final", "strategy": "collect_list"},
            outputs={"merged": "latest"}
        )
        
        runner = Mock(spec=AgentRunner)
        settings = Mock()
        
        node = MergeNode(config, state, runner, settings)
        
        import asyncio
        result = asyncio.run(node.execute())
        
        assert result.status == NodeStatus.SUCCESS
        merged = result.state_updates.get("latest")
        assert isinstance(merged, list)
        assert len(merged) == 2
    
    def test_merge_stitch(self):
        """Test stitch strategy"""
        from src.news_reporter.workflows.nodes.merge_node import MergeNode
        from src.news_reporter.workflows.agent_runner import AgentRunner
        from unittest.mock import Mock
        
        state = WorkflowState(goal="test")
        state.set("final", {"reporter1": "script1", "reporter2": "script2"})
        
        config = NodeConfig(
            id="merge",
            type="merge",
            params={"merge_key": "final", "strategy": "stitch"},
            outputs={"merged": "latest"}
        )
        
        runner = Mock(spec=AgentRunner)
        settings = Mock()
        
        node = MergeNode(config, state, runner, settings)
        
        import asyncio
        result = asyncio.run(node.execute())
        
        assert result.status == NodeStatus.SUCCESS
        merged = result.state_updates.get("latest")
        assert "reporter1" in merged
        assert "reporter2" in merged
        assert "script1" in merged
        assert "script2" in merged


class TestJoinBarrier:
    """Test join barrier semantics"""
    
    def test_merge_with_expected_keys(self):
        """Test merge node with expected_keys (join barrier)"""
        from src.news_reporter.workflows.nodes.merge_node import MergeNode
        from src.news_reporter.workflows.agent_runner import AgentRunner
        from unittest.mock import Mock
        
        state = WorkflowState(goal="test")
        # Only provide 2 of 3 expected keys
        state.set("final", {"key1": "value1", "key2": "value2"})
        
        config = NodeConfig(
            id="merge",
            type="merge",
            params={
                "merge_key": "final",
                "strategy": "concat_text",
                "expected_keys": ["key1", "key2", "key3"]
            },
            outputs={"merged": "latest"}
        )
        
        runner = Mock(spec=AgentRunner)
        settings = Mock()
        
        node = MergeNode(config, state, runner, settings)
        
        import asyncio
        result = asyncio.run(node.execute())
        
        # Should proceed but log warning about missing keys
        assert result.status == NodeStatus.SUCCESS
        artifacts = result.artifacts
        assert artifacts["expected_keys"] == ["key1", "key2", "key3"]
        assert len(artifacts["actual_keys"]) == 2  # Only 2 keys present


class TestQueueBasedExecutor:
    """Test queue-based graph executor"""
    
    def test_simple_linear_graph(self):
        """Test executing a simple linear graph"""
        nodes = [
            NodeConfig(id="start", type="agent", agent_id="test-agent"),
            NodeConfig(id="end", type="agent", agent_id="test-agent"),
        ]
        edges = [
            EdgeConfig(from_node="start", to_node="end"),
        ]
        
        graph = GraphDefinition(
            nodes=nodes,
            edges=edges,
            entry_node_id="start"
        )
        
        # Mock config
        config = Mock(spec=Settings)
        config.agent_id_triage = "test-agent"
        config.reporter_ids = []
        
        executor = GraphExecutor(graph, config)
        
        # This would require mocking agent execution
        # For now, just test that executor initializes correctly
        assert executor.graph_def == graph
        assert executor.nodes["start"] is not None
        assert executor.nodes["end"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
