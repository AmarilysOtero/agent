"""Test Phase 3: Loop handling, Fanout/Merge coordination, Join barriers, Error handling, Checkpointing"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.news_reporter.workflows.execution_tracker import (
    ExecutionTracker, FanoutTracker, LoopTracker, BranchTracker
)
from src.news_reporter.workflows.state_checkpoint import StateCheckpoint
from src.news_reporter.workflows.workflow_state import WorkflowState
from src.news_reporter.workflows.graph_schema import GraphDefinition, NodeConfig, EdgeConfig, GraphLimits
from src.news_reporter.workflows.execution_context import ExecutionContext
from src.news_reporter.workflows.node_result import NodeResult, NodeStatus


class TestExecutionTracker:
    """Test ExecutionTracker for fanout/loop coordination"""
    
    def test_register_fanout(self):
        """Test registering a fanout"""
        tracker = ExecutionTracker()
        fanout = tracker.register_fanout(
            fanout_node_id="fanout1",
            items=["item1", "item2"],
            branch_node_ids=["branch1"],
            merge_node_id="merge1"
        )
        
        assert fanout.fanout_node_id == "fanout1"
        assert fanout.items == ["item1", "item2"]
        assert fanout.merge_node_id == "merge1"
        assert tracker.get_fanout_tracker("fanout1") == fanout
    
    def test_register_branch(self):
        """Test registering a branch in a fanout"""
        tracker = ExecutionTracker()
        tracker.register_fanout("fanout1", ["item1"], ["branch1"], "merge1")
        
        branch = tracker.register_branch(
            fanout_node_id="fanout1",
            item="item1",
            branch_id="branch-123",
            branch_node_id="branch1"
        )
        
        assert branch.branch_id == "branch-123"
        assert branch.item == "item1"
        assert branch.completed is False
    
    def test_mark_branch_complete(self):
        """Test marking a branch as complete"""
        tracker = ExecutionTracker()
        tracker.register_fanout("fanout1", ["item1"], ["branch1"], "merge1")
        branch = tracker.register_branch("fanout1", "item1", "branch-123", "branch1")
        
        tracker.mark_branch_complete("branch-123", "result")
        
        assert branch.completed is True
        assert branch.result == "result"
    
    def test_all_branches_complete(self):
        """Test checking if all branches are complete"""
        tracker = ExecutionTracker()
        tracker.register_fanout("fanout1", ["item1", "item2"], ["branch1"], "merge1")
        branch1 = tracker.register_branch("fanout1", "item1", "branch-1", "branch1")
        branch2 = tracker.register_branch("fanout1", "item2", "branch-2", "branch1")
        
        fanout = tracker.get_fanout_tracker("fanout1")
        assert fanout.all_branches_complete() is False
        
        tracker.mark_branch_complete("branch-1")
        assert fanout.all_branches_complete() is False
        
        tracker.mark_branch_complete("branch-2")
        assert fanout.all_branches_complete() is True
    
    def test_register_loop(self):
        """Test registering a loop"""
        tracker = ExecutionTracker()
        loop = tracker.register_loop(
            loop_node_id="loop1",
            max_iters=5,
            body_node_id="body1"
        )
        
        assert loop.loop_node_id == "loop1"
        assert loop.max_iters == 5
        assert loop.body_node_id == "body1"
        assert tracker.get_loop_tracker("loop1") == loop
    
    def test_increment_loop_iteration(self):
        """Test incrementing loop iteration"""
        tracker = ExecutionTracker()
        tracker.register_loop("loop1", max_iters=5, body_node_id="body1")
        
        iter1 = tracker.increment_loop_iteration("loop1")
        assert iter1 == 1
        
        iter2 = tracker.increment_loop_iteration("loop1")
        assert iter2 == 2
        
        loop = tracker.get_loop_tracker("loop1")
        assert loop.current_iteration == 2


class TestStateCheckpoint:
    """Test StateCheckpoint for state persistence"""
    
    def test_save_and_load_checkpoint(self, tmp_path):
        """Test saving and loading a checkpoint"""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint = StateCheckpoint(str(checkpoint_dir))
        
        state = WorkflowState(goal="test goal")
        state.set("triage.preferred_agent", "sql")
        
        run_id = "test-run-123"
        checkpoint_path = checkpoint.save_checkpoint(run_id, state, {"test": "metadata"})
        
        assert checkpoint_path is not None
        assert checkpoint_dir.exists()
        
        # Load checkpoint
        checkpoint_data = checkpoint.load_checkpoint(run_id)
        assert checkpoint_data is not None
        assert checkpoint_data["run_id"] == run_id
        assert checkpoint_data["metadata"]["test"] == "metadata"
    
    def test_restore_state(self, tmp_path):
        """Test restoring state from checkpoint"""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint = StateCheckpoint(str(checkpoint_dir))
        
        state = WorkflowState(goal="test goal")
        state.set("triage.preferred_agent", "sql")
        
        run_id = "test-run-456"
        checkpoint.save_checkpoint(run_id, state)
        
        restored = checkpoint.restore_state(run_id)
        assert restored is not None
        assert restored.goal == "test goal"
        assert restored.get("triage.preferred_agent") == "sql"
    
    def test_list_checkpoints(self, tmp_path):
        """Test listing checkpoints"""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint = StateCheckpoint(str(checkpoint_dir))
        
        checkpoint.save_checkpoint("run1", WorkflowState(goal="goal1"))
        checkpoint.save_checkpoint("run2", WorkflowState(goal="goal2"))
        
        checkpoints = checkpoint.list_checkpoints()
        assert "run1" in checkpoints
        assert "run2" in checkpoints


class TestLoopHandling:
    """Test proper loop handling in executor"""
    
    def test_loop_tracker_iteration(self):
        """Test loop tracker tracks iterations correctly"""
        tracker = ExecutionTracker()
        loop = tracker.register_loop("loop1", max_iters=3, body_node_id="body1")
        
        # Simulate iterations
        for i in range(3):
            iter_num = tracker.increment_loop_iteration("loop1")
            tracker.set_loop_should_continue("loop1", iter_num < 3)
        
        assert loop.current_iteration == 3
        assert loop.should_continue is False


class TestFanoutMergeCoordination:
    """Test fanout/merge coordination"""
    
    def test_fanout_tracks_all_branches(self):
        """Test fanout tracks all branches correctly"""
        tracker = ExecutionTracker()
        fanout = tracker.register_fanout(
            fanout_node_id="fanout1",
            items=["item1", "item2", "item3"],
            branch_node_ids=["branch1"],
            merge_node_id="merge1"
        )
        
        # Register all branches
        for i, item in enumerate(["item1", "item2", "item3"]):
            tracker.register_branch("fanout1", item, f"branch-{i}", "branch1")
        
        assert len(fanout.branches) == 3
        assert fanout.all_branches_complete() is False
        
        # Complete all branches
        for i in range(3):
            tracker.mark_branch_complete(f"branch-{i}")
        
        assert fanout.all_branches_complete() is True


class TestJoinBarrier:
    """Test join barrier semantics"""
    
    def test_join_barrier_tracks_expected_keys(self):
        """Test join barrier tracks expected vs actual keys"""
        from src.news_reporter.workflows.nodes.merge_node import MergeNode
        from src.news_reporter.workflows.agent_runner import AgentRunner
        from unittest.mock import Mock
        
        state = WorkflowState(goal="test")
        # Provide 2 of 3 expected keys
        state.set("final", {"key1": "value1", "key2": "value2"})
        
        config = NodeConfig(
            id="merge",
            type="merge",
            params={
                "merge_key": "final",
                "strategy": "concat_text",
                "expected_keys": ["key1", "key2", "key3"],
                "timeout": 1000.0  # 1 second timeout
            },
            outputs={"merged": "latest"}
        )
        
        runner = Mock(spec=AgentRunner)
        settings = Mock()
        
        node = MergeNode(config, state, runner, settings)
        
        import asyncio
        result = asyncio.run(node.execute())
        
        # Should indicate waiting for keys
        assert result.status == NodeStatus.SUCCESS
        artifacts = result.artifacts
        assert "waiting_for_keys" in artifacts
        assert "key3" in artifacts["waiting_for_keys"]


class TestErrorHandling:
    """Test error handling and recovery"""
    
    def test_error_strategy_continue(self):
        """Test error strategy 'continue' allows execution to proceed"""
        # This would be tested in integration tests with actual executor
        # For now, just verify the structure
        pass
    
    def test_error_strategy_stop(self):
        """Test error strategy 'stop' halts execution on error"""
        # This would be tested in integration tests
        pass


class TestExecutionLimits:
    """Test execution limits and timeouts"""
    
    def test_graph_limits_initialization(self):
        """Test graph limits are properly initialized"""
        limits = GraphLimits(
            max_steps=500,
            timeout_ms=30000,
            max_parallel=10
        )
        
        assert limits.max_steps == 500
        assert limits.timeout_ms == 30000
        assert limits.max_parallel == 10
    
    def test_executor_uses_limits(self):
        """Test executor uses graph limits"""
        nodes = [NodeConfig(id="start", type="agent", agent_id="test-agent")]
        edges = []
        
        graph = GraphDefinition(
            nodes=nodes,
            edges=edges,
            entry_node_id="start",
            limits=GraphLimits(max_steps=100, timeout_ms=5000)
        )
        
        config = Mock()
        config.reporter_ids = []
        # Don't set checkpoint_dir to avoid Mock path issues
        config.checkpoint_dir = None
        
        from src.news_reporter.workflows.graph_executor import GraphExecutor
        executor = GraphExecutor(graph, config)
        
        assert executor.max_steps == 100
        assert executor.timeout_ms == 5000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
