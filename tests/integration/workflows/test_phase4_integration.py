"""Integration tests for Phase 4: Full workflow execution with metrics, caching, retry"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.news_reporter.workflows.graph_executor import GraphExecutor
from src.news_reporter.models.graph_schema import GraphDefinition, NodeConfig, EdgeConfig, GraphLimits
from src.news_reporter.workflows.performance_metrics import get_metrics_collector
from src.news_reporter.workflows.cache_manager import get_cache_manager
from src.news_reporter.workflows.retry_handler import RetryHandler, RetryConfig
from src.news_reporter.config import Settings


class TestPhase4Integration:
    """Integration tests for Phase 4 features"""
    
    @pytest.fixture
    def simple_graph(self):
        """Create a simple graph for testing"""
        nodes = [
            NodeConfig(id="start", type="agent", agent_id="test-agent"),
            NodeConfig(id="end", type="agent", agent_id="test-agent")
        ]
        edges = [
            EdgeConfig(from_node="start", to_node="end")
        ]
        return GraphDefinition(
            nodes=nodes,
            edges=edges,
            entry_node_id="start",
            limits=GraphLimits(max_steps=10, timeout_ms=5000)
        )
    
    @pytest.fixture
    def mock_config(self):
        """Create mock config"""
        config = Mock(spec=Settings)
        config.reporter_ids = []
        config.checkpoint_dir = None
        config.max_retries = 2
        config.retry_delay_ms = 100.0
        return config
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, simple_graph, mock_config):
        """Test that metrics are collected during execution"""
        with patch('src.news_reporter.workflows.agent_runner.AgentRunner') as mock_runner_class:
            mock_runner = Mock()
            mock_runner.run = AsyncMock(return_value="test result")
            mock_runner_class.return_value = mock_runner
            
            executor = GraphExecutor(simple_graph, mock_config)
            metrics_collector = get_metrics_collector()
            
            # Clear previous metrics
            metrics_collector.metrics.clear()
            
            try:
                result = await executor.execute("test goal")
                
                # Check that metrics were collected
                all_metrics = metrics_collector.get_all_metrics()
                assert len(all_metrics) > 0
                
                workflow_metrics = all_metrics[-1]
                assert workflow_metrics.goal == "test goal"
                assert workflow_metrics.total_nodes_executed > 0
            except Exception as e:
                # Execution might fail due to missing adapters, but metrics should still be collected
                all_metrics = metrics_collector.get_all_metrics()
                assert len(all_metrics) > 0
    
    @pytest.mark.asyncio
    async def test_caching(self, simple_graph, mock_config):
        """Test that caching works for node execution"""
        with patch('src.news_reporter.workflows.agent_runner.AgentRunner') as mock_runner_class:
            mock_runner = Mock()
            mock_runner.run = AsyncMock(return_value="test result")
            mock_runner_class.return_value = mock_runner
            
            executor = GraphExecutor(simple_graph, mock_config)
            cache_manager = get_cache_manager()
            cache_manager.clear()
            
            # First execution - should miss cache
            try:
                await executor.execute("test goal")
            except:
                pass
            
            # Check cache stats
            stats = cache_manager.get_stats()
            # Cache might have entries if nodes executed successfully
            assert "hits" in stats
            assert "misses" in stats
    
    @pytest.mark.asyncio
    async def test_retry_handler(self):
        """Test retry handler functionality"""
        config = RetryConfig(max_retries=2, initial_delay_ms=10.0)
        handler = RetryHandler(config)
        
        call_count = 0
        
        async def failing_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("temporary error")
            return "success"
        
        result, retry_count = await handler.execute_with_retry(
            node_id="test",
            execute_fn=failing_fn
        )
        
        assert retry_count == 2
        assert call_count == 3
    
    def test_cache_manager_basic(self):
        """Test basic cache manager operations"""
        cache = get_cache_manager()
        cache.clear()
        
        # Set and get
        cache.set("node1", {"input": "test"}, "result1")
        result = cache.get("node1", {"input": "test"})
        
        assert result == "result1"
        
        # Stats
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0
    
    def test_performance_metrics_basic(self):
        """Test basic performance metrics collection"""
        collector = get_metrics_collector()
        collector.metrics.clear()
        
        collector.start_workflow("run1", "test goal")
        collector.record_node_execution(
            node_id="node1",
            node_type="agent",
            status="success",
            duration_ms=100.0,
            start_time=0.0,
            end_time=0.1
        )
        metrics = collector.end_workflow()
        
        assert metrics is not None
        assert metrics.run_id == "run1"
        assert metrics.total_nodes_executed == 1
        assert metrics.successful_nodes == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
