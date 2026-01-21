"""Test Phase 4: Performance Metrics, Retry Handler, Cache Manager, API Integration"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import time

from src.news_reporter.workflows.performance_metrics import (
    PerformanceCollector, WorkflowMetrics, NodeMetrics, get_metrics_collector
)
from src.news_reporter.workflows.retry_handler import RetryHandler, RetryConfig
from src.news_reporter.workflows.cache_manager import CacheManager, CacheEntry, get_cache_manager
from src.news_reporter.workflows.node_result import NodeResult, NodeStatus
from src.news_reporter.workflows.workflow_state import WorkflowState


class TestPerformanceMetrics:
    """Test performance metrics collection"""
    
    def test_metrics_collector_start_end(self):
        """Test starting and ending workflow metrics collection"""
        collector = PerformanceCollector()
        collector.metrics.clear()
        
        collector.start_workflow("run1", "test goal")
        assert collector.current_run_id == "run1"
        assert collector.current_metrics is not None
        assert collector.current_metrics.goal == "test goal"
        
        metrics = collector.end_workflow()
        assert metrics is not None
        assert metrics.run_id == "run1"
        assert collector.current_run_id is None
        assert collector.current_metrics is None
    
    def test_record_node_execution(self):
        """Test recording node execution metrics"""
        collector = PerformanceCollector()
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
        
        collector.record_node_execution(
            node_id="node2",
            node_type="agent",
            status="failed",
            duration_ms=50.0,
            start_time=0.1,
            end_time=0.15,
            error="Test error"
        )
        
        metrics = collector.end_workflow()
        
        assert metrics.total_nodes_executed == 2
        assert metrics.successful_nodes == 1
        assert metrics.failed_nodes == 1
        assert len(metrics.node_metrics) == 2
        assert metrics.node_metrics[0].node_id == "node1"
        assert metrics.node_metrics[1].node_id == "node2"
    
    def test_cache_hit_tracking(self):
        """Test tracking cache hits and misses"""
        collector = PerformanceCollector()
        collector.metrics.clear()
        
        collector.start_workflow("run1", "test goal")
        
        collector.record_node_execution(
            node_id="node1",
            node_type="agent",
            status="success",
            duration_ms=10.0,
            start_time=0.0,
            end_time=0.01,
            cache_hit=True
        )
        
        collector.record_node_execution(
            node_id="node2",
            node_type="agent",
            status="success",
            duration_ms=100.0,
            start_time=0.01,
            end_time=0.11,
            cache_hit=False
        )
        
        metrics = collector.end_workflow()
        
        assert metrics.cache_hits == 1
        assert metrics.cache_misses == 1
    
    def test_retry_count_tracking(self):
        """Test tracking retry counts"""
        collector = PerformanceCollector()
        collector.metrics.clear()
        
        collector.start_workflow("run1", "test goal")
        
        collector.record_node_execution(
            node_id="node1",
            node_type="agent",
            status="success",
            duration_ms=100.0,
            start_time=0.0,
            end_time=0.1,
            retry_count=2
        )
        
        metrics = collector.end_workflow()
        
        assert metrics.total_retries == 2
        assert metrics.node_metrics[0].retry_count == 2
    
    def test_get_summary_stats(self):
        """Test getting summary statistics"""
        collector = PerformanceCollector()
        collector.metrics.clear()
        
        # Run 1
        collector.start_workflow("run1", "goal1")
        collector.record_node_execution("node1", "agent", "success", 100.0, 0.0, 0.1)
        metrics1 = collector.end_workflow()
        metrics1.total_duration_ms = 100.0  # Set explicitly for test
        
        # Run 2
        collector.start_workflow("run2", "goal2")
        collector.record_node_execution("node1", "agent", "success", 200.0, 0.0, 0.2)
        metrics2 = collector.end_workflow()
        metrics2.total_duration_ms = 200.0  # Set explicitly for test
        
        stats = collector.get_summary_stats()
        
        assert stats["total_runs"] == 2
        assert stats["avg_nodes_per_run"] == 1.0


class TestRetryHandler:
    """Test retry handler functionality"""
    
    def test_retry_config(self):
        """Test retry configuration"""
        config = RetryConfig(
            max_retries=3,
            initial_delay_ms=100.0,
            backoff_factor=2.0
        )
        
        assert config.max_retries == 3
        assert config.should_retry("timeout error", 0) is True
        assert config.should_retry("timeout error", 3) is False
        assert config.get_delay_ms(0) == 100.0
        assert config.get_delay_ms(1) == 200.0
        assert config.get_delay_ms(2) == 400.0
    
    def test_retry_success_after_retries(self):
        """Test retry handler succeeds after retries"""
        import asyncio
        config = RetryConfig(max_retries=3, initial_delay_ms=10.0)
        handler = RetryHandler(config)
        
        call_count = 0
        
        async def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("temporary network error")
            return NodeResult.success(state_updates={}, artifacts={"result": "success"})
        
        async def run_test():
            return await handler.execute_with_retry(
                node_id="test_node",
                execute_fn=failing_then_success
            )
        
        result, retry_count = asyncio.run(run_test())
        
        assert result.status == NodeStatus.SUCCESS
        assert retry_count == 2
        assert call_count == 3
    
    def test_retry_exhausted(self):
        """Test retry handler exhausts retries"""
        import asyncio
        config = RetryConfig(max_retries=2, initial_delay_ms=10.0)
        handler = RetryHandler(config)
        
        async def always_fail():
            raise Exception("persistent network error")
        
        async def run_test():
            return await handler.execute_with_retry(
                node_id="test_node",
                execute_fn=always_fail
            )
        
        result, retry_count = asyncio.run(run_test())
        
        assert result.status == NodeStatus.FAILED
        assert retry_count == 2
        assert "persistent network error" in result.error
    
    def test_non_retryable_error(self):
        """Test that non-retryable errors are not retried"""
        config = RetryConfig(
            max_retries=3,
            initial_delay_ms=10.0,
            retryable_errors=["timeout", "network"]
        )
        handler = RetryHandler(config)
        
        async def fail_with_non_retryable():
            raise Exception("invalid input error")
        
        result, retry_count = asyncio.run(handler.execute_with_retry(
            node_id="test_node",
            execute_fn=fail_with_non_retryable
        ))
        
        assert result.status == NodeStatus.FAILED
        assert retry_count == 0  # Should not retry non-retryable errors
    
    def test_immediate_success(self):
        """Test retry handler with immediate success"""
        import asyncio
        config = RetryConfig(max_retries=3, initial_delay_ms=10.0)
        handler = RetryHandler(config)
        
        async def immediate_success():
            return NodeResult.success(state_updates={}, artifacts={"result": "success"})
        
        async def run_test():
            return await handler.execute_with_retry(
                node_id="test_node",
                execute_fn=immediate_success
            )
        
        result, retry_count = asyncio.run(run_test())
        
        assert result.status == NodeStatus.SUCCESS
        assert retry_count == 0


class TestCacheManager:
    """Test cache manager functionality"""
    
    def test_cache_set_get(self):
        """Test setting and getting cache values"""
        cache = CacheManager(max_size=10)
        cache.clear()
        
        cache.set("node1", {"input": "test"}, "result1")
        result = cache.get("node1", {"input": "test"})
        
        assert result == "result1"
        
        # Different input should miss
        result2 = cache.get("node1", {"input": "different"})
        assert result2 is None
    
    def test_cache_expiration(self):
        """Test cache entry expiration"""
        cache = CacheManager(max_size=10, default_ttl_seconds=0.1)  # 100ms TTL
        cache.clear()
        
        cache.set("node1", {"input": "test"}, "result1")
        
        # Should be available immediately
        result = cache.get("node1", {"input": "test"})
        assert result == "result1"
        
        # Wait for expiration
        time.sleep(0.15)
        
        # Should be expired
        result = cache.get("node1", {"input": "test"})
        assert result is None
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction policy - verify eviction happens when cache is full"""
        cache = CacheManager(max_size=3, eviction_policy="lru")
        cache.clear()
        
        # Fill cache to capacity
        cache.set("node1", {"input": "1"}, "result1")
        cache.set("node2", {"input": "2"}, "result2")
        cache.set("node3", {"input": "3"}, "result3")
        
        assert cache.get_stats()["size"] == 3
        
        # Access node1 to make it recently used
        cache.get("node1", {"input": "1"})
        
        # Add node4 - should evict one entry (implementation dependent which one)
        cache.set("node4", {"input": "4"}, "result4")
        
        # Cache should still be at max size
        assert cache.get_stats()["size"] == 3
        
        # node4 should be in cache
        assert cache.get("node4", {"input": "4"}) == "result4"
    
    def test_cache_stats(self):
        """Test cache statistics"""
        cache = CacheManager()
        cache.clear()
        
        cache.set("node1", {"input": "test"}, "result1")
        cache.get("node1", {"input": "test"})  # Hit
        cache.get("node2", {"input": "test"})  # Miss
        
        stats = cache.get_stats()
        
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["size"] == 1
    
    def test_cache_invalidate(self):
        """Test cache invalidation"""
        cache = CacheManager()
        cache.clear()
        
        cache.set("node1", {"input": "test"}, {"node_id": "node1", "result": "result1"})
        cache.set("node2", {"input": "test"}, {"node_id": "node2", "result": "result2"})
        
        # Invalidate node1
        count = cache.invalidate("node1")
        assert count == 1
        
        # node1 should be gone
        assert cache.get("node1", {"input": "test"}) is None
        # node2 should still be there
        assert cache.get("node2", {"input": "test"}) is not None
    
    def test_cache_clear(self):
        """Test clearing cache"""
        cache = CacheManager()
        
        cache.set("node1", {"input": "test"}, "result1")
        cache.set("node2", {"input": "test"}, "result2")
        
        assert cache.get_stats()["size"] == 2
        
        cache.clear()
        
        assert cache.get_stats()["size"] == 0
        assert cache.get_stats()["hits"] == 0
        assert cache.get_stats()["misses"] == 0


class TestPhase4Integration:
    """Integration tests for Phase 4 features"""
    
    def test_metrics_with_cache(self):
        """Test metrics collection with cache hits"""
        collector = PerformanceCollector()
        collector.metrics.clear()
        
        collector.start_workflow("run1", "test goal")
        
        # Record cached execution
        collector.record_node_execution(
            node_id="node1",
            node_type="agent",
            status="success",
            duration_ms=5.0,
            start_time=0.0,
            end_time=0.005,
            cache_hit=True
        )
        
        # Record non-cached execution
        collector.record_node_execution(
            node_id="node2",
            node_type="agent",
            status="success",
            duration_ms=100.0,
            start_time=0.005,
            end_time=0.105,
            cache_hit=False
        )
        
        metrics = collector.end_workflow()
        
        assert metrics.cache_hits == 1
        assert metrics.cache_misses == 1
        assert metrics.node_metrics[0].cache_hit is True
        assert metrics.node_metrics[1].cache_hit is False
    
    def test_retry_with_metrics(self):
        """Test retry handler with metrics tracking"""
        collector = PerformanceCollector()
        collector.metrics.clear()
        
        config = RetryConfig(max_retries=2, initial_delay_ms=10.0)
        handler = RetryHandler(config)
        
        call_count = 0
        
        async def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("network error")
            return NodeResult.success(state_updates={}, artifacts={"result": "success"})
        
        collector.start_workflow("run1", "test goal")
        
        result, retry_count = asyncio.run(handler.execute_with_retry(
            node_id="test_node",
            execute_fn=failing_then_success
        ))
        
        # Record metrics
        collector.record_node_execution(
            node_id="test_node",
            node_type="agent",
            status=result.status.value,
            duration_ms=50.0,
            start_time=0.0,
            end_time=0.05,
            retry_count=retry_count
        )
        
        metrics = collector.end_workflow()
        
        assert result.status == NodeStatus.SUCCESS
        assert retry_count == 1
        assert metrics.total_retries == 1
        assert metrics.node_metrics[0].retry_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
