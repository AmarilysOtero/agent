"""Retry Handler - Retry mechanisms for failed node executions"""

from __future__ import annotations
from typing import Callable, Optional, Any, Dict, Tuple
import asyncio
import time
import logging
from functools import wraps

from .node_result import NodeResult, NodeStatus

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior"""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay_ms: float = 1000.0,
        max_delay_ms: float = 10000.0,
        backoff_factor: float = 2.0,
        retryable_errors: Optional[list[str]] = None
    ):
        self.max_retries = max_retries
        self.initial_delay_ms = initial_delay_ms
        self.max_delay_ms = max_delay_ms
        self.backoff_factor = backoff_factor
        self.retryable_errors = retryable_errors or ["timeout", "network", "rate_limit"]
    
    def should_retry(self, error: str, attempt: int) -> bool:
        """Determine if an error should be retried"""
        if attempt >= self.max_retries:
            return False
        
        error_lower = error.lower()
        return any(retryable in error_lower for retryable in self.retryable_errors)
    
    def get_delay_ms(self, attempt: int) -> float:
        """Calculate delay for retry attempt (exponential backoff)"""
        delay = self.initial_delay_ms * (self.backoff_factor ** attempt)
        return min(delay, self.max_delay_ms)


class RetryHandler:
    """Handles retries for node executions"""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
    
    async def execute_with_retry(
        self,
        node_id: str,
        execute_fn: Callable[[], Any],
        get_result_fn: Optional[Callable[[Any], NodeResult]] = None
    ) -> Tuple[NodeResult, int]:
        """
        Execute a function with retry logic.
        
        Args:
            node_id: Node identifier for logging
            execute_fn: Async function to execute
            get_result_fn: Optional function to convert result to NodeResult
        
        Returns:
            Tuple of (NodeResult, retry_count)
        """
        last_error = None
        retry_count = 0
        
        for attempt in range(self.config.max_retries + 1):
            try:
                result = await execute_fn()
                
                # Convert to NodeResult if needed
                if get_result_fn:
                    node_result = get_result_fn(result)
                elif isinstance(result, NodeResult):
                    node_result = result
                else:
                    # Assume success if no conversion function
                    node_result = NodeResult.success(state_updates={}, artifacts={"result": result})
                
                # If successful, return
                if node_result.status == NodeStatus.SUCCESS:
                    if retry_count > 0:
                        logger.info(f"Node {node_id} succeeded after {retry_count} retries")
                    return node_result, retry_count
                
                # Check if error is retryable
                error_msg = node_result.error or str(node_result.status)
                if not self.config.should_retry(error_msg, attempt):
                    logger.warning(
                        f"Node {node_id} failed with non-retryable error: {error_msg}"
                    )
                    return node_result, retry_count
                
                last_error = error_msg
                
            except Exception as e:
                error_msg = str(e)
                if not self.config.should_retry(error_msg, attempt):
                    logger.error(f"Node {node_id} failed with non-retryable exception: {e}")
                    return NodeResult.failed(error_msg), retry_count
                
                last_error = error_msg
            
            # Wait before retry
            if attempt < self.config.max_retries:
                retry_count += 1
                delay_ms = self.config.get_delay_ms(attempt)
                logger.info(
                    f"Node {node_id} failed (attempt {attempt + 1}/{self.config.max_retries + 1}). "
                    f"Retrying in {delay_ms:.0f}ms..."
                )
                await asyncio.sleep(delay_ms / 1000.0)
        
        # All retries exhausted
        logger.error(
            f"Node {node_id} failed after {retry_count} retries. Last error: {last_error}"
        )
        return NodeResult.failed(last_error or "Unknown error"), retry_count


def with_retry(
    max_retries: int = 3,
    initial_delay_ms: float = 1000.0,
    backoff_factor: float = 2.0
):
    """Decorator for adding retry logic to async functions"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_retries=max_retries,
                initial_delay_ms=initial_delay_ms,
                backoff_factor=backoff_factor
            )
            handler = RetryHandler(config)
            
            async def execute():
                return await func(*args, **kwargs)
            
            result, retry_count = await handler.execute_with_retry(
                node_id=kwargs.get("node_id", func.__name__),
                execute_fn=execute
            )
            return result
        
        return wrapper
    return decorator
