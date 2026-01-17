"""Workflow API Gateway - API gateway and rate limiting"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time

logger = logging.getLogger(__name__)


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"


@dataclass
class RateLimit:
    """Rate limit configuration"""
    limit_id: str
    strategy: RateLimitStrategy
    requests_per_window: int
    window_seconds: int = 60
    burst_size: Optional[int] = None  # For token bucket


@dataclass
class APIKey:
    """An API key"""
    key_id: str
    key_value: str
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    rate_limit: Optional[RateLimit] = None
    allowed_endpoints: List[str] = field(default_factory=list)  # Empty = all
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True


@dataclass
class RequestLog:
    """A request log entry"""
    request_id: str
    endpoint: str
    method: str
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    status_code: int = 200
    duration_ms: float = 0.0
    timestamp: Optional[datetime] = None


class WorkflowGateway:
    """API gateway and rate limiting for workflows"""
    
    def __init__(self):
        self.api_keys: Dict[str, APIKey] = {}  # key_value -> APIKey
        self.rate_limit_trackers: Dict[str, Dict[str, Any]] = {}  # key_id -> tracker
        self.request_logs: List[RequestLog] = []
        self._request_counter = 0
    
    def create_api_key(
        self,
        key_id: str,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        rate_limit: Optional[RateLimit] = None
    ) -> APIKey:
        """Create an API key"""
        import secrets
        key_value = secrets.token_urlsafe(32)
        
        api_key = APIKey(
            key_id=key_id,
            key_value=key_value,
            user_id=user_id,
            tenant_id=tenant_id,
            rate_limit=rate_limit,
            created_at=datetime.now()
        )
        
        self.api_keys[key_value] = api_key
        logger.info(f"Created API key: {key_id}")
        return api_key
    
    def validate_api_key(self, key_value: str) -> Optional[APIKey]:
        """Validate an API key"""
        api_key = self.api_keys.get(key_value)
        if not api_key or not api_key.is_active:
            return None
        
        if api_key.expires_at and datetime.now() > api_key.expires_at:
            return None
        
        return api_key
    
    def check_rate_limit(
        self,
        key_value: str,
        endpoint: str
    ) -> tuple[bool, Optional[str]]:
        """
        Check if request is within rate limit.
        
        Returns:
            (allowed, error_message)
        """
        api_key = self.validate_api_key(key_value)
        if not api_key:
            return False, "Invalid API key"
        
        if not api_key.rate_limit:
            return True, None  # No rate limit
        
        rate_limit = api_key.rate_limit
        key_id = api_key.key_id
        
        # Initialize tracker if needed
        if key_id not in self.rate_limit_trackers:
            self.rate_limit_trackers[key_id] = {
                "requests": [],
                "tokens": rate_limit.burst_size or rate_limit.requests_per_window
            }
        
        tracker = self.rate_limit_trackers[key_id]
        
        if rate_limit.strategy == RateLimitStrategy.FIXED_WINDOW:
            # Fixed window: count requests in current window
            window_start = time.time() - (time.time() % rate_limit.window_seconds)
            recent_requests = [
                r for r in tracker["requests"]
                if r >= window_start
            ]
            
            if len(recent_requests) >= rate_limit.requests_per_window:
                return False, f"Rate limit exceeded: {rate_limit.requests_per_window} requests per {rate_limit.window_seconds}s"
            
            tracker["requests"].append(time.time())
            # Clean old requests
            tracker["requests"] = [r for r in tracker["requests"] if r >= window_start]
        
        elif rate_limit.strategy == RateLimitStrategy.TOKEN_BUCKET:
            # Token bucket: refill tokens over time
            now = time.time()
            last_refill = tracker.get("last_refill", now)
            elapsed = now - last_refill
            
            # Refill tokens
            tokens_to_add = (elapsed / rate_limit.window_seconds) * rate_limit.requests_per_window
            tracker["tokens"] = min(
                tracker["tokens"] + tokens_to_add,
                rate_limit.burst_size or rate_limit.requests_per_window
            )
            tracker["last_refill"] = now
            
            if tracker["tokens"] < 1:
                return False, "Rate limit exceeded: token bucket empty"
            
            tracker["tokens"] -= 1
        
        return True, None
    
    def log_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> RequestLog:
        """Log an API request"""
        log = RequestLog(
            request_id=f"req_{self._request_counter}",
            endpoint=endpoint,
            method=method,
            user_id=user_id,
            tenant_id=tenant_id,
            status_code=status_code,
            duration_ms=duration_ms,
            timestamp=datetime.now()
        )
        
        self._request_counter += 1
        self.request_logs.append(log)
        
        # Keep only last 10000 logs
        if len(self.request_logs) > 10000:
            self.request_logs = self.request_logs[-10000:]
        
        return log
    
    def get_request_stats(
        self,
        endpoint: Optional[str] = None,
        user_id: Optional[str] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get request statistics"""
        cutoff = datetime.now() - timedelta(hours=hours)
        logs = [
            l for l in self.request_logs
            if l.timestamp and l.timestamp >= cutoff
        ]
        
        if endpoint:
            logs = [l for l in logs if l.endpoint == endpoint]
        
        if user_id:
            logs = [l for l in logs if l.user_id == user_id]
        
        total_requests = len(logs)
        successful = sum(1 for l in logs if 200 <= l.status_code < 300)
        failed = sum(1 for l in logs if l.status_code >= 400)
        avg_duration = sum(l.duration_ms for l in logs) / total_requests if total_requests > 0 else 0.0
        
        return {
            "total_requests": total_requests,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total_requests if total_requests > 0 else 0.0,
            "avg_duration_ms": avg_duration,
            "period_hours": hours
        }
    
    def revoke_api_key(self, key_value: str) -> bool:
        """Revoke an API key"""
        api_key = self.api_keys.get(key_value)
        if api_key:
            api_key.is_active = False
            logger.info(f"Revoked API key: {api_key.key_id}")
            return True
        return False


# Global gateway instance
_global_gateway = WorkflowGateway()


def get_workflow_gateway() -> WorkflowGateway:
    """Get the global workflow gateway instance"""
    return _global_gateway
