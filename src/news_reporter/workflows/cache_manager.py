"""Cache Manager - Caching for agent responses"""

from __future__ import annotations
from typing import Dict, Any, Optional, Callable
import hashlib
import json
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cache entry"""
    key: str
    value: Any
    created_at: float
    expires_at: Optional[float] = None
    access_count: int = 0
    last_accessed: float = 0.0
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def touch(self) -> None:
        """Update access time and count"""
        self.last_accessed = time.time()
        self.access_count += 1


class CacheManager:
    """Manages caching for agent responses"""
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl_seconds: Optional[int] = 3600,  # 1 hour
        eviction_policy: str = "lru"  # "lru" or "fifo"
    ):
        self.max_size = max_size
        self.default_ttl_seconds = default_ttl_seconds
        self.eviction_policy = eviction_policy
        self.cache: Dict[str, CacheEntry] = {}
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, node_id: str, inputs: Dict[str, Any]) -> str:
        """Generate cache key from node ID and inputs"""
        # Create a deterministic key from inputs
        key_data = {
            "node_id": node_id,
            "inputs": inputs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(self, node_id: str, inputs: Dict[str, Any]) -> Optional[Any]:
        """Get cached value"""
        key = self._generate_key(node_id, inputs)
        entry = self.cache.get(key)
        
        if entry is None:
            self.misses += 1
            return None
        
        if entry.is_expired():
            del self.cache[key]
            self.misses += 1
            return None
        
        entry.touch()
        self.hits += 1
        logger.debug(f"Cache hit for node {node_id}")
        return entry.value
    
    def set(
        self,
        node_id: str,
        inputs: Dict[str, Any],
        value: Any,
        ttl_seconds: Optional[int] = None
    ) -> None:
        """Set cached value"""
        key = self._generate_key(node_id, inputs)
        ttl = ttl_seconds or self.default_ttl_seconds
        
        expires_at = None
        if ttl:
            expires_at = time.time() + ttl
        
        # Evict if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict()
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            expires_at=expires_at
        )
        entry.touch()
        
        self.cache[key] = entry
        logger.debug(f"Cached value for node {node_id}")
    
    def _evict(self) -> None:
        """Evict an entry based on policy"""
        if not self.cache:
            return
        
        if self.eviction_policy == "lru":
            # Evict least recently used
            lru_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k].last_accessed
            )
            del self.cache[lru_key]
        elif self.eviction_policy == "fifo":
            # Evict first in (oldest created_at)
            fifo_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k].created_at
            )
            del self.cache[fifo_key]
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "eviction_policy": self.eviction_policy
        }
    
    def invalidate(self, node_id: Optional[str] = None) -> int:
        """
        Invalidate cache entries.
        
        Args:
            node_id: If provided, invalidate only entries for this node
        
        Returns:
            Number of entries invalidated
        """
        if node_id is None:
            count = len(self.cache)
            self.clear()
            return count
        
        # Invalidate entries for specific node
        keys_to_remove = [
            key for key, entry in self.cache.items()
            if isinstance(entry.value, dict) and entry.value.get("node_id") == node_id
        ]
        
        for key in keys_to_remove:
            del self.cache[key]
        
        logger.info(f"Invalidated {len(keys_to_remove)} cache entries for node {node_id}")
        return len(keys_to_remove)


# Global cache manager instance
_global_cache = CacheManager()


def get_cache_manager() -> CacheManager:
    """Get the global cache manager"""
    return _global_cache
