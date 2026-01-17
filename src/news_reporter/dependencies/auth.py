# dependencies/auth.py
"""
Canonical auth dependencies for FastAPI routers.

This module provides the single source of truth for authentication dependencies
used across all API routers to prevent drift and ensure consistency.

Uses lazy initialization to avoid Mongo connection failures at import time.
"""
from fastapi import HTTPException, Cookie
from typing import Optional, Tuple
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.collection import Collection
from bson import ObjectId
import logging

from ..config import Settings

logger = logging.getLogger(__name__)


# ===== Auth Principal Model =====

class UserPrincipal(BaseModel):
    """Authenticated user principal for dependency injection"""
    id: str
    email: Optional[str] = None


# ===== Lazy Database Connection =====

# Module-level cache for MongoDB client and collections
_auth_client: Optional[MongoClient] = None
_sessions_collection: Optional[Collection] = None
_users_collection: Optional[Collection] = None


def get_auth_collections() -> Tuple[Collection, Collection]:
    """
    Get auth collections with lazy initialization.
    
    Connects to MongoDB on first call and caches the client/collections.
    This prevents import-time connection failures.
    
    Returns:
        Tuple of (sessions_collection, users_collection)
        
    Raises:
        HTTPException: 503 if database connection fails
    """
    global _auth_client, _sessions_collection, _users_collection
    
    # Return cached collections if available
    if _sessions_collection is not None and _users_collection is not None:
        return _sessions_collection, _users_collection
    
    # Lazy init on first request
    try:
        settings = Settings.load()
        mongo_auth_url = settings.auth_api_url  # This is actually a MongoDB URI
        
        if not mongo_auth_url:
            logger.error("MONGO_AUTH_URL is not set in environment or .env")
            raise HTTPException(
                status_code=503,
                detail="Authentication service unavailable (configuration missing)"
            )
        
        # Create client and get collections
        _auth_client = MongoClient(mongo_auth_url)
        auth_db = _auth_client.get_database()
        _sessions_collection = auth_db["sessions"]
        _users_collection = auth_db["users"]
        
        logger.info("Auth collections initialized successfully")
        return _sessions_collection, _users_collection
        
    except Exception as e:
        logger.exception(f"Failed to connect to auth database: {e}")
        raise HTTPException(
            status_code=503,
            detail="Authentication service unavailable"
        )


# ===== Canonical Auth Dependency =====

def get_current_user(sid: Optional[str] = Cookie(None)) -> UserPrincipal:
    """
    Canonical auth dependency - extracts authenticated user from session cookie.
    
    This is the single source of truth for authentication used by all routers.
    Uses lazy initialization to avoid import-time Mongo connections.
    
    Args:
        sid: Session ID from HttpOnly cookie
        
    Returns:
        UserPrincipal with user id and email
        
    Raises:
        HTTPException: 401 if not authenticated or session invalid/expired
        HTTPException: 503 if database unavailable
    """
    if not sid:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Lazy load collections
    sessions_collection, users_collection = get_auth_collections()
    
    # Get session
    session = sessions_collection.find_one({"_id": sid})
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    # Check expiration - ensure timezone-aware comparison
    expires_at = session["expiresAt"]
    if expires_at.tzinfo is None:
        # MongoDB date is naive, assume UTC
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    
    if expires_at < datetime.now(timezone.utc):
        sessions_collection.delete_one({"_id": sid})
        raise HTTPException(status_code=401, detail="Session expired")
    
    # Get user
    user = users_collection.find_one({"_id": ObjectId(session["userId"])})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    # Return UserPrincipal
    return UserPrincipal(
        id=str(user["_id"]),
        email=user.get("email")
    )
