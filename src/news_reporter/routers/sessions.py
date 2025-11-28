# src/news_reporter/routers/sessions.py
"""FastAPI router for chat session management"""
from __future__ import annotations
import logging
from typing import List
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from ..models import ChatSession, ChatSessionSummary, CreateSessionRequest
from ..repository import LLMChatRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def get_repository(request: Request) -> LLMChatRepository:
    """Get chat repository from app state"""
    repo = request.app.state.chat_repository
    if not repo:
        raise HTTPException(
            status_code=503,
            detail="Chat session service is not available"
        )
    return repo


@router.get("", response_model=List[ChatSessionSummary])
async def list_sessions(
    user_id: str = Query(..., description="User ID to filter sessions"),
    request: Request = None
):
    """
    List all chat sessions for a user
    
    Args:
        user_id: User ID filter
        
    Returns:
        List of chat session summaries
    """
    try:
        repo = get_repository(request)
        sessions = repo.get_sessions(user_id)
        logger.info(f"Retrieved {len(sessions)} sessions for user {user_id}")
        return sessions
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


@router.post("", response_model=ChatSession)
async def create_session(
    create_request: CreateSessionRequest,
    request: Request = None
):
    """
    Create a new chat session
    
    Args:
        create_request: Session creation request
        
    Returns:
        Created chat session
    """
    try:
        repo = get_repository(request)
        session = repo.create_session(
            user_id=create_request.user_id,
            title=create_request.title or "New Chat"
        )
        logger.info(f"Created session {session.id} for user {create_request.user_id}")
        return session
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@router.get("/{session_id}", response_model=ChatSession)
async def get_session(
    session_id: str,
    user_id: str = Query(..., description="User ID for authorization"),
    request: Request = None
):
    """
    Get a single chat session with full message history
    
    Args:
        session_id: Session ID
        user_id: User ID for authorization check
        
    Returns:
        Chat session with messages
    """
    try:
        repo = get_repository(request)
        session = repo.get_session(session_id, user_id=user_id)
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found or access denied"
            )
        
        logger.info(f"Retrieved session {session_id} with {len(session.messages)} messages")
        return session
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")


class UpdateSessionRequest(BaseModel):
    title: str


@router.patch("/{session_id}", response_model=ChatSession)
async def update_session(
    session_id: str,
    update_request: UpdateSessionRequest,
    user_id: str = Query(..., description="User ID for authorization"),
    request: Request = None
):
    """
    Update a chat session (e.g. title)
    """
    try:
        repo = get_repository(request)
        
        # Verify ownership
        session = repo.get_session(session_id, user_id=user_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found or access denied"
            )
            
        # Update title
        repo.update_session_title(session_id, update_request.title)
        
        # Return updated session
        updated_session = repo.get_session(session_id, user_id=user_id)
        return updated_session
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update session: {str(e)}")


@router.delete("/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    user_id: str = Query(..., description="User ID for authorization"),
    request: Request = None
):
    """
    Delete a chat session
    """
    try:
        repo = get_repository(request)
        
        # Verify ownership
        session = repo.get_session(session_id, user_id=user_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found or access denied"
            )
            
        # Delete session
        repo.delete_session(session_id)
        
        logger.info(f"Deleted session {session_id} for user {user_id}")
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")
