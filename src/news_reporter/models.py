# src/news_reporter/models.py
"""Pydantic models for chat sessions and messages"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import uuid


class Message(BaseModel):
    """Represents a single message in a chat session"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sources: Optional[List[dict]] = None


class ChatSession(BaseModel):
    """Represents a chat session with metadata"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    title: str = "New Chat"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    messages: List[Message] = []


class ChatSessionSummary(BaseModel):
    """Summary view of a chat session (without full message history)"""
    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class CreateSessionRequest(BaseModel):
    """Request to create a new chat session"""
    user_id: str
    title: Optional[str] = "New Chat"


class AddMessageRequest(BaseModel):
    """Request to add a message to a session"""
    session_id: str
    role: str
    content: str
    sources: Optional[List[dict]] = None
