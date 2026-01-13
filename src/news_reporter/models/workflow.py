# src/news_reporter/models/workflow.py
"""Pydantic models for Workflow System (Phase 4)"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Literal
from datetime import datetime, timezone  # FIX D: Add timezone import
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ===== Node Result Models =====

class NodeError(BaseModel):
    """Structured error for node failure"""
    message: str
    details: str  # Type name or additional context


class NodeResult(BaseModel):
    """Result of a single node execution (terminal states only)"""
    status: Literal["succeeded", "failed"]
    inputs: Dict[str, str] = Field(default_factory=dict)  # {parentNodeId: outputString}
    output: Optional[str] = None  # PR5 Fix 2: Allow None for queued nodes
    outputTruncated: bool = False
    outputPreview: Optional[str] = None  # First N chars if truncated
    executionMs: float = 0.0  # PR5 Fix 2: Default 0.0 for queued
    startedAt: Optional[datetime] = None  # PR5 Fix 2: Allow None for queued nodes
    completedAt: Optional[datetime] = None  # PR5 Fix 2: Allow None for queued nodes
    logs: List[str] = Field(default_factory=list)  # Empty for MVP
    error: Optional[NodeError] = None


# ===== Workflow Models =====

class WorkflowGraph(BaseModel):
    """Graph structure (nodes + edges)"""
    nodes: List[dict] = Field(default_factory=list)
    edges: List[dict] = Field(default_factory=list)
    viewport: dict = Field(default_factory=dict)


class Workflow(BaseModel):
    """Workflow definition document"""
    id: Optional[str] = None  # MongoDB _id as string
    userId: str  # MUST be string per canonical contract
    name: str
    description: str = ""
    graph: WorkflowGraph = Field(default_factory=WorkflowGraph)
    validationStatus: Literal["unvalidated", "valid", "invalid"] = "unvalidated"  # PR5 Fix 1: Include 'unvalidated'
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))  # FIX D: timezone-aware
    updatedAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))  # FIX D: timezone-aware
    schemaVersion: int = 1

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ===== Workflow Run Models =====

class WorkflowRun(BaseModel):
    """Workflow execution run document"""
    id: Optional[str] = None  # MongoDB _id as string
    workflowId: str  # Reference to Workflow._id
    userId: str  # MUST be string per canonical contract
    status: Literal["queued", "running", "succeeded", "failed", "canceled"] = "queued"
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))  # FIX D: timezone-aware
    startedAt: Optional[datetime] = None
    completedAt: Optional[datetime] = None
    heartbeatAt: Optional[datetime] = None
    nodeResults: Dict[str, NodeResult] = Field(default_factory=dict)  # {nodeId: NodeResult}
    error: Optional[str] = None  # Run-level error (string summary)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ===== Request/Response Models =====

class WorkflowCreateRequest(BaseModel):
    """Request to create a new workflow"""
    name: str
    description: str = ""
    graph: WorkflowGraph = Field(default_factory=WorkflowGraph)


class WorkflowUpdateRequest(BaseModel):
    """Request to update workflow"""
    name: Optional[str] = None
    description: Optional[str] = None
    graph: Optional[WorkflowGraph] = None


class WorkflowRunCreateResponse(BaseModel):
    """Response when creating a workflow run"""
    runId: str


# ===== Validation Models =====

class ValidationIssue(BaseModel):
    """Structured validation error (FIX 3: align with validator output)"""
    code: str
    message: str
    nodeId: Optional[str] = None
    edgeId: Optional[str] = None


class ValidationResult(BaseModel):
    """Validation result response (FIX 3: use structured errors)"""
    valid: bool
    errors: List[ValidationIssue] = Field(default_factory=list)  # FIX 3: was List[str]
