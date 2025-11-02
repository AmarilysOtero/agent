"""Pydantic models for file structure and graph data"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class FileNode(BaseModel):
    """Model representing a file or directory node"""
    id: str
    type: str = Field(..., description="Either 'file' or 'directory'")
    name: str
    fullPath: str
    relativePath: str
    size: Optional[int] = None
    extension: Optional[str] = None
    modifiedTime: str
    createdAt: str
    source: str
    children: List['FileNode'] = []

    class Config:
        # Allow recursive models
        from_attributes = True


# Update forward reference after class definition
FileNode.model_rebuild()


class FileStructureRequest(BaseModel):
    """Request model for storing file structure"""
    data: FileNode
    metadata: Optional[Dict[str, Any]] = {}
    rag_data: Optional[Dict[str, Dict[str, Any]]] = {}
    machine_id: Optional[str] = None  # Machine ID to identify the client


class MachineRegistrationRequest(BaseModel):
    """Request model for machine registration (no body needed - uses fingerprint)"""
    pass


class GraphStats(BaseModel):
    """Model for graph statistics"""
    total_nodes: int
    total_files: int
    total_directories: int
    sources: List[str]


