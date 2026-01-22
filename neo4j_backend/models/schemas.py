"""Pydantic models for Neo4j GraphRAG API requests and responses"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class EntityModel(BaseModel):
    """Model for an extracted entity"""
    name: str = Field(..., description="Entity name")
    type: str = Field(..., description="Entity type (Person, Organization, Location, etc.)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence score")
    context: Optional[str] = Field(None, description="Text context where entity appears")
    extraction_method: Optional[str] = Field("llm", description="Extraction method used")


class RelationshipModel(BaseModel):
    """Model for a typed relationship between entities"""
    subject: str = Field(..., description="Subject entity name")
    relationship_type: str = Field(..., description="Type of relationship")
    object: str = Field(..., description="Object entity name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Relationship confidence score")
    source_chunk_id: Optional[str] = Field(None, description="Source chunk ID")


class ExtractEntitiesRequest(BaseModel):
    """Request model for entity extraction endpoint"""
    machine_id: Optional[str] = Field(None, description="Machine ID for scoping")
    file_path: str = Field(..., description="Path to file to extract entities from")
    entity_types: Optional[List[str]] = Field(
        None,
        description="List of entity types to extract (default: Person, Organization, Location, Concept, Event, Product)"
    )
    extract_relationships: bool = Field(
        False,
        description="Whether to also extract relationships between entities"
    )


class ExtractEntitiesResponse(BaseModel):
    """Response model for entity extraction endpoint"""
    success: bool = Field(..., description="Whether extraction was successful")
    file_path: str = Field(..., description="File path that was processed")
    chunks_processed: int = Field(..., description="Number of chunks processed")
    entities_created: int = Field(..., description="Number of entities created")
    mention_edges_created: int = Field(..., description="Number of chunk-to-entity edges created")
    relationships_created: int = Field(0, description="Number of typed relationships created")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")


class EntityStatsResponse(BaseModel):
    """Response model for entity statistics"""
    total_entities: int
    entities_by_type: Dict[str, int]
    total_mentions: int
    total_relationships: int


class ChunkEntitiesResponse(BaseModel):
    """Response model for getting entities in a chunk"""
    chunk_id: str
    entities: List[EntityModel]
    relationships: List[RelationshipModel]
