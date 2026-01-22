"""FastAPI router for Neo4j GraphRAG entity extraction and relationship management"""

import logging
import time
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from ..models.schemas import (
    ExtractEntitiesRequest,
    ExtractEntitiesResponse,
    EntityStatsResponse,
    ChunkEntitiesResponse
)
from ..database.operations import Neo4jOperations
from ..utils.llm_client import AzureOpenAIClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.post("/extract-entities", response_model=ExtractEntitiesResponse)
async def extract_entities_endpoint(request: ExtractEntitiesRequest):
    """
    Extract entities from a file's chunks and create entity graph
    
    This endpoint:
    1. Retrieves all chunks for the specified file
    2. Extracts entities from each chunk using Azure OpenAI
    3. Creates Entity nodes in Neo4j
    4. Creates (Chunk)-[:MENTIONS]->(Entity) edges
    5. Optionally extracts typed relationships between entities
    
    Args:
        request: ExtractEntitiesRequest with file_path and options
    
    Returns:
        ExtractEntitiesResponse with statistics on entities and relationships created
    """
    start_time = time.time()
    errors = []
    
    try:
        # Initialize clients
        db = Neo4jOperations()
        llm_client = AzureOpenAIClient()
        
        # Get chunks for the file
        chunks = db.get_chunks_for_file(request.file_path)
        
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found for file: {request.file_path}"
            )
        
        logger.info(f"Processing {len(chunks)} chunks for file: {request.file_path}")
        
        total_entities_created = 0
        total_mention_edges = 0
        total_relationships = 0
        
        # Process each chunk
        for chunk in chunks:
            chunk_id = chunk.get("id")
            chunk_text = chunk.get("text", "")
            
            if not chunk_text:
                continue
            
            try:
                # Extract entities from chunk
                entities = llm_client.extract_entities_from_chunk(
                    chunk_text=chunk_text,
                    chunk_id=chunk_id,
                    entity_types=request.entity_types
                )
                
                if entities:
                    # Create entity nodes
                    entity_ids = db.create_entity_nodes(
                        entities=entities,
                        source_chunk_id=chunk_id
                    )
                    total_entities_created += len(entity_ids)
                    
                    # Create mention edges
                    mention_count = db.create_mention_edges(
                        chunk_id=chunk_id,
                        entities=entities
                    )
                    total_mention_edges += mention_count
                    
                    # Extract relationships if requested
                    if request.extract_relationships and len(entities) >= 2:
                        relationships = llm_client.extract_relationships_from_chunk(
                            chunk_text=chunk_text,
                            chunk_id=chunk_id,
                            entities=entities
                        )
                        
                        if relationships:
                            rel_count = db.create_typed_relationships(relationships)
                            total_relationships += rel_count
                
            except Exception as e:
                error_msg = f"Error processing chunk {chunk_id}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        # Close database connection
        db.close()
        
        processing_time = time.time() - start_time
        
        return ExtractEntitiesResponse(
            success=True,
            file_path=request.file_path,
            chunks_processed=len(chunks),
            entities_created=total_entities_created,
            mention_edges_created=total_mention_edges,
            relationships_created=total_relationships,
            processing_time_seconds=processing_time,
            errors=errors
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in extract_entities_endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/entity-stats", response_model=EntityStatsResponse)
async def get_entity_stats():
    """
    Get statistics about entities in the graph
    
    Returns:
        EntityStatsResponse with counts of entities by type, mentions, and relationships
    """
    try:
        db = Neo4jOperations()
        
        # Get total entity count
        with db.driver.session() as session:
            result = session.run("MATCH (e:Entity) RETURN count(e) as total")
            total_entities = result.single()["total"]
            
            # Get entity counts by type
            result = session.run("""
                MATCH (e:Entity)
                RETURN e.type as type, count(e) as count
                ORDER BY count DESC
            """)
            entities_by_type = {record["type"]: record["count"] for record in result}
            
            # Get total mentions
            result = session.run("MATCH ()-[m:MENTIONS]->() RETURN count(m) as total")
            total_mentions = result.single()["total"]
            
            # Get total relationships (excluding MENTIONS)
            result = session.run("""
                MATCH ()-[r]->()
                WHERE type(r) <> 'MENTIONS' AND type(r) <> 'SEMANTICALLY_SIMILAR'
                RETURN count(r) as total
            """)
            total_relationships = result.single()["total"]
        
        db.close()
        
        return EntityStatsResponse(
            total_entities=total_entities,
            entities_by_type=entities_by_type,
            total_mentions=total_mentions,
            total_relationships=total_relationships
        )
    
    except Exception as e:
        logger.error(f"Error in get_entity_stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/chunk-entities/{chunk_id}")
async def get_chunk_entities(chunk_id: str):
    """
    Get all entities mentioned in a specific chunk
    
    Args:
        chunk_id: ID of the chunk
    
    Returns:
        List of entities with their mention metadata
    """
    try:
        db = Neo4jOperations()
        entities = db.get_entities_for_chunk(chunk_id)
        db.close()
        
        return {"chunk_id": chunk_id, "entities": entities}
    
    except Exception as e:
        logger.error(f"Error in get_chunk_entities: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/setup-constraints")
async def setup_constraints():
    """
    Set up Neo4j database constraints for entity nodes
    
    This should be called once during initial setup
    """
    try:
        db = Neo4jOperations()
        db.setup_constraints()
        db.close()
        
        return {"success": True, "message": "Constraints created successfully"}
    
    except Exception as e:
        logger.error(f"Error in setup_constraints: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
