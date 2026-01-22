"""Neo4j database operations for GraphRAG entity extraction and relationship management

This module provides operations for:
- Creating and managing entity nodes
- Creating chunk-to-entity mention edges
- Creating typed relationships between entities
- Entity canonicalization and deduplication
"""

import os
import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
import hashlib

logger = logging.getLogger(__name__)


class Neo4jOperations:
    """Neo4j database operations for GraphRAG"""
    
    def __init__(self):
        """Initialize Neo4j connection from environment variables"""
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close Neo4j driver connection"""
        if self.driver:
            self.driver.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    @staticmethod
    def _generate_entity_id(name: str, entity_type: str) -> str:
        """
        Generate a stable ID for an entity based on normalized name and type
        
        Args:
            name: Entity name
            entity_type: Entity type
        
        Returns:
            Stable entity ID (hash)
        """
        # Normalize: lowercase, strip whitespace
        normalized = f"{name.lower().strip()}_{entity_type.lower()}"
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def create_entity_nodes(
        self,
        entities: List[Dict[str, Any]],
        source_file_id: Optional[str] = None,
        source_chunk_id: Optional[str] = None
    ) -> List[str]:
        """
        Create Entity nodes in Neo4j
        
        Args:
            entities: List of entity dicts with keys: name, type, confidence, context
            source_file_id: Optional ID of source file
            source_chunk_id: Optional ID of source chunk
        
        Returns:
            List of created entity IDs
        """
        created_ids = []
        
        with self.driver.session() as session:
            for entity in entities:
                name = entity.get("name", "")
                entity_type = entity.get("type", "Unknown")
                confidence = entity.get("confidence", 0.0)
                context = entity.get("context", "")
                extraction_method = entity.get("extraction_method", "unknown")
                
                if not name:
                    continue
                
                entity_id = self._generate_entity_id(name, entity_type)
                
                # MERGE entity node (create if not exists, update if exists)
                query = """
                MERGE (e:Entity {id: $entity_id})
                ON CREATE SET
                    e.name = $name,
                    e.type = $entity_type,
                    e.created_at = datetime(),
                    e.confidence = $confidence,
                    e.extraction_method = $extraction_method,
                    e.mention_count = 1
                ON MATCH SET
                    e.updated_at = datetime(),
                    e.mention_count = e.mention_count + 1,
                    e.confidence = CASE 
                        WHEN $confidence > e.confidence THEN $confidence 
                        ELSE e.confidence 
                    END
                SET e:$label
                RETURN e.id as entity_id
                """
                
                # Add type-specific label (e.g., Entity:Person)
                label = f"Entity:{entity_type}"
                
                try:
                    result = session.run(
                        query.replace(":$label", f":{entity_type}"),  # Can't parameterize labels directly
                        entity_id=entity_id,
                        name=name,
                        entity_type=entity_type,
                        confidence=confidence,
                        extraction_method=extraction_method
                    )
                    
                    record = result.single()
                    if record:
                        created_ids.append(record["entity_id"])
                        logger.debug(f"Created/updated entity: {name} ({entity_type}) with ID {entity_id}")
                    
                except Exception as e:
                    logger.error(f"Error creating entity node for {name}: {e}")
        
        logger.info(f"Created/updated {len(created_ids)} entity nodes")
        return created_ids
    
    def create_mention_edges(
        self,
        chunk_id: str,
        entities: List[Dict[str, Any]]
    ) -> int:
        """
        Create (Chunk)-[:MENTIONS]->(Entity) edges
        
        Args:
            chunk_id: ID of the chunk
            entities: List of entity dicts
        
        Returns:
            Number of mention edges created
        """
        created_count = 0
        
        with self.driver.session() as session:
            for entity in entities:
                name = entity.get("name", "")
                entity_type = entity.get("type", "Unknown")
                confidence = entity.get("confidence", 0.0)
                context = entity.get("context", "")
                
                if not name:
                    continue
                
                entity_id = self._generate_entity_id(name, entity_type)
                
                # Create MENTIONS relationship
                query = """
                MATCH (c:Chunk {id: $chunk_id})
                MATCH (e:Entity {id: $entity_id})
                MERGE (c)-[m:MENTIONS]->(e)
                ON CREATE SET
                    m.confidence = $confidence,
                    m.context = $context,
                    m.created_at = datetime()
                RETURN m
                """
                
                try:
                    result = session.run(
                        query,
                        chunk_id=chunk_id,
                        entity_id=entity_id,
                        confidence=confidence,
                        context=context
                    )
                    
                    if result.single():
                        created_count += 1
                        logger.debug(f"Created MENTIONS edge from chunk {chunk_id} to entity {name}")
                    
                except Exception as e:
                    logger.error(f"Error creating MENTIONS edge for {name}: {e}")
        
        logger.info(f"Created {created_count} MENTIONS edges for chunk {chunk_id}")
        return created_count
    
    def create_typed_relationships(
        self,
        relationships: List[Dict[str, Any]]
    ) -> int:
        """
        Create typed relationships between entities
        
        Args:
            relationships: List of relationship dicts with keys:
                subject, relationship_type, object, confidence, source_chunk_id
        
        Returns:
            Number of relationships created
        """
        created_count = 0
        
        with self.driver.session() as session:
            for rel in relationships:
                subject_name = rel.get("subject", "")
                object_name = rel.get("object", "")
                rel_type = rel.get("relationship_type", "RELATED_TO")
                confidence = rel.get("confidence", 0.0)
                source_chunk_id = rel.get("source_chunk_id", "")
                
                if not subject_name or not object_name:
                    continue
                
                # Note: We need to find entities by name since we don't have their IDs
                # This is a simplification - in production, you'd want to resolve entities first
                query = f"""
                MATCH (subject:Entity)
                WHERE subject.name = $subject_name
                MATCH (object:Entity)
                WHERE object.name = $object_name
                MERGE (subject)-[r:{rel_type}]->(object)
                ON CREATE SET
                    r.confidence = $confidence,
                    r.source_chunk_id = $source_chunk_id,
                    r.created_at = datetime(),
                    r.extraction_method = 'llm'
                RETURN r
                """
                
                try:
                    result = session.run(
                        query,
                        subject_name=subject_name,
                        object_name=object_name,
                        confidence=confidence,
                        source_chunk_id=source_chunk_id
                    )
                    
                    if result.single():
                        created_count += 1
                        logger.debug(
                            f"Created {rel_type} relationship: {subject_name} -> {object_name}"
                        )
                    
                except Exception as e:
                    logger.error(
                        f"Error creating {rel_type} relationship "
                        f"({subject_name} -> {object_name}): {e}"
                    )
        
        logger.info(f"Created {created_count} typed relationships")
        return created_count
    
    def get_chunks_for_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a file
        
        Args:
            file_path: Path to the file
        
        Returns:
            List of chunk dicts with id, text, and metadata
        """
        query = """
        MATCH (c:Chunk)
        WHERE c.file_path = $file_path
        RETURN c.id as id, c.text as text, c.index as index,
               c.file_path as file_path, c.file_name as file_name
        ORDER BY c.index
        """
        
        with self.driver.session() as session:
            result = session.run(query, file_path=file_path)
            chunks = [dict(record) for record in result]
        
        logger.info(f"Retrieved {len(chunks)} chunks for file {file_path}")
        return chunks
    
    def get_entities_for_chunk(self, chunk_id: str) -> List[Dict[str, Any]]:
        """
        Get all entities mentioned in a chunk
        
        Args:
            chunk_id: Chunk ID
        
        Returns:
            List of entity dicts
        """
        query = """
        MATCH (c:Chunk {id: $chunk_id})-[m:MENTIONS]->(e:Entity)
        RETURN e.id as id, e.name as name, e.type as type,
               m.confidence as confidence, m.context as context
        """
        
        with self.driver.session() as session:
            result = session.run(query, chunk_id=chunk_id)
            entities = [dict(record) for record in result]
        
        return entities
    
    def setup_constraints(self):
        """Set up Neo4j constraints for entity nodes"""
        constraints = [
            "CREATE CONSTRAINT unique_entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT unique_chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"Created constraint: {constraint}")
                except Exception as e:
                    logger.warning(f"Could not create constraint (may already exist): {e}")
