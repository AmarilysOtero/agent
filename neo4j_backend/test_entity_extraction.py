#!/usr/bin/env python3
"""Test script for Neo4j GraphRAG entity extraction

This script tests the entity extraction functionality by:
1. Checking Neo4j connection
2. Testing entity extraction from sample text
3. Verifying entity nodes and relationships are created
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j_backend.database.operations import Neo4jOperations
from neo4j_backend.utils.llm_client import AzureOpenAIClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_neo4j_connection():
    """Test Neo4j database connection"""
    logger.info("Testing Neo4j connection...")
    try:
        db = Neo4jOperations()
        with db.driver.session() as session:
            result = session.run("RETURN 1 as test")
            assert result.single()["test"] == 1
        db.close()
        logger.info("✓ Neo4j connection successful")
        return True
    except Exception as e:
        logger.error(f"✗ Neo4j connection failed: {e}")
        return False


def test_entity_extraction():
    """Test entity extraction with sample text"""
    logger.info("Testing entity extraction...")
    
    sample_text = """
    Dr. Jane Smith works at Microsoft Corporation in Seattle, Washington.
    She is developing a new artificial intelligence framework called DeepMind.
    The project collaborates with researchers from Stanford University.
    """
    
    try:
        client = AzureOpenAIClient()
        entities = client.extract_entities_from_chunk(
            chunk_text=sample_text,
            chunk_id="test_chunk_001"
        )
        
        logger.info(f"✓ Extracted {len(entities)} entities:")
        for entity in entities:
            logger.info(f"  - {entity['name']} ({entity['type']}) [confidence: {entity['confidence']:.2f}]")
        
        return len(entities) > 0, entities
    
    except Exception as e:
        logger.error(f"✗ Entity extraction failed: {e}")
        return False, []


def test_entity_storage(entities):
    """Test storing entities in Neo4j"""
    logger.info("Testing entity storage...")
    
    try:
        db = Neo4jOperations()
        
        # Create test chunk node first
        with db.driver.session() as session:
            session.run("""
                MERGE (c:Chunk {id: 'test_chunk_001'})
                SET c.text = $text, c.file_path = '/test/sample.txt'
            """, text="Sample text for testing")
        
        # Create entity nodes
        entity_ids = db.create_entity_nodes(entities, source_chunk_id="test_chunk_001")
        logger.info(f"✓ Created {len(entity_ids)} entity nodes")
        
        # Create mention edges
        mention_count = db.create_mention_edges("test_chunk_001", entities)
        logger.info(f"✓ Created {mention_count} MENTIONS edges")
        
        db.close()
        return True
    
    except Exception as e:
        logger.error(f"✗ Entity storage failed: {e}")
        return False


def test_relationship_extraction(entities):
    """Test relationship extraction"""
    logger.info("Testing relationship extraction...")
    
    sample_text = """
    Dr. Jane Smith works at Microsoft Corporation in Seattle, Washington.
    She is developing a new artificial intelligence framework called DeepMind.
    """
    
    try:
        client = AzureOpenAIClient()
        relationships = client.extract_relationships_from_chunk(
            chunk_text=sample_text,
            chunk_id="test_chunk_001",
            entities=entities
        )
        
        logger.info(f"✓ Extracted {len(relationships)} relationships:")
        for rel in relationships:
            logger.info(
                f"  - {rel['subject']} --[{rel['relationship_type']}]--> {rel['object']} "
                f"[confidence: {rel['confidence']:.2f}]"
            )
        
        if relationships:
            # Store relationships
            db = Neo4jOperations()
            rel_count = db.create_typed_relationships(relationships)
            db.close()
            logger.info(f"✓ Created {rel_count} typed relationships in Neo4j")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Relationship extraction failed: {e}")
        return False


def test_setup_constraints():
    """Test setting up Neo4j constraints"""
    logger.info("Setting up Neo4j constraints...")
    try:
        db = Neo4jOperations()
        db.setup_constraints()
        db.close()
        logger.info("✓ Constraints set up successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Constraint setup failed: {e}")
        return False


def cleanup_test_data():
    """Clean up test data from Neo4j"""
    logger.info("Cleaning up test data...")
    try:
        db = Neo4jOperations()
        with db.driver.session() as session:
            # Delete test chunk and its relationships
            session.run("MATCH (c:Chunk {id: 'test_chunk_001'}) DETACH DELETE c")
            # Delete test entities (those without other mentions)
            session.run("""
                MATCH (e:Entity)
                WHERE NOT (e)<-[:MENTIONS]-()
                DETACH DELETE e
            """)
        db.close()
        logger.info("✓ Test data cleaned up")
    except Exception as e:
        logger.error(f"✗ Cleanup failed: {e}")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Neo4j GraphRAG Entity Extraction - Test Suite")
    print("="*60 + "\n")
    
    # Test 1: Neo4j connection
    if not test_neo4j_connection():
        print("\n✗ Tests failed: Cannot connect to Neo4j")
        print("  Make sure Neo4j is running and credentials are correct in .env")
        return False
    
    # Test 2: Set up constraints
    if not test_setup_constraints():
        print("\n✗ Tests failed: Cannot set up constraints")
        return False
    
    # Test 3: Entity extraction
    success, entities = test_entity_extraction()
    if not success:
        print("\n✗ Tests failed: Entity extraction failed")
        print("  Make sure Azure OpenAI credentials are correct in .env")
        return False
    
    # Test 4: Entity storage
    if not test_entity_storage(entities):
        print("\n✗ Tests failed: Entity storage failed")
        return False
    
    # Test 5: Relationship extraction
    if not test_relationship_extraction(entities):
        print("\n✗ Tests failed: Relationship extraction failed")
        return False
    
    # Cleanup
    cleanup_test_data()
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60 + "\n")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
