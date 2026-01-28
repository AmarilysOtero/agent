#!/usr/bin/env python3
"""
Build header vocabulary from Neo4j chunks via backend API.

Usage:
    python build_header_vocab.py
"""

import json
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import requests
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from src.news_reporter.tools.header_vocab import build_header_vocab, save_header_vocab

def get_chunks_from_neo4j_direct() -> List[Dict[str, Any]]:
    """Query Neo4j directly to get chunks."""
    try:
        from neo4j import GraphDatabase
    except ImportError:
        logger.error("neo4j package not installed")
        return []
    
    # Use Docker network name
    uri = "bolt://neo4j:7687"
    user = "neo4j"
    password = "password"
    
    try:
        logger.info(f"Connecting to Neo4j at {uri}...")
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        chunks = []
        with driver.session() as session:
            query = """
            MATCH (c:Chunk)
            RETURN 
                c.id as chunk_id,
                c.text as text,
                c.header_text as header_text,
                c.parent_headers as parent_headers
            LIMIT 5000
            """
            
            result = session.run(query)
            for record in result:
                chunk = {
                    "id": record["chunk_id"],
                    "text": record["text"],
                    "metadata": {
                        "header_text": record["header_text"],
                        "parent_headers": record["parent_headers"] or []
                    }
                }
                chunks.append(chunk)
        
        driver.close()
        logger.info(f"Loaded {len(chunks)} chunks from Neo4j")
        return chunks
        
    except Exception as e:
        logger.warning(f"Failed to connect to Neo4j: {e}")
        return []

def main():
    """Build and save header vocabulary."""
    logger.info("Starting header vocabulary build...")
    
    # Try to get chunks from Neo4j
    chunks = get_chunks_from_neo4j_direct()
    
    if not chunks:
        logger.error("❌ Failed to load chunks from Neo4j!")
        logger.error("   Make sure the Docker services are running:")
        logger.error("   cd c:\\Alexis\\Projects\\RAG_Infra")
        logger.error("   docker-compose -f docker-compose.dev.yml up -d")
        return False
    
    logger.info(f"✅ Loaded {len(chunks)} chunks from Neo4j")
    
    # Build vocabulary
    logger.info(f"Building vocabulary from {len(chunks)} chunks...")
    vocab = build_header_vocab(chunks, min_count=1, max_ngram=4)
    
    if not vocab:
        logger.error("No vocabulary phrases found!")
        return False
    
    logger.info(f"✅ Built vocabulary with {len(vocab)} unique phrases")
    
    # Top phrases
    top_phrases = sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:25]
    logger.info("📊 Top 25 phrases:")
    for i, (phrase, count) in enumerate(top_phrases, 1):
        logger.info(f"  {i:2d}. '{phrase}': {count}")
    
    # Save vocabulary
    output_path = Path(__file__).parent / "src" / "news_reporter" / "settings" / "header_vocab.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_header_vocab(vocab, str(output_path))
    logger.info(f"✅ Vocabulary saved to {output_path}")
    logger.info(f"   Size: {len(vocab)} phrases")
    logger.info(f"   This enables corpus-driven person name extraction")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Restart the Agent service")
    logger.info("  2. Test with queries like: 'What employees work at DXC?'")
    logger.info("  3. Graph traversal via WORKS_WITH relationships will now work")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
