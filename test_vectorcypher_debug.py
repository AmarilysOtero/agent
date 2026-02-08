#!/usr/bin/env python3
"""
Debug script to run the VectorCypher Retrieval query and capture [DEBUG] logs
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Configure logging to show DEBUG messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)-8s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(project_root / "logs" / "debug_vectorcypher.log", mode='w')
    ]
)

logger = logging.getLogger(__name__)

from src.news_reporter.retrieval.recursive_summarizer import recursive_summarize_files
from src.news_reporter.services.file_service import FileService
from src.news_reporter.retrieval.phase_3_expand import expand_chunks_in_files

async def main():
    logger.info("=== Starting VectorCypher Retrieval Debug ===")
    logger.info("Query: What is VectorCypher Retrieval")
    
    query = "What is VectorCypher Retrieval"
    
    # Get sample expanded file data
    file_service = FileService()
    
    # Try to get the Developers-Guide-GraphRAG.pdf
    # For demo purposes, create minimal test data
    expanded_files = {
        "test_file": {
            "file_id": "test-id",
            "file_name": "Developers-Guide-GraphRAG.pdf",
            "chunks": [
                {"chunk_id": f"chunk:{i}", "text": f"Sample chunk {i}"}
                for i in range(106)
            ]
        }
    }
    
    logger.debug("[DEBUG] Starting recursive_summarize_files with RLM enabled")
    
    try:
        results = await recursive_summarize_files(
            expanded_files=expanded_files,
            query=query,
            llm_client=None,
            model_deployment=None,
            rlm_enabled=True
        )
        
        logger.info(f"✓ Summarization complete. Results: {results}")
        
    except Exception as e:
        logger.error(f"✗ Error during summarization: {e}", exc_info=True)
    
    logger.info("=== Debug run complete ===")

if __name__ == "__main__":
    asyncio.run(main())
