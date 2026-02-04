"""Phase 3: Full File Expansion API - Expand entry chunks to all chunks per file."""

import logging
from typing import List, Dict, Optional
from neo4j import Driver

logger = logging.getLogger(__name__)


async def expand_to_full_files(
    entry_chunk_ids: List[str],
    neo4j_driver: Driver,
    include_metadata: bool = True
) -> Dict[str, Dict]:
    """
    Given entry chunks, fetch ALL chunks per file (Neo4j query).

    Phase 3: Full File Expansion
    - Query Neo4j for all chunks belonging to files that contain entry chunks
    - Return chunks grouped by file_id, ordered by chunk index
    - Enables RLM to expand from entry points to full file context

    Args:
        entry_chunk_ids: List of chunk UUIDs from retrieval
        neo4j_driver: Neo4j connection
        include_metadata: Whether to include metadata in returned chunks

    Returns:
        {
            file_id: {
                "chunks": [Chunk1, Chunk2, ...],
                "file_name": str,
                "total_chunks": int,
                "entry_chunk_count": int
            }
        }
        Chunks ordered by chunk_index per file

    Example:
        >>> result = await expand_to_full_files(
        ...     entry_chunk_ids=["chunk-1", "chunk-5", "chunk-12"],
        ...     neo4j_driver=driver
        ... )
        >>> for file_id, file_data in result.items():
        ...     print(f"{file_id}: {len(file_data['chunks'])} chunks")
    """
    logger.info(f"🔄 Phase 3: Starting file expansion for {len(entry_chunk_ids)} entry chunks")

    try:
        async with neo4j_driver.session() as session:
            # Step 1: Find all files containing the entry chunks
            logger.info("📍 Phase 3.1: Identifying source files from entry chunks...")
            
            find_files_query = """
            MATCH (f:File)-[:HAS_CHUNK]->(c:Chunk)
            WHERE c.id IN $entry_chunk_ids
            RETURN DISTINCT f.id as file_id, f.name as file_name
            """
            
            files_result = await session.run(
                find_files_query,
                {"entry_chunk_ids": entry_chunk_ids}
            )
            
            files = await files_result.data()
            logger.info(f"✅ Phase 3.1: Found {len(files)} source files")
            
            if not files:
                logger.warning("⚠️  Phase 3: No files found for entry chunks")
                return {}
            
            # Step 2: For each file, fetch ALL chunks ordered by index
            logger.info("📍 Phase 3.2: Fetching full chunk sets per file...")
            
            result = {}
            
            for file_info in files:
                file_id = file_info.get("file_id")
                file_name = file_info.get("file_name", "unknown")
                
                logger.info(f"  → Expanding file: {file_name} (ID: {file_id})")
                
                # Fetch all chunks for this file
                fetch_chunks_query = """
                MATCH (f:File)-[:HAS_CHUNK]->(c:Chunk)
                WHERE f.id = $file_id
                RETURN {
                    chunk_id: c.id,
                    chunk_index: c.index,
                    text: c.text,
                    embedding_id: c.embedding_id,
                    chunk_type: c.chunk_type,
                    metadata: properties(c)
                } as chunk_data
                ORDER BY c.index ASC
                """
                
                chunks_result = await session.run(
                    fetch_chunks_query,
                    {"file_id": file_id}
                )
                
                chunks = await chunks_result.data()
                
                # Count how many entry chunks are in this file
                entry_chunks_in_file = len([
                    c for c in chunks 
                    if c.get("chunk_data", {}).get("chunk_id") in entry_chunk_ids
                ])
                
                logger.info(
                    f"  ✅ File {file_name}: "
                    f"Expanded {entry_chunks_in_file} entry → {len(chunks)} total chunks"
                )
                
                result[file_id] = {
                    "chunks": [c.get("chunk_data") for c in chunks],
                    "file_name": file_name,
                    "total_chunks": len(chunks),
                    "entry_chunk_count": entry_chunks_in_file
                }
        
        # Summary logging
        total_expanded_chunks = sum(
            len(data["chunks"]) for data in result.values()
        )
        
        logger.info(
            f"✅ Phase 3: Expansion complete - "
            f"{len(entry_chunk_ids)} entry chunks → "
            f"{total_expanded_chunks} chunks across {len(result)} files"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Phase 3: Error during file expansion: {str(e)}", exc_info=True)
        raise


async def expand_with_chunks_only(
    entry_chunks: List[Dict],
    neo4j_driver: Driver
) -> Dict[str, Dict]:
    """
    Alternative expansion method: Given full chunk objects, expand to files.

    Args:
        entry_chunks: List of chunk dictionaries with chunk_id and file_id
        neo4j_driver: Neo4j connection

    Returns:
        Same format as expand_to_full_files()
    """
    entry_chunk_ids = [c.get("chunk_id") for c in entry_chunks]
    return await expand_to_full_files(entry_chunk_ids, neo4j_driver)


def filter_chunks_by_relevance(
    expanded_files: Dict[str, Dict],
    entry_chunk_ids: List[str],
    context_window: int = 3
) -> Dict[str, List[Dict]]:
    """
    Filter expanded chunks to maintain context around entry chunks.

    Phase 3 Optional: Selective context expansion
    - Keep entry chunks + surrounding context (before/after)
    - Useful for context windows with size limits

    Args:
        expanded_files: Output from expand_to_full_files()
        entry_chunk_ids: Original entry chunk IDs
        context_window: How many chunks before/after entry to include

    Returns:
        {file_id: [contextual_chunks]}
    """
    logger.info(
        f"🔍 Phase 3: Filtering chunks with context_window={context_window}"
    )
    
    result = {}
    
    for file_id, file_data in expanded_files.items():
        chunks = file_data.get("chunks", [])
        
        # Find indices of entry chunks
        entry_indices = [
            i for i, chunk in enumerate(chunks)
            if chunk.get("chunk_id") in entry_chunk_ids
        ]
        
        if not entry_indices:
            result[file_id] = chunks
            continue
        
        # Build set of indices to include (entry + context)
        include_indices = set()
        for idx in entry_indices:
            # Add entry chunk
            include_indices.add(idx)
            # Add context before
            for i in range(max(0, idx - context_window), idx):
                include_indices.add(i)
            # Add context after
            for i in range(idx + 1, min(len(chunks), idx + context_window + 1)):
                include_indices.add(i)
        
        filtered_chunks = [
            chunks[i] for i in sorted(include_indices)
        ]
        
        logger.info(
            f"  → File {file_id}: {len(entry_indices)} entry chunks, "
            f"{len(filtered_chunks)}/{len(chunks)} chunks with context"
        )
        
        result[file_id] = filtered_chunks
    
    return result
