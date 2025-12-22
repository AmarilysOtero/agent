"""Neo4j GraphRAG retrieval tool for Foundry AI agents

Provides hybrid GraphRAG retrieval from Neo4j:
- Vector seed (top-k chunks by similarity)
- Graph expansion (1-2 hops via SEMANTICALLY_SIMILAR)
- Re-rank with multi-signal scoring
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
import requests
import logging
import time

try:
    from ..config import Settings
except ImportError:
    import sys, pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root.parent))
    from src.news_reporter.config import Settings

logger = logging.getLogger(__name__)


class Neo4jGraphRAGRetriever:
    """
    Hybrid GraphRAG retrieval from Neo4j:
    - Vector seed (top-k chunks by similarity)
    - Graph expansion (1-2 hops via SEMANTICALLY_SIMILAR)
    - Re-rank with multi-signal scoring
    """
    
    def __init__(self, neo4j_api_url: str = None):
        """
        Args:
            neo4j_api_url: Base URL for Neo4j backend API (e.g., "http://localhost:8000")
        """
        settings = Settings.load()
        self.neo4j_api_url = neo4j_api_url or settings.neo4j_api_url
        if not self.neo4j_api_url:
            raise ValueError(
                "NEO4J_API_URL must be set in .env or passed to constructor. "
                "Example: http://localhost:8000"
            )
        
        # If running in Docker and URL uses localhost, replace with host.docker.internal
        if os.getenv("DOCKER_ENV") and "localhost" in self.neo4j_api_url:
            self.neo4j_api_url = self.neo4j_api_url.replace("localhost", "host.docker.internal")
            logger.info(f"Running in Docker - updated Neo4j URL to use host.docker.internal")
        
        # Remove trailing slash if present
        self.neo4j_api_url = self.neo4j_api_url.rstrip("/")
        logger.info(f"Neo4j GraphRAG retriever initialized with URL: {self.neo4j_api_url}")
    
    def hybrid_retrieve(
        self,
        query: str,
        top_k_vector: int = 10,
        max_hops: int = 2,
        similarity_threshold: float = 0.7,
        machine_id: Optional[str] = None,
        directory_path: Optional[str] = None,
        use_keyword_search: bool = True,
        keywords: Optional[List[str]] = None,
        keyword_match_type: str = "any",
        keyword_boost: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Hybrid GraphRAG retrieval with keyword + semantic search:
        1. Embed query → vector search (top-k chunks)
        2. Keyword search (text matching)
        3. Graph expansion (1-2 hops via relationships)
        4. Re-rank with hybrid scoring
        
        Args:
            query: Search query text
            top_k_vector: Number of top chunks to retrieve via vector search
            max_hops: Maximum graph hops for expansion (1 or 2)
            similarity_threshold: Minimum similarity for relationships
            machine_id: Optional machine ID to scope search
            directory_path: Optional directory path to scope search
            use_keyword_search: Enable keyword search alongside semantic
            keywords: Explicit keywords (auto-extracted if None)
            keyword_match_type: "any" (OR) or "all" (AND) matching
            keyword_boost: Weight for keyword matches in re-ranking (0.0 to 1.0)
        
        Returns:
            List of chunk dicts with: text, file_name, file_path, similarity, hybrid_score, metadata
        """
        try:
            # Call Neo4j GraphRAG API
            url = f"{self.neo4j_api_url}/api/graphrag/query"
            payload = {
                "query": query,
                "top_k_vector": top_k_vector,
                "max_hops": max_hops,
                "similarity_threshold": similarity_threshold,
                "use_keyword_search": use_keyword_search,
                "keyword_match_type": keyword_match_type,
                "keyword_boost": keyword_boost,
            }
            
            # Add optional parameters
            if machine_id:
                payload["machine_id"] = machine_id
            if directory_path:
                payload["directory_path"] = directory_path
            if keywords:
                payload["keywords"] = keywords
            
            logger.info(f"Querying Neo4j GraphRAG: '{query[:100]}...' (timeout: 120s)")
            start_time = time.time()
            # Increased timeout to 120 seconds for complex queries with graph expansion
            try:
                response = requests.post(url, json=payload, timeout=120.0)
                elapsed = time.time() - start_time
                logger.info(f"Neo4j GraphRAG request completed in {elapsed:.2f}s")
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.Timeout:
                elapsed = time.time() - start_time
                logger.error(f"Neo4j GraphRAG request timed out after {elapsed:.2f}s (timeout: 120s)")
                raise
            
            # Transform to match Azure Search format for compatibility
            results = []
            for chunk in data.get("results", []):
                results.append({
                    "id": chunk.get("id"),
                    "text": chunk.get("text", ""),
                    "file_name": chunk.get("file_name"),
                    "file_path": chunk.get("file_path"),
                    "directory_name": chunk.get("directory_name"),
                    "directory_path": chunk.get("directory_path"),
                    "similarity": chunk.get("similarity", 0.0),
                    "hybrid_score": chunk.get("hybrid_score", 0.0),
                    "metadata": {
                        "vector_score": chunk.get("metadata", {}).get("vector_score", 0.0),
                        "keyword_score": chunk.get("metadata", {}).get("keyword_score", 0.0),
                        "path_score": chunk.get("metadata", {}).get("path_score", 0.0),
                        "hop_count": chunk.get("metadata", {}).get("hop_count", 0),
                        "chunk_index": chunk.get("index"),
                        "file_id": chunk.get("file_id"),
                        "chunk_size": chunk.get("chunk_size")
                    }
                })
            
            logger.info(f"Retrieved {len(results)} chunks from Neo4j GraphRAG")
            return results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Neo4j GraphRAG API error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return []
        except Exception as e:
            logger.error(f"Neo4j GraphRAG retrieval error: {e}", exc_info=True)
            return []


def graphrag_search(
    query: str,
    top_k: int = 8,
    similarity_threshold: float = 0.7,
    machine_id: Optional[str] = None,
    directory_path: Optional[str] = None,
    use_keyword_search: bool = True,
    keywords: Optional[List[str]] = None,
    keyword_match_type: str = "any",
    keyword_boost: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Convenience function for GraphRAG search (matches Azure Search API style)
    
    Args:
        query: Search query text
        top_k: Number of results to return
        similarity_threshold: Minimum similarity for relationships
        machine_id: Optional machine ID to scope search
        directory_path: Optional directory path to scope search
        use_keyword_search: Enable keyword search alongside semantic
        keywords: Explicit keywords (auto-extracted if None)
        keyword_match_type: "any" (OR) or "all" (AND) matching
        keyword_boost: Weight for keyword matches in re-ranking (0.0 to 1.0)
    
    Returns:
        List of chunk results (compatible with Azure Search format)
    """
    retriever = Neo4jGraphRAGRetriever()
    return retriever.hybrid_retrieve(
        query=query,
        top_k_vector=top_k,
        max_hops=1,  # Reduced from 2 to 1 to limit indirect connections
        similarity_threshold=similarity_threshold,
        machine_id=machine_id,
        directory_path=directory_path,
        use_keyword_search=use_keyword_search,
        keywords=keywords,
        keyword_match_type=keyword_match_type,
        keyword_boost=keyword_boost
    )[:top_k]  # Limit to top_k

