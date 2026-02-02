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
    - Graph expansion (1-2 hops via discovered relationships)
    - Re-rank with multi-signal scoring
    - Dynamic schema discovery (no hardcoded node/relationship types)
    """
    
    def __init__(self, neo4j_api_url: Optional[str] = None):
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
        
        # Cache for discovered graph schema (node labels and relationship types)
        self._schema_cache = None
        
        # If running in Docker and URL uses localhost, replace with host.docker.internal
        if os.getenv("DOCKER_ENV") and "localhost" in self.neo4j_api_url:
            self.neo4j_api_url = self.neo4j_api_url.replace("localhost", "host.docker.internal")
            logger.info(f"Running in Docker - updated Neo4j URL to use host.docker.internal")
        
        # Remove trailing slash if present
        self.neo4j_api_url = self.neo4j_api_url.rstrip("/")
        logger.info(f"Neo4j GraphRAG retriever initialized with URL: {self.neo4j_api_url}")
    
    def discover_graph_schema(self) -> Dict[str, Any]:
        """Dynamically discover all node labels and relationship types in Neo4j graph.
        
        Returns dictionary with:
        - node_labels: List of all node labels (e.g., ['Chunk', 'File', 'Person'])
        - relationship_types: List of all relationship types (e.g., ['SEMANTICALLY_SIMILAR', 'HAS_CHUNK'])
        - relationship_counts: Count of each relationship type
        - sample_paths: Example graph paths showing actual structure
        
        No hardcoded types - discovers from actual database schema.
        """
        if self._schema_cache:
            return self._schema_cache
        
        try:
            url = f"{self.neo4j_api_url}/api/graphrag/schema"
            logger.info(f"🔍 [discover_graph_schema] Querying graph schema from: {url}")
            
            response = requests.get(url, timeout=10.0)
            response.raise_for_status()
            schema = response.json()
            
            # Cache the schema
            self._schema_cache = schema
            
            # Log discovered schema
            logger.info(f"📊 [GraphSchema] Discovered {len(schema.get('node_labels', []))} node labels: {schema.get('node_labels', [])}")
            logger.info(f"📊 [GraphSchema] Discovered {len(schema.get('relationship_types', []))} relationship types: {schema.get('relationship_types', [])}")
            print(f"📊 [GraphSchema] Node labels: {schema.get('node_labels', [])}")
            print(f"📊 [GraphSchema] Relationship types: {schema.get('relationship_types', [])}")
            
            if schema.get('relationship_counts'):
                logger.info(f"📊 [GraphSchema] Relationship counts:")
                for rel_type, count in schema.get('relationship_counts', {}).items():
                    logger.info(f"   - {rel_type}: {count:,} edges")
                    print(f"   - {rel_type}: {count:,} edges")
            
            return schema
            
        except Exception as e:
            logger.warning(f"⚠️ [discover_graph_schema] Failed to discover schema: {e}")
            # Return empty schema if discovery fails
            return {
                "node_labels": [],
                "relationship_types": [],
                "relationship_counts": {},
                "error": str(e)
            }
    
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
        keyword_boost: float = 0.3,
        is_person_query: bool = False,
        enable_coworker_expansion: bool = True,
        person_names: Optional[List[str]] = None,
        section_query: Optional[str] = None,
        use_section_routing: bool = False
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
            is_person_query: Flag indicating person-centric query for coworker expansion
            enable_coworker_expansion: Enable coworker graph traversal for person queries
            person_names: Resolved person names for entity-based expansion
            section_query: Optional structural section query (e.g., "Skills", "Industry Experience")
            use_section_routing: Enable section-aware retrieval routing
        
        Returns:
            List of chunk dicts with: text, file_name, file_path, similarity, hybrid_score, metadata
        """
        try:
            # Discover and log graph schema (cached after first call)
            schema = self.discover_graph_schema()
            if schema.get('relationship_types'):
                logger.info(f"🔗 [GraphStructure] Using graph with {len(schema.get('relationship_types', []))} relationship types for traversal")
                print(f"🔗 [GraphStructure] Graph relationships available: {', '.join(schema.get('relationship_types', [])[:5])}{'...' if len(schema.get('relationship_types', [])) > 5 else ''}")
            
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
                "is_person_query": is_person_query,
                "enable_coworker_expansion": enable_coworker_expansion,
                "use_section_routing": use_section_routing,
            }
            
            # Add optional parameters
            if machine_id:
                payload["machine_id"] = machine_id
            if directory_path:
                payload["directory_path"] = directory_path
            if keywords:
                payload["keywords"] = keywords
            if person_names:
                payload["person_names"] = person_names
            if section_query:
                payload["section_query"] = section_query
            
            logger.info(f"🔍 [hybrid_retrieve] Querying Neo4j GraphRAG API: {url}")
            logger.info(f"🔍 [hybrid_retrieve] Payload: {payload}")
            print(f"🔍 [hybrid_retrieve] Querying Neo4j GraphRAG API: {url}")
            print(f"🔍 [hybrid_retrieve] Payload: query='{query[:100]}...', top_k_vector={top_k_vector}, similarity_threshold={similarity_threshold}, keywords={keywords}")
            print(f"🔍 [hybrid_retrieve] SECTION: section_query={section_query}, use_section_routing={use_section_routing}")
            
            logger.info(f"Querying Neo4j GraphRAG: '{query[:100]}...' (timeout: 120s)")
            start_time = time.time()
            # Increased timeout to 120 seconds for complex queries with graph expansion
            try:
                response = requests.post(url, json=payload, timeout=120.0)
                elapsed = time.time() - start_time
                logger.info(f"Neo4j GraphRAG request completed in {elapsed:.2f}s")
                response.raise_for_status()
                data = response.json()
                
                logger.info(f"📊 [hybrid_retrieve] Neo4j API returned {len(data.get('results', []))} results")
                print(f"📊 [hybrid_retrieve] Neo4j API returned {len(data.get('results', []))} results")
                
                # Debug: log first result structure
                if data.get('results'):
                    first_chunk = data['results'][0]
                    logger.debug(f"🔍 [hybrid_retrieve] First chunk raw structure: {list(first_chunk.keys())}")
                    logger.debug(f"🔍 [hybrid_retrieve] First chunk header fields: header_text={first_chunk.get('header_text')}, header_level={first_chunk.get('header_level')}, parent_headers={first_chunk.get('parent_headers')}")
                    if first_chunk.get('metadata'):
                        logger.debug(f"🔍 [hybrid_retrieve] First chunk metadata keys: {list(first_chunk['metadata'].keys())}")
                
            except requests.exceptions.Timeout:
                elapsed = time.time() - start_time
                logger.error(f"Neo4j GraphRAG request timed out after {elapsed:.2f}s (timeout: 120s)")
                raise
            
            # Transform to match Azure Search format for compatibility
            results = []
            for i, chunk in enumerate(data.get("results", []), 1):
                chunk_result = {
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
                        "expansion_type": chunk.get("metadata", {}).get("expansion_type", "direct"),
                        "graph_path": chunk.get("metadata", {}).get("graph_path", []),
                        "relationship_types": chunk.get("metadata", {}).get("relationship_types", []),
                        "header_level": chunk.get("metadata", {}).get("header_level") or chunk.get("header_level"),
                        "header_text": chunk.get("metadata", {}).get("header_text") or chunk.get("header_text"),
                        "header_path": chunk.get("metadata", {}).get("header_path") or chunk.get("header_path"),
                        "parent_headers": chunk.get("metadata", {}).get("parent_headers") or chunk.get("parent_headers", []),
                        "chunk_index": chunk.get("index"),
                        "file_id": chunk.get("file_id"),
                        "chunk_size": chunk.get("chunk_size")
                    }
                }
                results.append(chunk_result)
                
                # Log first few chunks for debugging
                if i <= 3:
                    hop_count = chunk_result['metadata'].get('hop_count', 0)
                    expansion_type = chunk_result['metadata'].get('expansion_type', 'direct')
                    rel_types = chunk_result['metadata'].get('relationship_types', [])
                    graph_path = chunk_result['metadata'].get('graph_path', [])
                    
                    logger.info(
                        f"   Chunk {i} from API: similarity={chunk_result['similarity']:.3f}, "
                        f"hybrid_score={chunk_result['hybrid_score']:.3f}, "
                        f"hop_count={hop_count}, expansion_type={expansion_type}, "
                        f"relationships={rel_types}, graph_path_length={len(graph_path)}, "
                        f"file='{chunk_result['file_name']}', "
                        f"text_preview='{chunk_result['text'][:100]}...'"
                    )
                    print(
                        f"   Chunk {i} from API: similarity={chunk_result['similarity']:.3f}, "
                        f"hop_count={hop_count}, expansion={expansion_type}, "
                        f"relationships={rel_types if rel_types else 'none'}, "
                        f"file='{chunk_result['file_name']}'"
                    )
            
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
    keyword_boost: float = 0.3,
    is_person_query: bool = False,
    enable_coworker_expansion: bool = True,
    person_names: Optional[List[str]] = None,
    section_query: Optional[str] = None,
    use_section_routing: bool = False
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
        is_person_query: Flag indicating person-centric query for coworker expansion
        enable_coworker_expansion: Enable coworker graph traversal for person queries
        person_names: Resolved person names for entity-based expansion
        section_query: Optional structural section query (e.g., "Skills", "Industry Experience")
        use_section_routing: Enable section-aware retrieval routing
    
    Returns:
        List of chunk results (compatible with Azure Search format)
    """
    logger.info(f"🔍 [graphrag_search] Called with query='{query[:100]}...', top_k={top_k}, similarity_threshold={similarity_threshold}, is_person_query={is_person_query}, person_names={person_names}")
    print(f"🔍 [graphrag_search] Called with query='{query[:100]}...', is_person_query={is_person_query}, person_names={person_names}")
    
    retriever = Neo4jGraphRAGRetriever()
    results = retriever.hybrid_retrieve(
        query=query,
        top_k_vector=top_k,
        max_hops=1,  # Reduced from 2 to 1 to limit indirect connections
        similarity_threshold=similarity_threshold,
        machine_id=machine_id,
        directory_path=directory_path,
        use_keyword_search=use_keyword_search,
        keywords=keywords,
        keyword_match_type=keyword_match_type,
        keyword_boost=keyword_boost,
        is_person_query=is_person_query,
        enable_coworker_expansion=enable_coworker_expansion,
        person_names=person_names,
        section_query=section_query,
        use_section_routing=use_section_routing
    )
    
    logger.info(f"📊 [graphrag_search] hybrid_retrieve returned {len(results)} results")
    print(f"📊 [graphrag_search] hybrid_retrieve returned {len(results)} results")
    
    limited_results = results[:top_k]  # Limit to top_k
    logger.info(f"📊 [graphrag_search] Returning {len(limited_results)} results (limited from {len(results)})")
    print(f"📊 [graphrag_search] Returning {len(limited_results)} results (limited from {len(results)})")
    
    return limited_results

