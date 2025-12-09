"""Schema retrieval tool for SQL generation"""


from __future__ import annotations
from typing import List, Dict, Any, Optional
import requests
import logging
import time

try:
    from ..config import Settings
except ImportError:
    import sys
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root.parent))
    from src.news_reporter.config import Settings

logger = logging.getLogger(__name__)


class SchemaRetriever:
    """
    Schema retrieval from Neo4j backend for SQL generation
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
        
        # Remove trailing slash if present
        self.neo4j_api_url = self.neo4j_api_url.rstrip("/")
        logger.info(f"Schema retriever initialized with URL: {self.neo4j_api_url}")
    
    def get_relevant_schema(
        self,
        query: str,
        database_id: str,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        element_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve relevant schema elements for a natural language query
        
        Args:
            query: Natural language query
            database_id: Database configuration ID
            top_k: Number of schema elements to retrieve
            similarity_threshold: Minimum similarity score
            element_types: Types of elements to search (["table", "column", "metric"] or None for all)
        
        Returns:
            Dictionary with:
            - results: List of relevant schema elements
            - schema_slice: Focused schema slice with tables and columns
            - result_count: Number of results
        """
        try:
            url = f"{self.neo4j_api_url}/api/databases/{database_id}/schema/search"
            payload = {
                "query": query,
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
                "element_types": element_types,
                "use_keyword_search": True,
                "use_graph_expansion": True,
                "max_hops": 1
            }
            
            logger.info(f"Searching schema for query: '{query[:100]}...' (database: {database_id})")
            start_time = time.time()
            
            try:
                response = requests.post(url, json=payload, timeout=30.0)
                elapsed = time.time() - start_time
                logger.info(f"Schema search completed in {elapsed:.2f}s")
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.Timeout:
                elapsed = time.time() - start_time
                logger.error(f"Schema search timed out after {elapsed:.2f}s")
                raise
            
            logger.info(f"Retrieved {data.get('result_count', 0)} schema elements")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Schema retrieval API error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return {
                "results": [],
                "schema_slice": {"tables": []},
                "result_count": 0,
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Schema retrieval error: {e}", exc_info=True)
            return {
                "results": [],
                "schema_slice": {"tables": []},
                "result_count": 0,
                "error": str(e)
            }
    
    def format_schema_slice_for_llm(self, schema_slice: Dict[str, Any]) -> str:
        """
        Format schema slice as text for LLM SQL generation
        
        Args:
            schema_slice: Schema slice from get_relevant_schema
        
        Returns:
            Formatted text string describing the schema
        """
        tables = schema_slice.get("tables", [])
        if not tables:
            return "No relevant schema information found."
        
        lines = ["Relevant Database Schema:"]
        lines.append("=" * 60)
        
        for table in tables:
            table_name = table.get("name", "unknown")
            description = table.get("description", "")
            domain = table.get("domain", "")
            
            lines.append(f"\nTable: {table_name}")
            if description:
                lines.append(f"  Description: {description}")
            if domain:
                lines.append(f"  Domain: {domain}")
            
            columns = table.get("columns", [])
            if columns:
                lines.append("  Columns:")
                for col in columns:
                    col_name = col.get("name", "unknown")
                    col_type = col.get("data_type", "")
                    col_desc = col.get("description", "")
                    is_pk = col.get("is_primary_key", False)
                    nullable = col.get("nullable", True)
                    
                    col_line = f"    - {col_name} ({col_type})"
                    if is_pk:
                        col_line += " [PRIMARY KEY]"
                    if not nullable:
                        col_line += " [NOT NULL]"
                    if col_desc:
                        col_line += f" - {col_desc}"
                    lines.append(col_line)
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


def get_relevant_schema(
    query: str,
    database_id: str,
    top_k: int = 10,
    similarity_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Convenience function for schema retrieval (matches tool API style)
    
    Args:
        query: Natural language query
        database_id: Database configuration ID
        top_k: Number of results to return
        similarity_threshold: Minimum similarity score
    
    Returns:
        Schema retrieval results with schema slice
    """
    retriever = SchemaRetriever()
    return retriever.get_relevant_schema(
        query=query,
        database_id=database_id,
        top_k=top_k,
        similarity_threshold=similarity_threshold
    )

