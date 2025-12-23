"""Schema retrieval tool for SQL generation"""


from __future__ import annotations
from typing import List, Dict, Any, Optional
import requests
import logging
import time
import os

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
        
        # If running in Docker and URL uses localhost, replace with host.docker.internal
        if os.getenv("DOCKER_ENV") and "localhost" in self.neo4j_api_url:
            self.neo4j_api_url = self.neo4j_api_url.replace("localhost", "host.docker.internal")
            logger.info(f"Running in Docker - updated Neo4j URL to use host.docker.internal")
        
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
    
    def list_databases(self) -> List[Dict[str, Any]]:
        """
        List all available database configurations from Neo4j backend
        
        Returns:
            List of database configurations with their IDs and metadata
        """
        try:
            # Try different possible endpoints
            endpoints = [
                f"{self.neo4j_api_url}/api/databases",
                f"{self.neo4j_api_url}/api/v1/database/config",
                f"{self.neo4j_api_url}/api/v1/databases"
            ]
            
            for url in endpoints:
                try:
                    logger.info(f"Trying to list databases from: {url}")
                    response = requests.get(url, timeout=10.0)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Handle different response formats
                    if isinstance(data, list):
                        databases = data
                    elif isinstance(data, dict):
                        databases = data.get("databases", data.get("data", []))
                    else:
                        databases = []
                    
                    if databases:
                        logger.info(f"Found {len(databases)} available databases from {url}")
                        return databases
                except requests.exceptions.RequestException:
                    continue
            
            logger.warning("Could not list databases from any endpoint, will need database_id to be provided")
            return []
            
        except Exception as e:
            logger.error(f"Error listing databases: {e}", exc_info=True)
            return []
    
    def find_best_database(
        self,
        query: str,
        candidate_database_ids: Optional[List[str]] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.5
    ) -> Optional[str]:
        """
        Automatically find the best database_id for a query using priority order:
        1. PostgreSQL databases first
        2. CSV databases second  
        3. Other databases (vector similarity) third
        
        Args:
            query: Natural language query
            candidate_database_ids: Optional list of database IDs to search. If None, searches all databases.
            top_k: Number of schema elements to retrieve per database
            similarity_threshold: Minimum similarity score
        
        Returns:
            Best matching database_id, or None if no relevant schema found
        """
        # Get list of databases to search
        if candidate_database_ids is None:
            databases = self.list_databases()
            candidate_database_ids = []
            for db in databases:
                db_id = db.get("id") or db.get("database_id") or db.get("_id")
                if db_id:
                    candidate_database_ids.append(db_id)
        else:
            # If candidate_database_ids provided, still need to fetch full database info for categorization
            databases = self.list_databases()
        
        if not candidate_database_ids:
            logger.warning("No databases available to search - database listing may not be supported by backend API")
            logger.info("Auto-detection requires database listing endpoint. Falling back to provided database_id.")
            return None
        
        # Build dictionary of database info for categorization
        databases_dict = {db.get("id") or db.get("database_id") or db.get("_id"): db 
                         for db in databases 
                         if db.get("id") or db.get("database_id") or db.get("_id")}
        
        # Categorize databases by type (priority order: PostgreSQL -> CSV -> Others)
        postgresql_dbs = []
        csv_dbs = []
        other_dbs = []
        
        for db_id in candidate_database_ids:
            db_info = databases_dict.get(db_id, {})
            db_type = (db_info.get("databaseType") or db_info.get("database_type") or "").lower()
            db_name = (db_info.get("name") or db_id).lower()
            
            if "postgresql" in db_type or "postgres" in db_type:
                postgresql_dbs.append(db_id)
            elif "csv" in db_type or "csv" in db_name or ".csv" in db_name:
                csv_dbs.append(db_id)
            else:
                other_dbs.append(db_id)
        
        logger.info(f"Database priority order: {len(postgresql_dbs)} PostgreSQL, {len(csv_dbs)} CSV, {len(other_dbs)} other")
        logger.info(f"Searching for best database for query: '{query[:100]}...'")
        
        # Extract key terms from query for relevance checking
        query_lower = query.lower()
        # Extract meaningful words (longer than 2 chars, excluding common stop words)
        stop_words = {"how", "many", "what", "where", "when", "who", "which", "are", "is", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        words = query_lower.split()
        query_terms = set(word for word in words if len(word) > 2 and word not in stop_words)
        
        # Also add the full query as a phrase for better matching
        query_phrase = query_lower.strip()
        if len(query_phrase) > 5:  # Only add if meaningful length
            query_terms.add(query_phrase)
        
        logger.debug(f"Extracted query terms: {query_terms}")
        
        # Search in priority order: PostgreSQL -> CSV -> Others
        search_order = [
            ("PostgreSQL", postgresql_dbs),
            ("CSV", csv_dbs),
            ("Other", other_dbs)
        ]
        
        best_database_id = None
        best_score = 0
        best_schema_slice = None
        
        for category_name, db_list in search_order:
            if not db_list:
                continue
                
            logger.info(f"Searching {category_name} databases ({len(db_list)} databases)...")
            
            for db_id in db_list:
                try:
                    schema_result = self.get_relevant_schema(
                        query=query,
                        database_id=db_id,
                        top_k=top_k,
                        similarity_threshold=similarity_threshold
                    )
                    
                    result_count = schema_result.get("result_count", 0)
                    schema_slice = schema_result.get("schema_slice", {})
                    tables = schema_slice.get("tables", [])
                    
                    # Count actual tables found
                    table_count = len(tables) if tables else 0
                    
                    # If we found relevant tables in this category, use it (priority-based)
                    if table_count > 0:
                        # Calculate relevance score: table count + keyword matches
                        score = table_count
                        has_keyword_match = False
                        
                        # Check if table/column names contain query terms (boost score for better matches)
                        query_lower = query.lower()
                        for table in tables:
                            table_name = table.get("name", "").lower()
                            table_desc = table.get("description", "").lower()
                            
                            # Check if full query phrase appears (strongest match)
                            if query_lower in table_name or query_lower in table_desc:
                                score += 5  # Strong boost for full phrase match
                                has_keyword_match = True
                            
                            # Check if query terms appear in table name or description
                            for term in query_terms:
                                if term in table_name or term in table_desc:
                                    score += 2  # Boost for keyword matches
                                    has_keyword_match = True
                            
                            # Check columns too
                            for col in table.get("columns", []):
                                col_name = col.get("name", "").lower()
                                col_desc = col.get("description", "").lower()
                                
                                # Check if full query phrase appears
                                if query_lower in col_name or query_lower in col_desc:
                                    score += 3  # Strong boost for full phrase match in column
                                    has_keyword_match = True
                                
                                # Check individual terms
                                for term in query_terms:
                                    if term in col_name or term in col_desc:
                                        score += 1  # Smaller boost for column matches
                                        has_keyword_match = True
                        
                        # Only accept this database if:
                        # 1. It has keyword matches, OR
                        # 2. It's the last category ("Other") as a fallback
                        is_last_category = category_name == "Other"
                        if has_keyword_match or is_last_category:
                            logger.info(f"Found relevant schema in {category_name} database {db_id}: {table_count} tables, score: {score}, keyword_match: {has_keyword_match}")
                            
                            # Use the first database in this category with relevant tables
                            # (priority order means PostgreSQL wins over CSV, CSV wins over others)
                            best_database_id = db_id
                            best_score = score
                            best_schema_slice = schema_slice
                            break  # Found a match in this category, stop searching
                        else:
                            table_names = [t.get("name", "?") for t in tables[:3]]
                            logger.info(f"{category_name} database {db_id}: Found {table_count} tables ({', '.join(table_names)}) but no keyword matches with query terms {query_terms}, continuing search")
                    else:
                        logger.debug(f"{category_name} database {db_id}: No relevant tables found")
                        
                except Exception as e:
                    logger.debug(f"Error searching {category_name} database {db_id}: {e}")
                    continue
            
            # If we found a match in this category, stop searching other categories
            if best_database_id:
                break
        
        if best_database_id:
            logger.info(f"Selected database: {best_database_id} with relevance score: {best_score}")
            if best_schema_slice:
                table_names = [t.get("name", "?") for t in best_schema_slice.get("tables", [])]
                logger.info(f"Relevant tables: {', '.join(table_names[:5])}")
        else:
            logger.warning(f"No relevant schema found in any database for query: '{query[:100]}...'")
        
        return best_database_id


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

