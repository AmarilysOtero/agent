"""Schema retrieval tool for SQL generation"""


from __future__ import annotations
from typing import List, Dict, Any, Optional
import requests
import logging
import time
import os
import re
import json

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
            # First, test if the backend is reachable with a health check
            health_url = f"{self.neo4j_api_url}/health"
            try:
                logger.info(f"🔍 SchemaRetriever: Testing backend connectivity with health check: {health_url}")
                print(f"🔍 SchemaRetriever: Testing backend connectivity with health check: {health_url}")
                health_response = requests.get(health_url, timeout=5.0)
                logger.info(f"🔍 SchemaRetriever: Health check status: {health_response.status_code}")
                print(f"🔍 SchemaRetriever: Health check status: {health_response.status_code}")
            except Exception as e:
                logger.warning(f"⚠️ SchemaRetriever: Health check failed (non-critical): {e}")
                print(f"⚠️ SchemaRetriever: Health check failed (non-critical): {e}")
            
            # Primary endpoint - this is the correct one based on the API
            url = f"{self.neo4j_api_url}/api/databases"
            
            logger.info(f"🔍 SchemaRetriever: Attempting to list databases from: {url}")
            print(f"🔍 SchemaRetriever: Attempting to list databases from: {url}")
            
            try:
                # Increase timeout to 60 seconds for slow Neo4j queries
                response = requests.get(url, timeout=60.0)
                logger.info(f"🔍 SchemaRetriever: Response status: {response.status_code}")
                print(f"🔍 SchemaRetriever: Response status: {response.status_code}")
                
                response.raise_for_status()
                data = response.json()
                
                logger.info(f"🔍 SchemaRetriever: Response data type: {type(data)}, length: {len(data) if isinstance(data, list) else 'N/A'}")
                print(f"🔍 SchemaRetriever: Response data type: {type(data)}")
                
                # Handle different response formats
                if isinstance(data, list):
                    databases = data
                elif isinstance(data, dict):
                    databases = data.get("databases", data.get("data", []))
                else:
                    databases = []
                
                if databases:
                    logger.info(f"✅ SchemaRetriever: Found {len(databases)} available databases from {url}")
                    print(f"✅ SchemaRetriever: Found {len(databases)} available databases")
                    for db in databases[:5]:  # Show first 5
                        db_id = db.get("id") or db.get("database_id") or db.get("_id")
                        db_name = db.get("name", "Unknown")
                        db_type = db.get("database_type") or db.get("databaseType", "Unknown")
                        logger.info(f"   - Database: {db_name} (ID: {db_id}, Type: {db_type})")
                        print(f"   - Database: {db_name} (ID: {db_id}, Type: {db_type})")
                    return databases
                else:
                    logger.warning(f"⚠️ SchemaRetriever: Endpoint returned empty list")
                    print(f"⚠️ SchemaRetriever: Endpoint returned empty list")
                    return []
                    
            except requests.exceptions.Timeout:
                logger.error(f"❌ SchemaRetriever: Timeout after 30s trying to list databases from {url}")
                print(f"❌ SchemaRetriever: Timeout after 30s trying to list databases from {url}")
                return []
            except requests.exceptions.ConnectionError as e:
                logger.error(f"❌ SchemaRetriever: Connection error to {url}: {e}")
                print(f"❌ SchemaRetriever: Connection error to {url}: {e}")
                return []
            except requests.exceptions.HTTPError as e:
                logger.error(f"❌ SchemaRetriever: HTTP error from {url}: {e}")
                print(f"❌ SchemaRetriever: HTTP error from {url}: {e}")
                if hasattr(e.response, 'text'):
                    logger.error(f"   Response body: {e.response.text[:500]}")
                    print(f"   Response body: {e.response.text[:500]}")
                return []
            
        except Exception as e:
            logger.error(f"❌ SchemaRetriever: Error listing databases: {e}", exc_info=True)
            print(f"❌ SchemaRetriever: Error listing databases: {e}")
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
        logger.info(f"🔍 Searching for best database for query: '{query[:100]}...'")
        print(f"🔍 SchemaRetriever: Searching for best database for query: '{query[:100]}...'")
        print(f"   Priority order: {len(postgresql_dbs)} PostgreSQL, {len(csv_dbs)} CSV, {len(other_dbs)} other")
        
        # Extract key terms from query for relevance checking
        query_lower = query.lower()
        # Extract meaningful words (longer than 2 chars, excluding common stop words)
        stop_words = {"how", "many", "what", "where", "when", "who", "which", "are", "is", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "tell", "me", "show", "list", "get"}
        words = query_lower.split()
        query_terms = set(word for word in words if len(word) > 2 and word not in stop_words)
        
        # Add singular/plural variations for better matching
        # If query has "names", also check for "name"
        expanded_terms = set(query_terms)
        for term in query_terms:
            # Handle plural -> singular
            if term.endswith('s') and len(term) > 3:
                singular = term[:-1]  # Remove 's'
                expanded_terms.add(singular)
            # Handle singular -> plural (less common, but useful)
            if not term.endswith('s'):
                plural = term + 's'
                expanded_terms.add(plural)
        
        query_terms = expanded_terms
        
        # Also add the full query as a phrase for better matching
        query_phrase = query_lower.strip()
        if len(query_phrase) > 5:  # Only add if meaningful length
            query_terms.add(query_phrase)
        
        logger.info(f"🔑 Extracted query terms for schema matching: {query_terms}")
        logger.info(f"   Full query phrase: '{query_phrase}'")
        print(f"🔑 SchemaRetriever: Extracted query terms: {query_terms}")
        
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
                    logger.debug(f"🔎 Searching {category_name} database {db_id} for query: '{query[:100]}...'")
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
                    
                    # Log ALL schema information found
                    logger.info(f"📋 COMPLETE SCHEMA READ for {category_name} database {db_id}:")
                    logger.info(f"   - Result count: {result_count}")
                    logger.info(f"   - Tables found: {table_count}")
                    print(f"\n{'='*80}")
                    print(f"📋 COMPLETE SCHEMA READ for {category_name} database {db_id}:")
                    print(f"   - Result count: {result_count}")
                    print(f"   - Tables found: {table_count}")
                    print(f"{'='*80}")
                    
                    # Log the raw schema_slice structure
                    try:
                        schema_json = json.dumps(schema_slice, indent=2, default=str)
                        logger.info(f"📄 Raw schema_slice JSON:\n{schema_json}")
                        print(f"\n📄 Raw schema_slice structure:")
                        print(schema_json[:2000])  # First 2000 chars
                        if len(schema_json) > 2000:
                            print(f"... (truncated, full length: {len(schema_json)} chars)")
                    except Exception as e:
                        logger.warning(f"Could not serialize schema_slice: {e}")
                    
                    if table_count > 0:
                        for idx, table in enumerate(tables, 1):
                            table_name = table.get("name", "?")
                            table_desc = table.get("description", "")
                            table_domain = table.get("domain", "")
                            table_id = table.get("id", "?")
                            columns = table.get("columns", [])
                            
                            logger.info(f"\n   📊 Table {idx}: {table_name} (ID: {table_id})")
                            print(f"\n   📊 Table {idx}: {table_name} (ID: {table_id})")
                            
                            if table_desc:
                                logger.info(f"      Description: {table_desc}")
                                print(f"      Description: {table_desc}")
                            else:
                                logger.info(f"      Description: (none)")
                                print(f"      Description: (none)")
                            
                            if table_domain:
                                logger.info(f"      Domain: {table_domain}")
                                print(f"      Domain: {table_domain}")
                            
                            # Log all table metadata
                            table_metadata = {k: v for k, v in table.items() if k not in ['name', 'description', 'domain', 'id', 'columns']}
                            if table_metadata:
                                logger.info(f"      Other metadata: {table_metadata}")
                                print(f"      Other metadata: {table_metadata}")
                            
                            logger.info(f"      Columns ({len(columns)} total):")
                            print(f"      Columns ({len(columns)} total):")
                            
                            # Show ALL columns, not just first 15
                            for col_idx, col in enumerate(columns, 1):
                                col_name = col.get("name", "?")
                                col_type = col.get("data_type", "")
                                col_desc = col.get("description", "")
                                col_id = col.get("id", "?")
                                is_pk = col.get("is_primary_key", False)
                                nullable = col.get("nullable", True)
                                
                                col_info = f"         {col_idx}. {col_name} ({col_type})"
                                if is_pk:
                                    col_info += " [PRIMARY KEY]"
                                if not nullable:
                                    col_info += " [NOT NULL]"
                                if col_id:
                                    col_info += f" (ID: {col_id})"
                                
                                logger.info(col_info)
                                print(col_info)
                                
                                if col_desc:
                                    logger.info(f"            Description: {col_desc}")
                                    print(f"            Description: {col_desc}")
                                
                                # Log all column metadata
                                col_metadata = {k: v for k, v in col.items() if k not in ['name', 'data_type', 'description', 'id', 'is_primary_key', 'nullable']}
                                if col_metadata:
                                    logger.info(f"            Other metadata: {col_metadata}")
                                    print(f"            Other metadata: {col_metadata}")
                    else:
                        logger.info(f"   ⚠️  No tables found in schema_slice")
                        print(f"   ⚠️  No tables found in schema_slice")
                        # Log what was in schema_result anyway
                        logger.info(f"   Schema result keys: {list(schema_result.keys())}")
                        print(f"   Schema result keys: {list(schema_result.keys())}")
                        if "results" in schema_result:
                            results = schema_result.get("results", [])
                            logger.info(f"   Raw results count: {len(results)}")
                            print(f"   Raw results count: {len(results)}")
                            if results:
                                logger.info(f"   First result: {json.dumps(results[0], indent=2, default=str)}")
                                print(f"   First result sample:")
                                print(json.dumps(results[0], indent=2, default=str)[:500])
                    
                    print(f"{'='*80}\n")
                    
                    # If we found relevant tables in this category, use it (priority-based)
                    if table_count > 0:
                        # Calculate relevance score: table count + keyword matches
                        score = table_count
                        has_keyword_match = False
                        match_details = []
                        
                        # Check if table/column names contain query terms (boost score for better matches)
                        query_lower = query.lower()
                        logger.info(f"🔍 Starting keyword matching for {table_count} tables")
                        logger.info(f"   Query: '{query}'")
                        logger.info(f"   Query (lowercase): '{query_lower}'")
                        logger.info(f"   Query terms: {query_terms}")
                        print(f"\n🔍 Starting keyword matching for {table_count} tables")
                        print(f"   Query: '{query}'")
                        print(f"   Query (lowercase): '{query_lower}'")
                        print(f"   Query terms: {query_terms}")
                        print(f"   Tables to check: {[t.get('name', '?') for t in tables]}")
                        
                        try:
                            logger.info(f"   Entering table loop with {len(tables)} tables")
                            print(f"   Entering table loop with {len(tables)} tables", flush=True)
                            
                            for table_idx, table in enumerate(tables, 1):
                                logger.info(f"   [LOOP ITERATION {table_idx}] Processing table {table_idx} of {len(tables)}")
                                print(f"   [LOOP ITERATION {table_idx}] Processing table {table_idx} of {len(tables)}", flush=True)
                                
                                table_name = (table.get("name") or "").lower()
                                table_desc = (table.get("description") or "").lower()
                                logger.info(f"   Checking table: '{table.get('name')}' (lowercase: '{table_name}')")
                                print(f"   Checking table: '{table.get('name')}' (lowercase: '{table_name}')", flush=True)
                            
                            # Check if full query phrase appears (strongest match)
                            if query_lower in table_name or query_lower in table_desc:
                                score += 5  # Strong boost for full phrase match
                                has_keyword_match = True
                                match_details.append(f"Full phrase match in table '{table.get('name')}'")
                                logger.info(f"      ✅ Full phrase match!")
                                print(f"      ✅ Full phrase match!")
                            
                            # Check if query terms appear in table name or description
                            # Use word boundary matching for better accuracy
                            for term in query_terms:
                                # Check exact word match (with word boundaries) - strongest
                                word_pattern = r'\b' + re.escape(term) + r'\b'
                                word_match = re.search(word_pattern, table_name) or re.search(word_pattern, table_desc)
                                if word_match:
                                    score += 3  # Strong boost for exact word match
                                    has_keyword_match = True
                                    match_details.append(f"Exact word '{term}' in table '{table.get('name')}'")
                                    logger.info(f"      ✅ Exact word match: '{term}' in table '{table.get('name')}'")
                                    print(f"      ✅ Exact word match: '{term}' in table '{table.get('name')}'")
                                # Check substring match (for partial matches like "name" in "first_name")
                                elif term in table_name or term in table_desc:
                                    score += 2  # Boost for keyword matches
                                    has_keyword_match = True
                                    match_details.append(f"Keyword '{term}' in table '{table.get('name')}'")
                                    logger.info(f"      ✅ Substring match: '{term}' in table '{table.get('name')}'")
                                    print(f"      ✅ Substring match: '{term}' in table '{table.get('name')}'")
                            
                            # Check columns too
                            columns = table.get("columns", [])
                            logger.info(f"   Checking {len(columns)} columns in table '{table.get('name')}'")
                            print(f"   Checking {len(columns)} columns in table '{table.get('name')}'")
                            
                            for col in columns:
                                col_name = (col.get("name") or "").lower()
                                col_desc = (col.get("description") or "").lower()
                                logger.info(f"      Checking column: '{col.get('name')}' (lowercase: '{col_name}')")
                                print(f"      Checking column: '{col.get('name')}' (lowercase: '{col_name}')")
                                
                                # Check if full query phrase appears
                                if query_lower in col_name or query_lower in col_desc:
                                    score += 4  # Strong boost for full phrase match in column
                                    has_keyword_match = True
                                    match_details.append(f"Full phrase match in column '{col.get('name')}' of table '{table.get('name')}'")
                                    logger.info(f"         ✅ Full phrase match in column!")
                                    print(f"         ✅ Full phrase match in column!")
                                
                                # Check individual terms with word boundary matching
                                for term in query_terms:
                                    # Check exact word match (with word boundaries) - strongest
                                    word_pattern = r'\b' + re.escape(term) + r'\b'
                                    word_match = re.search(word_pattern, col_name) or re.search(word_pattern, col_desc)
                                    if word_match:
                                        score += 2  # Strong boost for exact word match in column
                                        has_keyword_match = True
                                        match_details.append(f"Exact word '{term}' in column '{col.get('name')}' of table '{table.get('name')}'")
                                        logger.info(f"         ✅ Exact word match: '{term}' in column '{col.get('name')}'")
                                        print(f"         ✅ Exact word match: '{term}' in column '{col.get('name')}'")
                                    # Check substring match (for partial matches like "name" in "first_name")
                                    elif term in col_name or term in col_desc:
                                        score += 1  # Smaller boost for column matches
                                        has_keyword_match = True
                                        match_details.append(f"Keyword '{term}' in column '{col.get('name')}' of table '{table.get('name')}'")
                                        logger.info(f"         ✅ Substring match: '{term}' in column '{col.get('name')}'")
                                        print(f"         ✅ Substring match: '{term}' in column '{col.get('name')}'")
                            
                            logger.info(f"   ✅ Completed table loop - checked {len(tables)} tables")
                            print(f"   ✅ Completed table loop - checked {len(tables)} tables", flush=True)
                            
                        except Exception as e:
                            logger.error(f"   ❌ ERROR in keyword matching loop: {e}", exc_info=True)
                            print(f"   ❌ ERROR in keyword matching loop: {e}")
                            import traceback
                            print(traceback.format_exc())
                            # Continue anyway - maybe we can still use the database
                        
                        # Log match details and final state
                        logger.info(f"\n📊 KEYWORD MATCHING SUMMARY:")
                        logger.info(f"   - Total tables checked: {len(tables)}")
                        logger.info(f"   - Total columns checked: {sum(len(t.get('columns', [])) for t in tables)}")
                        logger.info(f"   - Query terms used: {query_terms}")
                        logger.info(f"   - has_keyword_match: {has_keyword_match}")
                        logger.info(f"   - Match details count: {len(match_details)}")
                        logger.info(f"   - Final score: {score}")
                        print(f"\n📊 KEYWORD MATCHING SUMMARY:")
                        print(f"   - Total tables checked: {len(tables)}")
                        print(f"   - Total columns checked: {sum(len(t.get('columns', [])) for t in tables)}")
                        print(f"   - Query terms used: {query_terms}")
                        print(f"   - has_keyword_match: {has_keyword_match}")
                        print(f"   - Match details count: {len(match_details)}")
                        print(f"   - Final score: {score}")
                        
                        if match_details:
                            logger.info(f"   ✅ Keyword matches found ({len(match_details)}):")
                            print(f"   ✅ SchemaRetriever: Keyword matches found ({len(match_details)}):")
                            for detail in match_details[:20]:  # Show first 20 matches
                                logger.info(f"      - {detail}")
                                print(f"      - {detail}")
                            if len(match_details) > 20:
                                logger.info(f"      ... and {len(match_details) - 20} more matches")
                                print(f"      ... and {len(match_details) - 20} more matches")
                        else:
                            logger.warning(f"   ❌ No keyword matches found despite having tables with columns!")
                            logger.warning(f"   This might indicate a bug in the matching logic.")
                            print(f"   ❌ SchemaRetriever: No keyword matches found despite having tables with columns!")
                            print(f"   This might indicate a bug in the matching logic.")
                            
                            # Debug: Show what we're comparing
                            logger.info(f"   DEBUG - Sample comparison:")
                            if tables and tables[0].get("columns"):
                                sample_col = tables[0]["columns"][0]
                                sample_col_name = sample_col.get("name", "").lower()
                                logger.info(f"      Sample column name: '{sample_col.get('name')}' (lowercase: '{sample_col_name}')")
                                logger.info(f"      Query terms: {query_terms}")
                                for term in list(query_terms)[:3]:
                                    logger.info(f"      Does '{term}' in '{sample_col_name}'? {term in sample_col_name}")
                                    word_pattern = r'\b' + re.escape(term) + r'\b'
                                    word_match = re.search(word_pattern, sample_col_name)
                                    logger.info(f"      Regex match '{word_pattern}' in '{sample_col_name}'? {bool(word_match)}")
                                print(f"      Sample column name: '{sample_col.get('name')}' (lowercase: '{sample_col_name}')")
                                print(f"      Query terms: {query_terms}")
                                for term in list(query_terms)[:3]:
                                    print(f"      Does '{term}' in '{sample_col_name}'? {term in sample_col_name}")
                                    word_pattern = r'\b' + re.escape(term) + r'\b'
                                    word_match = re.search(word_pattern, sample_col_name)
                                    print(f"      Regex match '{word_pattern}' in '{sample_col_name}'? {bool(word_match)}")
                        
                        # Only accept this database if:
                        # 1. It has keyword matches, OR
                        # 2. It's the last category ("Other") as a fallback
                        is_last_category = category_name == "Other"
                        if has_keyword_match or is_last_category:
                            logger.info(f"Found relevant schema in {category_name} database {db_id}: {table_count} tables, score: {score}, keyword_match: {has_keyword_match}")
                            print(f"✅ SchemaRetriever: Found relevant schema in {category_name} database {db_id}: {table_count} tables, score: {score}, keyword_match: {has_keyword_match}")
                            
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
            logger.info(f"✅ Selected database: {best_database_id} with relevance score: {best_score}")
            print(f"\n{'='*80}")
            print(f"✅ FINAL SELECTED DATABASE: {best_database_id} (score: {best_score})")
            print(f"{'='*80}")
            
            if best_schema_slice:
                tables = best_schema_slice.get("tables", [])
                table_names = [t.get("name", "?") for t in tables]
                logger.info(f"📋 Relevant tables: {', '.join(table_names)}")
                print(f"📋 Relevant tables ({len(tables)}): {', '.join(table_names)}")
                
                # Show COMPLETE schema details for selected database
                logger.info(f"\n🔍 COMPLETE FINAL SCHEMA FOR SELECTED DATABASE {best_database_id}:")
                logger.info(f"   Total tables: {len(tables)}")
                print(f"\n🔍 COMPLETE FINAL SCHEMA FOR SELECTED DATABASE {best_database_id}:")
                print(f"   Total tables: {len(tables)}")
                
                # Log complete schema_slice as JSON
                try:
                    final_schema_json = json.dumps(best_schema_slice, indent=2, default=str)
                    logger.info(f"📄 Complete final schema_slice JSON:\n{final_schema_json}")
                    print(f"\n📄 Complete final schema_slice JSON:")
                    print(final_schema_json)
                except Exception as e:
                    logger.warning(f"Could not serialize final schema_slice: {e}")
                
                # Show detailed table information
                for idx, table in enumerate(tables, 1):
                    table_name = table.get("name", "?")
                    table_desc = table.get("description", "")
                    table_domain = table.get("domain", "")
                    table_id = table.get("id", "?")
                    columns = table.get("columns", [])
                    
                    logger.info(f"\n   📊 Table {idx}: {table_name} (ID: {table_id})")
                    print(f"\n   📊 Table {idx}: {table_name} (ID: {table_id})")
                    
                    if table_desc:
                        logger.info(f"      Description: {table_desc}")
                        print(f"      Description: {table_desc}")
                    if table_domain:
                        logger.info(f"      Domain: {table_domain}")
                        print(f"      Domain: {table_domain}")
                    
                    logger.info(f"      Columns ({len(columns)} total):")
                    print(f"      Columns ({len(columns)} total):")
                    
                    # Show ALL columns
                    for col_idx, col in enumerate(columns, 1):
                        col_name = col.get("name", "?")
                        col_type = col.get("data_type", "")
                        col_desc = col.get("description", "")
                        col_id = col.get("id", "?")
                        is_pk = col.get("is_primary_key", False)
                        nullable = col.get("nullable", True)
                        
                        col_info = f"         {col_idx}. {col_name} ({col_type})"
                        if is_pk:
                            col_info += " [PRIMARY KEY]"
                        if not nullable:
                            col_info += " [NOT NULL]"
                        if col_id:
                            col_info += f" (ID: {col_id})"
                        
                        logger.info(col_info)
                        print(col_info)
                        
                        if col_desc:
                            logger.info(f"            Description: {col_desc}")
                            print(f"            Description: {col_desc}")
                    
                    # Log all table metadata
                    table_metadata = {k: v for k, v in table.items() if k not in ['name', 'description', 'domain', 'id', 'columns']}
                    if table_metadata:
                        logger.info(f"      Other table metadata: {table_metadata}")
                        print(f"      Other table metadata: {table_metadata}")
                
                print(f"{'='*80}\n")
            else:
                logger.warning(f"⚠️  Selected database {best_database_id} but schema_slice is empty!")
                print(f"⚠️  Selected database {best_database_id} but schema_slice is empty!")
        else:
            logger.warning(f"❌ No relevant schema found in any database for query: '{query[:100]}...'")
            print(f"\n{'='*80}")
            print(f"❌ No relevant schema found in any database for query: '{query[:100]}...'")
            print(f"{'='*80}\n")
        
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

