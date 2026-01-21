"""Text-to-SQL tool that generates SQL from natural language and executes it"""

from __future__ import annotations
from typing import Dict, Any, Optional
import logging
import json
import os

try:
    import requests
except ImportError:
    requests = None

try:
    from ..config import Settings
    from .sql_generator import SQLGenerator
except ImportError:
    import sys
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root.parent))
    from src.news_reporter.config import Settings
    from src.news_reporter.tools_sql.sql_generator import SQLGenerator

logger = logging.getLogger(__name__)


class TextToSQLTool:
    """
    Tool that converts natural language to SQL, executes it, and returns results.
    Combines SQL generation with SQL execution.
    """
    
    def __init__(self, backend_url: Optional[str] = None):
        """
        Initialize Text-to-SQL tool
        
        Args:
            backend_url: URL of the neo4j_backend API (defaults to NEO4J_BACKEND_URL env var or config)
        """
        self.sql_generator = SQLGenerator()
        
        # Get backend URL from parameter, env var, or config
        if backend_url:
            self.backend_url = backend_url.rstrip("/")
        else:
            settings = Settings.load()
            self.backend_url = (
                os.getenv("NEO4J_BACKEND_URL") 
                or settings.neo4j_api_url 
                or "http://localhost:8000"
            ).rstrip("/")
        
        # If running in Docker and URL uses localhost, replace with host.docker.internal
        if os.getenv("DOCKER_ENV") and "localhost" in self.backend_url:
            self.backend_url = self.backend_url.replace("localhost", "host.docker.internal")
            logger.info(f"Running in Docker - updated backend URL to use host.docker.internal")
        
        logger.info(f"TextToSQLTool initialized with backend URL: {self.backend_url}")
    
    def query_database(
        self,
        natural_language_query: str,
        database_id: str,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        auto_detect_database: bool = True
    ) -> Dict[str, Any]:
        """
        Convert natural language to SQL, execute it, and return results.
        
        Args:
            natural_language_query: Natural language query (e.g., "list the names")
            database_id: Database configuration ID (if auto_detect_database is True and this database
                        has no relevant schema, will automatically search for a better database)
            top_k: Number of schema elements to retrieve for SQL generation
            similarity_threshold: Minimum similarity for schema retrieval
            auto_detect_database: If True, automatically find the best database if the provided one
                                has no relevant schema
        
        Returns:
            Dictionary with:
            - success: Whether the operation succeeded
            - generated_sql: The generated SQL query
            - explanation: Explanation of the SQL query
            - confidence: Confidence score from SQL generation (0.0 to 1.0)
            - results: Execution results (rows, columns, row_count) if successful
            - error: Error message if operation failed
            - database_id_used: The actual database_id used (may differ from input if auto-detected)
        """
        try:
            actual_database_id = database_id
            
            # 1. If auto_detect is enabled, always search for the best database
            if auto_detect_database:
                logger.info(f"Auto-detecting best database for query: '{natural_language_query[:100]}...'")
                from .schema_retrieval import SchemaRetriever
                schema_retriever = SchemaRetriever()
                
                # Get initial schema to compare
                initial_schema = schema_retriever.get_relevant_schema(
                    query=natural_language_query,
                    database_id=database_id,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold
                )
                initial_tables = initial_schema.get("schema_slice", {}).get("tables", [])
                initial_table_count = len(initial_tables) if initial_tables else 0
                
                # Search for best database across all databases
                # Use lower similarity threshold for database search to cast wider net
                best_db_id = schema_retriever.find_best_database(
                    query=natural_language_query,
                    candidate_database_ids=None,  # Search all databases
                    top_k=top_k * 2,  # Get more results when searching
                    similarity_threshold=max(0.3, similarity_threshold - 0.2)  # Lower threshold for search
                )
                
                # Use best database if it's different and has more relevant tables
                if best_db_id and best_db_id != database_id:
                    # Get schema from best database to compare
                    best_schema = schema_retriever.get_relevant_schema(
                        query=natural_language_query,
                        database_id=best_db_id,
                        top_k=top_k,
                        similarity_threshold=similarity_threshold
                    )
                    best_tables = best_schema.get("schema_slice", {}).get("tables", [])
                    best_table_count = len(best_tables) if best_tables else 0
                    
                    # Use best database if it has more relevant tables
                    if best_table_count > initial_table_count:
                        logger.info(f"Found better database: {best_db_id} ({best_table_count} tables) vs {database_id} ({initial_table_count} tables)")
                        actual_database_id = best_db_id
                    else:
                        logger.info(f"Initial database {database_id} ({initial_table_count} tables) is best, keeping it")
                elif best_db_id == database_id:
                    logger.info(f"Initial database {database_id} is the best match")
                else:
                    logger.info(f"No better database found, using initial database {database_id}")
            
            # 2. Generate SQL with the selected database_id
            logger.info(f"Generating SQL for query: '{natural_language_query[:100]}...' (database_id: {actual_database_id})")
            sql_result = self.sql_generator.generate_sql(
                query=natural_language_query,
                database_id=actual_database_id,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            if sql_result.get("error") or not sql_result.get("sql"):
                error_msg = sql_result.get("error", "Failed to generate SQL")
                logger.error(f"SQL generation failed: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "generated_sql": None,
                    "explanation": sql_result.get("explanation"),
                    "confidence": sql_result.get("confidence", 0.0),
                    "results": None,
                    "database_id_used": actual_database_id
                }
            
            generated_sql = sql_result["sql"]
            logger.info(f"Generated SQL: {generated_sql[:200]}...")
            
            # 3. Execute SQL via backend API
            logger.info(f"Executing SQL for database_id: {actual_database_id}")
            execution_result = self._execute_sql(actual_database_id, generated_sql)
            
            # 4. Normalize results format for Agent compatibility
            # Backend API returns list for SELECT queries, dict for DML
            # Agent expects dict with 'rows', 'columns', 'row_count' for SELECT
            results_data = execution_result.get("data")
            query_type = execution_result.get("query_type", "SELECT")
            
            if results_data is not None:
                if isinstance(results_data, list):
                    # SELECT query - convert list to dict format
                    if len(results_data) > 0:
                        # Extract columns from first row
                        columns = list(results_data[0].keys()) if isinstance(results_data[0], dict) else []
                        results_data = {
                            "rows": results_data,
                            "columns": columns,
                            "row_count": len(results_data)
                        }
                    else:
                        # Empty result set
                        results_data = {
                            "rows": [],
                            "columns": [],
                            "row_count": 0
                        }
                elif isinstance(results_data, dict):
                    # DML query or already formatted - keep as is
                    # But ensure it has row_count if it doesn't
                    if "row_count" not in results_data and "rows" in results_data:
                        results_data["row_count"] = len(results_data.get("rows", []))
            
            # 5. Combine results
            result = {
                "success": execution_result.get("success", False),
                "generated_sql": generated_sql,
                "explanation": sql_result.get("explanation"),
                "confidence": sql_result.get("confidence", 0.0),
                "results": results_data,
                "error": execution_result.get("error"),
                "database_id_used": actual_database_id
            }
            
            if result["success"]:
                logger.info(f"SQL execution successful. Query type: {execution_result.get('query_type')}")
            else:
                logger.warning(f"SQL execution failed: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.exception(f"Error in query_database: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "generated_sql": None,
                "explanation": None,
                "confidence": 0.0,
                "results": None
            }
    
    def _execute_sql(self, database_id: str, sql_query: str) -> Dict[str, Any]:
        """
        Execute SQL query via neo4j_backend API
        
        Args:
            database_id: Database configuration ID
            sql_query: SQL query to execute
        
        Returns:
            Dictionary with execution results from backend API
        """
        if requests is None:
            logger.error("requests library not available. Install with: pip install requests")
            return {
                "success": False,
                "error": "requests library not available",
                "data": None
            }
        
        url = f"{self.backend_url}/api/databases/{database_id}/execute"
        
        try:
            logger.debug(f"Calling backend API: {url}")
            response = requests.post(
                url,
                json={"query": sql_query},
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            
            # Check HTTP status
            response.raise_for_status()
            
            # Parse JSON response
            result = response.json()
            logger.debug(f"Backend API response: success={result.get('success')}, query_type={result.get('query_type')}")
            return result
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout calling backend API: {url}")
            return {
                "success": False,
                "error": "Backend API request timed out",
                "data": None
            }
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error calling backend API: {url} - {e}")
            return {
                "success": False,
                "error": f"Could not connect to backend API at {self.backend_url}. Please check if the backend is running.",
                "data": None
            }
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error calling backend API: {url} - {e}")
            try:
                error_detail = response.json().get("detail", str(e))
            except:
                error_detail = str(e)
            return {
                "success": False,
                "error": f"Backend API error: {error_detail}",
                "data": None
            }
        except Exception as e:
            logger.exception(f"Unexpected error calling backend API: {e}")
            return {
                "success": False,
                "error": f"Unexpected error executing SQL: {str(e)}",
                "data": None
            }


def query_database(natural_language_query: str, database_id: str) -> str:
    """
    Foundry tool function for text-to-SQL and execution.
    
    This function is registered with Foundry agents as a callable tool.
    It converts natural language to SQL, executes it, and returns results as a JSON string.
    
    The system automatically searches for the best database that contains relevant schema.
    If the provided database_id has no relevant schema, it will automatically search across
    all available databases to find the one with the most relevant tables and columns.
    
    Args:
        natural_language_query: Natural language query (e.g., "how many 4Runner TRD Pro are there?")
        database_id: Database configuration ID stored in Neo4j. The system will try this database first,
                    but will automatically search for a better match if no relevant schema is found.
                    You can provide any database_id as a starting point.
    
    Returns:
        JSON string with SQL query, execution results, and metadata (including database_id_used)
    """
    tool = TextToSQLTool()
    result = tool.query_database(natural_language_query, database_id, auto_detect_database=True)
    
    # Format as JSON string for Foundry
    return json.dumps(result, indent=2)










