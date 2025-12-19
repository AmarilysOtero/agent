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
        
        logger.info(f"TextToSQLTool initialized with backend URL: {self.backend_url}")
    
    def query_database(
        self,
        natural_language_query: str,
        database_id: str,
        top_k: int = 10,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Convert natural language to SQL, execute it, and return results.
        
        Args:
            natural_language_query: Natural language query (e.g., "list the names")
            database_id: Database configuration ID
            top_k: Number of schema elements to retrieve for SQL generation
            similarity_threshold: Minimum similarity for schema retrieval
        
        Returns:
            Dictionary with:
            - success: Whether the operation succeeded
            - generated_sql: The generated SQL query
            - explanation: Explanation of the SQL query
            - confidence: Confidence score from SQL generation (0.0 to 1.0)
            - results: Execution results (rows, columns, row_count) if successful
            - error: Error message if operation failed
        """
        try:
            # 1. Generate SQL from natural language
            logger.info(f"Generating SQL for query: '{natural_language_query[:100]}...' (database_id: {database_id})")
            sql_result = self.sql_generator.generate_sql(
                query=natural_language_query,
                database_id=database_id,
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
                    "results": None
                }
            
            generated_sql = sql_result["sql"]
            logger.info(f"Generated SQL: {generated_sql[:200]}...")
            
            # 2. Execute SQL via backend API
            logger.info(f"Executing SQL for database_id: {database_id}")
            execution_result = self._execute_sql(database_id, generated_sql)
            
            # 3. Combine results
            result = {
                "success": execution_result.get("success", False),
                "generated_sql": generated_sql,
                "explanation": sql_result.get("explanation"),
                "confidence": sql_result.get("confidence", 0.0),
                "results": execution_result.get("data"),
                "error": execution_result.get("error")
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
    
    Args:
        natural_language_query: Natural language query (e.g., "list the names")
        database_id: Database configuration ID
    
    Returns:
        JSON string with SQL query, execution results, and metadata
    """
    tool = TextToSQLTool()
    result = tool.query_database(natural_language_query, database_id)
    
    # Format as JSON string for Foundry
    return json.dumps(result, indent=2)
