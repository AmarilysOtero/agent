"""SQL generation tool using LLM with schema slices"""

from __future__ import annotations
from typing import Dict, Any, Optional
import logging

try:
    from ..config import Settings
    from .schema_retrieval import SchemaRetriever
except ImportError:
    import sys
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root.parent))
    from src.news_reporter.config import Settings
    from src.news_reporter.tools_sql.schema_retrieval import SchemaRetriever

logger = logging.getLogger(__name__)


class SQLGenerator:
    """
    Generate SQL from natural language using LLM and schema slices
    """
    
    def __init__(self):
        """Initialize SQL generator"""
        settings = Settings.load()
        self.schema_retriever = SchemaRetriever()
    
    def generate_sql(
        self,
        query: str,
        database_id: str,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate SQL from natural language query
        
        Args:
            query: Natural language query
            database_id: Database configuration ID
            top_k: Number of schema elements to retrieve
            similarity_threshold: Minimum similarity for schema retrieval
            model: LLM model to use (optional, uses default from settings)
        
        Returns:
            Dictionary with:
            - sql: Generated SQL query
            - schema_slice: Schema slice used
            - explanation: Explanation of the query
            - confidence: Confidence score (0.0 to 1.0)
        """
        try:
            # 1. Retrieve relevant schema
            logger.info(f"Retrieving schema for query: '{query[:100]}...'")
            schema_result = self.schema_retriever.get_relevant_schema(
                query=query,
                database_id=database_id,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            if schema_result.get("error"):
                return {
                    "sql": None,
                    "error": schema_result.get("error"),
                    "schema_slice": {"tables": []},
                    "explanation": "Failed to retrieve schema information",
                    "confidence": 0.0
                }
            
            schema_slice = schema_result.get("schema_slice", {})
            if "tables" not in schema_slice:
                schema_slice["tables"] = []
            
            if not schema_slice.get("tables"):
                return {
                    "sql": None,
                    "error": "No relevant schema elements found",
                    "schema_slice": {"tables": []},
                    "explanation": "Could not find relevant tables or columns for the query",
                    "confidence": 0.0
                }
            
            # 2. Format schema for LLM
            schema_text = self.schema_retriever.format_schema_slice_for_llm(schema_slice)
            
            # 3. Generate SQL using LLM
            logger.info("Generating SQL using LLM...")
            sql_prompt = self._build_sql_prompt(query, schema_text, database_id)
            
            # Use Azure OpenAI or OpenAI API
            sql_result = self._call_llm(sql_prompt, model=model)
            
            # If LLM failed to generate valid SQL, try simple fallback generation
            if not sql_result.get("sql") or sql_result.get("sql") == "SELECT ...":
                logger.warning("LLM returned invalid SQL, attempting fallback generation")
                fallback_sql = self._generate_fallback_sql(query, schema_slice)
                if fallback_sql:
                    sql_result["sql"] = fallback_sql
                    sql_result["explanation"] = "Generated SQL using fallback logic"
                    sql_result["confidence"] = 0.6
            
            # Post-process SQL: quote identifiers for PostgreSQL
            if sql_result.get("sql"):
                sql_result["sql"] = self._quote_identifiers(sql_result["sql"], schema_slice, database_id)
            
            return {
                "sql": sql_result.get("sql"),
                "schema_slice": schema_slice,
                "explanation": sql_result.get("explanation", ""),
                "confidence": sql_result.get("confidence", 0.8),
                "query": query
            }
            
        except Exception as e:
            logger.error(f"SQL generation error: {e}", exc_info=True)
            return {
                "sql": None,
                "error": str(e),
                "schema_slice": {"tables": []},
                "explanation": f"Error generating SQL: {str(e)}",
                "confidence": 0.0
            }
    
    def _build_sql_prompt(self, query: str, schema_text: str, database_id: str) -> str:
        """Build prompt for LLM SQL generation"""
        # Check if this is PostgreSQL (which requires quoted identifiers for case-sensitive names)
        is_postgresql = False
        try:
            schema_result = self.schema_retriever.get_relevant_schema(
                query="", database_id=database_id, top_k=1, similarity_threshold=0.0
            )
            # Try to get database type from schema retriever
            # This is a workaround - ideally we'd pass db_type directly
            is_postgresql = True  # Assume PostgreSQL for now, we'll handle it in post-processing
        except:
            pass
        
        postgresql_note = ""
        if is_postgresql:
            postgresql_note = "\n- For PostgreSQL: Use quoted identifiers (double quotes) for table and column names to preserve case sensitivity, e.g., SELECT name FROM \"Employee\""
        
        prompt = f"""Generate a SQL query for this request.

Database Schema:
{schema_text}

User Request: {query}

CRITICAL: You must respond with ONLY valid JSON, nothing else. Do not include the prompt, instructions, or any other text.

Example response format (replace with actual SQL):
{{
    "sql": "SELECT name FROM \"Employee\"",
    "explanation": "This query retrieves all names from the Employee table",
    "confidence": 0.9
}}

IMPORTANT RULES:
- The "sql" field must contain a COMPLETE, executable SQL query (NOT "SELECT ..." or any placeholder)
- Use the exact table and column names from the schema above{postgresql_note}
- For PostgreSQL databases, ALWAYS use double quotes around table and column names: SELECT "name" FROM "Employee"
- For "list the names" or similar queries, use: SELECT "name" FROM "table_name"
- Return ONLY the JSON object, no other text before or after"""
        return prompt
    
    def _quote_identifiers(self, sql: str, schema_slice: Dict[str, Any], database_id: str) -> str:
        """
        Post-process SQL to add quotes around identifiers for PostgreSQL.
        This ensures case-sensitive table and column names work correctly.
        """
        import re
        
        if not sql:
            return sql
        
        # Check if database is PostgreSQL (we'll assume it is for now, or check via schema_retriever)
        # For now, always apply quoting as it's safe for PostgreSQL and won't break other DBs if done carefully
        try:
            # Get all table and column names from schema
            table_names = []
            column_names = []
            
            for table in schema_slice.get("tables", []):
                table_name = table.get("name")
                if table_name:
                    table_names.append(table_name)
                    # Also add lowercase version for matching
                    table_names.append(table_name.lower())
                
                for col in table.get("columns", []):
                    col_name = col.get("name") if isinstance(col, dict) else str(col)
                    if col_name:
                        column_names.append(col_name)
                        column_names.append(col_name.lower())
            
            # Remove duplicates while preserving order
            table_names = list(dict.fromkeys(table_names))
            column_names = list(dict.fromkeys(column_names))
            
            # Sort by length (longest first) to avoid partial matches
            table_names.sort(key=len, reverse=True)
            column_names.sort(key=len, reverse=True)
            
            result = sql
            
            # First, check if SQL already has quoted identifiers - if so, don't add more
            # Simple heuristic: if we see patterns like "identifier", assume it's already quoted
            has_quoted_identifiers = bool(re.search(r'"[A-Za-z_][A-Za-z0-9_]*"', result))
            
            if has_quoted_identifiers:
                logger.info("SQL already contains quoted identifiers, skipping quote addition")
                return result
            
            # Build a map of lowercase names to original case names
            table_name_map = {}
            for table in schema_slice.get("tables", []):
                table_name = table.get("name")
                if table_name:
                    table_name_map[table_name.lower()] = table_name
            
            column_name_map = {}
            for table in schema_slice.get("tables", []):
                for col in table.get("columns", []):
                    col_name = col.get("name") if isinstance(col, dict) else str(col)
                    if col_name:
                        column_name_map[col_name.lower()] = col_name
            
            # Quote table names (case-insensitive match, but preserve original case)
            # Only match unquoted identifiers
            for table_lower, table_original in table_name_map.items():
                # Match table name that is NOT already quoted
                # Pattern: not preceded by quote, word boundary, table name, word boundary, not followed by quote
                pattern = r'(?<!")\b' + re.escape(table_lower) + r'\b(?!")'
                
                def make_table_replacer(orig_name):
                    def replace_table(match):
                        return f'"{orig_name}"'
                    return replace_table
                
                result = re.sub(pattern, make_table_replacer(table_original), result, flags=re.IGNORECASE)
            
            # Quote column names
            for col_lower, col_original in column_name_map.items():
                # Match column name that is NOT already quoted
                pattern = r'(?<!")\b' + re.escape(col_lower) + r'\b(?!")'
                
                def make_column_replacer(orig_name):
                    def replace_column(match):
                        return f'"{orig_name}"'
                    return replace_column
                
                result = re.sub(pattern, make_column_replacer(col_original), result, flags=re.IGNORECASE)
            
            logger.info(f"Quoted identifiers in SQL: {sql[:100]} -> {result[:100]}")
            return result
            
        except Exception as e:
            logger.warning(f"Error quoting identifiers in SQL: {e}. Returning original SQL.")
            return sql
    
    def _call_llm(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Call LLM to generate SQL using Foundry with timeout"""
        import threading
        import time
        
        try:
            from ..foundry_runner import run_foundry_agent
            
            settings = Settings.load()
            
            agent_id = settings.agent_id_aisearch if hasattr(settings, 'agent_id_aisearch') and settings.agent_id_aisearch else None
            
            if not agent_id:
                raise ValueError("No Foundry agent ID available for SQL generation. Please configure agent_id_aisearch or create a dedicated SQL agent.")
            
            # IMPORTANT: Tell the LLM to generate ACTUAL SQL, not placeholders
            system_prompt = """You are a SQL expert. Generate valid, executable SQL queries based on natural language requests. 
IMPORTANT: You MUST provide the complete, actual SQL query in the "sql" field - NOT a placeholder like "SELECT ...". 
The SQL must be ready to execute. Format your response as JSON: {"sql": "SELECT name FROM Employee", "explanation": "...", "confidence": 0.9}"""
            
            result_container = {"content": None, "error": None, "completed": False}
            
            def run_agent():
                try:
                    logger.info(f"Starting Foundry agent call for SQL generation (timeout: 90s)")
                    start_time = time.time()
                    result_container["content"] = run_foundry_agent(
                        agent_id=agent_id,
                        user_content=prompt,
                        system_hint=system_prompt
                    )
                    elapsed = time.time() - start_time
                    logger.info(f"Foundry agent call completed in {elapsed:.2f}s")
                    result_container["completed"] = True
                except Exception as e:
                    result_container["error"] = e
                    result_container["completed"] = True
                    logger.error(f"Foundry agent call failed: {e}", exc_info=True)
            
            thread = threading.Thread(target=run_agent, daemon=True)
            thread.start()
            thread.join(timeout=90.0)
            
            if thread.is_alive():
                logger.error("LLM call timed out after 90 seconds - Foundry agent may be stuck")
                return {
                    "sql": None,
                    "explanation": "SQL generation timed out after 90 seconds. The query may be too complex or the LLM service is slow. Please try again or simplify your query.",
                    "confidence": 0.0
                }
            
            if not result_container["completed"]:
                logger.error("LLM call thread completed but result_container not marked as completed")
                return {
                    "sql": None,
                    "explanation": "SQL generation encountered an unexpected error. Please try again.",
                    "confidence": 0.0
                }
            
            if result_container["error"]:
                raise result_container["error"]
            
            if result_container["content"] is None:
                raise RuntimeError("LLM call returned no content")
            
            return self._parse_llm_response(result_container["content"])
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            return {
                "sql": None,
                "explanation": f"Failed to generate SQL: {str(e)}",
                "confidence": 0.0
            }
    
    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response to extract SQL and explanation"""
        import json
        import re
        
        logger.debug(f"Parsing LLM response (length: {len(content)}): {content[:500]}...")
        
        # First, try to find JSON that contains "sql" key - this is most reliable
        # Look for the pattern: "sql": "SELECT ..."
        sql_json_match = re.search(r'"sql"\s*:\s*"([^"]+)"', content, re.IGNORECASE | re.DOTALL)
        if sql_json_match:
            # Found SQL in JSON format, now extract the full JSON object
            sql_value = sql_json_match.group(1)
            # Find the JSON object containing this SQL
            # Look backwards and forwards from the match to find the full JSON object
            match_start = sql_json_match.start()
            match_end = sql_json_match.end()
            
            # Find the opening brace before this match
            brace_start = content.rfind('{', 0, match_start)
            if brace_start >= 0:
                # Find the closing brace after this match
                brace_count = 1
                brace_end = match_end
                while brace_end < len(content) and brace_count > 0:
                    if content[brace_end] == '{':
                        brace_count += 1
                    elif content[brace_end] == '}':
                        brace_count -= 1
                    brace_end += 1
                
                if brace_count == 0:
                    try:
                        json_str = content[brace_start:brace_end]
                        parsed = json.loads(json_str)
                        sql = parsed.get("sql", "").strip()
                        
                        # Clean SQL - remove quotes if present
                        if sql.startswith('"') and sql.endswith('"'):
                            sql = sql[1:-1]
                        
                        if sql and sql != "SELECT ..." and not sql.startswith("SELECT ...") and len(sql) > 10:
                            if re.match(r'^\s*SELECT\s+', sql, re.IGNORECASE):
                                logger.info(f"Extracted SQL from JSON (method 1): {sql[:100]}...")
                                return {
                                    "sql": sql,
                                    "explanation": parsed.get("explanation", ""),
                                    "confidence": parsed.get("confidence", 0.8)
                                }
                    except (json.JSONDecodeError, KeyError, AttributeError) as e:
                        logger.debug(f"JSON extraction from SQL match failed: {e}")
        
        # Remove common prompt prefixes that LLM might include
        content_cleaned = content
        prompt_markers = [
            r'You are a SQL expert[^}]*',
            r'Database Schema:[^}]*',
            r'User Query:[^}]*',
            r'User Request:[^}]*',
            r'Instructions:[^}]*',
            r'Format your response[^}]*',
            r'SQL Query:[^}]*',
            r'CRITICAL:[^}]*',
            r'IMPORTANT RULES:[^}]*',
            r'Example response format[^}]*',
        ]
        for marker in prompt_markers:
            content_cleaned = re.sub(marker, '', content_cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Try to find JSON object with balanced braces (most reliable)
        brace_count = 0
        start_idx = -1
        json_candidates = []
        
        for i, char in enumerate(content_cleaned):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx >= 0:
                    json_candidates.append((start_idx, i+1))
                    start_idx = -1
        
        # Try each JSON candidate
        for start, end in json_candidates:
            try:
                json_str = content_cleaned[start:end]
                parsed = json.loads(json_str)
                sql = parsed.get("sql", "").strip()
                
                # Check if SQL is valid (not a placeholder)
                if sql and sql != "SELECT ..." and not sql.startswith("SELECT ...") and len(sql) > 10:
                    if re.match(r'^\s*SELECT\s+', sql, re.IGNORECASE):
                        logger.info(f"Extracted SQL from JSON: {sql[:100]}...")
                        return {
                            "sql": sql,
                            "explanation": parsed.get("explanation", ""),
                            "confidence": parsed.get("confidence", 0.8)
                        }
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                logger.debug(f"JSON candidate failed: {e}")
                continue
        
        # Try regex patterns for JSON extraction - handle escaped quotes and newlines
        json_patterns = [
            # Pattern 1: Full JSON with all fields
            r'\{\s*"sql"\s*:\s*"((?:[^"\\]|\\.)+)"\s*,\s*"explanation"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"confidence"\s*:\s*([\d.]+)\s*\}',
            # Pattern 2: JSON with just sql field
            r'\{\s*"sql"\s*:\s*"((?:[^"\\]|\\.)+)"[^}]*\}',
            # Pattern 3: More flexible - handle multiline SQL
            r'"sql"\s*:\s*"((?:[^"\\]|\\.)+)"',
        ]
        
        for pattern in json_patterns:
            json_match = re.search(pattern, content_cleaned, re.DOTALL | re.IGNORECASE)
            if json_match:
                try:
                    sql = json_match.group(1).strip()
                    # Unescape JSON string properly
                    sql = sql.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
                    
                    # Remove surrounding quotes if present
                    if sql.startswith('"') and sql.endswith('"'):
                        sql = sql[1:-1]
                    
                    if sql and sql != "SELECT ..." and not sql.startswith("SELECT ...") and len(sql) > 10:
                        if re.match(r'^\s*SELECT\s+', sql, re.IGNORECASE):
                            explanation = json_match.group(2) if len(json_match.groups()) > 1 and json_match.lastindex >= 2 else ""
                            confidence = float(json_match.group(3)) if len(json_match.groups()) > 2 and json_match.lastindex >= 3 else 0.8
                            logger.info(f"Extracted SQL from regex JSON: {sql[:100]}...")
                            return {
                                "sql": sql,
                                "explanation": explanation,
                                "confidence": confidence
                            }
                except (IndexError, ValueError) as e:
                    logger.debug(f"Regex JSON extraction failed: {e}")
                    continue
        
        # Fallback: extract SQL from code blocks
        sql_patterns = [
            r'```sql\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```',
        ]
        
        for pattern in sql_patterns:
            sql_match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if sql_match:
                sql = sql_match.group(1).strip()
                if sql and sql != "SELECT ..." and len(sql) > 10:
                    if re.match(r'^\s*SELECT\s+', sql, re.IGNORECASE):
                        logger.info(f"Extracted SQL from code block: {sql[:100]}...")
                        return {
                            "sql": sql,
                            "explanation": "Generated SQL query",
                            "confidence": 0.7
                        }
        
        # Look for SELECT statement directly
        select_match = re.search(r'(SELECT\s+[^;\n]+(?:FROM|JOIN)[^;]+)', content, re.DOTALL | re.IGNORECASE)
        if select_match:
            sql = select_match.group(1).strip()
            if sql and sql != "SELECT ..." and len(sql) > 10:
                logger.info(f"Extracted SQL from direct SELECT: {sql[:100]}...")
                return {
                    "sql": sql,
                    "explanation": "Generated SQL query",
                    "confidence": 0.6
                }
        
        logger.warning(f"Could not extract valid SQL from LLM response. Full content: {content}")
        return {
            "sql": None,
            "explanation": "Could not parse SQL from LLM response. The response may not contain valid SQL.",
            "confidence": 0.0
        }
    
    def _generate_fallback_sql(self, query: str, schema_slice: Dict[str, Any]) -> Optional[str]:
        """Generate simple SQL as fallback when LLM fails"""
        import re
        
        query_lower = query.lower()
        tables = schema_slice.get("tables", [])
        
        if not tables:
            return None
        
        # "list the names" or "show names" -> SELECT name FROM table
        if re.search(r'\b(list|show|get|display|find)\s+(the\s+)?names?\b', query_lower):
            for table in tables:
                table_name = table.get("name", "")
                columns = table.get("columns", [])
                for col in columns:
                    if col.get("name", "").lower() == "name":
                        return f'SELECT name FROM "{table_name}"'
            # If no "name" column, use first table and first column
            if tables:
                table_name = tables[0].get("name", "")
                columns = tables[0].get("columns", [])
                if columns:
                    col_name = columns[0].get("name", "")
                    return f'SELECT {col_name} FROM "{table_name}"'
        
        # "list all" or "show all" -> SELECT * FROM table
        if re.search(r'\b(list|show|get|display)\s+(all|everything)\b', query_lower):
            if tables:
                table_name = tables[0].get("name", "")
                return f'SELECT * FROM "{table_name}"'
        
        # Default: SELECT all columns from first table
        if tables:
            table_name = tables[0].get("name", "")
            columns = tables[0].get("columns", [])
            if columns:
                col_names = [col.get("name", "") for col in columns]
                cols_str = ", ".join(f'"{col}"' for col in col_names)
                return f'SELECT {cols_str} FROM "{table_name}"'
            else:
                return f'SELECT * FROM "{table_name}"'
        
        return None


def generate_sql(
    query: str,
    database_id: str,
    top_k: int = 10,
    similarity_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Convenience function for SQL generation
    
    Args:
        query: Natural language query
        database_id: Database configuration ID
        top_k: Number of schema elements to retrieve
        similarity_threshold: Minimum similarity for schema retrieval
    
    Returns:
        SQL generation result with SQL query and metadata
    """
    generator = SQLGenerator()
    return generator.generate_sql(
        query=query,
        database_id=database_id,
        top_k=top_k,
        similarity_threshold=similarity_threshold
    )
