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
    from src.news_reporter.tools.schema_retrieval import SchemaRetriever

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
                    "schema_slice": schema_result.get("schema_slice", {}),
                    "explanation": "Failed to retrieve schema information",
                    "confidence": 0.0
                }
            
            schema_slice = schema_result.get("schema_slice", {})
            if not schema_slice.get("tables"):
                return {
                    "sql": None,
                    "error": "No relevant schema elements found",
                    "schema_slice": schema_slice,
                    "explanation": "Could not find relevant tables or columns for the query",
                    "confidence": 0.0
                }
            
            # 2. Format schema for LLM
            schema_text = self.schema_retriever.format_schema_slice_for_llm(schema_slice)
            
            # 3. Generate SQL using LLM
            logger.info("Generating SQL using LLM...")
            sql_prompt = self._build_sql_prompt(query, schema_text)
            
            # Use Azure OpenAI or OpenAI API
            sql_result = self._call_llm(sql_prompt, model=model)
            
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
                "schema_slice": {},
                "explanation": f"Error generating SQL: {str(e)}",
                "confidence": 0.0
            }
    
    def _build_sql_prompt(self, query: str, schema_text: str) -> str:
        """Build prompt for LLM SQL generation"""
        prompt = f"""You are a SQL expert. Generate a SQL query based on the user's natural language request.

Database Schema:
{schema_text}

User Query: {query}

Instructions:
1. Analyze the user's query and identify which tables and columns are needed
2. Generate a valid SQL query that answers the question
3. Use proper SQL syntax for the database type
4. Include appropriate JOINs if multiple tables are needed
5. Add WHERE clauses for filtering if mentioned in the query
6. Use aggregate functions (COUNT, SUM, AVG, etc.) if the query asks for statistics
7. Provide a brief explanation of what the query does

Format your response as JSON:
{{
    "sql": "SELECT ...",
    "explanation": "This query ...",
    "confidence": 0.9
}}

SQL Query:"""
        return prompt
    
    def _call_llm(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Call LLM to generate SQL"""
        try:
            settings = Settings.load()
            
            # Try Azure OpenAI first
            if hasattr(settings, 'azure_openai_endpoint') and settings.azure_openai_endpoint:
                return self._call_azure_openai(prompt, model or settings.azure_openai_deployment)
            else:
                # Fallback to OpenAI
                import openai
                client = openai.OpenAI(api_key=settings.openai_api_key)
                
                response = client.chat.completions.create(
                    model=model or "gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a SQL expert. Generate valid SQL queries based on natural language requests."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                
                content = response.choices[0].message.content
                return self._parse_llm_response(content)
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            return {
                "sql": None,
                "explanation": f"Failed to generate SQL: {str(e)}",
                "confidence": 0.0
            }
    
    def _call_azure_openai(self, prompt: str, deployment: str) -> Dict[str, Any]:
        """Call Azure OpenAI"""
        try:
            import requests
            settings = Settings.load()
            
            url = f"{settings.azure_openai_endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions"
            headers = {
                "api-key": settings.azure_openai_api_key,
                "Content-Type": "application/json"
            }
            params = {
                "api-version": settings.azure_openai_api_version
            }
            data = {
                "messages": [
                    {"role": "system", "content": "You are a SQL expert. Generate valid SQL queries based on natural language requests."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            response = requests.post(url, headers=headers, params=params, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            content = result["choices"][0]["message"]["content"]
            return self._parse_llm_response(content)
        except Exception as e:
            logger.error(f"Azure OpenAI call failed: {e}", exc_info=True)
            raise
    
    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response to extract SQL and explanation"""
        import json
        import re
        
        # Try to extract JSON from response
        json_match = re.search(r'\{[^{}]*"sql"[^{}]*\}', content, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return {
                    "sql": parsed.get("sql", "").strip(),
                    "explanation": parsed.get("explanation", ""),
                    "confidence": parsed.get("confidence", 0.8)
                }
            except json.JSONDecodeError:
                pass
        
        # Fallback: extract SQL from code blocks
        sql_match = re.search(r'```sql\n(.*?)\n```', content, re.DOTALL)
        if sql_match:
            return {
                "sql": sql_match.group(1).strip(),
                "explanation": content.replace(sql_match.group(0), "").strip(),
                "confidence": 0.7
            }
        
        # Last resort: return content as SQL
        return {
            "sql": content.strip(),
            "explanation": "Generated SQL query",
            "confidence": 0.5
        }


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

