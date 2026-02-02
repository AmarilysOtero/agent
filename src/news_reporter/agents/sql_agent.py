"""SQL Agent for PostgreSQL, CSV, and Vector search fallback."""

import logging
from typing import Optional, List, Dict, Any

from .utils import (
    infer_header_from_chunk,
    extract_person_names_and_mode,
    filter_results_by_exact_match,
)

logger = logging.getLogger(__name__)


class SQLAgent:
    """SQL agent with fallback chain: PostgreSQL SQL → CSV → Vector search"""
    
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, query: str, database_id: Optional[str] = None) -> str:
        """
        Run query with fallback chain: PostgreSQL SQL → CSV → Vector
        
        Args:
            query: Natural language query
            database_id: Optional database ID (if None, will auto-detect)
        
        Returns:
            Results from first successful source in fallback chain
        """
        logger.info(f"🤖 [AGENT INVOKED] SQLAgent (ID: {self._id}, database_id: {database_id})")
        print(f"🤖 [AGENT INVOKED] SQLAgent (ID: {self._id}, database_id: {database_id})")
        print("SQLAgent: using Foundry agent:", self._id)  # keep print
        
        # Step 1: Try PostgreSQL SQL first
        sql_result = None
        if database_id:
            try:
                from ..tools_sql.text_to_sql_tool import TextToSQLTool
                sql_tool = TextToSQLTool()
                logger.info(f"SQLAgent: Attempting PostgreSQL SQL query with database_id: {database_id}")
                sql_result = sql_tool.query_database(
                    natural_language_query=query,
                    database_id=database_id,
                    auto_detect_database=True
                )
                
                # Check if SQL query was successful and has results
                logger.info(f"🔍 SQLAgent: SQL query result - success: {sql_result.get('success')}, has_results: {bool(sql_result.get('results'))}")
                print(f"🔍 SQLAgent: SQL query result - success: {sql_result.get('success')}, has_results: {bool(sql_result.get('results'))}")
                
                if sql_result.get("success") and sql_result.get("results"):
                    results_data = sql_result.get("results", {})
                    row_count = results_data.get("row_count", 0)
                    
                    if row_count > 0:
                        logger.info(f"✅ SQLAgent: PostgreSQL SQL query successful with {row_count} rows")
                        print(f"✅ SQLAgent: PostgreSQL SQL query successful with {row_count} rows")
                        # Format SQL results
                        findings = []
                        findings.append(
                            f"═══════════════════════════════════════════════════════════\n"
                            f"**POSTGRESQL SQL QUERY RESULTS:**\n"
                            f"Database: {sql_result.get('database_id_used', database_id)}\n"
                            f"Rows: {row_count}\n"
                            f"═══════════════════════════════════════════════════════════\n\n"
                        )
                        
                        # Add result rows
                        rows = results_data.get("rows", [])
                        columns = results_data.get("columns", [])
                        
                        if rows:
                            findings.append("**Results:**\n")
                            for i, row in enumerate(rows[:20], 1):  # Show first 20 rows
                                row_str = " | ".join([f"{col}: {row.get(col, 'N/A')}" for col in columns])
                                findings.append(f"{i}. {row_str}")
                            
                            if len(rows) > 20:
                                findings.append(f"\n... and {len(rows) - 20} more rows")
                        
                        if sql_result.get("explanation"):
                            findings.append(f"\n**Explanation:** {sql_result.get('explanation')}")
                        
                        return "\n".join(findings)
                    else:
                        logger.warning(f"⚠️ SQLAgent: PostgreSQL SQL returned 0 rows, falling back to CSV")
                        print(f"⚠️ SQLAgent: PostgreSQL SQL returned 0 rows, falling back to CSV")
                else:
                    error_msg = sql_result.get("error", "Unknown error")
                    logger.warning(f"⚠️ SQLAgent: PostgreSQL SQL failed: {error_msg}, falling back to CSV")
                    print(f"⚠️ SQLAgent: PostgreSQL SQL failed: {error_msg}, falling back to CSV")
            except Exception as e:
                logger.warning(f"⚠️ SQLAgent: PostgreSQL SQL query failed: {e}, falling back to CSV", exc_info=True)
                print(f"⚠️ SQLAgent: PostgreSQL SQL query failed: {e}, falling back to CSV")
        else:
            logger.warning(f"⚠️ SQLAgent: No database_id provided, skipping SQL and trying CSV")
            print(f"⚠️ SQLAgent: No database_id provided, skipping SQL and trying CSV")
        
        # Step 2: Try CSV query
        try:
            from ..tools.neo4j_graphrag import graphrag_search
            from ..tools.csv_query import (
                query_requires_exact_numbers,
                query_requires_list,
                extract_csv_path_from_rag_results,
                extract_filter_value_from_query,
                sum_numeric_columns,
                get_distinct_values,
                is_csv_specific_query
            )
            csv_query_available = True
        except ImportError:
            logger.warning("CSV query tools not available, skipping CSV queries")
            csv_query_available = False
        
        if csv_query_available:
            # Get some initial results to find CSV path
            person_names, is_person_query = extract_person_names_and_mode(query)
            keywords = person_names if (is_person_query and person_names) else None
            initial_results = graphrag_search(
                query=query,
                top_k=5,
                similarity_threshold=0.7,
                keywords=keywords,
                keyword_match_type="any"
            )
            
            if initial_results:
                csv_path = extract_csv_path_from_rag_results(initial_results)
                
                if csv_path:
                    needs_exact = query_requires_exact_numbers(query)
                    needs_list = query_requires_list(query)
                    
                    # Try CSV exact numbers query
                    if needs_exact:
                        try:
                            filter_columns = ['Model', 'Product', 'Category', 'Item', 'Name', 'Type']
                            filters = {}
                            
                            for col in filter_columns:
                                value = extract_filter_value_from_query(query, col, initial_results)
                                if value:
                                    filters[col] = value
                                    break
                            
                            if filters:
                                exact_result = sum_numeric_columns(
                                    file_path=csv_path,
                                    filters=filters,
                                    exclude_columns=list(filters.keys()) + ['Factory Location', 'Location', 'Region']
                                )
                                
                                if 'error' not in exact_result and exact_result.get('total', 0) > 0:
                                    filter_str = ', '.join([f"{k}={v}" for k, v in filters.items()])
                                    logger.info(f"✅ SQLAgent: CSV query successful: total={exact_result['total']}")
                                    return (
                                        f"═══════════════════════════════════════════════════════════\n"
                                        f"**CSV QUERY RESULTS:**\n"
                                        f"Total = {exact_result['total']:,} units\n"
                                        f"Filter: {filter_str}\n"
                                        f"Summed across {exact_result.get('columns_summed', 0)} numeric columns\n"
                                        f"═══════════════════════════════════════════════════════════\n"
                                    )
                        except Exception as e:
                            logger.warning(f"SQLAgent: CSV exact query failed: {e}, falling back to vector")
                    
                    # Try CSV list query
                    if needs_list:
                        try:
                            list_columns = ['Model', 'Product', 'Category', 'Item', 'Name', 'Type']
                            column_to_list = None
                            query_lower = query.lower()
                            
                            for col in list_columns:
                                if col.lower() in query_lower:
                                    column_to_list = col
                                    break
                            
                            if not column_to_list:
                                column_to_list = 'Model'  # Default
                            
                            list_result = get_distinct_values(file_path=csv_path, column=column_to_list)
                            
                            if 'error' not in list_result and list_result.get('values'):
                                values_str = ', '.join(list_result['values'][:20])
                                if list_result['count'] > 20:
                                    values_str += f", ... and {list_result['count'] - 20} more"
                                logger.info(f"✅ SQLAgent: CSV list query successful: {list_result['count']} values")
                                return (
                                    f"═══════════════════════════════════════════════════════════\n"
                                    f"**CSV LIST RESULTS ({column_to_list} column):**\n"
                                    f"Total distinct values: {list_result['count']}\n"
                                    f"Values: {values_str}\n"
                                    f"═══════════════════════════════════════════════════════════\n"
                                )
                        except Exception as e:
                            logger.warning(f"SQLAgent: CSV list query failed: {e}, falling back to vector")
        
        # Step 3: Fall back to Vector/GraphRAG search
        logger.info("SQLAgent: Falling back to Vector/GraphRAG search")
        from ..tools.neo4j_graphrag import graphrag_search
        
        # Use context-aware name extraction
        person_names, is_person_query = extract_person_names_and_mode(query)
        keywords = person_names if (is_person_query and person_names) else None
        keyword_boost = 0.4 if keywords else 0.0
        
        results = graphrag_search(
            query=query,
            top_k=12,
            similarity_threshold=0.75,
            keywords=keywords,
            keyword_match_type="any",
            keyword_boost=keyword_boost
        )
        
        if not results:
            return "No results found in PostgreSQL SQL, CSV, or Vector search."
        
        # Pass is_person_query to filter so it knows whether to enforce name matching
        filtered_results = filter_results_by_exact_match(
            results, query, min_similarity=0.3, 
            is_person_query=is_person_query, person_names=person_names
        )
        filtered_results = filtered_results[:8]
        
        if not filtered_results:
            return "No relevant results found after filtering in Vector search."
        
        findings = []
        findings.append("**Vector/GraphRAG Search Results (fallback):**\n")
        
        for res in filtered_results:
            text = res.get("text", "").replace("\n", " ")
            file_path = res.get("file_path", "")
            file_name = res.get("file_name", "Unknown")
            similarity = res.get("similarity", 0.0)
            
            if file_path and file_path.lower().endswith('.csv'):
                source_note = "[CSV Data]"
            elif file_path and file_path.lower().endswith('.pdf'):
                source_note = "[PDF Document]"
            elif file_path and file_path.lower().endswith(('.doc', '.docx')):
                source_note = "[Word Document]"
            else:
                source_note = "[Document]"
            
            source_info = file_path if file_path else file_name
            if len(text) > 2000:
                findings.append(f"- {source_note} {source_info}: {text[:2000]}...")
                logger.info(f"📝 Truncated chunk text from {len(text)} to 2000 characters for file: {file_name}")
            else:
                findings.append(f"- {source_note} {source_info}: {text}")
                logger.info(f"📝 Included full chunk text ({len(text)} characters) for file: {file_name}")
        
        return "\n".join(findings)
