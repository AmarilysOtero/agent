"""Neo4j GraphRAG Agent for Neo4j GraphRAG retrieval."""

import logging
from typing import Optional, List, Dict, Any, Tuple, Union

from .utils import (
    infer_header_from_chunk,
    extract_person_names_and_mode,
    filter_results_by_exact_match,
)
from ..tools.header_vocab import extract_attribute_keywords

logger = logging.getLogger(__name__)


class Neo4jGraphRAGAgent:
    """Search agent using Neo4j GraphRAG retrieval"""
    
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(
        self,
        query: str,
        high_recall_mode: bool = False,
        return_results: bool = False,
    ) -> Union[str, Tuple[str, List[Dict[str, Any]]]]:
        logger.info(f"🤖 [AGENT INVOKED] Neo4jGraphRAGAgent (ID: {self._id})")
        print(f"🤖 [AGENT INVOKED] Neo4jGraphRAGAgent (ID: {self._id})")
        print("Neo4jGraphRAGAgent: using Foundry agent:", self._id)  # keep print
        from ..tools.neo4j_graphrag import graphrag_search
        
        # Import CSV query tools for exact numerical calculations and list queries
        try:
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
        
        # Extract person names and determine if query is person-centric
        person_names, is_person_query = extract_person_names_and_mode(query)
        
        # Build keywords for search (RLM-style: person names + attribute keywords, same as AiSearchAgent)
        keywords: List[str] = []
        if is_person_query and person_names:
            keywords.extend(person_names)
        attribute_kws = extract_attribute_keywords(query)
        if attribute_kws:
            keywords.extend(attribute_kws)
            logger.info(f"🔍 [Neo4jGraphRAGAgent] Added attribute keywords: {attribute_kws}")
        keywords = list(dict.fromkeys(keywords)) if keywords else []
        keywords = keywords if keywords else None  # API expects None when no keywords
        keyword_boost = 0.4 if keywords else 0.0
        
        top_k = 18 if high_recall_mode else 12
        similarity_threshold = 0.6 if high_recall_mode else 0.75
        logger.info(
            "🔍 [Neo4jGraphRAGAgent] Calling graphrag_search with: "
            f"top_k={top_k}, similarity_threshold={similarity_threshold}, keywords={keywords}, "
            f"keyword_boost={keyword_boost}, high_recall_mode={high_recall_mode}"
        )
        # Use improved search parameters to reduce false matches
        results = graphrag_search(
            query=query,
            top_k=top_k,  # Get more results initially for filtering
            similarity_threshold=similarity_threshold,
            keywords=keywords,
            keyword_match_type="any",
            keyword_boost=keyword_boost
        )

        if not results:
            return ("No results found in Neo4j GraphRAG.", []) if return_results else "No results found in Neo4j GraphRAG."

        # Filter results - mode-aware filtering based on query type
        filtered_results = filter_results_by_exact_match(
            results, 
            query, 
            min_similarity=0.3,
            is_person_query=is_person_query,
            person_names=person_names
        )
        
        # Limit to top 8 after filtering
        filtered_results = filtered_results[:8]

        if not filtered_results:
            logger.warning(f"No relevant results found after filtering (had {len(results)} initial results)")
            return ("No relevant results found in Neo4j GraphRAG after filtering.", []) if return_results else "No relevant results found in Neo4j GraphRAG after filtering."

        # Check if query requires exact numerical calculation or list query and try CSV query
        exact_answer = None
        list_answer = None
        
        # Handle CSV queries if available
        if csv_query_available:
            needs_exact = query_requires_exact_numbers(query)
            needs_list = query_requires_list(query)
            
            # Handle "list all" queries
            if needs_list and filtered_results:
                logger.info(f"Query requires list, attempting CSV distinct values query for: '{query[:100]}...'")
                try:
                    csv_path = extract_csv_path_from_rag_results(filtered_results)
                    if csv_path:
                        list_columns = ['Model', 'Product', 'Category', 'Item', 'Name', 'Type']
                        column_to_list = self._detect_list_column(query, list_columns)
                        
                        list_result = get_distinct_values(
                            file_path=csv_path,
                            column=column_to_list
                        )
                        
                        if 'error' not in list_result and list_result.get('values'):
                            list_answer = {
                                'values': list_result['values'],
                                'count': list_result.get('count', 0),
                                'column': column_to_list
                            }
                            logger.info(f"✅ CSV list query successful: {list_answer['count']} distinct values")
                except Exception as e:
                    logger.error(f"❌ CSV list query failed: {e}", exc_info=True)
            
            # Handle exact numerical queries
            if needs_exact and filtered_results:
                logger.info(f"Query requires exact numbers, attempting CSV query for: '{query[:100]}...'")
                try:
                    csv_path = extract_csv_path_from_rag_results(filtered_results)
                    if csv_path:
                        filter_columns = ['Model', 'Product', 'Category', 'Item', 'Name', 'Type']
                        filters = {}
                        
                        for col in filter_columns:
                            value = extract_filter_value_from_query(query, col, filtered_results)
                            if value:
                                filters[col] = value
                                logger.info(f"Extracted filter: {col} = '{value}'")
                                break
                        
                        if filters:
                            exact_result = sum_numeric_columns(
                                file_path=csv_path,
                                filters=filters,
                                exclude_columns=list(filters.keys()) + ['Factory Location', 'Location', 'Region']
                            )
                            
                            if 'error' not in exact_result and exact_result.get('total', 0) > 0:
                                exact_answer = {
                                    'total': exact_result['total'],
                                    'breakdown': exact_result.get('breakdown', {}),
                                    'filters': filters,
                                    'columns_summed': exact_result.get('columns_summed', 0)
                                }
                                logger.info(f"✅ CSV query successful: total={exact_answer['total']}")
                            elif 'error' not in exact_result and exact_result.get('total', 0) == 0:
                                is_csv_specific = is_csv_specific_query(query)
                                if is_csv_specific:
                                    exact_answer = {
                                        'total': 0,
                                        'no_data': True,
                                        'message': f"No matching data found in CSV for filter: {filters}",
                                        'filters': filters,
                                        'source': 'csv'
                                    }
                except Exception as e:
                    logger.error(f"❌ CSV query failed: {e}", exc_info=True)

        findings = []
        
        # Add list answer at the top if available
        if list_answer:
            values_str = ', '.join(list_answer['values'][:20])
            if list_answer['count'] > 20:
                values_str += f", ... and {list_answer['count'] - 20} more"
            findings.append(
                f"═══════════════════════════════════════════════════════════\n"
                f"**COMPLETE LIST ({list_answer['column']} column):**\n"
                f"Total distinct values: {list_answer['count']}\n"
                f"Values: {values_str}\n"
                f"═══════════════════════════════════════════════════════════\n\n"
                f"**Context from documents:**\n"
            )
            logger.info(f"✅ Added list answer to response: {list_answer['count']} distinct values")
        
        # Add exact answer at the top if available
        if exact_answer:
            if exact_answer.get('no_data'):
                filter_str = ', '.join([f"{k}={v}" for k, v in exact_answer['filters'].items()])
                findings.append(
                    f"═══════════════════════════════════════════════════════════\n"
                    f"**NO DATA FOUND IN CSV:**\n"
                    f"Filter: {filter_str}\n"
                    f"Message: {exact_answer.get('message', 'No matching data found')}\n"
                    f"═══════════════════════════════════════════════════════════\n\n"
                    f"**Note:** This query is CSV-specific. No matching data found in the CSV file.\n"
                    f"**Context from documents (if any):**\n"
                )
            else:
                filter_str = ', '.join([f"{k}={v}" for k, v in exact_answer['filters'].items()])
                findings.append(
                    f"═══════════════════════════════════════════════════════════\n"
                    f"**EXACT NUMERICAL ANSWER:**\n"
                    f"Total = {exact_answer['total']:,} units\n"
                    f"Filter: {filter_str}\n"
                    f"Summed across {exact_answer['columns_summed']} numeric columns\n"
                    f"═══════════════════════════════════════════════════════════\n\n"
                    f"**Context from documents:**\n"
                )
                logger.info(f"✅ Added exact answer to response: {exact_answer['total']:,} units")
        
        # Add document chunks
        for res in filtered_results:
            text = res.get("text", "").replace("\n", " ")
            file_name = res.get("file_name", "Unknown")
            directory = res.get("directory_name", "")
            file_path = res.get("file_path", "")
            similarity = res.get("similarity", 0.0)
            
            # Add source type indicator
            if file_path and file_path.lower().endswith('.csv'):
                source_note = "[CSV Data]"
            elif file_path and file_path.lower().endswith('.pdf'):
                source_note = "[PDF Document]"
            elif file_path and file_path.lower().endswith(('.doc', '.docx')):
                source_note = "[Word Document]"
            elif file_path and file_path.lower().endswith(('.xls', '.xlsx')):
                source_note = "[Excel Document]"
            else:
                source_note = "[Document]"
            
            # Format source info
            if file_path:
                source_info = file_path
            elif directory:
                source_info = f"{directory}/{file_name}"
            else:
                source_info = file_name
            
            # Log and add to findings
            logger.info(f"📝 Adding chunk ({len(text)} chars) from {file_name}")
            
            if len(text) > 2000:
                findings.append(f"- {source_note} {source_info}: {text[:2000]}...")
            else:
                findings.append(f"- {source_note} {source_info}: {text}")

        context_text = "\n".join(findings)
        if return_results:
            return context_text, filtered_results
        return context_text
    
    def _detect_list_column(self, query: str, list_columns: List[str]) -> str:
        """Detect which column to list from query."""
        query_lower = query.lower()
        for col in list_columns:
            if col.lower() in query_lower:
                logger.info(f"Detected column to list from query: {col}")
                return col
        
        # If not found, try common patterns
        if 'model' in query_lower or 'car' in query_lower:
            return 'Model'
        elif 'product' in query_lower:
            return 'Product'
        elif 'category' in query_lower:
            return 'Category'
        else:
            # Default to first common column
            return 'Model'
