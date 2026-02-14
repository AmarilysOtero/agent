"""AI Search Agent for Neo4j GraphRAG retrieval."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

from .utils import (
    infer_header_from_chunk,
    extract_person_names_and_mode,
    filter_results_by_exact_match,
)

logger = logging.getLogger(__name__)


class AiSearchAgent:
    """Search agent using Neo4j GraphRAG retrieval"""
    
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(
        self,
        query: str,
        high_recall_mode: bool = False,
        return_results: bool = False
    ) -> Union[str, Tuple[str, List[Dict[str, Any]]]]:
        logger.info(f"🤖 [AGENT INVOKED] AiSearchAgent (ID: {self._id})")
        print(f"🤖 [AGENT INVOKED] AiSearchAgent (ID: {self._id})")
        print("AiSearchAgent: using Foundry agent:", self._id)  # keep print
        from ..tools.neo4j_graphrag import graphrag_search
        # ...existing code...
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
        from ..tools.header_vocab import extract_attribute_keywords
        person_names, is_person_query = extract_person_names_and_mode(query)
        # ...existing code...
        # Build keywords and keyword_boost before using them
        keywords = []
        if is_person_query and person_names:
            keywords.extend(person_names)
        # Add attribute keywords for richer matching
        attribute_kws = extract_attribute_keywords(query)
        if attribute_kws:
            keywords.extend(attribute_kws)
            logger.info(f"🔍 [AiSearchAgent] Added attribute keywords: {attribute_kws}")
        # Remove duplicates while preserving order
        keywords = list(dict.fromkeys(keywords))
        keyword_boost = 0.4 if keywords else 0.0

        # PHASE 5: Query Classification for Structural Routing
        query_intent = self._classify_query_intent(query, person_names or [])
        logger.info(f"🔍 [QueryClassification] Intent: {query_intent['type']}, routing: {query_intent['routing']}")
        print(f"🔍 [QueryClassification] Intent: {query_intent['type']}, routing: {query_intent['routing']}")
        top_k = 20 if high_recall_mode else 12
        similarity_threshold = 0.6 if high_recall_mode else 0.75
        logger.info(
            "🔍 [AiSearchAgent] Calling graphrag_search with: "
            f"top_k={top_k}, similarity_threshold={similarity_threshold}, keywords={keywords}, "
            f"keyword_boost={keyword_boost}, is_person_query={is_person_query}, person_names={person_names}, "
            f"high_recall_mode={high_recall_mode}"
        )
        print(
            "🔍 [AiSearchAgent] Calling graphrag_search with: "
            f"keywords={keywords}, keyword_boost={keyword_boost}, is_person_query={is_person_query}, "
            f"person_names={person_names}, high_recall_mode={high_recall_mode}"
        )
        results = graphrag_search(
            query=query,
            top_k=top_k,  # Get more results initially for filtering
            similarity_threshold=similarity_threshold,
            keywords=keywords,
            keyword_match_type="any",
            keyword_boost=keyword_boost,
            is_person_query=is_person_query,
            enable_coworker_expansion=True,  # Enable coworker expansion for person queries
            person_names=person_names
        )
        logger.info(f"📊 [AiSearchAgent] GraphRAG search returned {len(results)} results")
        print(f"📊 [AiSearchAgent] GraphRAG search returned {len(results)} results")
        if not results:
            # Always log attempted chunks, even if empty
            logger.info(f"[DEBUG] No results returned from graphrag_search. Query: {query}")
            # Write to chunk log (simulate chunk_logger usage)
            try:
                from ..chunk_logger import log_chunk_selection
                log_chunk_selection(query, [], high_recall_mode, candidate_chunks=[])
            except Exception as e:
                logger.warning(f"[DEBUG] Could not write empty chunk log: {e}")
            return (
                "No results found in Neo4j GraphRAG.",
                []
            ) if return_results else "No results found in Neo4j GraphRAG."
        # ...existing code...
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

        # Log all candidate chunk similarities and IDs before filtering
        logger.info(f"[DEBUG] Candidate chunks before filtering: " + ", ".join([
            f"ID={r.get('chunk_id', r.get('id', 'N/A'))}, sim={r.get('similarity', 0.0):.3f}" for r in results
        ]))
        filtered_results = filter_results_by_exact_match(
            results, 
            query, 
            min_similarity=0.3,
            is_person_query=is_person_query,
            person_names=person_names
        )
        
        # Extract person names and determine if query is person-centric
        # Uses corpus-learned vocabulary for context-aware classification
        from ..tools.header_vocab import extract_attribute_keywords
        
        person_names, is_person_query = extract_person_names_and_mode(query)
        if not filtered_results:
            logger.warning(f"⚠️ [AiSearchAgent] No relevant results found after filtering (had {len(results)} initial results)")
            print(f"⚠️ [AiSearchAgent] No relevant results found after filtering (had {len(results)} initial results)")
            # Always log attempted chunks, even if none pass filter
            try:
                from ..chunk_logger import log_chunk_selection
                log_chunk_selection(query, [], high_recall_mode, candidate_chunks=results)
            except Exception as e:
                logger.warning(f"[DEBUG] Could not write empty chunk log: {e}")
            return (
                "No relevant results found in Neo4j GraphRAG after filtering.",
                []
            ) if return_results else "No relevant results found in Neo4j GraphRAG after filtering."
        # For "Tell me Kevin Skills" → keywords=['kevin', 'skills']
        keywords = []
        if is_person_query and person_names:
            keywords.extend(person_names)
        
        # Add attribute keywords for richer matching
        attribute_kws = extract_attribute_keywords(query)
        if attribute_kws:
            keywords.extend(attribute_kws)
            logger.info(f"🔍 [AiSearchAgent] Added attribute keywords: {attribute_kws}")
        
        # Remove duplicates while preserving order
        keywords = list(dict.fromkeys(keywords))
        
        keyword_boost = 0.4 if keywords else 0.0
        
        # PHASE 5: Query Classification for Structural Routing
        query_intent = self._classify_query_intent(query, person_names or [])
        logger.info(f"🔍 [QueryClassification] Intent: {query_intent['type']}, routing: {query_intent['routing']}")
        print(f"🔍 [QueryClassification] Intent: {query_intent['type']}, routing: {query_intent['routing']}")
        
        top_k = 20 if high_recall_mode else 12
        similarity_threshold = 0.6 if high_recall_mode else 0.75
        logger.info(
            "🔍 [AiSearchAgent] Calling graphrag_search with: "
            f"top_k={top_k}, similarity_threshold={similarity_threshold}, keywords={keywords}, "
            f"keyword_boost={keyword_boost}, is_person_query={is_person_query}, person_names={person_names}, "
            f"high_recall_mode={high_recall_mode}"
        )
        print(
            "🔍 [AiSearchAgent] Calling graphrag_search with: "
            f"keywords={keywords}, keyword_boost={keyword_boost}, is_person_query={is_person_query}, "
            f"person_names={person_names}, high_recall_mode={high_recall_mode}"
        )
        
        # Call graphrag_search with base parameters
        results = graphrag_search(
            query=query,
            top_k=top_k,  # Get more results initially for filtering
            similarity_threshold=similarity_threshold,
            keywords=keywords,
            keyword_match_type="any",
            keyword_boost=keyword_boost,
            is_person_query=is_person_query,
            enable_coworker_expansion=True,  # Enable coworker expansion for person queries
            person_names=person_names
        )

        logger.info(f"📊 [AiSearchAgent] GraphRAG search returned {len(results)} results")
        print(f"📊 [AiSearchAgent] GraphRAG search returned {len(results)} results")
        if results:
            logger.info(f"[DEBUG] Pre-filter chunk IDs: {[r.get('chunk_id', r.get('id', 'N/A')) for r in results]}")
        else:
            logger.info("[DEBUG] No results returned from graphrag_search.")

        if not results:
            return (
                "No results found in Neo4j GraphRAG.",
                []
            ) if return_results else "No results found in Neo4j GraphRAG."

        # Filter results - mode-aware filtering based on query type
        logger.info(f"🔍 [AiSearchAgent] Filtering {len(results)} results (is_person_query={is_person_query})")
        print(f"🔍 [AiSearchAgent] Filtering {len(results)} results (is_person_query={is_person_query})")
        
        filtered_results = filter_results_by_exact_match(
            results, 
            query, 
            min_similarity=0.3,
            is_person_query=is_person_query,
            person_names=person_names
        )
        
        logger.info(f"✅ [AiSearchAgent] After filtering: {len(filtered_results)} results (from {len(results)} initial)")
        print(f"✅ [AiSearchAgent] After filtering: {len(filtered_results)} results (from {len(results)} initial)")
        if filtered_results:
            logger.info(f"[DEBUG] Post-filter chunk IDs: {[r.get('chunk_id', r.get('id', 'N/A')) for r in filtered_results]}")
        else:
            logger.info("[DEBUG] No chunks passed the filter.")
        # Limit to top 8 after filtering
        filtered_results = filtered_results[:8]

        if not filtered_results:
            logger.warning(f"⚠️ [AiSearchAgent] No relevant results found after filtering (had {len(results)} initial results)")
            print(f"⚠️ [AiSearchAgent] No relevant results found after filtering (had {len(results)} initial results)")
            return (
                "No relevant results found in Neo4j GraphRAG after filtering.",
                []
            ) if return_results else "No relevant results found in Neo4j GraphRAG after filtering."

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
                        # Get distinct values
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
                                # CSV returned 0
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
        
        # When RLM is off but this is a person+attribute query, expand matched files so we include
        # sections like Skills that may not be in the initial top-k (e.g. "tell me Alexis Skills").
        # Skip expansion when rlm_mode is "standard" (old flow: semantic+graph only).
        use_expanded_context = False
        try:
            from ..config import Settings
            rlm_mode = getattr(Settings.load(), "rlm_mode", "standard")
        except Exception:
            rlm_mode = "standard"
        if not high_recall_mode and is_person_query and filtered_results and rlm_mode != "standard":
            file_ids = []
            for r in filtered_results:
                fid = (
                    r.get("file_id")
                    or (r.get("metadata") or {}).get("file_id")
                    or r.get("file_name")
                )
                if fid and str(fid).strip() not in ("", "unknown", "None") and "graph_connection" not in str(fid):
                    file_ids.append(fid)
            file_ids = list(dict.fromkeys(file_ids))
            if file_ids:
                try:
                    from ..tools.neo4j_graphrag import expand_files_via_api
                    expanded = await asyncio.to_thread(expand_files_via_api, file_ids)
                    if expanded:
                        expanded_file_ids = set(expanded.keys())
                        max_chunks_total = 30
                        max_chars_total = 45_000
                        chunks_added = 0
                        chars_added = 0
                        for fid, file_data in expanded.items():
                            if chunks_added >= max_chunks_total or chars_added >= max_chars_total:
                                break
                            file_name = file_data.get("file_name", fid)
                            for ch in file_data.get("chunks", [])[:25]:
                                if chunks_added >= max_chunks_total or chars_added >= max_chars_total:
                                    break
                                text = (ch.get("text") or "").replace("\n", " ")
                                if not text:
                                    continue
                                source_note = "[PDF Document]" if (file_name or "").lower().endswith(".pdf") else "[Document]"
                                source_info = file_name or fid
                                snippet = text[:2000] + "..." if len(text) > 2000 else text
                                findings.append(f"- {source_note} {source_info}: {snippet}")
                                chunks_added += 1
                                chars_added += len(text)
                        # Include any filtered result not from an expanded file (e.g. graph_connection)
                        for res in filtered_results:
                            fid = res.get("file_id") or (res.get("metadata") or {}).get("file_id")
                            if fid not in expanded_file_ids:
                                text = res.get("text", "").replace("\n", " ")
                                file_name = res.get("file_name", "Unknown")
                                file_path = res.get("file_path", "")
                                source_note = "[PDF Document]" if (file_path or "").lower().endswith(".pdf") else "[Document]"
                                source_info = file_path or file_name
                                snippet = text[:2000] + "..." if len(text) > 2000 else text
                                findings.append(f"- {source_note} {source_info}: {snippet}")
                        use_expanded_context = True
                        logger.info(f"🔍 [AiSearchAgent] Person-query expansion: {len(file_ids)} files -> {chunks_added} chunks in context")
                except Exception as e:
                    logger.warning(f"Person-query expansion failed: {e}; using search results only", exc_info=True)
        
        # Add document chunks (skip when we already built context from expanded files)
        if not use_expanded_context:
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
    
    def _classify_query_intent(self, query: str, person_names: List[str]) -> dict:
        """Determine if query is section-based or semantic."""
        query_lower = query.lower()
        
        # Attribute keywords (structural sections)
        attribute_keywords = [
            'skill', 'experience', 'education', 'qualification',
            'role', 'position', 'project', 'certification',
            'background', 'expertise', 'training', 'achievement',
            'industry', 'professional', 'technical', 'employment',
            'work', 'career', 'competenc', 'capabilit'
        ]
        
        # Check for person name
        has_person = any(name.lower() in query_lower for name in person_names)
        
        # Check for attribute keyword
        has_attribute = any(keyword in query_lower for keyword in attribute_keywords)
        
        # Extract attribute phrase if present
        attribute_match = None
        if has_attribute:
            attribute_match = self._extract_attribute_phrase(query, attribute_keywords)
        
        # Classify
        if has_person and has_attribute:
            return {
                'type': 'section_based_scoped',
                'routing': 'hard',
                'person_names': person_names,
                'section_query': attribute_match,
                'file_scope': True
            }
        elif has_attribute and not has_person:
            return {
                'type': 'section_based_cross_document',
                'routing': 'hard',
                'section_query': attribute_match,
                'file_scope': False
            }
        else:
            return {
                'type': 'semantic',
                'routing': 'soft',
                'section_query': None,
                'file_scope': has_person
            }
    
    def _extract_attribute_phrase(self, query: str, attribute_keywords: List[str]) -> str:
        """Extract section-like phrase around attribute keyword."""
        words = query.lower().split()
        
        # Find first keyword that appears
        for i, word in enumerate(words):
            word_clean = word.strip('.,!?;:\'"')
            # Check if any attribute keyword is in this word
            for keyword in attribute_keywords:
                if keyword in word_clean:
                    # Take 1-2 words before + keyword + 1 word after
                    start = max(0, i - 1)
                    end = min(len(words), i + 2)
                    phrase = ' '.join(words[start:end])
                    # Clean up
                    phrase = phrase.strip('.,!?;:\'"')
                    return phrase
        
        # Fallback: return first keyword found
        for keyword in attribute_keywords:
            if keyword in query.lower():
                return keyword
        
        return "attribute"
    
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
