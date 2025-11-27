from __future__ import annotations
import json
import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field, ValidationError
from ..foundry_runner import run_foundry_agent, run_foundry_agent_json

logger = logging.getLogger(__name__)


def extract_person_names(query: str) -> List[str]:
    """Extract potential person names from query (capitalized words that might be names)
    
    Args:
        query: User query text
        
    Returns:
        List of potential person names (capitalized words)
    """
    # Split query into words
    words = query.split()
    # Extract capitalized words that are likely names (length > 2, starts with capital)
    names = [w.strip('.,!?;:') for w in words if w and w[0].isupper() and len(w.strip('.,!?;:')) > 2]
    # Remove common words that start with capital but aren't names
    common_words = {'The', 'This', 'That', 'These', 'Those', 'What', 'When', 'Where', 'Who', 'Why', 'How', 'Tell', 'Show', 'Give', 'Find', 'Search', 'Get'}
    names = [n for n in names if n not in common_words]
    return names


def filter_results_by_exact_match(results: List[Dict[str, Any]], query: str, min_similarity: float = 0.88) -> List[Dict[str, Any]]:
    """Filter search results to require query name appears in chunk text or very high similarity
    
    Args:
        results: List of search result dictionaries
        query: Original query text
        min_similarity: Minimum similarity to keep result without exact match
        
    Returns:
        Filtered list of results
    """
    if not results:
        return results
    
    # Extract potential name words from query using the same function that filters common words
    names = extract_person_names(query)
    query_words = [n.lower() for n in names]
    
    # If no capitalized words, apply minimum similarity threshold only
    if not query_words:
        # Still filter out very low similarity results
        return [res for res in results if res.get("similarity", 0.0) >= 0.3]
    
    # Get first name (first name word) - critical for distinguishing names
    first_name = query_words[0] if query_words else None
    last_name = query_words[-1] if len(query_words) > 1 else None
    
    logger.info(f"Filtering {len(results)} results for query '{query}' (first_name='{first_name}', last_name='{last_name}')")
    
    filtered = []
    for res in results:
        text = res.get("text", "").lower()
        similarity = res.get("similarity", 0.0)
        
        # Apply absolute minimum similarity threshold (reject very low scores)
        if similarity < 0.3:
            logger.debug(f"Filtered out result: similarity={similarity:.3f} < 0.3 (absolute minimum)")
            continue
        
        # Check if first name appears in text (required for name queries)
        # This prevents "Axel Torres" from matching "Alexis Torres" queries
        first_name_found = first_name in text if first_name else True
        
        # If we have both first and last name, require both to match
        if first_name and last_name:
            last_name_found = last_name in text
            name_match = first_name_found and last_name_found
        else:
            # Only first name available, require it to match
            name_match = first_name_found
        
        # Keep if: (name matches AND similarity >= 0.3) OR similarity is very high (>= min_similarity)
        # Lower threshold for name matches to allow more results through
        if (name_match and similarity >= 0.3) or similarity >= min_similarity:
            filtered.append(res)
            logger.debug(f"Kept result: similarity={similarity:.3f}, first_name_match={first_name_found}, name_match={name_match}")
        else:
            logger.info(f"Filtered out result: similarity={similarity:.3f}, first_name_match={first_name_found}, name_match={name_match}, text_preview={text[:100]}")
    
    logger.info(f"Filtered {len(results)} results down to {len(filtered)} results")
    return filtered
    
    return filtered

# Lazy import for Azure Search to avoid import errors when using Neo4j only
def _get_hybrid_search():
    try:
        from ..tools.azure_search import hybrid_search
        return hybrid_search
    except ImportError as e:
        logger.warning(f"Azure Search not available: {e}")
        return None

# ---------- TRIAGE (Foundry) ----------

class IntentResult(BaseModel):
    intents: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""
    targets: list[str] = Field(default_factory=list)

class TriageAgent:
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, goal: str) -> IntentResult:
        content = f"Classify and return JSON only. User goal: {goal}"
        print("TriageAgent: using Foundry agent:", self._id)  # keep print
        try:
            raw = run_foundry_agent(self._id, content).strip()
        except RuntimeError as e:
            logger.error("TriageAgent Foundry error: %s", e)
            raise RuntimeError(
                f"Triage agent failed: {str(e)}. "
                "Please check your Foundry access and agent configuration."
            ) from e
        print("Triage raw:", raw)  # keep print
        try:
            data = json.loads(raw)
            return IntentResult(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error("Triage parse error: %s", e)
            return IntentResult(intents=["unknown"], confidence=0.0, rationale="parse_error")

# ---------- AI SEARCH (Foundry) ----------

class AiSearchAgent:
    """Search agent using Azure AI Search"""
    
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, query: str) -> str:
        print("AiSearchAgent: using Foundry agent:", self._id)  # keep print
        hybrid_search = _get_hybrid_search()
        if hybrid_search is None:
            return "Azure Search is not available. Please configure Azure Search or use Neo4j search instead."
        
        results = hybrid_search(
            search_text=query,
            top_k=8,
            select=["file_name", "content", "url", "last_modified"],
            semantic=False
        )

        if not results:
            return "No results found in Azure AI Search."

        findings = []
        for res in results:
            content = (res.get("content") or "").replace("\n", " ")
            findings.append(f"- {res.get('file_name')}: {content[:300]}...")

        # print("AiSearchAgent list of sources/content\n\n" + "\n".join(findings))
        return "\n".join(findings)


# ---------- NEO4J GRAPHRAG SEARCH (Foundry) ----------

class Neo4jGraphRAGAgent:
    """Search agent using Neo4j GraphRAG retrieval"""
    
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, query: str) -> str:
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
                get_distinct_values
            )
            csv_query_available = True
        except ImportError:
            logger.warning("CSV query tools not available, skipping CSV queries")
            csv_query_available = False
        
        # Extract person names from query for keyword filtering
        person_names = extract_person_names(query)
        
        # Use improved search parameters to reduce false matches
        results = graphrag_search(
            query=query,
            top_k=12,  # Get more results initially for filtering
            similarity_threshold=0.75,  # Increased from 0.7 to reduce false matches
            keywords=person_names if person_names else None,
            keyword_match_type="any",
            keyword_boost=0.4  # Increase keyword weight for name matching
        )

        if not results:
            return "No results found in Neo4j GraphRAG."

        # Filter results to require exact name match or very high similarity
        filtered_results = filter_results_by_exact_match(
            results, 
            query, 
            min_similarity=0.7  # Very high threshold for results without exact match
        )
        
        # Limit to top 8 after filtering
        filtered_results = filtered_results[:8]

        if not filtered_results:
            logger.warning(f"No relevant results found after filtering (had {len(results)} initial results)")
            return "No relevant results found in Neo4j GraphRAG after filtering."

        # Check if query requires exact numerical calculation or list query and try CSV query
        exact_answer = None
        list_answer = None
        
        # Always log the check status for debugging
        if csv_query_available:
            needs_exact = query_requires_exact_numbers(query)
            needs_list = query_requires_list(query)
            logger.warning(f"🔍 CSV query check: available={csv_query_available}, needs_exact={needs_exact}, needs_list={needs_list}, query='{query[:50]}...', has_results={bool(filtered_results)}")
        else:
            needs_exact = False
            needs_list = False
            logger.warning(f"🔍 CSV query check: available={csv_query_available} (tools not imported)")
        
        # Handle "list all" queries (e.g., "name all models")
        if csv_query_available and needs_list and filtered_results:
            logger.info(f"Query requires list, attempting CSV distinct values query for: '{query[:100]}...'")
            try:
                csv_path = extract_csv_path_from_rag_results(filtered_results)
                logger.info(f"Extracted CSV path: {csv_path}")
                
                if csv_path:
                    # Try to detect which column to list (Model, Product, Category, etc.)
                    list_columns = ['Model', 'Product', 'Category', 'Item', 'Name', 'Type']
                    column_to_list = None
                    
                    # Check query for column name hints
                    query_lower = query.lower()
                    for col in list_columns:
                        if col.lower() in query_lower:
                            column_to_list = col
                            logger.info(f"Detected column to list from query: {column_to_list}")
                            break
                    
                    # If not found, try common patterns
                    if not column_to_list:
                        if 'model' in query_lower or 'car' in query_lower:
                            column_to_list = 'Model'
                        elif 'product' in query_lower:
                            column_to_list = 'Product'
                        elif 'category' in query_lower:
                            column_to_list = 'Category'
                        else:
                            # Default to first common column
                            column_to_list = 'Model'
                            logger.info(f"Using default column: {column_to_list}")
                    
                    # Get distinct values
                    list_result = get_distinct_values(
                        file_path=csv_path,
                        column=column_to_list
                    )
                    
                    logger.info(f"CSV distinct values result - count: {list_result.get('count', 'N/A')}, error: {list_result.get('error', 'none')}")
                    
                    if 'error' not in list_result and list_result.get('values'):
                        list_answer = {
                            'values': list_result['values'],
                            'count': list_result.get('count', 0),
                            'column': column_to_list
                        }
                        logger.info(f"✅ CSV list query successful: {list_answer['count']} distinct values in column '{column_to_list}'")
                    else:
                        error_msg = list_result.get('error', 'unknown')
                        logger.warning(f"❌ CSV list query returned no data or error: {error_msg}")
            except Exception as e:
                logger.error(f"❌ CSV list query failed: {e}", exc_info=True)
        
        # Handle exact numerical queries (e.g., "how many")
        if csv_query_available and needs_exact and filtered_results:
            logger.info(f"Query requires exact numbers, attempting CSV query for: '{query[:100]}...'")
            try:
                csv_path = extract_csv_path_from_rag_results(filtered_results)
                logger.info(f"Extracted CSV path: {csv_path}")
                
                if csv_path:
                    # Try to extract filter values from query (common column names)
                    filter_columns = ['Model', 'Product', 'Category', 'Item', 'Name', 'Type']
                    filters = {}
                    
                    for col in filter_columns:
                        value = extract_filter_value_from_query(query, col, filtered_results)
                        if value:
                            filters[col] = value
                            logger.info(f"Extracted filter: {col} = '{value}'")
                            break  # Use first match
                    
                    if filters:
                        logger.info(f"Calling sum_numeric_columns with filters: {filters}, file: {csv_path}")
                        # Get exact numerical answer using pandas
                        exact_result = sum_numeric_columns(
                            file_path=csv_path,
                            filters=filters,
                            exclude_columns=list(filters.keys()) + ['Factory Location', 'Location', 'Region']
                        )
                        
                        logger.info(f"CSV query result - total: {exact_result.get('total', 'N/A')}, error: {exact_result.get('error', 'none')}, columns_summed: {exact_result.get('columns_summed', 'N/A')}")
                        
                        if 'error' not in exact_result and exact_result.get('total', 0) > 0:
                            exact_answer = {
                                'total': exact_result['total'],
                                'breakdown': exact_result.get('breakdown', {}),
                                'filters': filters,
                                'columns_summed': exact_result.get('columns_summed', 0)
                            }
                            logger.info(f"✅ CSV query successful: total={exact_answer['total']}, columns={exact_answer['columns_summed']}")
                        else:
                            error_msg = exact_result.get('error', 'unknown')
                            logger.warning(f"❌ CSV query returned no data or error: {error_msg}. Full result keys: {list(exact_result.keys())}")
                            if 'metadata' in exact_result:
                                logger.warning(f"   Metadata: {exact_result['metadata']}")
                    else:
                        logger.warning(f"❌ Could not extract filter values from query. Tried columns: {filter_columns}")
                        logger.warning(f"   Query: '{query}'")
                        logger.warning(f"   RAG results count: {len(filtered_results)}")
                else:
                    logger.warning("❌ No CSV path found in RAG results")
                    logger.warning(f"   Available file_paths: {[r.get('file_path') for r in filtered_results[:3]]}")
            except Exception as e:
                logger.error(f"❌ CSV query failed, falling back to RAG only: {e}", exc_info=True)
        elif csv_query_available and not needs_exact:
            logger.info("CSV query available but query does not require exact numbers")
        elif not csv_query_available:
            logger.warning("CSV query tools not available")

        findings = []
        
        # Add list answer at the top if available (for "list all" queries)
        if list_answer:
            values_str = ', '.join(list_answer['values'][:20])  # Show first 20
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
        
        # Add exact answer at the top if available (make it very prominent)
        if exact_answer:
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
        
        if not exact_answer and not list_answer:
            logger.warning("⚠️ No CSV query answer available - response will use RAG chunks only")
        
        for res in filtered_results:
            text = res.get("text", "").replace("\n", " ")
            file_name = res.get("file_name", "Unknown")
            directory = res.get("directory_name", "")
            file_path = res.get("file_path", "")
            
            # Include comprehensive metadata for agent context
            metadata_parts = []
            if "hybrid_score" in res and res["hybrid_score"] is not None:
                metadata_parts.append(f"score:{res['hybrid_score']:.2f}")
            if "similarity" in res and res["similarity"] is not None:
                metadata_parts.append(f"similarity:{res['similarity']:.2f}")
            if "metadata" in res and res["metadata"]:
                meta = res["metadata"]
                if meta.get("hop_count", 0) > 0:
                    metadata_parts.append(f"hops:{meta['hop_count']}")
                if meta.get("vector_score") is not None:
                    metadata_parts.append(f"vector:{meta['vector_score']:.3f}")
                if meta.get("keyword_score") is not None:
                    metadata_parts.append(f"keyword:{meta['keyword_score']:.3f}")
                if meta.get("path_score") is not None:
                    metadata_parts.append(f"path:{meta['path_score']:.3f}")
                if meta.get("chunk_index") is not None:
                    metadata_parts.append(f"chunk_idx:{meta['chunk_index']}")
                if meta.get("chunk_size") is not None:
                    metadata_parts.append(f"size:{meta['chunk_size']}")
                if meta.get("file_id"):
                    metadata_parts.append(f"file_id:{meta['file_id']}")
            
            metadata_str = f" [{', '.join(metadata_parts)}]" if metadata_parts else ""
            
            # Format source info with file path (preferred) or directory/name
            if file_path:
                source_info = file_path
            elif directory:
                source_info = f"{directory}/{file_name}"
            else:
                source_info = file_name
            
            findings.append(f"- {source_info}: {text[:300]}...{metadata_str}")

        return "\n".join(findings)


# ---------- REPORTER (Foundry) ----------

class NewsReporterAgent:
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, topic: str, latest_news: str) -> str:
        content = (
            f"Topic: {topic}\n"
            f"Latest info:\n{latest_news}\n"
            # "Write a 60-90s news broadcast script."
            "Write a description about the information in the tone of a news reporter." 
        )
        print("NewsReporterAgent: using Foundry agent:", self._id)  # keep print
        try:
            return run_foundry_agent(self._id, content)
        except RuntimeError as e:
            logger.error("NewsReporterAgent Foundry error: %s", e)
            raise RuntimeError(
                f"Reporter agent failed: {str(e)}. "
                "Please check your Foundry access and agent configuration."
            ) from e

# ---------- REVIEWER (Foundry, strict JSON) ----------

class ReviewAgent:
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, topic: str, candidate_script: str) -> dict:
        """
        Foundry system prompt already defines the JSON schema. We still remind at user layer.
        Returns a dict with keys: decision, reason, suggested_changes, revised_script.
        Return ONLY STRICT JSON (no markdown, no prose) as per your schema.
        """
        prompt = (
            f"Topic: {topic}\n\n"
            f"Candidate script:\n{candidate_script}\n\n"
            # "Evaluate factual accuracy, clarity, neutral tone, explicit dates, and 60-90s length. "
            "Evaluate factual accuracy, relevance, and tone of a news reporter. " 
            "Return ONLY STRICT JSON (no markdown, no prose) as per your schema."
        )
        print("ReviewAgent: using Foundry agent:", self._id)  # keep print
        try:
            data = run_foundry_agent_json(
                self._id,
                prompt,
                system_hint="You are a reviewer that returns STRICT JSON only."
            )
        except RuntimeError as e:
            logger.error("ReviewAgent Foundry error: %s", e)
            raise RuntimeError(
                f"Review agent failed: {str(e)}. "
                "Please check your Foundry access and agent configuration."
            ) from e
        
        try:
            if not isinstance(data, dict) or "decision" not in data:
                raise ValueError("Invalid JSON shape from reviewer")
            decision = (data.get("decision") or "revise").lower()
            return {
                "decision": decision if decision in {"accept", "revise"} else "revise",
                "reason": data.get("reason", ""),
                "suggested_changes": data.get("suggested_changes", ""),
                "revised_script": data.get("revised_script", candidate_script),
            }
        except Exception as e:
            logger.error("Review parse error: %s", e)
            # Fail-safe: accept last script to avoid infinite loops
            return {
                "decision": "accept",
                "reason": "parse_error",
                "suggested_changes": "",
                "revised_script": candidate_script,
            }
