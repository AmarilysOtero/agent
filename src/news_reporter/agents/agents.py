from __future__ import annotations
import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError
from ..foundry_runner import run_foundry_agent, run_foundry_agent_json
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.core.exceptions import HttpResponseError

logger = logging.getLogger(__name__)

def _load_env():
    """Load .env from the project root"""
    try:
        from dotenv import load_dotenv
        here = Path(__file__).resolve()
        candidates = [
            here.parents[2] / ".env",  # repo root
            here.parents[1] / ".env",
            Path.cwd() / ".env",
        ]
        for p in candidates:
            if p.exists():
                load_dotenv(p)
                logger.info(f"Loaded .env from: {p}")
                break
    except Exception as e:
        logger.warning(f"Failed to load .env: {e}")


_load_env()

def get_ai_project_client() -> AIProjectClient:
    conn = os.getenv("AI_PROJECT_CONNECTION_STRING")
    if not conn:
        raise ValueError("AI_PROJECT_CONNECTION_STRING is not set")

    from ..tools.util import parse_connection_string
    parts = parse_connection_string(conn)
    credential = DefaultAzureCredential()

    return AIProjectClient(
        endpoint=parts["endpoint"],
        project=parts["project"],
        subscription_id=parts["subscription_id"],
        resource_group_name=parts["resource_group"],
        account_name=parts["account"],
        credential=credential,
    )

def list_agents_from_foundry() -> List[Dict[str, Any]]:
    """
    List all available agents from Azure AI Foundry.
    
    Returns:
        List of agent dictionaries with id, name, model, description, etc.
    """
    try:
        # Ensure OPENAI_API_VERSION is set for the SDK
        if "OPENAI_API_VERSION" not in os.environ:
            os.environ["OPENAI_API_VERSION"] = "2024-05-01-preview"
            logger.info("Set default OPENAI_API_VERSION to 2024-05-01-preview for agents")

        client = get_ai_project_client()

        listing = client.agents.list_agents()
        print(f"Listing: {listing}")

        agents: List[Dict[str, Any]] = []

        # In the current SDK, `listing` is an iterable of Agent objects.
        # We still keep a fallback for any response shape that has `.data`/`.value`.
        iterable = None

        # If the SDK ever returns a wrapper with `.data` or `.value`, use it
        maybe_data = getattr(listing, "data", None) or getattr(listing, "value", None)
        if maybe_data is not None:
            iterable = maybe_data
        else:
            # Normal case: pageable iterator
            iterable = listing

        for agent in iterable:
            agent_dict = {
                "id": getattr(agent, "id", None) or getattr(agent, "value", None) or "",
                "name": getattr(agent, "name", "Unknown"),
                "model": getattr(agent, "model", ""),
                "description": getattr(agent, "description", ""),
                "created_at": getattr(agent, "created_at", None),
                "instructions": getattr(agent, "instructions", ""),
            }
            agents.append(agent_dict)

        logger.info(
            "Successfully listed %d agents from Foundry: %s",
            len(agents),
            [a["name"] for a in agents],
        )
        return agents

    except HttpResponseError as e:
        logger.error(f"HTTP error listing agents from Foundry: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to list agents from Foundry: {e}")
        raise

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
    # Expanded to include query verbs like "List", "Name", etc.
    common_words = {
        'The', 'This', 'That', 'These', 'Those', 
        'What', 'When', 'Where', 'Who', 'Why', 'How', 
        'Tell', 'Show', 'Give', 'Find', 'Search', 'Get',
        'List', 'Name', 'Count', 'Sum', 'Calculate', 'Compute',
        'Return', 'Display', 'Print', 'Output'
    }
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
    
    logger.info(f"🔍 [filter_results_by_exact_match] Filtering {len(results)} results for query '{query}'")
    logger.info(f"🔍 [filter_results_by_exact_match] Extracted names: {names}, query_words: {query_words}")
    logger.info(f"🔍 [filter_results_by_exact_match] first_name='{first_name}', last_name='{last_name}', min_similarity={min_similarity}")
    print(f"🔍 [filter_results_by_exact_match] Filtering {len(results)} results for query '{query}'")
    print(f"🔍 [filter_results_by_exact_match] first_name='{first_name}', last_name='{last_name}', min_similarity={min_similarity}")
    
    filtered = []
    for i, res in enumerate(results, 1):
        text = res.get("text", "").lower()
        similarity = res.get("similarity", 0.0)
        file_name = res.get("file_name", "?")
        text_preview = text[:100].replace("\n", " ")
        
        # Apply absolute minimum similarity threshold (reject very low scores)
        if similarity < 0.3:
            logger.info(f"❌ [filter] Result {i} FILTERED OUT: similarity={similarity:.3f} < 0.3 (absolute minimum), file='{file_name}'")
            print(f"❌ [filter] Result {i} FILTERED OUT: similarity={similarity:.3f} < 0.3 (absolute minimum), file='{file_name}'")
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
        
        # Also check if file name contains the person's name (useful when text matching fails)
        # This helps when the chunk text doesn't contain the name but the file name does
        file_name_lower = file_name.lower() if file_name else ""
        file_contains_name = False
        if first_name and last_name:
            # Check if file contains both names, or at least the last name (common in file names)
            file_contains_name = (first_name in file_name_lower and last_name in file_name_lower) or \
                                 (last_name in file_name_lower)  # Last name alone is often in file names
        elif first_name:
            file_contains_name = first_name in file_name_lower
        
        # Keep if: 
        # 1. Name matches in text AND similarity >= 0.3, OR
        # 2. File name contains the person's name AND similarity >= 0.4 (slightly higher for file match), OR
        # 3. Similarity is very high (>= min_similarity)
        if (name_match and similarity >= 0.3) or \
           (file_contains_name and similarity >= 0.4) or \
           similarity >= min_similarity:
            filtered.append(res)
            logger.info(
                f"✅ [filter] Result {i} KEPT: similarity={similarity:.3f}, first_name_match={first_name_found}, "
                f"last_name_match={last_name_found if (first_name and last_name) else 'N/A'}, "
                f"name_match={name_match}, file_contains_name={file_contains_name}, file='{file_name}'"
            )
            print(
                f"✅ [filter] Result {i} KEPT: similarity={similarity:.3f}, first_name_match={first_name_found}, "
                f"name_match={name_match}, file_contains_name={file_contains_name}, file='{file_name}'"
            )
        else:
            logger.info(
                f"❌ [filter] Result {i} FILTERED OUT: similarity={similarity:.3f}, first_name_match={first_name_found}, "
                f"last_name_match={last_name_found if (first_name and last_name) else 'N/A'}, "
                f"name_match={name_match}, file_contains_name={file_contains_name}, similarity >= min_similarity={similarity >= min_similarity}, "
                f"file='{file_name}', text_preview='{text_preview}...'"
            )
            print(
                f"❌ [filter] Result {i} FILTERED OUT: similarity={similarity:.3f}, first_name_match={first_name_found}, "
                f"name_match={name_match}, file_contains_name={file_contains_name}, file='{file_name}'"
            )
    
    logger.info(f"📊 [filter_results_by_exact_match] Filtered {len(results)} results down to {len(filtered)} results")
    print(f"📊 [filter_results_by_exact_match] Filtered {len(results)} results down to {len(filtered)} results")
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
    intents: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""
    targets: List[str] = Field(default_factory=list)
    database_type: Optional[str] = None  # "postgresql", "csv", "other"
    database_id: Optional[str] = None  # Best matching database ID
    preferred_agent: Optional[str] = None  # "sql", "csv", "vector"

class TriageAgent:
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, goal: str) -> IntentResult:
        logger.info(f"🤖 [AGENT INVOKED] TriageAgent (ID: {self._id})")
        print(f"🤖 [AGENT INVOKED] TriageAgent (ID: {self._id})")
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
            
            # If "ai_search" intent OR "unknown" intent, detect best database/schema for routing
            # (unknown queries might be search queries that weren't classified correctly)
            intents = data.get("intents", [])
            if "ai_search" in intents or ("unknown" in intents and len(intents) == 1):
                print(f"🔍 TriageAgent: Starting schema detection for query: '{goal}'")  # Always print
                try:
                    from ..tools_sql.schema_retrieval import SchemaRetriever
                    logger.info(f"🔍 TriageAgent: Starting schema detection for query: '{goal}'")
                    schema_retriever = SchemaRetriever()
                    
                    # List all available databases for debug
                    print("📊 TriageAgent: Listing available databases...")  # Always print
                    all_databases = schema_retriever.list_databases()
                    logger.info(f"📊 TriageAgent: Found {len(all_databases)} available databases")
                    print(f"📊 TriageAgent: Found {len(all_databases)} available databases")
                    for db in all_databases:
                        db_id = db.get("id") or db.get("database_id") or db.get("_id")
                        db_name = db.get("name", "Unknown")
                        db_type = db.get("databaseType") or db.get("database_type", "Unknown")
                        logger.info(f"   - Database: {db_name} (ID: {db_id}, Type: {db_type})")
                        print(f"   - Database: {db_name} (ID: {db_id}, Type: {db_type})")
                    
                    print(f"🔎 TriageAgent: Searching for best database...")
                    best_db_id = schema_retriever.find_best_database(query=goal)
                    print(f"🔎 TriageAgent: Best database ID: {best_db_id}")
                    
                    if best_db_id:
                        # Get database type from database list
                        databases = schema_retriever.list_databases()
                        db_info = next(
                            (db for db in databases 
                             if (db.get("id") or db.get("database_id") or db.get("_id")) == best_db_id),
                            {}
                        )
                        db_type = (db_info.get("databaseType") or db_info.get("database_type") or "").lower()
                        db_name = (db_info.get("name") or best_db_id).lower()
                        
                        logger.info(f"✅ TriageAgent: Selected database '{db_name}' (ID: {best_db_id}, Type: {db_type})")
                        print(f"✅ TriageAgent: Selected database '{db_name}' (ID: {best_db_id}, Type: {db_type})")
                        
                        if "postgresql" in db_type or "postgres" in db_type:
                            data["database_type"] = "postgresql"
                            data["preferred_agent"] = "sql"
                            print(f"✅ TriageAgent: Set preferred_agent='sql', database_type='postgresql'")
                        elif "csv" in db_type or "csv" in db_name or ".csv" in db_name:
                            data["database_type"] = "csv"
                            data["preferred_agent"] = "csv"
                            print(f"✅ TriageAgent: Set preferred_agent='csv', database_type='csv'")
                        else:
                            data["database_type"] = "other"
                            data["preferred_agent"] = "vector"
                            print(f"✅ TriageAgent: Set preferred_agent='vector', database_type='other'")
                        
                        data["database_id"] = best_db_id
                        logger.info(f"TriageAgent detected database_type={data.get('database_type')}, preferred_agent={data.get('preferred_agent')}, database_id={best_db_id}")
                        print(f"TriageAgent final: database_type={data.get('database_type')}, preferred_agent={data.get('preferred_agent')}, database_id={best_db_id}")
                    else:
                        logger.info("TriageAgent: No best database found, will use default routing")
                        print("TriageAgent: No best database found, will use default routing")
                except Exception as e:
                    logger.error(f"TriageAgent: Schema detection failed, falling back to default: {e}", exc_info=True)
                    print(f"❌ TriageAgent: Schema detection failed: {e}")
                    import traceback
                    print(traceback.format_exc())
                    # Fall back to default behavior (no schema routing)
            
            result = IntentResult(**data)
            logger.info(f"📋 TriageAgent: Final IntentResult - intents={result.intents}, preferred_agent={result.preferred_agent}, database_id={result.database_id}, database_type={getattr(result, 'database_type', 'N/A')}, confidence={result.confidence}")
            print(f"📋 TriageAgent: Final IntentResult - intents={result.intents}, preferred_agent={result.preferred_agent}, database_id={result.database_id}, database_type={getattr(result, 'database_type', 'N/A')}, confidence={result.confidence}")
            print(f"📋 TriageAgent: Full result dump: {result.model_dump()}")
            return result
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error("Triage parse error: %s", e)
            return IntentResult(intents=["unknown"], confidence=0.0, rationale="parse_error")

# ---------- AI SEARCH (Foundry) ----------

class AiSearchAgent:
    """Search agent using Neo4j GraphRAG retrieval"""
    
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, query: str) -> str:
        logger.info(f"🤖 [AGENT INVOKED] AiSearchAgent (ID: {self._id})")
        print(f"🤖 [AGENT INVOKED] AiSearchAgent (ID: {self._id})")
        print("AiSearchAgent: using Foundry agent:", self._id)  # keep print
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
        
        # Extract person names from query for keyword filtering
        person_names = extract_person_names(query)
        
        logger.info(f"🔍 [AiSearchAgent] Starting search for query: '{query}'")
        logger.info(f"🔍 [AiSearchAgent] Extracted person names: {person_names}")
        print(f"🔍 [AiSearchAgent] Starting search for query: '{query}'")
        print(f"🔍 [AiSearchAgent] Extracted person names: {person_names}")
        
        # Use improved search parameters to reduce false matches
        logger.info(f"🔍 [AiSearchAgent] Calling graphrag_search with: top_k=12, similarity_threshold=0.75, keywords={person_names}, keyword_boost=0.4")
        print(f"🔍 [AiSearchAgent] Calling graphrag_search with: top_k=12, similarity_threshold=0.75, keywords={person_names}, keyword_boost=0.4")
        
        results = graphrag_search(
            query=query,
            top_k=12,  # Get more results initially for filtering
            similarity_threshold=0.75,  # Increased from 0.7 to reduce false matches
            keywords=person_names if person_names else None,
            keyword_match_type="any",
            keyword_boost=0.4  # Increase keyword weight for name matching
        )

        logger.info(f"📊 [AiSearchAgent] GraphRAG search returned {len(results)} results")
        print(f"📊 [AiSearchAgent] GraphRAG search returned {len(results)} results")
        
        # Log detailed information about each result
        if results:
            logger.info(f"📋 [AiSearchAgent] Detailed results from GraphRAG:")
            print(f"📋 [AiSearchAgent] Detailed results from GraphRAG:")
            for i, res in enumerate(results, 1):
                similarity = res.get("similarity", 0.0)
                hybrid_score = res.get("hybrid_score", 0.0)
                text_preview = res.get("text", "")[:150].replace("\n", " ")
                file_name = res.get("file_name", "?")
                file_path = res.get("file_path", "?")
                chunk_id = res.get("id", "?")
                metadata = res.get("metadata", {})
                vector_score = metadata.get("vector_score", 0.0)
                keyword_score = metadata.get("keyword_score", 0.0)
                
                logger.info(
                    f"   Result {i}: similarity={similarity:.3f}, hybrid_score={hybrid_score:.3f}, "
                    f"vector_score={vector_score:.3f}, keyword_score={keyword_score:.3f}, "
                    f"file='{file_name}', chunk_id='{chunk_id[:50]}...', "
                    f"text_preview='{text_preview}...'"
                )
                print(
                    f"   Result {i}: similarity={similarity:.3f}, hybrid_score={hybrid_score:.3f}, "
                    f"vector_score={vector_score:.3f}, keyword_score={keyword_score:.3f}, "
                    f"file='{file_name}', text_preview='{text_preview}...'"
                )
        else:
            logger.warning(f"⚠️ [AiSearchAgent] No results returned from GraphRAG search")
            print(f"⚠️ [AiSearchAgent] No results returned from GraphRAG search")

        if not results:
            return "No results found in Neo4j GraphRAG."

        # Filter results to require exact name match or very high similarity
        logger.info(f"🔍 [AiSearchAgent] Filtering {len(results)} results with filter_results_by_exact_match (min_similarity=0.7)")
        print(f"🔍 [AiSearchAgent] Filtering {len(results)} results with filter_results_by_exact_match (min_similarity=0.7)")
        
        filtered_results = filter_results_by_exact_match(
            results, 
            query, 
            min_similarity=0.7  # Very high threshold for results without exact match
        )
        
        logger.info(f"✅ [AiSearchAgent] After filtering: {len(filtered_results)} results (from {len(results)} initial)")
        print(f"✅ [AiSearchAgent] After filtering: {len(filtered_results)} results (from {len(results)} initial)")
        
        # Log filtered results details
        if filtered_results:
            logger.info(f"📋 [AiSearchAgent] Filtered results details:")
            print(f"📋 [AiSearchAgent] Filtered results details:")
            for i, res in enumerate(filtered_results, 1):
                similarity = res.get("similarity", 0.0)
                text_preview = res.get("text", "")[:150].replace("\n", " ")
                file_name = res.get("file_name", "?")
                logger.info(f"   Filtered {i}: similarity={similarity:.3f}, file='{file_name}', text_preview='{text_preview}...'")
                print(f"   Filtered {i}: similarity={similarity:.3f}, file='{file_name}', text_preview='{text_preview}...'")
        
        # Limit to top 8 after filtering
        filtered_results = filtered_results[:8]

        if not filtered_results:
            logger.warning(f"⚠️ [AiSearchAgent] No relevant results found after filtering (had {len(results)} initial results)")
            print(f"⚠️ [AiSearchAgent] No relevant results found after filtering (had {len(results)} initial results)")
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
                        elif 'error' not in exact_result and exact_result.get('total', 0) == 0:
                            # CSV returned 0 - check if query is CSV-specific
                            is_csv_specific = is_csv_specific_query(query)
                            
                            if is_csv_specific:
                                # Query is CSV-specific, so 0 means no data in CSV
                                exact_answer = {
                                    'total': 0,
                                    'no_data': True,
                                    'message': f"No matching data found in CSV for filter: {filters}",
                                    'filters': filters,
                                    'source': 'csv'
                                }
                                logger.warning(f"⚠️ CSV-specific query returned 0 - no data found in CSV")
                            else:
                                # General query - CSV has no matches, but data might be in PDFs/other docs
                                # Filter out CSV chunks from RAG results to avoid confusion
                                original_count = len(filtered_results)
                                filtered_results = [
                                    res for res in filtered_results 
                                    if not res.get('file_path', '').lower().endswith('.csv')
                                ]
                                filtered_count = len(filtered_results)
                                
                                if filtered_count < original_count:
                                    logger.info(f"ℹ️ CSV query returned 0, filtered out {original_count - filtered_count} CSV chunks, keeping {filtered_count} non-CSV chunks (PDFs, Word docs, etc.)")
                                else:
                                    logger.info(f"ℹ️ CSV query returned 0, no CSV chunks to filter (all chunks are from other sources)")
                        else:
                            error_msg = exact_result.get('error', 'unknown')
                            logger.warning(f"❌ CSV query returned error: {error_msg}. Full result keys: {list(exact_result.keys())}")
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
            if exact_answer.get('no_data'):
                # CSV-specific query with no data
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
                logger.info(f"⚠️ Added 'no data' message to response for CSV-specific query")
            else:
                # Successful CSV query
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
        
        # Add note if CSV returned 0 for general query (CSV chunks already filtered out)
        if not exact_answer and not list_answer and csv_query_available and needs_exact:
            # Check if we filtered out CSV chunks (this means CSV returned 0 for general query)
            csv_chunks_in_results = any(
                res.get('file_path', '').lower().endswith('.csv') 
                for res in filtered_results
            )
            if not csv_chunks_in_results and len(filtered_results) > 0:
                findings.append(
                    f"**Note:** CSV query returned no matches for the specified filters. "
                    f"Searching in other document sources (PDFs, Word documents, etc.)...\n\n"
                )
                logger.info(f"ℹ️ Added note that CSV had no matches, using other document sources")
        
        if not exact_answer and not list_answer:
            logger.warning("⚠️ No CSV query answer available - response will use RAG chunks only")
        
        for res in filtered_results:
            text = res.get("text", "").replace("\n", " ")
            file_name = res.get("file_name", "Unknown")
            directory = res.get("directory_name", "")
            file_path = res.get("file_path", "")
            
            # Add source type indicator to help agent distinguish data sources
            if file_path.lower().endswith('.csv'):
                source_note = "[CSV Data]"
            elif file_path.lower().endswith('.pdf'):
                source_note = "[PDF Document]"
            elif file_path.lower().endswith(('.doc', '.docx')):
                source_note = "[Word Document]"
            elif file_path.lower().endswith(('.xls', '.xlsx')):
                source_note = "[Excel Document]"
            else:
                source_note = "[Document]"
            
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
                    metadata_parts.append(f"chunk_index:{meta['chunk_index']}")
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
            
            # Use 2000 character limit to include full chunk text for detailed information
            if len(text) > 2000:
                findings.append(f"- {source_note} {source_info}: {text[:2000]}...{metadata_str}")
                logger.info(f"📝 Truncated chunk text from {len(text)} to 2000 characters for file: {file_name}")
            else:
                findings.append(f"- {source_note} {source_info}: {text}{metadata_str}")
                logger.info(f"📝 Included full chunk text ({len(text)} characters) for file: {file_name}")

        return "\n".join(findings)


# ---------- SQL AGENT (PostgreSQL → CSV → Vector Fallback) ----------

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
                logger.info(f"🔍 SQLAgent: SQL query result - success: {sql_result.get('success')}, has_results: {bool(sql_result.get('results'))}, generated_sql: {sql_result.get('generated_sql', 'N/A')[:100]}")
                print(f"🔍 SQLAgent: SQL query result - success: {sql_result.get('success')}, has_results: {bool(sql_result.get('results'))}")
                print(f"🔍 SQLAgent: Generated SQL: {sql_result.get('generated_sql', 'N/A')}")
                
                if sql_result.get("success") and sql_result.get("results"):
                    results_data = sql_result.get("results", {})
                    row_count = results_data.get("row_count", 0)
                    
                    logger.info(f"🔍 SQLAgent: Results data - row_count: {row_count}, columns: {results_data.get('columns', [])}")
                    print(f"🔍 SQLAgent: Results data - row_count: {row_count}, columns: {results_data.get('columns', [])}")
                    
                    if row_count > 0:
                        logger.info(f"✅ SQLAgent: PostgreSQL SQL query successful with {row_count} rows")
                        print(f"✅ SQLAgent: PostgreSQL SQL query successful with {row_count} rows")
                        # Format SQL results
                        findings = []
                        findings.append(
                            f"═══════════════════════════════════════════════════════════\n"
                            f"**POSTGRESQL SQL QUERY RESULTS:**\n"
                            f"Database: {sql_result.get('database_id_used', database_id)}\n"
                            f"SQL: {sql_result.get('generated_sql', 'N/A')}\n"
                            f"Rows: {row_count}\n"
                            f"═══════════════════════════════════════════════════════════\n\n"
                        )
                        
                        # Add result rows
                        rows = results_data.get("rows", [])
                        columns = results_data.get("columns", [])
                        
                        if rows:
                            # Format as table
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
        
        # Step 2: Try CSV query (reuse AiSearchAgent CSV logic)
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
            person_names = extract_person_names(query)
            initial_results = graphrag_search(
                query=query,
                top_k=5,
                similarity_threshold=0.7,
                keywords=person_names if person_names else None,
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
        
        # Step 3: Fall back to Vector/GraphRAG search (reuse AiSearchAgent logic)
        logger.info("SQLAgent: Falling back to Vector/GraphRAG search")
        from ..tools.neo4j_graphrag import graphrag_search
        
        person_names = extract_person_names(query)
        results = graphrag_search(
            query=query,
            top_k=12,
            similarity_threshold=0.75,
            keywords=person_names if person_names else None,
            keyword_match_type="any",
            keyword_boost=0.4
        )
        
        if not results:
            return "No results found in PostgreSQL SQL, CSV, or Vector search."
        
        filtered_results = filter_results_by_exact_match(results, query, min_similarity=0.7)
        filtered_results = filtered_results[:8]
        
        if not filtered_results:
            return "No relevant results found after filtering in Vector search."
        
        findings = []
        findings.append("**Vector/GraphRAG Search Results (fallback):**\n")
        
        for res in filtered_results:
            text = res.get("text", "").replace("\n", " ")
            file_path = res.get("file_path", "")
            file_name = res.get("file_name", "Unknown")
            
            if file_path.lower().endswith('.csv'):
                source_note = "[CSV Data]"
            elif file_path.lower().endswith('.pdf'):
                source_note = "[PDF Document]"
            elif file_path.lower().endswith(('.doc', '.docx')):
                source_note = "[Word Document]"
            else:
                source_note = "[Document]"
            
            source_info = file_path if file_path else file_name
            # Use 2000 character limit to include full chunk text for detailed information
            if len(text) > 2000:
                findings.append(f"- {source_note} {source_info}: {text[:2000]}...")
                logger.info(f"📝 Truncated chunk text from {len(text)} to 2000 characters for file: {file_name}")
            else:
                findings.append(f"- {source_note} {source_info}: {text}")
                logger.info(f"📝 Included full chunk text ({len(text)} characters) for file: {file_name}")
        
        return "\n".join(findings)


# ---------- NEO4J GRAPHRAG SEARCH (Foundry) ----------

class Neo4jGraphRAGAgent:
    """Search agent using Neo4j GraphRAG retrieval"""
    
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, query: str) -> str:
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
                        elif 'error' not in exact_result and exact_result.get('total', 0) == 0:
                            # CSV returned 0 - check if query is CSV-specific
                            is_csv_specific = is_csv_specific_query(query)
                            
                            if is_csv_specific:
                                # Query is CSV-specific, so 0 means no data in CSV
                                exact_answer = {
                                    'total': 0,
                                    'no_data': True,
                                    'message': f"No matching data found in CSV for filter: {filters}",
                                    'filters': filters,
                                    'source': 'csv'
                                }
                                logger.warning(f"⚠️ CSV-specific query returned 0 - no data found in CSV")
                            else:
                                # General query - CSV has no matches, but data might be in PDFs/other docs
                                # Filter out CSV chunks from RAG results to avoid confusion
                                original_count = len(filtered_results)
                                filtered_results = [
                                    res for res in filtered_results 
                                    if not res.get('file_path', '').lower().endswith('.csv')
                                ]
                                filtered_count = len(filtered_results)
                                
                                if filtered_count < original_count:
                                    logger.info(f"ℹ️ CSV query returned 0, filtered out {original_count - filtered_count} CSV chunks, keeping {filtered_count} non-CSV chunks (PDFs, Word docs, etc.)")
                                else:
                                    logger.info(f"ℹ️ CSV query returned 0, no CSV chunks to filter (all chunks are from other sources)")
                        else:
                            error_msg = exact_result.get('error', 'unknown')
                            logger.warning(f"❌ CSV query returned error: {error_msg}. Full result keys: {list(exact_result.keys())}")
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
            if exact_answer.get('no_data'):
                # CSV-specific query with no data
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
                logger.info(f"⚠️ Added 'no data' message to response for CSV-specific query")
            else:
                # Successful CSV query
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
        
        # Add note if CSV returned 0 for general query (CSV chunks already filtered out)
        if not exact_answer and not list_answer and csv_query_available and needs_exact:
            # Check if we filtered out CSV chunks (this means CSV returned 0 for general query)
            csv_chunks_in_results = any(
                res.get('file_path', '').lower().endswith('.csv') 
                for res in filtered_results
            )
            if not csv_chunks_in_results and len(filtered_results) > 0:
                findings.append(
                    f"**Note:** CSV query returned no matches for the specified filters. "
                    f"Searching in other document sources (PDFs, Word documents, etc.)...\n\n"
                )
                logger.info(f"ℹ️ Added note that CSV had no matches, using other document sources")
        
        if not exact_answer and not list_answer:
            logger.warning("⚠️ No CSV query answer available - response will use RAG chunks only")
        
        for res in filtered_results:
            text = res.get("text", "").replace("\n", " ")
            file_name = res.get("file_name", "Unknown")
            directory = res.get("directory_name", "")
            file_path = res.get("file_path", "")
            
            # Add source type indicator to help agent distinguish data sources
            if file_path.lower().endswith('.csv'):
                source_note = "[CSV Data]"
            elif file_path.lower().endswith('.pdf'):
                source_note = "[PDF Document]"
            elif file_path.lower().endswith(('.doc', '.docx')):
                source_note = "[Word Document]"
            elif file_path.lower().endswith(('.xls', '.xlsx')):
                source_note = "[Excel Document]"
            else:
                source_note = "[Document]"
            
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
            
            # Use 2000 character limit to include full chunk text for detailed information
            if len(text) > 2000:
                findings.append(f"- {source_note} {source_info}: {text[:2000]}...{metadata_str}")
                logger.info(f"📝 Truncated chunk text from {len(text)} to 2000 characters for file: {file_name}")
            else:
                findings.append(f"- {source_note} {source_info}: {text}{metadata_str}")
                logger.info(f"📝 Included full chunk text ({len(text)} characters) for file: {file_name}")

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
        logger.info(f"🤖 [AGENT INVOKED] NewsReporterAgent (ID: {self._id})")
        print(f"🤖 [AGENT INVOKED] NewsReporterAgent (ID: {self._id})")
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
        logger.info(f"🤖 [AGENT INVOKED] ReviewAgent (ID: {self._id})")
        print(f"🤖 [AGENT INVOKED] ReviewAgent (ID: {self._id})")
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
