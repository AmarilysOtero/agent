from __future__ import annotations
import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from pydantic import BaseModel, Field, ValidationError
from ..foundry_runner import run_foundry_agent, run_foundry_agent_json

logger = logging.getLogger(__name__)

def infer_header_from_chunk(text: str, file_name: str = "") -> tuple[str, list[str]]:
    """Infer header context from chunk text when header_text is N/A.
    
    Generic patterns only—no hardcoded keywords.
    Pattern detection: "Lines that look like headers" (short, capitalized, structured).
    
    Args:
        text: Chunk text to analyze
        file_name: Original file name for additional context
        
    Returns:
        Tuple of (inferred_header, parent_headers)
    """
    if not text or not text.strip():
        return "N/A", []
    
    lines = text.split('\n')
    inferred_header = "N/A"
    parent_headers = []
    
    # Look for header patterns in the first few lines (purely structural, no keywords)
    for line in lines[:5]:  # Check first 5 lines
        stripped = line.strip()
        if not stripped:
            continue
        
        # Generic header detection (no domain keywords):
        # 1. All caps short line
        # 2. Title case short line
        # 3. Line ending with colon or dash (structural marker)
        
        is_all_caps = stripped.isupper() and len(stripped) > 1
        is_title_case_short = (
            len(stripped) < 80 and
            stripped[0].isupper() and
            not stripped.endswith('.') and
            not stripped[0].isdigit()
        )
        has_structural_marker = stripped.endswith((':',  '-', '–', '—'))
        
        # If any generic header pattern matches
        if (is_all_caps or 
            (is_title_case_short and (len(stripped) < 50 or has_structural_marker or ' ' not in stripped))):
            
            inferred_header = stripped
            logger.debug(f"[InferHeader] Detected header from generic pattern: '{inferred_header}'")
            break
    
    # If still no header but file name has context, use it
    if inferred_header == "N/A" and file_name:
        # Extract meaningful parts from file name
        name_parts = file_name.replace('.pdf', '').replace('.docx', '').replace('.xlsx', '')
        if name_parts and len(name_parts) > 5:  # Reasonable file name
            inferred_header = f"[From {name_parts}]"
            logger.debug(f"[InferHeader] Using file name as context: '{inferred_header}'")
    
    return inferred_header, parent_headers


# Import context-aware person name extraction from header_vocab module
# This uses corpus-learned vocabulary instead of hardcoded keyword lists
try:
    from ..tools.header_vocab import extract_person_names_and_mode, extract_person_names
except ImportError:
    # Fallback if header_vocab module not available
    logger.warning("header_vocab module not available, using basic name extraction")
    
    def extract_person_names(query: str) -> List[str]:
        """Basic fallback: Extract capitalized words that might be names"""
        words = query.split()
        names = [w.strip('.,!?;:') for w in words if w and w[0].isupper() and len(w.strip('.,!?;:')) > 2]
        return names
    
    def extract_person_names_and_mode(query: str, vocab_set: Optional[Set[str]] = None) -> Tuple[List[str], bool]:
        """Fallback: Extract names and always return is_person_query=False"""
        return extract_person_names(query), False


def filter_results_by_exact_match(
    results: List[Dict[str, Any]], 
    query: str, 
    min_similarity: float = 0.88,
    is_person_query: Optional[bool] = None,
    person_names: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Filter search results based on query type (person-centric vs generic).
    
    For person-centric queries: requires name to appear in chunk or file scope match.
    For attribute queries (skills/experience): trusts file scope - keeps chunks from person's file
    For generic queries: only applies similarity threshold, no name enforcement.
    
    Args:
        results: List of search result dictionaries
        query: Original query text
        min_similarity: Minimum similarity to keep result without exact match (person mode)
        is_person_query: If provided, uses this instead of re-detecting
        person_names: If provided, uses these instead of re-extracting
        
    Returns:
        Filtered list of results
    """
    if not results:
        return results
    
    # Use provided values or detect from query
    if is_person_query is None or person_names is None:
        detected_names, detected_mode = extract_person_names_and_mode(query)
        if person_names is None:
            person_names = detected_names
        if is_person_query is None:
            is_person_query = detected_mode
    
    logger.info(f"🔍 [filter_results_by_exact_match] Filtering {len(results)} results for query '{query}'")
    logger.info(f"🔍 [filter_results_by_exact_match] is_person_query={is_person_query}, person_names={person_names}")
    print(f"🔍 [filter_results_by_exact_match] is_person_query={is_person_query}, person_names={person_names}")
    
    # ✅ GENERIC MODE: No name enforcement, just similarity threshold
    if not is_person_query:
        logger.info(f"📋 [filter] Generic mode - only applying similarity threshold >= 0.3")
        print(f"📋 [filter] Generic mode - only applying similarity threshold >= 0.3")
        
        # SPECIAL CASE: Detect relationship queries even in generic mode
        # If query asks about "relationship between X and Y", keep low-similarity results
        relationship_keywords = ['relationship', 'connection', 'connect', 'between', 'work together']
        query_lower = query.lower()
        is_relationship_query_generic = any(keyword in query_lower for keyword in relationship_keywords)
        
        # Count potential names in query
        import re
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', query)
        common_words = {'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 
                        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'I', 'Is', 'Are'}
        potential_names = [w for w in capitalized_words if w not in common_words]
        
        # For relationship queries with multiple names, be more lenient with similarity
        if is_relationship_query_generic and len(potential_names) >= 2:
            logger.info(f"🔗 [filter] Detected relationship query in generic mode: {potential_names}")
            print(f"🔗 [filter] Relationship query with potential names: {potential_names}")
            # Keep results that mention any of the potential names, even with low similarity
            # Also look for organization keywords (DXC, work, employment, role, position) for relationship context
            filtered = []
            org_keywords = ['dxc', 'work', 'position', 'role', 'employment', 'company', 'organization', 'currently', 'hired', 'hired at', 'employed']
            
            for res in results:
                source = res.get("source") or res.get("metadata", {}).get("source")
                if source == "graph_supporting_evidence":
                    filtered.append(res)
                    continue
                
                text = res.get("text", "").lower()
                header_text = (res.get("header_text") or res.get("metadata", {}).get("header_text") or "").lower()
                file_name = (res.get("file") or "").lower()
                
                # Check if ANY potential name appears in text, header, or file
                name_found = any(name.lower() in text or name.lower() in header_text or name.lower() in file_name for name in potential_names)
                
                # Check if this chunk has organization context (employment, work details)
                has_org_context = any(keyword in text or keyword in header_text for keyword in org_keywords)
                
                # Keep if:
                # 1. Name is found (high priority) - include intro sections showing the person
                # 2. Name + org context found - shows employment/role information (BEST for relationships)
                # 3. High similarity (standard threshold)
                # OR just org keyword + name in file/header (employment sections for people)
                if name_found or (has_org_context and any(n.lower() in file_name for n in potential_names)) or res.get("similarity", 0.0) >= 0.3:
                    filtered.append(res)
                    if name_found and has_org_context:
                        logger.info(f"✅ [filter] Kept result (relationship query, NAME+ORG): {res.get('header_text', '')[:50]}")
                    elif name_found:
                        logger.info(f"✅ [filter] Kept result (relationship query, NAME): {res.get('header_text', '')[:50]}")
                    elif has_org_context:
                        logger.info(f"✅ [filter] Kept result (relationship query, ORG): {res.get('header_text', '')[:50]}")
        else:
            # Standard generic mode: just similarity threshold
            filtered = [
                res for res in results 
                if res.get("similarity", 0.0) >= 0.3 or res.get("source") == "graph_supporting_evidence"
            ]
        
        logger.info(f"📊 [filter_results_by_exact_match] Generic mode: kept {len(filtered)} of {len(results)} results")
        print(f"📊 [filter_results_by_exact_match] Generic mode: kept {len(filtered)} of {len(results)} results")
        return filtered
    
    # ✅ PERSON MODE: Lenient matching with file scope awareness
    person_names_lower = [n.lower() for n in (person_names or [])]
    
    if not person_names_lower:
        # Person mode but no valid names - fall back to similarity only
        logger.info(f"📋 [filter] Person mode but no person names - using similarity threshold")
        return [res for res in results if res.get("similarity", 0.0) >= 0.3]
    
    # Detect if query is asking about relationships between multiple people
    # Keywords: relationship, connection, work together, know each other, etc.
    # Also check for multiple capitalized words that look like names
    relationship_keywords = [
        'relationship', 'connection', 'connect', 'work together', 'worked with',
        'know each other', 'collaborate', 'related', 'between'
    ]
    query_lower = query.lower()
    
    # Count potential person names in query (capitalized words that aren't common words)
    import re
    capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', query)
    common_words = {'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 
                    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'I', 'Is', 'Are'}
    potential_names = [w for w in capitalized_words if w not in common_words]
    
    is_relationship_query = (
        (len(potential_names) >= 2 or len(person_names_lower) >= 2) and 
        any(keyword in query_lower for keyword in relationship_keywords)
    )
    
    # Detect if query contains attribute keywords (skills, experience, education, etc.)
    # For attribute queries, we're less strict about name appearing in chunk text
    attribute_keywords = [
        'skill', 'experience', 'education', 'qualification', 'certification',
        'role', 'position', 'project', 'achievement', 'responsibility',
        'background', 'expertise', 'ability', 'competency', 'proficiency'
    ]
    is_attribute_query = any(keyword in query_lower for keyword in attribute_keywords)
    
    logger.info(f"🔍 [filter] Person mode: person_names={person_names_lower}, potential_names={potential_names}, is_attribute_query={is_attribute_query}, is_relationship_query={is_relationship_query}")
    print(f"🔍 [filter] Person mode: person_names={person_names_lower}, potential_names={potential_names}, is_attribute_query={is_attribute_query}, is_relationship_query={is_relationship_query}")
    
    filtered = []
    for i, res in enumerate(results, 1):
        # Always preserve graph supporting evidence (high-confidence graph facts)
        source = res.get("source") or res.get("metadata", {}).get("source")
        if source == "graph_supporting_evidence":
            filtered.append(res)
            logger.info(f"✅ [filter] Result {i} KEPT (GRAPH EVIDENCE): source={source}")
            print(f"✅ [filter] Result {i} KEPT (GRAPH EVIDENCE)")
            continue
        
        # RAW DEBUG: Print all keys and a sample of the dict for Result 3
        if i == 3:
            logger.info(f"[DEBUG-KEYS-3] Result 3 top-level keys: {list(res.keys())}")
            logger.info(f"[DEBUG-KEYS-3] Result 3 metadata keys: {list(res.get('metadata', {}).keys())}")
            logger.info(f"[DEBUG-RAW-3] header_text in res? {res.get('header_text')}")
            logger.info(f"[DEBUG-RAW-3] header_text in metadata? {res.get('metadata', {}).get('header_text')}")
            logger.info(f"[DEBUG-RAW-3] keyword_score in res? {res.get('keyword_score')}")
            logger.info(f"[DEBUG-RAW-3] keyword_score in metadata? {res.get('metadata', {}).get('keyword_score')}")
            print(f"[DEBUG-KEYS-3] Result 3 keys: {list(res.keys())}")
        
        text = res.get("text", "").lower()
        similarity = res.get("similarity", 0.0)
        
        # Get metadata dict first
        metadata = res.get("metadata", {})
        
        # Normalize field names - check metadata first, then top-level
        hybrid_score = res.get("hybrid_score") or metadata.get("hybrid_score") or similarity
        keyword_score = metadata.get("keyword_score") or res.get("keyword_score") or 0.0
        
        file_name = res.get("file_name") or res.get("file") or "?"
        
        # Header text - check metadata first, then top-level
        header_raw = metadata.get("header_text") or res.get("header_text") or res.get("header") or ""
        header_text = header_raw.lower() if header_raw else ""
        
        # Keywords - check metadata first, then top-level
        keywords_raw = metadata.get("keywords") or res.get("keywords") or []
        keywords_list = [k.lower() for k in keywords_raw] if keywords_raw else []
        
        # DEBUG: Print what we're actually receiving
        logger.info(f"[DEBUG] Result {i}: header_text='{header_raw}' (from metadata or top-level), keywords={keywords_raw[:3] if keywords_raw else []}, hybrid={hybrid_score:.3f}, kw_score={keyword_score:.3f}")
        print(f"[DEBUG] Result {i}: header='{header_raw}', kw_score={keyword_score:.3f}, hybrid={hybrid_score:.3f}")
        
        # Check if ANY of the person names appear in text or header
        # Also check for potential names from query if it's a relationship query
        names_to_check = person_names_lower.copy()
        if is_relationship_query and potential_names:
            names_to_check.extend([n.lower() for n in potential_names])
        
        name_found_in_text = any(name in text for name in names_to_check)
        name_found_in_header = any(name in header_text for name in names_to_check)
        
        # Check file name for person's name
        file_name_lower = file_name.lower() if file_name else ""
        name_found_in_file = any(name in file_name_lower for name in person_names_lower)
        
        # Check if this chunk is actually about the requested attribute
        is_attribute_match = (
            any(attr in header_text for attr in attribute_keywords) or
            any(attr in keywords_list for attr in attribute_keywords)
        )
        
        # DEBUG: Show attribute detection
        if is_attribute_query:
            logger.info(f"[DEBUG] Result {i} attr detection: is_attr_match={is_attribute_match}, header_text='{header_text}', has_attr_keyword={any(attr in keywords_list for attr in attribute_keywords)}")
            print(f"[DEBUG] Result {i} attr: match={is_attribute_match}, header='{header_text}'")
        
        # 🔥 ATTRIBUTE QUERY MODE: Structural chunks (Skills, Experience) don't need name in text
        # They score low on embeddings but high on keyword/header signals
        if is_attribute_query and name_found_in_file and is_attribute_match:
            # Keep if ANY signal is strong (header match, keyword score, or hybrid score)
            if header_text and any(attr in header_text for attr in attribute_keywords):
                # Strong signal: header explicitly says "Skills", "Experience", etc.
                filtered.append(res)
                logger.info(
                    f"✅ [filter] Result {i} KEPT (ATTRIBUTE): sim={similarity:.3f}, hybrid={hybrid_score:.3f}, "
                    f"header='{res.get('header_text', '')}', file='{file_name}'"
                )
                print(f"✅ [filter] Result {i} KEPT (ATTRIBUTE): header='{res.get('header_text', '')}', hybrid={hybrid_score:.3f}")
                continue
            elif keyword_score > 0 or hybrid_score >= 0.45:
                # Good keyword/hybrid match from person's file
                filtered.append(res)
                logger.info(
                    f"✅ [filter] Result {i} KEPT (ATTRIBUTE): sim={similarity:.3f}, hybrid={hybrid_score:.3f}, "
                    f"kw_score={keyword_score:.3f}, file='{file_name}'"
                )
                print(f"✅ [filter] Result {i} KEPT (ATTRIBUTE): kw={keyword_score:.3f}, hybrid={hybrid_score:.3f}")
                continue
            else:
                logger.info(
                    f"❌ [filter] Result {i} FILTERED OUT (WEAK ATTRIBUTE): sim={similarity:.3f}, "
                    f"hybrid={hybrid_score:.3f}, kw={keyword_score:.3f}"
                )
                continue
        
        # 🔥 PERSON IDENTITY MODE: Requires name in text/header for verification
        # For relationship queries: Keep ALL chunks that mention ANY of the people, regardless of similarity
        # Standard similarity threshold only for non-relationship queries
        if not is_relationship_query and similarity < 0.3:
            logger.info(f"❌ [filter] Result {i} FILTERED OUT: similarity={similarity:.3f} < 0.3")
            continue
        
        name_match = name_found_in_text or name_found_in_header
        
        # For relationship queries: Keep if ANY person name matches
        if is_relationship_query and name_match:
            filtered.append(res)
            logger.info(
                f"✅ [filter] Result {i} KEPT (RELATIONSHIP QUERY): similarity={similarity:.3f}, "
                f"name_in_text={name_found_in_text}, name_in_header={name_found_in_header}, "
                f"file='{file_name}'"
            )
            print(f"✅ [filter] Result {i} KEPT (RELATIONSHIP): sim={similarity:.3f}, name_match={name_match}")
        # For other person queries: Keep if name matches or high similarity
        elif not is_relationship_query and (name_match or similarity >= min_similarity):
            filtered.append(res)
            logger.info(
                f"✅ [filter] Result {i} KEPT (PERSON): similarity={similarity:.3f}, "
                f"name_in_text={name_found_in_text}, name_in_header={name_found_in_header}, "
                f"file='{file_name}'"
            )
            print(f"✅ [filter] Result {i} KEPT (PERSON): sim={similarity:.3f}, name_match={name_match}")
        else:
            logger.info(
                f"❌ [filter] Result {i} FILTERED OUT: similarity={similarity:.3f}, "
                f"name_match={name_match}, file='{file_name}'"
            )
            print(f"❌ [filter] Result {i} FILTERED OUT: sim={similarity:.3f}, name_match={name_match}")
    
    logger.info(f"📊 [filter_results_by_exact_match] Person mode: kept {len(filtered)} of {len(results)} results")
    print(f"📊 [filter_results_by_exact_match] Person mode: kept {len(filtered)} of {len(results)} results")
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
        
        # Extract person names and determine if query is person-centric
        # Uses corpus-learned vocabulary for context-aware classification
        from ..tools.header_vocab import extract_attribute_keywords
        
        person_names, is_person_query = extract_person_names_and_mode(query)
        
        logger.info(f"🔍 [AiSearchAgent] Starting search for query: '{query}'")
        logger.info(f"🔍 [AiSearchAgent] Extracted person names: {person_names}, is_person_query: {is_person_query}")
        print(f"🔍 [AiSearchAgent] Starting search for query: '{query}'")
        print(f"🔍 [AiSearchAgent] person_names={person_names}, is_person_query={is_person_query}")
        
        # Build keyword list: person names + attribute keywords (e.g., skills)
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
        logger.info(f"🔍 [QueryClassification] Intent: {query_intent['type']}, routing: {query_intent['routing']}, section_query: {query_intent.get('section_query')}")
        print(f"🔍 [QueryClassification] Intent: {query_intent['type']}, routing: {query_intent['routing']}, section_query: {query_intent.get('section_query')}")
        
        logger.info(f"🔍 [AiSearchAgent] Calling graphrag_search with: top_k=12, similarity_threshold=0.75, keywords={keywords}, keyword_boost={keyword_boost}, is_person_query={is_person_query}, person_names={person_names}, section_query={query_intent.get('section_query')}, use_section_routing={query_intent['routing'] == 'hard'}")
        print(f"🔍 [AiSearchAgent] Calling graphrag_search with: keywords={keywords}, keyword_boost={keyword_boost}, is_person_query={is_person_query}, person_names={person_names}, routing={query_intent['routing']}")
        
        # Call graphrag_search with section routing parameters for hard routing
        results = graphrag_search(
            query=query,
            top_k=12,  # Get more results initially for filtering
            similarity_threshold=0.75,  # Increased from 0.7 to reduce false matches
            keywords=keywords,
            keyword_match_type="any",
            keyword_boost=keyword_boost,
            is_person_query=is_person_query,
            enable_coworker_expansion=True,  # Enable coworker expansion for person queries
            person_names=person_names,
            section_query=query_intent.get('section_query') if query_intent['routing'] == 'hard' else None,
            use_section_routing=query_intent['routing'] == 'hard'
        )

        logger.info(f"📊 [AiSearchAgent] GraphRAG search returned {len(results)} results")
        print(f"📊 [AiSearchAgent] GraphRAG search returned {len(results)} results")
        
        # ========== SUMMARY ANALYSIS: Semantic, Keyword, Graph ==========
        logger.info(f"")
        logger.info(f"🔍 [RETRIEVAL SUMMARY] Analyzing {len(results)} results...")
        print(f"🔍 [RETRIEVAL SUMMARY] Analyzing {len(results)} results...")
        
        # Collect statistics for each signal type
        semantic_stats = []
        keyword_stats = []
        graph_stats = []
        
        for res in results:
            similarity = res.get("similarity", 0.0)
            vector_score = res.get("metadata", {}).get("vector_score", 0.0)
            keyword_score = res.get("metadata", {}).get("keyword_score", 0.0)
            hybrid_score = res.get("hybrid_score", 0.0)
            source = res.get("source", "vector")
            
            # Track semantic signals
            if vector_score > 0 or similarity > 0:
                semantic_stats.append({
                    'similarity': similarity,
                    'vector_score': vector_score,
                    'file': res.get("file_name", "?"),
                    'header': res.get("metadata", {}).get("header_text", "N/A")
                })
            
            # Track keyword signals
            if keyword_score > 0:
                keyword_stats.append({
                    'keyword_score': keyword_score,
                    'file': res.get("file_name", "?"),
                    'header': res.get("metadata", {}).get("header_text", "N/A")
                })
            
            # Track graph signals
            if source == 'graph_traversal':
                graph_stats.append({
                    'hybrid_score': hybrid_score,
                    'entity1': res.get("metadata", {}).get("entity1", "?"),
                    'entity2': res.get("metadata", {}).get("entity2", "?"),
                    'connection': res.get("metadata", {}).get("connection_type", "?"),
                    'via': res.get("metadata", {}).get("via", "?")
                })
        
        # Log semantic analysis
        logger.info(f"")
        logger.info(f"📊 [SEMANTIC ANALYSIS]")
        logger.info(f"   Total with semantic signals: {len(semantic_stats)}")
        if semantic_stats:
            avg_similarity = sum(s['similarity'] for s in semantic_stats) / len(semantic_stats)
            max_similarity = max(s['similarity'] for s in semantic_stats)
            logger.info(f"   Avg similarity: {avg_similarity:.3f}, Max: {max_similarity:.3f}")
            logger.info(f"   Top semantic matches:")
            for s in sorted(semantic_stats, key=lambda x: x['similarity'], reverse=True)[:3]:
                logger.info(f"     - {s['file']}: similarity={s['similarity']:.3f}, header='{s['header']}'")
        
        # Log keyword analysis
        logger.info(f"")
        logger.info(f"🔑 [KEYWORD ANALYSIS]")
        logger.info(f"   Total with keyword matches: {len(keyword_stats)}")
        if keyword_stats:
            avg_keyword = sum(k['keyword_score'] for k in keyword_stats) / len(keyword_stats)
            max_keyword = max(k['keyword_score'] for k in keyword_stats)
            logger.info(f"   Avg keyword score: {avg_keyword:.3f}, Max: {max_keyword:.3f}")
            logger.info(f"   Top keyword matches:")
            for k in sorted(keyword_stats, key=lambda x: x['keyword_score'], reverse=True)[:3]:
                logger.info(f"     - {k['file']}: keyword_score={k['keyword_score']:.3f}, header='{k['header']}'")
        
        # Log graph analysis
        logger.info(f"")
        logger.info(f"🔗 [GRAPH ANALYSIS]")
        logger.info(f"   Total graph connections found: {len(graph_stats)}")
        if graph_stats:
            logger.info(f"   Graph connections:")
            for g in graph_stats:
                logger.info(f"     - {g['entity1']} <-[::{g['connection']}]-> {g['entity2']} via {g['via']}")
        else:
            logger.info(f"   No graph connections discovered (single entity or no relationships found)")
        
        logger.info(f"")
        
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
                
                # Try to get header_text from metadata first, then from top-level
                header_text = metadata.get("header_text", res.get("header_text", "N/A"))
                parent_headers = metadata.get("parent_headers", res.get("parent_headers", []))
                
                logger.info(
                    f"   Result {i}: similarity={similarity:.3f}, hybrid_score={hybrid_score:.3f}, "
                    f"vector_score={vector_score:.3f}, keyword_score={keyword_score:.3f}, "
                    f"header_text='{header_text}', parent_headers={parent_headers}, "
                    f"file='{file_name}', chunk_id='{chunk_id[:50]}...', "
                    f"text_preview='{text_preview}...'"
                )
                print(
                    f"   Result {i}: similarity={similarity:.3f}, hybrid_score={hybrid_score:.3f}, "
                    f"vector_score={vector_score:.3f}, keyword_score={keyword_score:.3f}, "
                    f"header_text='{header_text}', parent_headers={parent_headers}, "
                    f"file='{file_name}', text_preview='{text_preview}...'"
                )
        else:
            logger.warning(f"⚠️ [AiSearchAgent] No results returned from GraphRAG search")
            print(f"⚠️ [AiSearchAgent] No results returned from GraphRAG search")

        if not results:
            return "No results found in Neo4j GraphRAG."

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
        
        # PHASE 5: Additional filtering based on query intent (structural vs semantic)
        if query_intent['routing'] == 'hard':
            # Hard routing: strict filtering for section-based queries
            logger.info(f"🔍 [IntentFilter] Applying HARD routing filter for {query_intent['type']}")
            intent_filtered = []
            for res in filtered_results:
                header_text = res.get("metadata", {}).get("header_text", "").lower()
                section_query_lower = (query_intent.get('section_query') or "").lower()
                
                # For hard routing, keep results that match the section query
                if section_query_lower and section_query_lower in header_text:
                    intent_filtered.append(res)
                    logger.debug(f"  ✓ Kept: header='{header_text}' (matches section_query='{section_query_lower}')")
                elif not section_query_lower:
                    # If no section query, keep all
                    intent_filtered.append(res)
            
            if intent_filtered:
                filtered_results = intent_filtered
                logger.info(f"  ✅ Hard routing filter: {len(intent_filtered)} results kept (exact section match)")
            else:
                # Fallback: use all results if none match exactly
                logger.warning(f"  ⚠️ Hard routing filter: no exact matches, using all results")
        else:
            # Soft routing: semantic-based, use all filtered results
            logger.info(f"🔍 [IntentFilter] Using SOFT routing (semantic) for {query_intent['type']}")
        
        # ========== FILTER SUMMARY ANALYSIS ==========
        logger.info(f"")
        logger.info(f"📊 [FILTER SUMMARY]")
        
        # Analyze what passed filtering
        passed_by_similarity = 0
        passed_by_name = 0
        passed_by_graph = 0
        removed_count = len(results) - len(filtered_results)
        
        for res in filtered_results:
            source = res.get("source", "vector")
            similarity = res.get("similarity", 0.0)
            
            if source == 'graph_traversal':
                passed_by_graph += 1
            elif similarity >= 0.3:
                passed_by_similarity += 1
            else:
                passed_by_name += 1
        
        logger.info(f"   Kept {len(filtered_results)} results:")
        logger.info(f"     - By similarity (>= 0.3): {passed_by_similarity}")
        logger.info(f"     - By name matching: {passed_by_name}")
        logger.info(f"     - By graph discovery: {passed_by_graph}")
        logger.info(f"   Removed {removed_count} results (low similarity, no match)")
        
        # Show what was removed
        if removed_count > 0:
            removed_results = [r for r in results if r not in filtered_results]
            logger.info(f"   Sample removed (low similarity):")
            for r in removed_results[:3]:
                logger.info(f"     - {r.get('file_name', '?')}: similarity={r.get('similarity', 0):.3f}, header='{r.get('metadata', {}).get('header_text', 'N/A')}'")
        
        logger.info(f"")
        
        # Log filtered results details
        if filtered_results:
            logger.info(f"📋 [AiSearchAgent] Filtered results details:")
            print(f"📋 [AiSearchAgent] Filtered results details:")
            for i, res in enumerate(filtered_results, 1):
                similarity = res.get("similarity", 0.0)
                text_preview = res.get("text", "")[:150].replace("\n", " ")
                file_name = res.get("file_name", "?")
                metadata = res.get("metadata", {})
                header_text = metadata.get("header_text", "N/A")
                parent_headers = metadata.get("parent_headers", [])
                logger.info(f"   Filtered {i}: similarity={similarity:.3f}, header_text='{header_text}', parent_headers={parent_headers}, file='{file_name}', text_preview='{text_preview}...'")
                print(f"   Filtered {i}: similarity={similarity:.3f}, header_text='{header_text}', parent_headers={parent_headers}, file='{file_name}', text_preview='{text_preview}...'")
        
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
            similarity = res.get("similarity", 0.0)
            
            # Add source type indicator to help agent distinguish data sources
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
            
            # Include comprehensive metadata for agent context
            metadata_parts = []
            if "hybrid_score" in res and res["hybrid_score"] is not None:
                metadata_parts.append(f"score:{res['hybrid_score']:.2f}")
            if "similarity" in res and res["similarity"] is not None:
                metadata_parts.append(f"similarity:{res['similarity']:.2f}")
            if "metadata" in res and res["metadata"]:
                meta = res["metadata"]
                # Try to get header_text from metadata first, then from top-level
                header_text = meta.get("header_text", res.get("header_text", "N/A"))
                parent_headers = meta.get("parent_headers", res.get("parent_headers", []))
                
                # Infer header if missing (for PDFs without font metadata)
                if header_text == "N/A":
                    chunk_text = res.get("text", "")
                    inferred_header, _ = infer_header_from_chunk(chunk_text, file_name)
                    if inferred_header != "N/A":
                        header_text = inferred_header
                        logger.debug(f"[AiSearchAgent] Inferred header from chunk text: '{header_text}'")
                
                # Debug log for each chunk being added to findings
                logger.info(f"🔍 [AiSearchAgent] Adding chunk to findings: similarity={similarity:.3f}, header_text='{header_text}', parent_headers={parent_headers}, file='{file_name}'")
                print(f"🔍 [AiSearchAgent] Adding chunk: similarity={similarity:.3f}, header_text='{header_text}', parent_headers={parent_headers}")
                
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
                if header_text != "N/A":
                    metadata_parts.append(f"header:{header_text}")
                if parent_headers:
                    metadata_parts.append(f"parents:{len(parent_headers)}")
            
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
    
    def _classify_query_intent(self, query: str, person_names: List[str]) -> dict:
        """Determine if query is section-based or semantic
        
        Detects:
        1. Section-based with person scope (person + attribute)
        2. Section-based cross-document (attribute only, no person)
        3. Semantic-based (general question)
        
        Examples:
            "Kevin's Industry Experience" → section_based_scoped
            "What are Kevin's skills?" → section_based_scoped
            "Tell me about machine learning" → semantic
            "All resumes with Python skills" → section_based_cross_document
            
        Args:
            query: User query text
            person_names: List of person names extracted from query
            
        Returns:
            Dictionary with type, routing, section_query, file_scope
        """
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
        """Extract section-like phrase around attribute keyword
        
        Example:
            "Kevin's industry experience" → "industry experience"
            "technical skills summary" → "technical skills"
            
        Args:
            query: User query
            attribute_keywords: List of attribute keywords
            
        Returns:
            Extracted attribute phrase or first keyword found
        """
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
            # For CSV queries, we don't need person-mode filtering
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
        
        # Step 3: Fall back to Vector/GraphRAG search (reuse AiSearchAgent logic)
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
            metadata = res.get("metadata", {})
            # Try to get header_text from metadata first, then from top-level
            header_text = metadata.get("header_text", res.get("header_text", "N/A"))
            parent_headers = metadata.get("parent_headers", res.get("parent_headers", []))
            
            # Infer header if missing (for PDFs without font metadata)
            if header_text == "N/A":
                chunk_text = res.get("text", "")
                inferred_header, _ = infer_header_from_chunk(chunk_text, file_name)
                if inferred_header != "N/A":
                    header_text = inferred_header
                    logger.debug(f"[SQLAgent] Inferred header from chunk text: '{header_text}'")
            
            # Debug log for each chunk being added to findings
            logger.info(f"🔍 [SQLAgent] Adding chunk to findings: similarity={similarity:.3f}, header_text='{header_text}', parent_headers={parent_headers}, file='{file_name}'")
            print(f"🔍 [SQLAgent] Adding chunk: similarity={similarity:.3f}, header_text='{header_text}', parent_headers={parent_headers}")
            
            if file_path and file_path.lower().endswith('.csv'):
                source_note = "[CSV Data]"
            elif file_path and file_path.lower().endswith('.pdf'):
                source_note = "[PDF Document]"
            elif file_path and file_path.lower().endswith(('.doc', '.docx')):
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
        
        # Extract person names and determine if query is person-centric
        # Uses corpus-learned vocabulary for context-aware classification
        person_names, is_person_query = extract_person_names_and_mode(query)
        
        # Only use keywords for person-centric queries
        keywords = person_names if (is_person_query and person_names) else None
        keyword_boost = 0.4 if keywords else 0.0
        
        # Use improved search parameters to reduce false matches
        results = graphrag_search(
            query=query,
            top_k=12,  # Get more results initially for filtering
            similarity_threshold=0.75,  # Increased from 0.7 to reduce false matches
            keywords=keywords,
            keyword_match_type="any",
            keyword_boost=keyword_boost
        )

        if not results:
            return "No results found in Neo4j GraphRAG."

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
            similarity = res.get("similarity", 0.0)
            
            # Add source type indicator to help agent distinguish data sources
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
            
            # Include comprehensive metadata for agent context
            metadata_parts = []
            if "hybrid_score" in res and res["hybrid_score"] is not None:
                metadata_parts.append(f"score:{res['hybrid_score']:.2f}")
            if "similarity" in res and res["similarity"] is not None:
                metadata_parts.append(f"similarity:{res['similarity']:.2f}")
            if "metadata" in res and res["metadata"]:
                meta = res["metadata"]
                # Try to get header_text from metadata first, then from top-level
                header_text = meta.get("header_text", res.get("header_text", "N/A"))
                parent_headers = meta.get("parent_headers", res.get("parent_headers", []))
                
                # Infer header if missing (for PDFs without font metadata)
                if header_text == "N/A":
                    chunk_text = res.get("text", "")
                    inferred_header, _ = infer_header_from_chunk(chunk_text, file_name)
                    if inferred_header != "N/A":
                        header_text = inferred_header
                        logger.debug(f"[Neo4jGraphRAGAgent] Inferred header from chunk text: '{header_text}'")
                
                # Debug log for each chunk being added to findings
                logger.info(f"🔍 [Neo4jGraphRAGAgent] Adding chunk to findings: similarity={similarity:.3f}, header_text='{header_text}', parent_headers={parent_headers}, file='{file_name}'")
                print(f"🔍 [Neo4jGraphRAGAgent] Adding chunk: similarity={similarity:.3f}, header_text='{header_text}', parent_headers={parent_headers}")
                
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
                if header_text != "N/A":
                    metadata_parts.append(f"header:{header_text}")
                if parent_headers:
                    metadata_parts.append(f"parents:{len(parent_headers)}")
            
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


# ---------- ASSISTANT (Foundry) ----------

class AssistantAgent:
    """Generate natural language responses using RAG context"""
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, query: str, context: str) -> str:
        logger.info(f"🤖 [AGENT INVOKED] AssistantAgent (ID: {self._id})")
        print(f"🤖 [AGENT INVOKED] AssistantAgent (ID: {self._id})")
        
        # If no context found, allow LLM to provide helpful general guidance
        context_instruction = "the context above" if context and context.strip() else "general knowledge"
        fallback_permission = "" if context and context.strip() else "\n- If no specific documentation is available, you may provide general best-practice guidance."
        
        prompt = (
            f"User Question: {query}\n\n"
            f"Retrieved Context:\n{context if context and context.strip() else '(No specific documentation found in knowledge base)'}\n\n"
            "Instructions:\n"
            f"- Answer the user's question using {context_instruction}\n"
            f"- Be conversational, concise, and accurate{fallback_permission}\n"
            "- Cite specific details from the context when available\n"
            "- If citing context, mention the source"
        )
        print("AssistantAgent: using Foundry agent:", self._id)  # keep print
        try:
            return run_foundry_agent(self._id, prompt)
        except RuntimeError as e:
            logger.error("AssistantAgent Foundry error: %s", e)
            raise RuntimeError(
                f"Assistant agent failed: {str(e)}. "
                "Please check your Foundry access and agent configuration."
            ) from e

# ---------- REVIEWER (Foundry, strict JSON) ----------

class ReviewAgent:
    """Review assistant responses for accuracy and completeness"""
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, query: str, candidate_response: str) -> dict:
        """
        Review assistant response and decide if it needs improvement.
        Returns a dict with keys: decision, reason, suggested_changes, revised_script.
        """
        prompt = (
            f"User Query: {query}\n\n"
            f"Assistant Response:\n{candidate_response}\n\n"
            "Review the response for:\n"
            "1. Accuracy - Does it correctly answer the question?\n"
            "2. Completeness - Is the answer sufficient?\n"
            "3. Clarity - Is it easy to understand?\n\n"
            "Return ONLY STRICT JSON (no markdown, no prose) with keys:\n"
            '"decision": "accept" or "revise"\n'
            '"reason": brief explanation\n'
            '"suggested_changes": what to improve (empty if accept)\n'
            '"revised_script": improved version (empty if accept)'
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
                "revised_script": data.get("revised_script", candidate_response),
            }
        except Exception as e:
            logger.error("Review parse error: %s", e)
            # Fail-safe: accept last response to avoid infinite loops
            return {
                "decision": "accept",
                "reason": "parse_error",
                "suggested_changes": "",
                "revised_script": candidate_response,
            }


# Backward compatibility alias
NewsReporterAgent = AssistantAgent
