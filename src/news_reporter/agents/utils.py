"""Shared utilities for agent classes."""

from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional, Tuple, Set

logger = logging.getLogger(__name__)


def infer_header_from_chunk(text: str, file_name: str = "") -> Tuple[str, List[str]]:
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


def extract_person_names_and_mode(
    query: str, vocab_set: Optional[Set[str]] = None
) -> Tuple[List[str], bool]:
    """Extract person names and determine if query is person-centric.
    
    Uses corpus-learned vocabulary instead of hardcoded keyword lists.
    
    Args:
        query: User query text
        vocab_set: Optional pre-loaded vocabulary set
        
    Returns:
        Tuple of (names_list, is_person_query)
    """
    try:
        from ..tools.header_vocab import extract_person_names_and_mode as _extract
        return _extract(query, vocab_set)
    except ImportError:
        # Fallback: basic name extraction
        logger.warning("header_vocab module not available, using basic name extraction")
        words = query.split()
        names = [w.strip('.,!?;:') for w in words if w and w[0].isupper() and len(w.strip('.,!?;:')) > 2]
        return names, False


def extract_person_names(query: str) -> List[str]:
    """Extract potential person names from query (backward compatibility)."""
    names, _ = extract_person_names_and_mode(query)
    return names


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
    import re
    
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
        relationship_keywords = ['relationship', 'connection', 'connect', 'between', 'work together']
        query_lower = query.lower()
        is_relationship_query_generic = any(keyword in query_lower for keyword in relationship_keywords)
        
        # Count potential names in query
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', query)
        common_words = {'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 
                        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'I', 'Is', 'Are'}
        potential_names = [w for w in capitalized_words if w not in common_words]
        
        # For relationship queries with multiple names, be more lenient with similarity
        if is_relationship_query_generic and len(potential_names) >= 2:
            logger.info(f"🔗 [filter] Detected relationship query in generic mode: {potential_names}")
            print(f"🔗 [filter] Relationship query with potential names: {potential_names}")
            # Keep results that mention any of the potential names, even with low similarity
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
                name_found = any(name.lower() in text or name.lower() in header_text or name.lower() in file_name for name in potential_names if name)
                
                # Check if this chunk has organization context (employment, work details)
                has_org_context = any(keyword in text or keyword in header_text for keyword in org_keywords)
                
                # Keep if name found, or org context + name in file, or high similarity
                if name_found or (has_org_context and any(n.lower() in file_name for n in potential_names if n)) or res.get("similarity", 0.0) >= 0.3:
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
    person_names_lower = [n.lower() for n in (person_names or []) if n]
    
    if not person_names_lower:
        # Person mode but no valid names - fall back to similarity only
        logger.info(f"📋 [filter] Person mode but no person names - using similarity threshold")
        return [res for res in results if res.get("similarity", 0.0) >= 0.3]
    
    # Detect if query is asking about relationships between multiple people
    relationship_keywords = [
        'relationship', 'connection', 'connect', 'work together', 'worked with',
        'know each other', 'collaborate', 'related', 'between'
    ]
    query_lower = query.lower()
    
    # Count potential person names in query (capitalized words that aren't common words)
    capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', query)
    common_words = {'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 
                    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'I', 'Is', 'Are'}
    potential_names = [w for w in capitalized_words if w not in common_words]
    
    is_relationship_query = (
        (len(potential_names) >= 2 or len(person_names_lower) >= 2) and 
        any(keyword in query_lower for keyword in relationship_keywords)
    )
    
    # Detect if query contains attribute keywords (skills, experience, education, etc.)
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
            names_to_check.extend([n.lower() for n in potential_names if n])
        
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
        
        # 🔥 PERSON IDENTITY MODE: Requires name in text/header or file match for verification
        # For relationship queries: Keep ALL chunks that mention ANY of the people, regardless of similarity
        name_match = name_found_in_text or name_found_in_header
        name_or_file_match = name_match or name_found_in_file

        if not is_relationship_query and not name_or_file_match:
            logger.info(
                f"❌ [filter] Result {i} FILTERED OUT: no name/file match, sim={similarity:.3f}, file='{file_name}'"
            )
            print(f"❌ [filter] Result {i} FILTERED OUT: no name/file match")
            continue

        # Standard similarity threshold only for non-relationship queries
        if not is_relationship_query and similarity < 0.3:
            logger.info(f"❌ [filter] Result {i} FILTERED OUT: similarity={similarity:.3f} < 0.3")
            continue
        
        # For relationship queries: Keep if ANY person name matches
        if is_relationship_query and name_match:
            filtered.append(res)
            logger.info(
                f"✅ [filter] Result {i} KEPT (RELATIONSHIP QUERY): similarity={similarity:.3f}, "
                f"name_in_text={name_found_in_text}, name_in_header={name_found_in_header}, "
                f"file='{file_name}'"
            )
            print(f"✅ [filter] Result {i} KEPT (RELATIONSHIP): sim={similarity:.3f}, name_match={name_match}")
        # For other person queries: Keep if name or file scope matches
        elif not is_relationship_query and name_or_file_match:
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
