"""
Header Vocabulary Module - Corpus-Driven Query Classification

This module builds and uses a vocabulary of document header phrases learned from 
your actual knowledge base. This eliminates hardcoded keyword lists and makes
query classification generic across any document type (resumes, SOPs, policies, etc.).

Usage:
    # At ingestion time (after chunking):
    from header_vocab import build_header_vocab, save_header_vocab
    vocab = build_header_vocab(chunks)
    save_header_vocab(vocab, "header_vocab.json")
    
    # At query time:
    from header_vocab import load_header_vocab, match_following_header_phrase
    vocab_set = load_header_vocab("header_vocab.json")
    matched = match_following_header_phrase(tokens, idx, vocab_set)
"""

import json
import re
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Module-level cache for loaded vocabulary
_HEADER_VOCAB_CACHE: Optional[Set[str]] = None
_HEADER_VOCAB_PATH: Optional[str] = None


def normalize_header(s: str) -> str:
    """Normalize a header string for consistent matching.
    
    Args:
        s: Raw header text
        
    Returns:
        Normalized lowercase string with cleaned punctuation
    """
    s = (s or "").strip().lower()
    # Remove bracket-y noise, punctuation -> spaces
    s = re.sub(r"[\[\]\(\)\{\}<>]", " ", s)
    # Keep unicode letters + digits broadly
    s = re.sub(r"[^a-z0-9áéíóúñüàèìòùâêîôûäëïöü\s\-:/]", " ", s)
    s = s.replace(":", " ").replace("/", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def header_ngrams(header: str, min_n: int = 1, max_n: int = 4) -> List[str]:
    """Extract n-grams from a normalized header string.
    
    Args:
        header: Normalized header text
        min_n: Minimum n-gram length (tokens)
        max_n: Maximum n-gram length (tokens)
        
    Returns:
        List of n-gram phrases
    """
    toks = header.split()
    out = []
    for n in range(min_n, max_n + 1):
        for i in range(0, len(toks) - n + 1):
            out.append(" ".join(toks[i:i + n]))
    return out


def build_header_vocab(
    chunks: Iterable[Dict[str, Any]], 
    min_count: int = 3,
    max_ngram: int = 4
) -> Dict[str, int]:
    """Build header vocabulary from document chunks.
    
    Iterates all chunks and extracts header phrases from metadata.
    Uses n-grams so "professional experience" is matchable even if 
    the full header is "Professional Work Experience".
    
    Args:
        chunks: Iterable of chunk dictionaries with metadata
        min_count: Minimum frequency to include phrase (filters rare headers)
        max_ngram: Maximum n-gram length to extract
        
    Returns:
        Dictionary mapping header phrases to their frequency counts
    """
    c = Counter()

    for ch in chunks:
        meta = ch.get("metadata", {}) or {}

        candidates = []
        
        # Get header_text from metadata or top-level
        ht = meta.get("header_text") or ch.get("header_text")
        if ht and ht != "N/A":
            candidates.append(ht)

        # Get parent_headers
        parents = meta.get("parent_headers") or ch.get("parent_headers") or []
        candidates.extend([p for p in parents if p and p != "N/A"])

        # Get inferred header if available
        inf = meta.get("inferred_header") or ch.get("inferred_header")
        if inf and inf != "N/A":
            candidates.append(inf)

        for raw in candidates:
            norm = normalize_header(raw)
            if not norm:
                continue

            # Count ngrams so "professional experience" is matchable
            for g in header_ngrams(norm, 1, max_ngram):
                c[g] += 1

    # Generic pruning: remove ultra-rare phrases (statistical, not hardcoded)
    vocab = {k: v for k, v in c.items() if v >= min_count}
    
    logger.info(f"[HeaderVocab] Built vocabulary with {len(vocab)} phrases from {sum(c.values())} total occurrences")
    
    return vocab


def save_header_vocab(vocab: Dict[str, int], path: str, top_k: int = 5000) -> None:
    """Save header vocabulary to JSON file.
    
    Args:
        vocab: Vocabulary dictionary (phrase -> count)
        path: Output file path
        top_k: Maximum number of phrases to keep (size control)
    """
    # Keep the most common phrases
    items = sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(dict(items), ensure_ascii=False, indent=2), 
        encoding="utf-8"
    )
    
    logger.info(f"[HeaderVocab] Saved {len(items)} phrases to {path}")


def load_header_vocab(path: Optional[str] = None, force_reload: bool = False) -> Set[str]:
    """Load header vocabulary from JSON file.
    
    Uses module-level caching to avoid repeated file reads.
    
    Args:
        path: Path to vocabulary JSON file. If None, uses default location.
        force_reload: Force reload even if cached
        
    Returns:
        Set of vocabulary phrases for O(1) lookup
    """
    global _HEADER_VOCAB_CACHE, _HEADER_VOCAB_PATH
    
    # Use default path if not specified
    if path is None:
        # Try common locations
        candidates = [
            Path(__file__).parent.parent / "settings" / "header_vocab.json",
            Path(__file__).parent / "header_vocab.json",
            Path("header_vocab.json"),
        ]
        for p in candidates:
            if p.exists():
                path = str(p)
                break
    
    # Return cached if available and path matches
    if not force_reload and _HEADER_VOCAB_CACHE is not None and _HEADER_VOCAB_PATH == path:
        return _HEADER_VOCAB_CACHE
    
    # Load from file
    if path and Path(path).exists():
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            _HEADER_VOCAB_CACHE = set(data.keys())
            _HEADER_VOCAB_PATH = path
            logger.info(f"[HeaderVocab] Loaded {len(_HEADER_VOCAB_CACHE)} phrases from {path}")
            return _HEADER_VOCAB_CACHE
        except Exception as e:
            logger.warning(f"[HeaderVocab] Failed to load vocabulary from {path}: {e}")
    
    # Return empty set if file doesn't exist (graceful degradation)
    logger.warning(f"[HeaderVocab] No vocabulary file found, using empty vocabulary")
    _HEADER_VOCAB_CACHE = set()
    _HEADER_VOCAB_PATH = path
    return _HEADER_VOCAB_CACHE


def tokenize_query(q: str) -> List[str]:
    """Tokenize a query string into words.
    
    Args:
        q: Query string
        
    Returns:
        List of word tokens
    """
    q = (q or "").strip()
    # Generic tokenization: words/numbers, keep unicode letters
    toks = re.findall(r"[A-Za-z0-9ÁÉÍÓÚÑÜáéíóúñüàèìòùâêîôûäëïöü]+", q)
    return toks


def match_following_header_phrase(
    tokens: List[str], 
    start_idx: int, 
    vocab_set: Optional[Set[str]] = None,
    max_len: int = 4
) -> Optional[str]:
    """Check if tokens following start_idx form a known header phrase.
    
    Args:
        tokens: List of query tokens
        start_idx: Index of the candidate name token
        vocab_set: Set of known header phrases. If None, loads from file.
        max_len: Maximum phrase length to check
        
    Returns:
        Matched header phrase, or None if no match
    """
    if vocab_set is None:
        vocab_set = load_header_vocab()
    
    if not vocab_set:
        return None
    
    # Look ahead 1..max_len tokens after start_idx
    for n in range(1, max_len + 1):
        j = start_idx + 1
        if j + n > len(tokens):
            break
        phrase = " ".join(tokens[j:j + n]).lower()
        if phrase in vocab_set:
            return phrase
    
    return None


def _normalize_possessive(token: str) -> str:
    """Remove possessive suffixes from a token.
    
    Args:
        token: Word token (e.g., "John's", "Alexis'")
        
    Returns:
        Token without possessive suffix
    """
    return re.sub(r"(?:'s|')$", "", token)


def extract_person_names_and_mode(query: str, vocab_set: Optional[Set[str]] = None) -> Tuple[List[str], bool]:
    """Extract person names and determine if query is person-centric.
    
    This is the main entry point for context-aware query classification.
    Returns both extracted names and a boolean indicating if the query
    is asking about a specific person.
    
    Detection logic:
    1. Multi-token name spans (2-4 capitalized tokens, not query words) → person-centric
    2. Single-token name followed by known header phrase → person-centric
    3. Single-token name in context-rich queries (e.g., "who does X work with") → person-centric
    4. No name signals → NOT person-centric
    
    Args:
        query: User query text
        vocab_set: Optional pre-loaded vocabulary set
        
    Returns:
        Tuple of (names_list, is_person_query)
    """
    q = (query or "").strip()
    if not q:
        return [], False
    
    # Load vocabulary if not provided
    if vocab_set is None:
        vocab_set = load_header_vocab()
    
    # Common query words that should not be treated as names
    query_words = {
        'who', 'what', 'where', 'when', 'why', 'how', 'is', 'are', 'am', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'doing', 'will', 'would',
        'could', 'should', 'can', 'shall', 'may', 'might', 'must', 'the', 'a', 'an',
        'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
        'about', 'show', 'tell', 'find', 'get', 'give', 'as', 'if', 'this', 'that',
        'these', 'those', 'me', 'you', 'him', 'her', 'us', 'them', 'it', 'my', 'your',
        'his', 'her', 'its', 'our', 'their', 'all', 'each', 'every', 'both', 'any',
        'some', 'more', 'most', 'other', 'another', 'such', 'no', 'not', 'only',
        'then', 'now', 'just', 'also', 'still', 'up', 'down', 'out', 'over', 'under',
        'through', 'before', 'after', 'during', 'including', 'without', 'between',
    }
    
    # Tokenize case-sensitively to detect actual capitalization
    # Split by spaces/punctuation to maintain case
    import re as re_module
    raw_tokens = re_module.split(r'[\s\-,;.!?()]+', q.strip())
    raw_tokens = [t.strip() for t in raw_tokens if t.strip()]
    
    # ---- 1) Multi-token capitalized name spans (2-4 tokens) ----
    for i in range(len(raw_tokens) - 1):
        # Look for 2-4 consecutive tokens that start with uppercase
        span_len = 1
        while (i + span_len < len(raw_tokens) and 
               raw_tokens[i + span_len] and 
               raw_tokens[i + span_len][0].isupper() and
               span_len < 4):
            span_len += 1
        
        if span_len >= 2:  # Found potential multi-token name
            name_tokens = raw_tokens[i:i + span_len]
            
            # Filter out if ANY token is a common query word
            if any(t.lower() in query_words for t in name_tokens):
                continue
            
            # CORPUS-DRIVEN CHECK: If the last token appears in learned vocabulary,
            # it's likely a document header (learned from actual corpus),
            # not part of a person name. Extract just the name part.
            if vocab_set:
                last_token_lower = name_tokens[-1].lower()
                # Check if this token appears in any vocabulary phrase
                # (using corpus-learned header patterns, not hardcoded lists)
                if any(last_token_lower in phrase for phrase in vocab_set):
                    person_tokens = name_tokens[:-1]
                    if person_tokens:
                        clean_name = " ".join(_normalize_possessive(t.strip(".,!?;:()[]{}")) for t in person_tokens)
                        logger.debug(f"[PersonMode] Name '{clean_name}' followed by vocab phrase containing '{name_tokens[-1]}'")
                        return [clean_name], True
            
            # Valid multi-token name found (e.g., "John Smith", "Maria Rodriguez")
            clean_name = " ".join(_normalize_possessive(t.strip(".,!?;:()[]{}")) for t in name_tokens)
            logger.debug(f"[PersonMode] Multi-token name found: '{clean_name}'")
            return [clean_name], True
    
    # ---- 2) Single-token name detection ----
    for i, tok in enumerate(raw_tokens):
        # Skip short tokens or query words
        if len(tok) <= 2 or tok.lower() in query_words:
            continue
        
        # Check if token starts with uppercase (potential name)
        if tok and tok[0].isupper():
            clean_name = _normalize_possessive(tok.strip(".,!?;:()[]{}"))
            
            # Get context: tokens after this token
            context_after = [raw_tokens[j].lower() for j in range(i + 1, min(len(raw_tokens), i + 5))]
            
            if not context_after:
                continue
            
            # ---- 2a) Header vocabulary matching (if available) ----
            if vocab_set:
                # Check if next token(s) form a known header phrase
                for n in range(1, min(5, len(context_after) + 1)):
                    phrase = " ".join(context_after[:n])
                    if phrase in vocab_set:
                        logger.debug(f"[PersonMode] Single-token name '{clean_name}' followed by header '{phrase}'")
                        return [clean_name], True
            
            # ---- 2b) Intent-based matching: single-token name in person-oriented queries ----
            person_intent_keywords = {'work', 'colleague', 'team', 'project', 'experience', 'skills', 'history', 'background'}
            
            # Check for person intent keywords in the context
            for keyword in person_intent_keywords:
                if keyword in context_after:
                    logger.debug(f"[PersonMode] Single-token name '{clean_name}' followed by intent keyword '{keyword}'")
                    return [clean_name], True
    
    # ---- 3) No person signals detected ----
    logger.debug(f"[PersonMode] No person signals in query: '{q[:50]}...'")
    return [], False


# Backward compatibility: keep old function signature but use new logic
def extract_person_names(query: str) -> List[str]:
    """Extract potential person names from query.
    
    DEPRECATED: Use extract_person_names_and_mode() instead for proper
    context-aware filtering.
    
    Args:
        query: User query text
        
    Returns:
        List of potential person names
    """
    names, _ = extract_person_names_and_mode(query)
    return names
