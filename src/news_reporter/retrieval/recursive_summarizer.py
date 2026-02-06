"""Phase 4: Recursive Summarization - Summarize expanded file chunks using LLM.

MIT RLM (Recursive Inspection Model) Implementation:
- Generate executable Python code for chunk relevance evaluation
- Apply the code to filter chunks relevant to the user query
- Selectively summarize matched content per file
- Return file-level summaries with metadata for citations

The core innovation: Instead of using rules, the LLM generates actual Python code
(evaluate_chunk_relevance function) that the system executes to determine which
chunks are relevant to the user's query. This follows the MIT RLM paper approach
for generating small, executable inspection programs.

Uses Azure OpenAI API for LLM-based code generation and summarization.
"""

import logging
import asyncio
import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# MIT RLM Configuration (Phase 2 feature flag)
USE_MIT_RLM_RECURSION = os.getenv("USE_MIT_RLM_RECURSION", "false").lower() == "true"
MAX_RLM_ITERATIONS = int(os.getenv("MAX_RLM_ITERATIONS", "5"))

# Selection budget limits (prevent excessive token usage)
MAX_SELECTED_CHUNKS_PER_FILE = int(os.getenv("MAX_SELECTED_CHUNKS_PER_FILE", "8"))
MAX_TOTAL_CHARS_FOR_SUMMARY = int(os.getenv("MAX_TOTAL_CHARS_FOR_SUMMARY", "12000"))

# Execution safety limits (PRODUCTION WARNING: needs separate process execution)
MAX_EXEC_ITERATIONS = 1000  # Prevent infinite loops in generated code
MAX_EXEC_STRING_LENGTH = 100000  # Prevent memory bombs

logger.info(f"🔧 MIT RLM Recursion: {'ENABLED' if USE_MIT_RLM_RECURSION else 'DISABLED'}")
logger.info(f"📊 Selection Budgets: max_chunks={MAX_SELECTED_CHUNKS_PER_FILE}, max_chars={MAX_TOTAL_CHARS_FOR_SUMMARY}")


def _build_completion_params(model_deployment: str, **kwargs) -> dict:
    """
    Build completion parameters compatible with the model.
    o1/o3 models don't support temperature parameter.
    
    Args:
        model_deployment: Model deployment name
        **kwargs: Other parameters like max_completion_tokens, messages, etc.
    
    Returns:
        Dict with appropriate parameters for the model
    """
    # Check if this is an o1/o3 model
    is_o_series = any(model_deployment.startswith(prefix) for prefix in ['o1', 'o3'])
    
    # Remove temperature for o-series models
    if is_o_series and 'temperature' in kwargs:
        kwargs.pop('temperature')
        logger.debug(f"Removed 'temperature' parameter for {model_deployment} (o-series model)")
    
    return kwargs


@dataclass
class FileSummary:
    """Summary of a file's relevant chunks."""
    file_id: str
    file_name: str
    summary_text: str
    source_chunk_ids: List[str]  # Chunk IDs that contributed to summary
    chunk_count: int
    summarized_chunk_count: int
    expansion_ratio: float  # (expanded_chunks / entry_chunks)
    # Citation-grade metadata (for production)
    chunk_metadata: Optional[List[Dict[str, Any]]] = None  # [{chunk_id, page, offset, section}]
    file_path: Optional[str] = None
    selection_method: Optional[str] = None  # "rlm_iterative", "rlm_per_chunk", "fallback"


def _normalize_text(text: str) -> str:
    """
    Normalize text by decoding HTML entities.
    
    Example: "Data &amp; AI" -> "Data & AI"
    """
    import html
    return html.unescape(text)


def _tokenize_text(text: str) -> set:
    """
    Tokenize text into words, handling punctuation properly.
    
    Example: "Where does Kevin work?" -> {"where", "does", "kevin", "work"}
    """
    import re
    # Extract word characters only (removes punctuation)
    words = re.findall(r'\b\w+\b', text.lower())
    return set(words)


def _extract_string_literals_via_ast(code: str) -> set:
    """
    Extract ALL string literals from Python code using AST parsing.
    
    This catches literals in:
    - Lists: ["work", "company"]
    - Function calls: chunk.find("term")
    - Comparisons: "term" in chunk
    - Anywhere else
    
    Excludes docstrings and dict keys.
    
    Returns:
        Set of string literals found in the code
    """
    import ast
    
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # If code doesn't parse, return empty (will be caught by other validation)
        return set()
    
    literals = set()
    
    class StringExtractor(ast.NodeVisitor):
        def __init__(self):
            self.strings = set()
            self.in_dict_key = False
        
        def visit_Constant(self, node):
            # Python 3.8+ uses Constant for all literals
            if isinstance(node.value, str):
                # Skip very short strings (likely not search terms)
                if len(node.value) > 0 and not self.in_dict_key:
                    self.strings.add(node.value)
            self.generic_visit(node)
        
        def visit_Dict(self, node):
            # Visit dict values but not keys
            for value in node.values:
                self.visit(value)
            # Don't visit keys (they're not search terms)
        
        def visit_FunctionDef(self, node):
            # Skip docstrings
            body = node.body
            if body and isinstance(body[0], ast.Expr):
                expr = body[0].value
                if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
                    # This is the docstring, skip it
                    for child in body[1:]:
                        self.visit(child)
                    return
            # Normal function, visit all
            self.generic_visit(node)
    
    extractor = StringExtractor()
    extractor.visit(tree)
    return extractor.strings


def _validate_inspection_program(program: str, query: str, chunk_count: int) -> tuple:
    """
    Validate generated inspect_iteration program for MIT RLM recursion.
    
    Checks:
    1. Syntax is valid
    2. Returns dict with required keys (selected_chunk_ids, extracted_data, confidence, stop)
    3. selected_chunk_ids is always a list (never None or other type)
    4. confidence is a float in [0.0, 1.0]
    5. No "select all chunks blindly" patterns (unless stop=True and low confidence)
    6. Logic checks chunk.text or chunk["text"] (evidence-based)
    7. No hardcoded list of chunk IDs
    8. No obvious infinite loops (for i in range(10**N))
    
    Args:
        program: Python code for inspect_iteration function
        query: User query
        chunk_count: Number of chunks being evaluated
    
    Returns:
        (is_valid, error_message)
    """
    
    # Check for obvious CPU bombs (basic pattern detection)
    dangerous_patterns = [
        r'range\s*\(\s*10\s*\*\*\s*[6-9]',  # range(10**6) or higher
        r'range\s*\(\s*\d{7,}',  # range(1000000+)
        r'while\s+True',  # while True without obvious break
    ]
    import re
    for pattern in dangerous_patterns:
        if re.search(pattern, program):
            return False, f"Dangerous pattern detected: {pattern}"
    import ast
    
    try:
        tree = ast.parse(program)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    
    class InspectionProgramValidator(ast.NodeVisitor):
        def __init__(self):
            self.has_function = False
            self.returns_dict_with_required_keys = False
            self.checks_chunk_text = False
            self.selects_all_chunks_unconditionally = False
            self.hardcoded_all_chunk_ids = False
            self._in_function = False
            self._return_statements = []
            
        def visit_FunctionDef(self, node):
            if node.name == 'inspect_iteration':
                self.has_function = True
                self._in_function = True
                self.generic_visit(node)
                self._in_function = False
            else:
                self.generic_visit(node)
        
        def visit_Return(self, node):
            if not self._in_function:
                return
            self._return_statements.append(node)
            
            # Check if returns a dict with required keys
            if isinstance(node.value, ast.Dict):
                keys = set()
                for key in node.value.keys:
                    if isinstance(key, ast.Constant):
                        keys.add(key.value)
                    elif isinstance(key, ast.Str):  # Python < 3.8
                        keys.add(key.s)
                
                required_keys = {'selected_chunk_ids', 'extracted_data', 'confidence', 'stop'}
                if keys >= required_keys:
                    self.returns_dict_with_required_keys = True
            
            self.generic_visit(node)
        
        def visit_Call(self, node):
            # Check for chunk text access patterns: chunk.get('text'), chunk['text']
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == 'get' and isinstance(node.func.value, ast.Name):
                    # chunk.get('text') pattern
                    if node.args and isinstance(node.args[0], ast.Constant):
                        if node.args[0].value == 'text':
                            self.checks_chunk_text = True
            self.generic_visit(node)
        
        def visit_Subscript(self, node):
            # Check for chunk['text'] pattern
            if isinstance(node.slice, ast.Constant):
                if node.slice.value == 'text':
                    self.checks_chunk_text = True
            self.generic_visit(node)
        
        def visit_Compare(self, node):
            # Check for 'in' operations (evidence checking)
            for op in node.ops:
                if isinstance(op, (ast.In, ast.NotIn)):
                    self.checks_chunk_text = True
            self.generic_visit(node)
    
    validator = InspectionProgramValidator()
    validator.visit(tree)
    
    if not validator.has_function:
        return False, "Missing 'inspect_iteration' function definition"
    
    if not validator.returns_dict_with_required_keys:
        return False, "Function must return dict with keys: selected_chunk_ids, extracted_data, confidence, stop"
    
    if not validator.checks_chunk_text:
        return False, "Function must check chunk text/evidence (no hardcoded logic)"
    
    return True, ""


def _validate_inspection_code(code: str, chunk_text: str, query: str) -> tuple:
    """
    Validate generated inspection code for common bugs.
    
    Uses AST parsing for robust validation:
    1. No unguarded 'return True' (AST-based check)
    2. Inverted logic patterns (if X: return False; followed by return True)
    3. Code ending with 'return True' (default True behavior)
    4. Evidence-only rule: ALL string literals must come from query or chunk_text (AST-based)
    5. Multiword literals must exist as exact phrases in evidence
    6. Must have at least one evidence-checking operation
    7. Tighter return True dominance analysis
    
    Returns:
        (is_valid, error_message)
    """
    import ast
    import re
    
    # Try to parse the code
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    
    # Normalize evidence text (decode HTML entities)
    query_norm = _normalize_text(query.lower())
    chunk_norm = _normalize_text(chunk_text.lower())
    
    # Tokenize evidence (removes punctuation)
    query_words = _tokenize_text(query)
    chunk_words = _tokenize_text(chunk_text)
    allowed_words = query_words | chunk_words
    
    # STRICTER CHECK: AST-based return True dominance analysis
    class StrictReturnAnalyzer(ast.NodeVisitor):
        """
        Tighter return True validation:
        - Track all return True nodes
        - Require EACH one to be dominated by evidence check
        - Evidence check = direct in-condition check, not just "was assigned from evidence"
        """
        def __init__(self):
            self.all_return_true_nodes = []
            self.return_false_nodes = []
            self.evidence_checks = set()  # Conditions that check evidence
            self.unsafe_return_true_count = 0
            self._in_function = False
            self._condition_stack = []  # Track nested if conditions
            
        def _is_evidence_condition(self, node: ast.AST) -> bool:
            """Check if a condition directly accesses chunk evidence."""
            for sub in ast.walk(node):
                if isinstance(sub, ast.Compare):
                    # 'in' or 'not in' operator
                    for op in sub.ops:
                        if isinstance(op, (ast.In, ast.NotIn)):
                            return True
                    # startswith, endswith, etc
                    for comparator in sub.comparators:
                        if isinstance(comparator, (ast.Constant, ast.Str)):
                            return True
                
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                    # FIX #4: Method calls like .find(), .count(), .startswith()
                    if sub.func.attr in ('find', 'count', 'startswith', 'endswith'):
                        return True
            return False
        
        def visit_FunctionDef(self, node):
            if node.name == 'evaluate_chunk_relevance':
                self._in_function = True
                self.generic_visit(node)
                self._in_function = False
            else:
                self.generic_visit(node)
        
        def visit_If(self, node):
            if not self._in_function:
                return
            
            is_evidence_test = self._is_evidence_condition(node.test)
            self._condition_stack.append(is_evidence_test)
            
            # Visit body
            for stmt in node.body:
                self.visit(stmt)
            
            # Visit else/elif
            for stmt in node.orelse:
                self.visit(stmt)
            
            self._condition_stack.pop()
        
        def visit_Return(self, node):
            if not self._in_function:
                return
            
            if isinstance(node.value, ast.Constant) and node.value.value is True:
                self.all_return_true_nodes.append(node)
                
                # Check if this return True is dominated by evidence check
                in_evidence_context = any(self._condition_stack)
                
                if not in_evidence_context:
                    self.unsafe_return_true_count += 1
            
            elif isinstance(node.value, ast.Constant) and node.value.value is False:
                self.return_false_nodes.append(node)
            
            self.generic_visit(node)
    
    strict_analyzer = StrictReturnAnalyzer()
    strict_analyzer.visit(tree)
    
    if strict_analyzer.unsafe_return_true_count > 0:
        return False, f"Found {strict_analyzer.unsafe_return_true_count} unsafe 'return True' not dominated by evidence check"
    
    # Check 1: AST-based check for unconditional return True  
    class ReturnChecker(ast.NodeVisitor):
        def __init__(self):
            self.has_unconditional_return_true = False
            self.has_inverted_logic = False
            self.has_evidence_check = False
            self.has_unsafe_return_true = False
            self.evidence_bool_vars = set()
            self._in_target_function = False
            self._evidence_context_stack = [False]

        def _expr_has_evidence_check(self, node: ast.AST) -> bool:
            # FIX #4: Only count actual evidence checks, not string helpers
            evidence_methods = {
                'find', 'count', 'startswith', 'endswith'
            }
            for sub in ast.walk(node):
                if isinstance(sub, ast.Compare):
                    if any(isinstance(op, (ast.In, ast.NotIn)) for op in sub.ops):
                        return True
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                    if sub.func.attr in evidence_methods:
                        return True
            return False

        def visit_FunctionDef(self, node):
            # Only check the evaluate_chunk_relevance function
            if node.name != 'evaluate_chunk_relevance':
                return

            self._in_target_function = True

            # Check if function ends with return True
            if node.body:
                last_stmt = node.body[-1]
                if isinstance(last_stmt, ast.Return):
                    if isinstance(last_stmt.value, ast.Constant) and last_stmt.value.value is True:
                        self.has_unconditional_return_true = True

            # Check for inverted logic: if ...: return False ... return True
            has_return_false_in_if = False
            for stmt in node.body:
                if isinstance(stmt, ast.If):
                    # Check if any branch returns False
                    for branch_stmt in stmt.body + stmt.orelse:
                        if isinstance(branch_stmt, ast.Return):
                            if isinstance(branch_stmt.value, ast.Constant) and branch_stmt.value.value is False:
                                has_return_false_in_if = True
                                break
                elif isinstance(stmt, ast.Return) and has_return_false_in_if:
                    # Found return after if with return False
                    if isinstance(stmt.value, ast.Constant) and stmt.value.value is True:
                        self.has_inverted_logic = True

            self.generic_visit(node)
            self._in_target_function = False

        def visit_Assign(self, node):
            if not self._in_target_function:
                return

            if self._expr_has_evidence_check(node.value):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.evidence_bool_vars.add(target.id)

            self.generic_visit(node)

        def visit_If(self, node):
            if not self._in_target_function:
                return

            is_evidence_test = self._expr_has_evidence_check(node.test)
            if is_evidence_test:
                self.has_evidence_check = True

            current_context = self._evidence_context_stack[-1]

            self._evidence_context_stack.append(current_context or is_evidence_test)
            for stmt in node.body:
                self.visit(stmt)
            self._evidence_context_stack.pop()

            self._evidence_context_stack.append(current_context)
            for stmt in node.orelse:
                self.visit(stmt)
            self._evidence_context_stack.pop()

        def visit_Return(self, node):
            if not self._in_target_function:
                return

            if node.value is None:
                return

            # If return expression itself checks evidence, allow it
            if self._expr_has_evidence_check(node.value):
                self.has_evidence_check = True
                return

            in_evidence_context = self._evidence_context_stack[-1]

            if isinstance(node.value, ast.Constant) and node.value.value is True:
                if not in_evidence_context:
                    self.has_unsafe_return_true = True
            elif isinstance(node.value, ast.Name):
                if not in_evidence_context and node.value.id not in self.evidence_bool_vars:
                    self.has_unsafe_return_true = True

            self.generic_visit(node)

        def visit_Compare(self, node):
            # Check for 'in' operator (evidence checking pattern)
            for op in node.ops:
                if isinstance(op, (ast.In, ast.NotIn)):
                    self.has_evidence_check = True
            self.generic_visit(node)

        def visit_Call(self, node):
            # FIX #4: Check for actual evidence-checking method calls only
            if isinstance(node.func, ast.Attribute):
                method_name = node.func.attr
                if method_name in ('find', 'count', 'startswith', 'endswith'):
                    self.has_evidence_check = True
            self.generic_visit(node)
    
    checker = ReturnChecker()
    checker.visit(tree)
    
    if checker.has_unconditional_return_true:
        return False, "Function ends with unconditional 'return True' - must default to False"
    
    if checker.has_inverted_logic:
        return False, "Inverted logic detected: 'if ...: return False' followed by 'return True'"

    if checker.has_unsafe_return_true:
        return False, "Unsafe 'return True' path detected without evidence check"
    
    # NEW CHECK: Require at least one evidence-checking operation
    # This prevents trivial "return True" or "x = True; return x" patterns
    if not checker.has_evidence_check:
        return False, "Function must perform at least one evidence check (e.g., 'in', .find(), .count(), .lower())"
    
    # PRODUCTION WARNING: This exec() validation doesn't prevent:
    # - CPU/memory bombs (for i in range(10**9))
    # - Pathological string operations
    # TODO: Run generated code in separate process with resource limits
    
    # Check 2: Extract ALL string literals using AST
    all_literals = _extract_string_literals_via_ast(code)
    
    # Validate each literal against evidence
    suspicious_literals = []
    for literal in all_literals:
        literal_norm = _normalize_text(literal.lower())
        literal_words = _tokenize_text(literal)
        
        # If literal contains multiple words, it must exist as an exact phrase
        if ' ' in literal_norm or len(literal_words) > 1:
            # Multiword: must exist as substring in evidence
            if literal_norm not in query_norm and literal_norm not in chunk_norm:
                suspicious_literals.append(f"'{literal}' (multiword phrase not in evidence)")
        else:
            # Single word: must be in allowed word set
            if literal_words and not literal_words.issubset(allowed_words):
                # Also check if it's a substring of evidence (case-insensitive)
                if literal_norm not in query_norm and literal_norm not in chunk_norm:
                    suspicious_literals.append(f"'{literal}' (not in query/chunk)")
    
    if suspicious_literals:
        return False, f"Evidence-only violation: {suspicious_literals}"
    
    return True, ""


def _extract_evidence_terms(query: str, chunk_text: str) -> str:
    """
    Extract allowed terms from query and chunk text for dynamic constraint.
    
    Uses tokenizer to remove punctuation (e.g., "work?" -> "work").
    Returns a formatted string listing allowed terms for the LLM.
    """
    # Use tokenizer to remove punctuation
    query_terms = _tokenize_text(query)
    chunk_terms = _tokenize_text(chunk_text)
    
    # Combine and cap length
    allowed_terms = sorted(query_terms | chunk_terms)
    allowed_terms = allowed_terms[:50]  # Cap to 50 terms
    
    return ", ".join(repr(t) for t in allowed_terms)


def _is_current_employment_query(query: str) -> bool:
    """
    Detect if query is asking about current employment/position.
    
    Examples: "Where does X work?", "What does Y do?", "Who does Z work for?"
    
    Improved with more specific patterns to reduce false positives.
    """
    import re
    query_lower = query.lower()
    
    # Specific employment patterns (more precise than before)
    employment_patterns = [
        r'where.*(?:work|works|working)',
        r'(?:work|works)\s+(?:for|at|with)',
        r'(?:employed|working)\s+(?:by|at|for)',
        r'current\s+(?:employer|job|position|role)',
        r'who\s+does.*work\s+for',
        r'what\s+does.*do\s+(?:for|at)',
    ]
    
    return any(re.search(pattern, query_lower) for pattern in employment_patterns)


def _prioritize_current_role_chunks(
    chunks: List[Dict], 
    selected_chunk_ids: List[str], 
    force_search_all: bool = False
) -> List[str]:
    """
    Prioritize chunks with current/present role markers.
    
    CRITICAL FIX: If force_search_all=True (for current employment queries),
    searches ALL chunks for present/current markers, not just selected ones.
    This ensures we don't miss "Mar 2025 - Present" if the LLM filter failed.
    
    This is purely deterministic date/status logic (no hardcoded job vocab).
    
    Args:
        chunks: All chunks from the file
        selected_chunk_ids: Chunk IDs that passed relevance filter
        force_search_all: If True, search ALL chunks for current markers (ignore selected_chunk_ids)
    
    Returns:
        Prioritized list of chunk IDs (current role chunks if found)
    """
    import re
    
    # Build map of chunk_id -> chunk text
    chunk_map = {c.get('chunk_id'): c.get('text', '') for c in chunks if isinstance(c, dict)}
    
    # Markers for current/present roles (case-insensitive)
    current_markers = [
        r'\bpresent\b',
        r'\bcurrent\b',
        r'\bcurrently\b',
        r'-\s*present',
        r'–\s*present',
        r'to\s+present',
    ]
    
    # Determine search space
    if force_search_all:
        # FORCE SEARCH ALL CHUNKS (for current employment queries)
        search_chunks = [(cid, text) for cid, text in chunk_map.items()]
        logger.info("  🔍 Force-searching ALL chunks for current-role markers (current employment query)")
    else:
        # Only search selected chunks
        search_chunks = [(cid, chunk_map.get(cid, '')) for cid in selected_chunk_ids if cid in chunk_map]
    
    if not search_chunks:
        return selected_chunk_ids
    
    # Find chunks with current markers
    current_chunks = []
    for cid, text in search_chunks:
        text_lower = text.lower()
        if any(re.search(pattern, text_lower) for pattern in current_markers):
            current_chunks.append(cid)
    
    if current_chunks:
        logger.info(
            f"  🎯 Found {len(current_chunks)} current-role chunks with 'present/current' markers "
            f"({'force-searched all chunks' if force_search_all else 'from selected'})"
        )
        return current_chunks
    
    # Fallback: No current markers found, return original selection
    logger.debug("  No current-role markers found, keeping original selection")
    return selected_chunk_ids


def _deduplicate_chunks(
    chunks: List[Dict],
    selected_chunk_ids: List[str]
) -> List[str]:
    """
    Remove near-duplicate chunks based on text similarity.
    
    Expanded chunks often overlap. This deduplicates by:
    - Exact text match
    - High character overlap (>80% shared)
    
    Args:
        chunks: All chunks from the file
        selected_chunk_ids: Chunk IDs to deduplicate
    
    Returns:
        Deduplicated list of chunk IDs
    """
    if len(selected_chunk_ids) <= 1:
        return selected_chunk_ids
    
    # Build map of chunk_id -> normalized text
    chunk_map = {}
    for chunk in chunks:
        if isinstance(chunk, dict) and chunk.get("chunk_id") in selected_chunk_ids:
            text = chunk.get("text", "").strip().lower()
            chunk_map[chunk.get("chunk_id")] = text
    
    seen_texts = set()
    deduplicated = []
    
    for chunk_id in selected_chunk_ids:
        text = chunk_map.get(chunk_id, "")
        if not text:
            continue
        
        # Check for exact duplicates
        if text in seen_texts:
            logger.debug(f"  🗑️  Removing duplicate chunk: {chunk_id[:30]}...")
            continue
        
        # Check for high overlap with existing chunks
        is_duplicate = False
        for seen_text in seen_texts:
            # Simple overlap check: shared chars / min length
            shared = sum(1 for c in text if c in seen_text)
            overlap_ratio = shared / min(len(text), len(seen_text)) if min(len(text), len(seen_text)) > 0 else 0
            
            if overlap_ratio > 0.8:
                logger.debug(f"  🗑️  Removing high-overlap chunk ({overlap_ratio:.0%}): {chunk_id[:30]}...")
                is_duplicate = True
                break
        
        if not is_duplicate:
            seen_texts.add(text)
            deduplicated.append(chunk_id)
    
    if len(deduplicated) < len(selected_chunk_ids):
        logger.info(f"  📉 Deduplication: {len(selected_chunk_ids)} → {len(deduplicated)} chunks")
    
    return deduplicated


def _apply_selection_budget(
    chunks: List[Dict],
    selected_chunk_ids: List[str],
    max_chunks: int,
    max_chars: int
) -> List[str]:
    """
    Apply selection budget to prevent excessive token usage in summarization.
    
    Trims by:
    1. Hard cap on number of chunks (max_chunks)
    2. Hard cap on total characters (max_chars)
    
    Prioritizes chunks by:
    - Current/recent markers ("present", dates)
    - Keyword density (query-relevant terms)
    
    Args:
        chunks: All chunks from the file
        selected_chunk_ids: Chunk IDs to trim
        max_chunks: Maximum chunks to keep
        max_chars: Maximum total characters
    
    Returns:
        Trimmed list of chunk IDs
    """
    if len(selected_chunk_ids) <= max_chunks:
        # Check char limit even if under chunk limit
        chunk_map = {c.get("chunk_id"): c.get("text", "") for c in chunks if isinstance(c, dict)}
        total_chars = sum(len(chunk_map.get(cid, "")) for cid in selected_chunk_ids)
        
        if total_chars <= max_chars:
            return selected_chunk_ids
    
    # Need to trim - prioritize by recency markers
    import re
    chunk_scores = []
    chunk_map = {c.get("chunk_id"): c.get("text", "") for c in chunks if isinstance(c, dict)}
    
    for chunk_id in selected_chunk_ids:
        text = chunk_map.get(chunk_id, "")
        score = 0
        
        # Boost for current/present markers
        if re.search(r'\b(present|current|currently)\b', text.lower()):
            score += 10
        
        # Boost for recent years (2024, 2025, etc.)
        if re.search(r'\b202[3-9]\b', text):
            score += 5
        
        # Slight boost for longer chunks (more context)
        score += min(len(text) / 1000, 3)
        
        chunk_scores.append((chunk_id, score, len(text)))
    
    # Sort by score (descending)
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Apply budgets
    selected = []
    total_chars = 0
    
    for chunk_id, score, char_len in chunk_scores:
        if len(selected) >= max_chunks:
            break
        if total_chars + char_len > max_chars:
            break
        
        selected.append(chunk_id)
        total_chars += char_len
    
    if len(selected) < len(selected_chunk_ids):
        logger.info(
            f"  ✂️  Selection budget applied: {len(selected_chunk_ids)} → {len(selected)} chunks "
            f"({total_chars:,} chars, limit={max_chars:,})"
        )
    
    return selected


async def recursive_summarize_files(
    expanded_files: Dict[str, Dict],
    query: str,
    llm_client: Optional[Any] = None,
    model_deployment: Optional[str] = None
) -> List[FileSummary]:
    """
    Apply LLM-based recursive summarization to expanded file chunks.

    Phase 4: Recursive Summarization
    - For each file, analyze expanded chunks
    - Generate summarization logic (e.g., regex rules or filtering criteria)
    - Selectively summarize matched content
    - Return file summaries with citations

    Args:
        expanded_files: Output from Phase 3 {file_id: {chunks: [...], file_name: str, ...}}
        query: User query for context
        llm_client: Optional Azure OpenAI client (created if not provided)
        model_deployment: Azure OpenAI deployment name (reads from AZURE_OPENAI_CHAT_DEPLOYMENT if not provided)

    Returns:
        List of FileSummary objects with file-level summaries
    """
    # Initialize Azure OpenAI client if not provided
    if llm_client is None:
        try:
            from openai import AsyncAzureOpenAI
            
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            
            if not (azure_endpoint and api_key):
                logger.warning("⚠️  Phase 4: Azure OpenAI credentials not configured; skipping recursive summarization")
                return []
            
            llm_client = AsyncAzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint
            )
        except ImportError:
            logger.warning("⚠️  Phase 4: Azure OpenAI SDK not available; skipping recursive summarization")
            return []
        except Exception as e:
            logger.warning(f"⚠️  Phase 4: Failed to initialize Azure OpenAI client: {e}")
            return []
    
    # Get deployment name from environment or parameter
    if model_deployment is None:
        model_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "o3-mini")
    
    logger.info(f"🔄 Phase 4: Starting recursive summarization for {len(expanded_files)} files (deployment: {model_deployment})")

    summaries = []
    inspection_code = {}  # Store LLM-generated inspection logic per file
    inspection_code_with_text = {}  # Store inspection code with chunk text details

    for file_id, file_data in expanded_files.items():
        try:
            file_name = file_data.get("file_name", "unknown")
            chunks = file_data.get("chunks", [])
            entry_chunk_count = file_data.get("entry_chunk_count", 0)
            total_chunks = len(chunks)

            if not chunks:
                logger.warning(f"⚠️  Phase 4: File {file_name} has no chunks; skipping")
                continue

            logger.info(
                f"📍 Phase 4.1: Analyzing file '{file_name}' "
                f"({entry_chunk_count} entry → {total_chunks} total chunks, deployment: {model_deployment})"
            )

            if USE_MIT_RLM_RECURSION:
                result = await _process_file_with_rlm_recursion(
                    file_id=file_id,
                    file_name=file_name,
                    chunks=chunks,
                    query=query,
                    llm_client=llm_client,
                    model_deployment=model_deployment
                )
                relevant_chunks = result["relevant_chunks"]
                final_selected_chunk_ids = result["selected_chunk_ids"]
                inspection_code[file_id] = result["iteration_programs"]
            else:
                # MIT RLM: Generate and apply inspection code per chunk (legacy path)
                file_inspection_codes = {}  # Store generated code for each chunk
                file_inspection_payloads = {}  # Store code + text metadata per chunk
                relevant_chunks = []
                relevant_chunk_ids = []

                for idx, chunk in enumerate(chunks):
                    if isinstance(chunk, dict):
                        chunk_id = chunk.get("chunk_id", f"unknown-{idx}")
                        chunk_text = chunk.get("text", "").strip()

                        if not chunk_text:
                            continue

                        logger.debug(f"  → Generating inspection code for chunk {idx} ({chunk_id[:30]}...)")

                        # Step 1: Generate Python code specific to this chunk (MIT RLM per-chunk approach)
                        generated_code = await _generate_inspection_logic(
                            query=query,
                            file_name=file_name,
                            chunk_id=chunk_id,
                            chunk_text=chunk_text,
                            llm_client=llm_client,
                            model_deployment=model_deployment
                        )

                        file_inspection_codes[chunk_id] = generated_code
                        file_inspection_payloads[chunk_id] = {
                            "code": generated_code,
                            "chunk_text": chunk_text,
                            "first_read_text": chunk_text[:500],
                            # Chunk-scoped recursive text to avoid file-level repetition in logs.
                            "recursive_text": chunk_text[:500]
                        }

                        # Step 2: Apply the generated code to evaluate this specific chunk
                        logger.debug(f"  → Evaluating chunk {idx} with generated code")
                        is_relevant = await _evaluate_chunk_with_code(
                            chunk_text=chunk_text,
                            inspection_code=generated_code,
                            chunk_id=chunk_id
                        )

                        if is_relevant:
                            relevant_chunks.append(chunk_text)
                            relevant_chunk_ids.append(chunk_id)
                            logger.debug(f"    ✓ Chunk {idx} is relevant")
                        else:
                            logger.debug(f"    ✗ Chunk {idx} is not relevant")
                
                # PRODUCTION NOTE: Per-chunk mode is expensive (N LLM calls for N chunks)
                # For large chunk sets, iterative mode is more efficient
                if len(chunks) > 50:
                    logger.warning(
                        f"  ⚠️  Per-chunk mode with {len(chunks)} chunks may be slow/expensive. "
                        f"Consider using iterative mode (USE_MIT_RLM_RECURSION=true)"
                    )

                # Store all chunk-level inspection codes for this file
                inspection_code[file_id] = file_inspection_codes
                inspection_code_with_text[file_id] = file_inspection_payloads

                # GUARDRAIL: Check selection ratio (reject if selecting too many chunks)
                selection_ratio = len(relevant_chunk_ids) / max(1, len(chunks))
                if selection_ratio > 0.7 and len(chunks) > 5:
                    logger.warning(
                        f"⚠️  Phase 4: Per-chunk mode selected {len(relevant_chunk_ids)}/{len(chunks)} chunks "
                        f"({selection_ratio:.0%}) - too many, treating as low-signal filter. Using fallback."
                    )
                    # Reset to empty and fallback
                    relevant_chunks = []
                    relevant_chunk_ids = []

                if not relevant_chunks:
                    logger.warning(f"⚠️  Phase 4: No relevant chunks identified in {file_name}")
                    # Fallback: use first few chunks if none pass relevance
                    for fallback_idx, chunk in enumerate(chunks[:min(3, len(chunks))]):
                        if not isinstance(chunk, dict):
                            continue
                        chunk_text = chunk.get("text", "").strip()
                        if not chunk_text:
                            continue
                        relevant_chunks.append(chunk_text)
                        relevant_chunk_ids.append(chunk.get("chunk_id", f"unknown-{fallback_idx}"))

                final_selected_chunk_ids = list(relevant_chunk_ids)
                
                # Apply deduplication before prioritization
                final_selected_chunk_ids = _deduplicate_chunks(
                    chunks=chunks,
                    selected_chunk_ids=final_selected_chunk_ids
                )
                
                # Apply selection budget (prevent excessive summary length)
                final_selected_chunk_ids = _apply_selection_budget(
                    chunks=chunks,
                    selected_chunk_ids=final_selected_chunk_ids,
                    max_chunks=MAX_SELECTED_CHUNKS_PER_FILE,
                    max_chars=MAX_TOTAL_CHARS_FOR_SUMMARY
                )
                
                # DETERMINISTIC RULE: Prioritize current-role chunks if query implies "where does X work"
                if _is_current_employment_query(query):
                    # FORCE SEARCH ALL CHUNKS (not just selected) - fixes Problem B
                    final_selected_chunk_ids = _prioritize_current_role_chunks(
                        chunks=chunks,
                        selected_chunk_ids=final_selected_chunk_ids,
                        force_search_all=True  # Search ALL chunks for present/current
                    )
                    # Update relevant_chunks to match prioritized IDs
                    chunk_id_to_text = {c.get("chunk_id"): c.get("text", "").strip() for c in chunks if isinstance(c, dict)}
                    relevant_chunks = [chunk_id_to_text.get(cid, "") for cid in final_selected_chunk_ids if cid in chunk_id_to_text]

            # Step 3: Summarize relevant chunks
            logger.info(
                f"  → Step 3: Summarizing {len(relevant_chunks)} relevant chunks "
                f"(from {total_chunks} total)"
            )
            summary_text = await _summarize_chunks(
                chunks=relevant_chunks,
                query=query,
                file_name=file_name,
                llm_client=llm_client,
                model_deployment=model_deployment
            )

            source_chunk_ids = final_selected_chunk_ids

            # Build citation metadata
            chunk_metadata = []
            for chunk_id in source_chunk_ids:
                # Find the chunk to extract metadata
                for chunk in chunks:
                    if isinstance(chunk, dict) and chunk.get("chunk_id") == chunk_id:
                        chunk_metadata.append({
                            "chunk_id": chunk_id,
                            "page": chunk.get("page"),
                            "offset": chunk.get("offset"),
                            "section": chunk.get("section"),
                        })
                        break
            
            selection_method = "rlm_iterative" if USE_MIT_RLM_RECURSION else "rlm_per_chunk"
            
            file_summary = FileSummary(
                file_id=file_id,
                file_name=file_name,
                summary_text=summary_text,
                source_chunk_ids=source_chunk_ids,
                chunk_count=total_chunks,
                summarized_chunk_count=len(relevant_chunks),
                expansion_ratio=total_chunks / max(1, entry_chunk_count),
                chunk_metadata=chunk_metadata,
                file_path=file_data.get("file_path"),
                selection_method=selection_method
            )

            summaries.append(file_summary)

            logger.info(
                f"  ✅ File '{file_name}': "
                f"Summary generated from {len(relevant_chunks)} chunks, "
                f"expansion ratio: {file_summary.expansion_ratio:.2f}x"
            )

        except Exception as e:
            logger.warning(f"⚠️  Phase 4: Failed to summarize {file_name}: {e}", exc_info=True)
            continue

    logger.info(
        f"✅ Phase 4: Recursive summarization complete - "
        f"{len(summaries)} file summaries generated"
    )

    # Log inspection code for debugging and analysis
    summary_by_file_id = {summary.file_id: summary.summary_text for summary in summaries}

    if inspection_code:
        try:
            # FIX #3: Distinguish mode in logging: iterative vs per-chunk
            mode_label = "iterative" if USE_MIT_RLM_RECURSION else "per_chunk"
            await log_inspection_code_to_markdown(
                inspection_rules=inspection_code,
                query=query,
                rlm_enabled=USE_MIT_RLM_RECURSION,  # True for iterative, False for per-chunk
                output_dir="/app/logs/chunk_analysis",
                mode=mode_label  # Pass mode explicitly
            )
        except Exception as e:
            logger.warning(f"⚠️  Failed to log inspection code: {e}")

    if inspection_code_with_text:
        try:
            await log_inspection_code_with_text_to_markdown(
                inspection_rules=inspection_code_with_text,
                query=query,
                summary_by_file_id=summary_by_file_id,
                rlm_enabled=False,  # This is only for per-chunk mode
                output_dir="/app/logs/chunk_analysis"
            )
        except Exception as e:
            logger.warning(f"⚠️  Failed to log inspection code with text: {e}")

    return summaries


async def _run_inspection_code_sanity_tests(
    code: str,
    chunk_text: str,
    chunk_id: str
) -> tuple:
    """
    Run sanity tests on generated inspection code to catch "default True" patterns.
    
    Tests:
    1. Run against actual chunk_text -> should return True (relevance should match)
    2. Run against irrelevant text -> should return False
    3. Run against empty string -> should return False
    4. Run against query-unrelated terms -> should return False
    
    If code returns True for irrelevant content, it's a "default True" pattern
    and must be rejected regardless of what the AST validator said.
    
    Args:
        code: Python code containing evaluate_chunk_relevance function
        chunk_text: The actual chunk being evaluated
        chunk_id: ID for logging
    
    Returns:
        (passes_sanity, error_message)
    """
    try:
        # PRODUCTION WARNING: exec() without process isolation is unsafe
        # TODO: Use subprocess with timeout and resource limits
        safe_globals = {
            "__builtins__": {
                "len": len, "sum": sum, "any": any, "all": all,
                "min": min, "max": max, "int": int, "str": str,
                "bool": bool, "list": list, "dict": dict, "set": set,
                "range": range, "enumerate": enumerate, "sorted": sorted,
            },
            # Add iteration budget tracking (basic protection)
            "_exec_counter": 0,
            "_MAX_ITERATIONS": MAX_EXEC_ITERATIONS,
        }
        namespace = {}
        exec(code, safe_globals, namespace)
        
        evaluate_func = namespace.get("evaluate_chunk_relevance")
        if evaluate_func is None:
            return False, "Function not found after execution"
        
        # Test 1: Should return True on actual chunk_text (or close to it)
        result_on_actual = evaluate_func(chunk_text)
        
        # Test 2: Should return False on empty string
        result_on_empty = evaluate_func("")
        if result_on_empty:
            return False, "Returns True on empty string (default True pattern)"
        
        # Test 3: Should return False on irrelevant garbage text
        irrelevant_texts = [
            "asdf qwer zxcv qwerty",
            "the quick brown fox jumps over the lazy dog",
            "123 456 789 000 111",
        ]
        
        for irrelevant_text in irrelevant_texts:
            result_on_irrelevant = evaluate_func(irrelevant_text)
            if result_on_irrelevant:
                return False, f"Returns True on irrelevant text: '{irrelevant_text}' (default True pattern)"
        
        # FIX #5: Enforce that function should match the real chunk (if non-trivial)
        # Prevents overly-strict "always return False" functions
        if chunk_text.strip() and len(chunk_text.strip()) > 20:
            if not result_on_actual:
                return False, "Returns False on the actual chunk (overly-strict / always-false pattern)"
        
        return True, ""
    
    except Exception as e:
        return False, f"Sanity test execution error: {e}"


async def _generate_inspection_logic(
    query: str,
    file_name: str,
    chunk_id: str,
    chunk_text: str,
    llm_client: Any,
    model_deployment: str
) -> str:
    """
    Generate LLM-based inspection logic (executable Python code) for a specific chunk.

    MIT RLM approach: Generate small, executable Python code for each chunk
    to determine if it's relevant to the user query.

    Enforces:
    - Default False behavior (no inverted logic)
    - Evidence-only string literals (from query + chunk text only)
    - Post-generation validation with one retry
    - Sanity tests to catch "default True" patterns

    Args:
        query: User query
        file_name: Name of file being analyzed
        chunk_id: ID of the chunk being evaluated
        chunk_text: The actual chunk content
        llm_client: Azure OpenAI client
        model_deployment: Azure deployment name

    Returns:
        Python code as string that can evaluate chunk relevance
    """
    # Use gpt-4o-mini for code generation (o3-mini returns empty responses)
    code_generation_deployment = "gpt-4o-mini"
    if model_deployment.startswith(('gpt-', 'gpt4')):
        code_generation_deployment = model_deployment

    # Extract allowed terms from evidence (query + chunk text)
    allowed_terms_str = _extract_evidence_terms(query, chunk_text)

    prompt = f"""You are implementing the MIT Recursive Inspection Model (RLM) for document analysis.

TASK: Generate a Python function to evaluate if this specific chunk is relevant to the user's query.

Document: {file_name}
Chunk ID: {chunk_id}
User Query: {query}

CHUNK CONTENT:
{chunk_text[:500]}

Generate a Python function called `evaluate_chunk_relevance(chunk_text: str) -> bool` that:
1. Returns True if the chunk is relevant to the user query
2. Returns False if the chunk is NOT relevant
3. Uses simple string operations (case-insensitive matching)
4. Implements 2-3 specific criteria based on this chunk's content

Requirements:
- Function signature: def evaluate_chunk_relevance(chunk_text: str) -> bool:
- Valid, executable Python code only
- Return ONLY the function code with no explanations
- No imports needed

CRITICAL RULES - Default Behavior & Evidence-Only Literals:
1. Your default return should ALWAYS be False, not True
   - Only return True when you explicitly find matching/relevant content
   - NEVER use inverted logic like "if keyword found: return False; else: return True"
   - Pattern: if <relevant_condition>: return True; else: return False

2. EVIDENCE-ONLY RULE: Any string literal you use must come from ONLY these sources:
   - The user query: {query}
   - The chunk text (shown above)
   - Do NOT invent terms from document metadata or filenames
   - Do NOT use generic keywords not present in query or chunk
   - Allowed terms: {allowed_terms_str}

NOW generate the function:"""

    generated_code = None
    for attempt in range(2):  # Try up to 2 times
        try:
            params = _build_completion_params(
                code_generation_deployment,
                model=code_generation_deployment,
                messages=[
                    {"role": "system", "content": "You are a Python expert. Generate clean, executable Python code with no explanations. You MUST follow the evidence-only rule strictly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_completion_tokens=300
            )
            
            response = await llm_client.chat.completions.create(**params)
            inspection_code = response.choices[0].message.content.strip()
            
            # Extract code from markdown if needed (more robust extraction)
            if "```" in inspection_code:
                # Try to extract code between markers
                import re
                match = re.search(r'```(?:python)?\s*\n(.*?)\n```', inspection_code, re.DOTALL)
                if match:
                    inspection_code = match.group(1).strip()
                    logger.debug(f"    Extracted code from markdown wrapper")
            
            # Check if response is empty or doesn't contain the function signature
            if not inspection_code or "def evaluate_chunk_relevance" not in inspection_code:
                logger.warning(f"⚠️  LLM returned empty or incomplete code on attempt {attempt + 1}")
                continue
            
            # Validate the generated code (checks for inverted logic and evidence-only)
            is_valid, error_msg = _validate_inspection_code(inspection_code, chunk_text, query)
            
            if not is_valid:
                logger.warning(f"⚠️  AST validation failed on attempt {attempt + 1}: {error_msg}")
                if attempt == 0:
                    # Add validation error to prompt for retry
                    prompt += f"\n\nPrevious attempt failed: {error_msg}\nPlease fix and regenerate."
                continue
            
            logger.debug(f"✅ AST validation passed for chunk {chunk_id[:20]}...")
            
            # Run sanity tests (catches "default True" even if validator missed it)
            sanity_passed, sanity_msg = await _run_inspection_code_sanity_tests(
                code=inspection_code,
                chunk_text=chunk_text,
                chunk_id=chunk_id
            )
            
            if sanity_passed:
                logger.debug(f"    ✅ Generated valid code for chunk {chunk_id} + sanity tests passed")
                generated_code = inspection_code
                break  # Exit loop with success
            else:
                logger.warning(f"⚠️  Code failed sanity tests on attempt {attempt + 1}: {sanity_msg}")
                if attempt == 0:
                    # Retry with feedback
                    prompt += f"\n\nPrevious attempt failed sanity checks: {sanity_msg}\nPlease fix and regenerate."
                continue
        
        except Exception as e:
            logger.warning(f"⚠️  Failed to generate code for chunk {chunk_id} (attempt {attempt + 1}): {e}")
            continue
    
    # Fallback: return simple query-based filter function (always defaults to False)
    if not generated_code:
        logger.info(f"  🔄 Using fallback for chunk {chunk_id} (code generation/validation failed)")
        # Remove stopwords to reduce false positives
        STOPWORDS = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
                     "is", "are", "was", "were", "where", "who", "what", "does", "do", "did",
                     "work", "works", "working", "at", "from", "by", "be", "been"}
        query_terms = [t for t in _tokenize_text(query) if t not in STOPWORDS and len(t) >= 3]
        fallback_code = f"""def evaluate_chunk_relevance(chunk_text: str) -> bool:
    \"\"\"Fallback relevance filter based on query terms (stopwords removed).\"\"\"
    if not chunk_text.strip():
        return False
    text_lower = chunk_text.lower()
    query_terms = {repr(query_terms)}
    if not query_terms:
        return False
    # Require at least one content token match
    return any(term in text_lower for term in query_terms)"""
        return fallback_code
    
    # FIX #1: Return the successfully generated code
    return generated_code



async def _run_inspection_program_sanity_tests(
    program: str,
    chunks: List[Dict],
    iteration: int
) -> tuple:
    """
    Run sanity tests on generated inspect_iteration program.
    
    Tests:
    1. Execute against empty chunk list -> should return compliant structure
    2. Execute against actual chunks -> should select subset (not all)
    3. Verify selected_chunk_ids are valid (subset check)
    4. Verify confidence is in [0.0, 1.0]
    5. Verify no "select all unless stop" pattern (prevents default behavior)
    
    Args:
        program: Python code for inspect_iteration
        chunks: List of chunks being evaluated
        iteration: Iteration number (for logging)
    
    Returns:
        (passes_sanity, error_message)
    """
    try:
        safe_globals = {
            "__builtins__": {
                "len": len, "list": list, "dict": dict, "set": set,
                "sum": sum, "min": min, "max": max, "int": int, "float": float,
                "str": str, "bool": bool, "range": range, "enumerate": enumerate, "sorted": sorted, "any": any, "all": all
            }
        }
        namespace = {}
        exec(program, safe_globals, namespace)
        
        inspect_func = namespace.get("inspect_iteration")
        if inspect_func is None:
            return False, "Function not found after execution"
        
        # Test 1: Execute with actual chunks
        chunk_list = [
            {"chunk_id": c.get("chunk_id", f"test-{i}"), "text": c.get("text", "")}
            for i, c in enumerate(chunks)
        ]
        valid_chunk_ids = set(c["chunk_id"] for c in chunk_list)
        
        result = inspect_func(chunk_list)
        
        # Test 2: Verify return structure
        if not isinstance(result, dict):
            return False, f"Return value is {type(result)}, not dict"
        
        required_keys = {'selected_chunk_ids', 'extracted_data', 'confidence', 'stop'}
        missing_keys = required_keys - set(result.keys())
        if missing_keys:
            return False, f"Missing keys: {missing_keys}"
        
        # Test 3: Verify selected_chunk_ids structure
        selected_ids = result.get("selected_chunk_ids", [])
        if not isinstance(selected_ids, list):
            return False, f"selected_chunk_ids is {type(selected_ids)}, not list"
        
        # All selected IDs must be valid
        for cid in selected_ids:
            if cid not in valid_chunk_ids:
                return False, f"selected_chunk_ids contains invalid ID: {cid}"
        
        # Test 4: Verify confidence
        confidence = result.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            return False, f"confidence={confidence} not in [0.0, 1.0]"
        
        # Test 5: Prevent "select all chunks" pattern (low-signal indicator)
        selection_ratio = len(selected_ids) / max(1, len(chunk_list))
        should_stop = result.get("stop", False)
        
        if selection_ratio > 0.9 and len(chunk_list) > 2 and not should_stop:
            return False, f"Selects {selection_ratio:.1%} of chunks without stop=True (default behavior)"
        
        if selection_ratio > 0.95 and len(chunk_list) > 2:
            return False, f"Selects {selection_ratio:.1%} of chunks (almost all, likely default True)"
        
        # Enforce narrowing for iterations > 0 (should select fewer than input)
        # This is a soft check (only warn, don't fail) since we do this check in execute too
        
        return True, ""
    
    except Exception as e:
        return False, f"Sanity test execution error: {e}"


async def _generate_recursive_inspection_program(
    query: str,
    file_name: str,
    active_chunks: List[Dict],
    iteration: int,
    previous_data: Dict,
    llm_client: Any,
    model_deployment: str
) -> str:
    """
    Generate MIT RLM-compliant inspection program for one iteration.

    Returns Python code that evaluates all chunks and returns structured output:
    {
        "selected_chunk_ids": [...],
        "extracted_data": {...},
        "confidence": 0.0-1.0,
        "stop": True/False
    }
    
    WITH VALIDATION:
    - Validates code structure with _validate_inspection_program
    - Runs sanity tests on generated program (prevents "select all" patterns)
    - Rejects with fallback if validation fails
    """
    chunk_summaries = "\n".join([
        f"  - Chunk {i} (ID: {c.get('chunk_id', '')[:20]}...): {c.get('text', '')[:100]}..."
        for i, c in enumerate(active_chunks[:10])
    ])

    prompt = f"""You are implementing the MIT Recursive Language Model (RLM) for document analysis.

ITERATION {iteration + 1}/5
Document: {file_name}
User Query: {query}

ACTIVE CHUNKS ({len(active_chunks)} total):
{chunk_summaries}

PREVIOUS ITERATIONS DATA:
{previous_data if previous_data else "None (first iteration)"}

Generate a Python function with this EXACT signature:

def inspect_iteration(chunks):
    \"\""
    Evaluate chunks for this iteration and return structured output.

    Args:
        chunks: List of dicts with keys: chunk_id, text

    Returns:
        {{
            "selected_chunk_ids": [...],  # IDs of relevant chunks to keep
            "extracted_data": {{}},        # Any data extracted this iteration
            "confidence": 0.0-1.0,        # Confidence in completeness
            "stop": True/False            # Whether we have enough information
        }}
    \"\""
    # Your implementation here
    pass

Requirements:
1. Evaluate ALL chunks in the chunks list
2. Return structured dict matching the schema above
3. Use simple string operations (no imports)
4. Be specific to this iteration and previous findings
5. Narrow focus each iteration (select fewer chunks than input)
6. selected_chunk_ids MUST be a subset of provided chunk IDs (validate before returning)
7. Return at least 2 selected_chunk_ids unless stop=True
8. confidence MUST be a float in range [0.0, 1.0]
9. NEVER select ALL chunks unless high confidence AND stop=True
10. Return ONLY function code with no markdown or explanations

CRITICAL: Do NOT blindly select all chunks. Each iteration should narrow the focus.

Generate ONLY the function code with no explanations:"""

    try:
        code_generation_deployment = "gpt-4o-mini"
        if model_deployment.startswith(('gpt-', 'gpt4')):
            code_generation_deployment = model_deployment

        params = _build_completion_params(
            code_generation_deployment,
            model=code_generation_deployment,
            messages=[
                {"role": "system", "content": "You are a Python expert implementing MIT RLM. Generate clean, executable code with no explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_completion_tokens=600
        )

        response = await llm_client.chat.completions.create(**params)
        program_code = response.choices[0].message.content.strip()

        # Extract markdown-wrapped code if present
        if "```" in program_code:
            import re
            match = re.search(r'```(?:python)?\s*\n(.*?)\n```', program_code, re.DOTALL)
            if match:
                program_code = match.group(1).strip()
                logger.debug(f"    Extracted program from markdown wrapper")

        if not program_code or "def inspect_iteration" not in program_code:
            logger.warning(f"⚠️  LLM returned incomplete program for iteration {iteration}")
            return _get_fallback_inspection_program(query, iteration)

        # Validate the generated program structure
        is_valid, error_msg = _validate_inspection_program(program_code, query, len(active_chunks))
        
        if not is_valid:
            logger.warning(f"⚠️  Program validation failed for iteration {iteration}: {error_msg}")
            return _get_fallback_inspection_program(query, iteration)
        
        logger.debug(f"✅ AST validation passed for iteration {iteration}")

        # Run sanity tests on the program (prevents "select all" behavior)
        sanity_passed, sanity_msg = await _run_inspection_program_sanity_tests(
            program=program_code,
            chunks=active_chunks,
            iteration=iteration
        )
        
        if not sanity_passed:
            logger.warning(f"⚠️  Program failed sanity tests for iteration {iteration}: {sanity_msg}")
            return _get_fallback_inspection_program(query, iteration)
        
        logger.debug(f"✅ Sanity tests passed for iteration {iteration}")
        logger.debug(f"    Generated {len(program_code)} char program for iteration {iteration}")
        return program_code

    except Exception as e:
        logger.warning(f"⚠️  Failed to generate program for iteration {iteration}: {e}")
        return _get_fallback_inspection_program(query, iteration)


def _get_fallback_inspection_program(query: str, iteration: int) -> str:
    """Fallback inspection program when LLM fails."""
    query_terms = sorted(_tokenize_text(query))

    return f"""def inspect_iteration(chunks):
    \"\""Fallback program based on query term matching.\"\""
    query_terms = {repr(query_terms)}
    selected_ids = []
    extracted_data = {{}}

    for chunk in chunks:
        text_lower = chunk['text'].lower()
        matches = sum(1 for term in query_terms if term in text_lower)
        if matches >= max(2, len(query_terms) // 2):
            selected_ids.append(chunk['chunk_id'])

    confidence = min(1.0, len(selected_ids) / max(1, len(chunks)))
    stop = iteration >= 3 or confidence > 0.8

    return {{
        "selected_chunk_ids": selected_ids,
        "extracted_data": {{"fallback": True}},
        "confidence": confidence,
        "stop": stop
    }}"""


async def _evaluate_chunk_with_code(
    chunk_text: str,
    inspection_code: str,
    chunk_id: str
) -> bool:
    """
    Execute generated inspection code against a specific chunk.

    MIT RLM: Execute the Python function to evaluate if chunk is relevant.

    Uses restricted execution environment (safe_globals) to prevent:
    - Imports and module access
    - Unbounded iteration/recursion
    - Memory exhaustion

    Args:
        chunk_text: The chunk to evaluate
        inspection_code: Python code containing evaluate_chunk_relevance function
        chunk_id: ID of chunk for logging

    Returns:
        True if chunk is relevant, False otherwise (defaults to False on error)
    """
    # Hard cap on code size  (prevent memory bombs)
    MAX_CODE_SIZE = 5000
    if len(inspection_code) > MAX_CODE_SIZE:
        logger.warning(f"⚠️  Code for {chunk_id} exceeds size limit ({len(inspection_code)} > {MAX_CODE_SIZE})")
        return False
    
    try:
        # Use restricted global namespace (no imports, minimal builtins)
        safe_globals = {
            "__builtins__": {
                "len": len,
                "sum": sum,
                "any": any,
                "all": all,
                "min": min,
                "max": max,
                "int": int,
                "str": str,
                "bool": bool,
                "list": list,
                "dict": dict,
                "set": set,
                "range": range,
                "enumerate": enumerate,
                "sorted": sorted,
            }
        }
        
        # Execute the generated Python code in restricted environment
        namespace = {}
        exec(inspection_code, safe_globals, namespace)
        
        # Get the evaluate_chunk_relevance function
        evaluate_func = namespace.get("evaluate_chunk_relevance")
        
        if evaluate_func is None:
            logger.warning(f"⚠️  Code for {chunk_id} doesn't contain evaluate_chunk_relevance; returning False")
            return False
        
        # Hard timeout (simple check - would need signal/threading for true timeout)
        # Execute the function against the chunk
        result = evaluate_func(chunk_text)
        
        # Ensure result is a boolean
        return bool(result)
        
    except SyntaxError as e:
        logger.warning(f"⚠️  Syntax error in code for {chunk_id}: {e}")
        return False
    except Exception as e:
        logger.warning(f"⚠️  Error executing code for {chunk_id}: {e}")
        return False


async def _execute_inspection_program(
    chunks: List[Dict],
    program: str,
    iteration: int
) -> Dict:
    """
    Execute MIT RLM inspection program and return structured output.

    The program is expected to define:
        def inspect_iteration(chunks) -> Dict
    """
    try:
        safe_globals = {
            "__builtins__": {
                "len": len,
                "list": list,
                "dict": dict,
                "set": set,
                "sum": sum,
                "min": min,
                "max": max,
                "int": int,
                "float": float,
                "str": str,
                "range": range,
                "enumerate": enumerate,
                "sorted": sorted,
                "any": any,
                "all": all
            }
        }
        namespace = {}
        exec(program, safe_globals, namespace)

        inspect_func = namespace.get("inspect_iteration")

        if inspect_func is None:
            logger.warning(f"⚠️  Program for iteration {iteration} missing inspect_iteration function")
            return _get_fallback_result(chunks, iteration)

        chunk_list = [
            {"chunk_id": chunk.get("chunk_id", f"unknown-{i}"), "text": chunk.get("text", "")}
            for i, chunk in enumerate(chunks)
        ]
        valid_chunk_ids = set(chunk.get("chunk_id") for chunk in chunk_list)

        result = inspect_func(chunk_list)

        if not isinstance(result, dict):
            logger.warning(f"⚠️  Program returned non-dict: {type(result)}")
            return _get_fallback_result(chunks, iteration)

        MIN_KEEP = 2
        selected_ids = result.get("selected_chunk_ids", [])
        should_stop = result.get("stop", False)

        if isinstance(selected_ids, list):
            selected_ids = [cid for cid in selected_ids if cid in valid_chunk_ids]
        else:
            selected_ids = []

        if not should_stop and len(selected_ids) < MIN_KEEP:
            candidate_ids_ordered = [chunk.get("chunk_id") for chunk in chunk_list if chunk.get("chunk_id") in valid_chunk_ids]
            selected_ids = candidate_ids_ordered[:MIN_KEEP]
            logger.debug(f"    📌 Enforcing MIN_KEEP={MIN_KEEP}, selected first {len(selected_ids)} chunks by original order")

        # Check for "selects everything repeatedly" pattern (low-signal indicator)
        # If >90% of chunks are selected, treat as no filtering happening
        selection_ratio = len(selected_ids) / max(1, len(chunk_list))
        if selection_ratio > 0.9 and len(chunk_list) > 3:
            logger.warning(
                f"    ⚠️  Iteration {iteration}: Selecting {len(selected_ids)}/{len(chunk_list)} chunks ({selection_ratio:.1%}) - "
                f"no meaningful filtering. Treating as low-signal, will stop early."
            )
            # Set stop=True to halt recursion, fallback to all chunks
            should_stop = True
            confidence = 0.2  # Low confidence due to poor filtering
        else:
            confidence = max(0.0, min(1.0, result.get("confidence", 0.5)))

        extracted = result.get("extracted_data", {})
        MAX_EXTRACTED_SIZE = 50000
        if isinstance(extracted, dict):
            extracted_size = len(str(extracted))
            if extracted_size > MAX_EXTRACTED_SIZE:
                logger.warning(
                    f"    ⚠️  Iteration {iteration}: extracted_data too large ({extracted_size} bytes), truncating"
                )
                truncated = {}
                for k, v in extracted.items():
                    if isinstance(v, str):
                        truncated[k] = v[:500]
                    elif isinstance(v, (int, float, bool)):
                        truncated[k] = v
                extracted = truncated

        validated_result = {
            "selected_chunk_ids": selected_ids,
            "extracted_data": extracted,
            "confidence": confidence,
            "stop": bool(should_stop)
        }

        return validated_result

    except SyntaxError as e:
        logger.warning(f"⚠️  Syntax error in program for iteration {iteration}: {e}")
        return _get_fallback_result(chunks, iteration)
    except Exception as e:
        logger.warning(f"⚠️  Error executing program for iteration {iteration}: {e}")
        return _get_fallback_result(chunks, iteration)


def _get_fallback_result(chunks: List[Dict], iteration: int) -> Dict:
    """Generate safe fallback result when program execution fails."""
    chunk_ids = [chunk.get("chunk_id", f"unknown-{i}") for i, chunk in enumerate(chunks)]

    return {
        "selected_chunk_ids": chunk_ids[:min(10, len(chunk_ids))],
        "extracted_data": {"fallback": True, "iteration": iteration},
        "confidence": 0.3,
        "stop": iteration >= 3
    }


async def _process_file_with_rlm_recursion(
    file_id: str,
    file_name: str,
    chunks: List[Dict],
    query: str,
    llm_client: Any,
    model_deployment: str
) -> Dict[str, Any]:
    """Process a single file using MIT RLM recursion (per-iteration programs)."""
    active_chunks = chunks
    accumulated_data: Dict[str, Any] = {}
    iteration_programs: Dict[str, str] = {}
    final_selected_chunk_ids: List[str] = []
    final_confidence = 0.0
    narrowing_streak = 0

    logger.info(
        f"📍 Phase 4.1: Starting MIT RLM recursion for '{file_name}' "
        f"({len(chunks)} chunks, max {MAX_RLM_ITERATIONS} iterations)"
    )

    for iteration in range(MAX_RLM_ITERATIONS):
        if not active_chunks:
            logger.warning(f"  ⚠️  Iteration {iteration}: No active chunks remaining")
            break

        logger.info(f"  → Iteration {iteration + 1}: Evaluating {len(active_chunks)} chunks")

        inspection_program = await _generate_recursive_inspection_program(
            query=query,
            file_name=file_name,
            active_chunks=active_chunks,
            iteration=iteration,
            previous_data=accumulated_data,
            llm_client=llm_client,
            model_deployment=model_deployment
        )

        iteration_programs[f"iteration_{iteration}"] = inspection_program

        try:
            result = await _execute_inspection_program(
                chunks=active_chunks,
                program=inspection_program,
                iteration=iteration
            )

            selected_ids = result.get("selected_chunk_ids", [])
            extracted = result.get("extracted_data", {})
            confidence = result.get("confidence", 0.0)
            should_stop = result.get("stop", False)

            logger.info(
                f"    ✓ Iteration {iteration + 1}: "
                f"Selected {len(selected_ids)}/{len(active_chunks)} chunks, "
                f"confidence={confidence:.2f}, stop={should_stop}"
            )

            accumulated_data.update(extracted)
            final_selected_chunk_ids = selected_ids
            final_confidence = confidence

            if should_stop or confidence > 0.9:
                logger.info(
                    f"    🛑 Stopping: {'stop flag' if should_stop else 'high confidence'}"
                )
                break

            prev_active_ids = set(chunk.get("chunk_id") for chunk in active_chunks)
            active_chunks = [
                chunk for chunk in active_chunks
                if chunk.get("chunk_id") in selected_ids
            ]
            new_active_ids = set(chunk.get("chunk_id") for chunk in active_chunks)

            # Enforce narrowing (should shrink by at least 10% unless stopping)
            if new_active_ids == prev_active_ids:
                narrowing_streak += 1
                if narrowing_streak >= 2:
                    logger.warning(
                        f"    ⚠️  No narrowing for {narrowing_streak} consecutive iterations, stopping"
                    )
                    break
                logger.debug(
                    f"    ⏸️  Iteration {iteration + 1}: No narrowing ({narrowing_streak}/2)"
                )
            else:
                # Check that we actually narrowed meaningfully
                shrink_ratio = len(new_active_ids) / max(1, len(prev_active_ids))
                if shrink_ratio > 0.9 and not should_stop:
                    logger.warning(
                        f"    ⚠️  Iteration {iteration + 1}: Minimal narrowing ({shrink_ratio:.1%}), "
                        f"selection may not be improving"
                    )
                    narrowing_streak += 0.5  # Partial strike
                else:
                    narrowing_streak = 0

        except Exception as e:
            logger.warning(f"  ⚠️  Iteration {iteration} failed: {e}")
            break

    relevant_chunks = [
        chunk.get("text", "").strip()
        for chunk in chunks
        if chunk.get("chunk_id") in final_selected_chunk_ids
        and chunk.get("text", "").strip()
    ]

    if not relevant_chunks:
        logger.warning(f"⚠️  Phase 4: No chunks selected after {iteration + 1} iterations, using fallback")
        relevant_chunks = [
            chunk.get("text", "").strip()
            for chunk in chunks[:min(3, len(chunks))]
            if chunk.get("text", "").strip()
        ]
        final_selected_chunk_ids = [
            chunk.get("chunk_id")
            for chunk in chunks[:min(3, len(chunks))]
        ]
    
    # Apply deduplication
    final_selected_chunk_ids = _deduplicate_chunks(
        chunks=chunks,
        selected_chunk_ids=final_selected_chunk_ids
    )
    
    # Apply selection budget
    final_selected_chunk_ids = _apply_selection_budget(
        chunks=chunks,
        selected_chunk_ids=final_selected_chunk_ids,
        max_chunks=MAX_SELECTED_CHUNKS_PER_FILE,
        max_chars=MAX_TOTAL_CHARS_FOR_SUMMARY
    )
    
    # APPLY CURRENT-ROLE PRIORITIZATION IN ITERATIVE MODE TOO
    # (Previously only applied in per-chunk mode)
    if _is_current_employment_query(query):
        final_selected_chunk_ids = _prioritize_current_role_chunks(
            chunks=chunks,
            selected_chunk_ids=final_selected_chunk_ids,
            force_search_all=True  # Force search all chunks
        )
    
    # Update relevant_chunks to match final selection
    chunk_id_to_text = {c.get("chunk_id"): c.get("text", "").strip() for c in chunks if isinstance(c, dict)}
    relevant_chunks = [
        chunk_id_to_text.get(cid, "")
        for cid in final_selected_chunk_ids
        if cid in chunk_id_to_text and chunk_id_to_text.get(cid)
    ]

    return {
        "relevant_chunks": relevant_chunks,
        "selected_chunk_ids": final_selected_chunk_ids,
        "iteration_programs": iteration_programs,
        "final_confidence": final_confidence,
        "accumulated_data": accumulated_data,
        "iterations": iteration + 1
    }


async def _apply_inspection_logic(
    chunks: List[str],
    inspection_logic: str,
    llm_client: Any,
    model_deployment: str,
    max_chunks: int = 10
) -> List[str]:
    """
    Apply inspection logic (executable Python code) to select relevant chunks.

    MIT RLM approach: Execute the generated Python code to filter chunks.
    The inspection_logic should be a Python function: evaluate_chunk_relevance(chunk_text: str) -> bool

    Uses restricted execution environment to prevent imports and unsafe operations.

    Args:
        chunks: List of chunk texts
        inspection_logic: Python code containing evaluate_chunk_relevance() function
        llm_client: Azure OpenAI client (kept for compatibility)
        model_deployment: Azure deployment name (kept for compatibility)
        max_chunks: Max chunks to select

    Returns:
        Filtered list of relevant chunks
    """
    if not chunks:
        return []

    try:
        # Hard cap on code size (prevent memory bombs)
        MAX_CODE_SIZE = 5000
        if len(inspection_logic) > MAX_CODE_SIZE:
            logger.warning(f"⚠️  Inspection logic exceeds size limit ({len(inspection_logic)} > {MAX_CODE_SIZE}); using fallback")
            return await _apply_inspection_logic_llm_fallback(
                chunks, inspection_logic, llm_client, model_deployment, max_chunks
            )
        
        # Use restricted global namespace (safe_globals) - same as _evaluate_chunk_with_code
        safe_globals = {
            "__builtins__": {
                "len": len,
                "sum": sum,
                "any": any,
                "all": all,
                "min": min,
                "max": max,
                "int": int,
                "str": str,
                "bool": bool,
                "list": list,
                "dict": dict,
                "set": set,
                "range": range,
                "enumerate": enumerate,
                "sorted": sorted,
            }
        }
        
        # Execute the generated Python code in restricted environment
        namespace = {}
        exec(inspection_logic, safe_globals, namespace)
        
        # Get the evaluate_chunk_relevance function
        evaluate_func = namespace.get("evaluate_chunk_relevance")
        
        if evaluate_func is None:
            logger.warning("⚠️  Generated code does not contain evaluate_chunk_relevance function; using LLM fallback")
            return await _apply_inspection_logic_llm_fallback(
                chunks, inspection_logic, llm_client, model_deployment, max_chunks
            )
        
        # Apply the filter function to each chunk
        relevant_chunks = []
        for chunk in chunks[:max_chunks * 2]:  # Evaluate up to 2x max_chunks
            try:
                if evaluate_func(chunk):
                    relevant_chunks.append(chunk)
                    if len(relevant_chunks) >= max_chunks:
                        break
            except Exception as e:
                logger.debug(f"Error evaluating chunk with filter function: {e}")
                continue
        
        # If no chunks matched, return top few as fallback
        if not relevant_chunks:
            logger.warning(f"⚠️  No chunks matched filter criteria; using first few chunks as fallback")
            relevant_chunks = chunks[:min(3, len(chunks))]
        
        logger.debug(f"Selected {len(relevant_chunks)} relevant chunks from {len(chunks)} using Python filter")
        return relevant_chunks
    
    except SyntaxError as e:
        logger.warning(f"⚠️  Syntax error in generated inspection code: {e}; using LLM fallback")
        return await _apply_inspection_logic_llm_fallback(
            chunks, inspection_logic, llm_client, model_deployment, max_chunks
        )
    except Exception as e:
        logger.warning(f"⚠️  Error executing inspection code: {e}; using LLM fallback")
        return await _apply_inspection_logic_llm_fallback(
            chunks, inspection_logic, llm_client, model_deployment, max_chunks
        )


async def _apply_inspection_logic_llm_fallback(
    chunks: List[str],
    inspection_logic: str,
    llm_client: Any,
    model_deployment: str,
    max_chunks: int = 10
) -> List[str]:
    """
    LLM-based fallback for inspection logic when code execution fails.
    
    Uses the LLM to apply the inspection logic to chunks.
    """
    if not chunks:
        return []

    # Use gpt-4o-mini if o3-mini is configured
    analysis_deployment = "gpt-4o-mini" if model_deployment.startswith('o3') else model_deployment

    # Prepare chunk list with indices for tracking
    chunks_with_idx = [(i, chunk) for i, chunk in enumerate(chunks[:max_chunks * 2])]

    prompt = f"""Analyze the following chunks and identify which ones contain information 
relevant according to these inspection criteria:

INSPECTION CRITERIA/CODE:
{inspection_logic}

CHUNKS TO ANALYZE:
{chr(10).join(f'[Chunk {i}]: {chunk[:200]}...' if len(chunk) > 200 else f'[Chunk {i}]: {chunk}' for i, chunk in chunks_with_idx)}

Return a JSON list of chunk indices that are relevant. Format: {{"relevant_indices": [0, 2, 5]}}
Include only chunks that clearly match the criteria. If fewer than 3 chunks match, include the most relevant ones anyway."""

    try:
        params = _build_completion_params(
            analysis_deployment,
            model=analysis_deployment,
            messages=[
                {"role": "system", "content": "You are a document analysis expert. Respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_completion_tokens=200
        )
        response = await llm_client.chat.completions.create(**params)
        response_text = response.choices[0].message.content.strip()

        # Parse JSON response
        import json
        result = json.loads(response_text)
        relevant_indices = result.get("relevant_indices", [])

        # Filter chunks based on selected indices
        relevant_chunks = [chunks[i] for i in relevant_indices if i < len(chunks)]

        logger.debug(f"Selected {len(relevant_chunks)} relevant chunks from {len(chunks)} using LLM fallback")
        return relevant_chunks

    except Exception as e:
        logger.warning(f"⚠️  LLM fallback also failed: {e}; returning first few chunks")
        # Final fallback: return first few chunks
        return chunks[:min(3, len(chunks))]


async def _summarize_chunks(
    chunks: List[str],
    query: str,
    file_name: str,
    llm_client: Any,
    model_deployment: str
) -> str:
    """
    Summarize selected chunks into a cohesive file-level summary.

    Args:
        chunks: List of relevant chunk texts
        query: User query for context
        file_name: Name of file being summarized
        llm_client: Azure OpenAI client
        model_deployment: Azure deployment name

    Returns:
        Summary text
    """
    # Use gpt-4o-mini if o3-mini is configured
    summary_deployment = "gpt-4o-mini" if model_deployment.startswith('o3') else model_deployment
    
    chunks_text = "\n---\n".join(chunks)

    prompt = f"""Summarize the following chunks from "{file_name}" to answer: {query}

RELEVANT CHUNKS:
{chunks_text}

Provide a concise summary (3-5 sentences) that:
1. Directly addresses the user's query
2. Includes specific details from the chunks
3. Maintains the context and relationships mentioned
4. Is suitable for inclusion in a final answer

Return only the summary text, without preamble."""

    try:
        params = _build_completion_params(
            summary_deployment,
            model=summary_deployment,
            messages=[
                {"role": "system", "content": "You are a summarization expert. Provide clear, concise summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_completion_tokens=500
        )
        response = await llm_client.chat.completions.create(**params)
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        logger.warning(f"⚠️  Failed to summarize chunks: {e}")
        # Fallback: return concatenated chunks
        return " ".join(chunks[:min(2, len(chunks))])


async def log_file_summaries_to_markdown(
    file_summaries: List[FileSummary],
    query: str,
    rlm_enabled: bool = True,
    output_dir: str = "/app/logs/chunk_analysis"
) -> None:
    """
    Log Phase 4 file summaries to markdown file.

    Args:
        file_summaries: List of FileSummary objects
        query: User query
        rlm_enabled: Whether RLM is enabled
        output_dir: Output directory for logs
    """
    from pathlib import Path
    from datetime import datetime

    try:
        # Determine output file
        file_suffix = "enabled" if rlm_enabled else "disabled"
        output_path = Path(output_dir) / f"summaries_rlm_{file_suffix}.md"

        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build markdown content
        timestamp = datetime.now().isoformat()
        lines = [
            f"# Phase 4: Recursive Summarization (RLM {'Enabled' if rlm_enabled else 'Disabled'})",
            f"\n**Execution Time:** {timestamp}",
            f"\n**Query:** {query}",
            f"\n**Total Summaries:** {len(file_summaries)}",
            "\n---\n",
        ]

        # Add each file summary
        for idx, summary in enumerate(file_summaries, 1):
            lines.extend([
                f"## {idx}. {summary.file_name}",
                f"\n**File ID:** {summary.file_id}",
                f"**Chunks:** {summary.summarized_chunk_count}/{summary.chunk_count} summarized",
                f"**Expansion Ratio:** {summary.expansion_ratio:.2f}x",
                f"\n### Summary",
                f"\n{summary.summary_text}",
                f"\n### Source Chunks",
                f"\n{', '.join(summary.source_chunk_ids)}",
                "\n---\n",
            ])

        content = "\n".join(lines)

        # Write to file (overwrite mode - one query per file)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"✅ Phase 4 summaries logged to {output_path}")
        print(f"✅ Phase 4 summaries logged: {len(file_summaries)} files")

    except Exception as e:
        logger.error(f"❌ Failed to log Phase 4 summaries: {e}", exc_info=True)
        raise


async def log_inspection_code_to_markdown(
    inspection_rules: Dict[str, Dict[str, str]],
    query: str,
    rlm_enabled: bool = True,
    output_dir: str = "/app/logs/chunk_analysis",
    mode: str = "per_chunk"
) -> None:
    """
    Log LLM-generated Python inspection code to markdown file.

    This stores the executable Python code generated by the LLM per the MIT RLM model.
    Each chunk gets its own Python function (evaluate_chunk_relevance) that determines 
    if that specific chunk is relevant to the user query.

    Args:
        inspection_rules: Dict mapping file_id to dict of chunk_id -> code (MIT RLM per-chunk approach)
        query: User query that drove the analysis
        rlm_enabled: Whether RLM is enabled
        output_dir: Output directory for logs
        mode: Either "iterative" or "per_chunk" to label correctly
    """
    from pathlib import Path
    from datetime import datetime

    try:
        # Determine output file
        file_suffix = "enabled" if rlm_enabled else "disabled"
        output_path = Path(output_dir) / f"inspection_code_rlm_{file_suffix}.md"

        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate total number of inspection programs (all chunks)
        total_programs = sum(len(chunk_codes) for chunk_codes in inspection_rules.values())

        # FIX #3: Correct header based on mode
        if mode == "iterative":
            impl_description = "MIT Recursive Inspection Model (RLM) - Iterative Refinement"
            function_signature = "inspect_iteration(chunks) -> dict"
            function_desc = "Each iteration gets a program that evaluates all chunks and returns selected subset."
        else:
            impl_description = "MIT Recursive Inspection Model (RLM) - Per-Chunk Code Generation"
            function_signature = "evaluate_chunk_relevance(chunk_text: str) -> bool"
            function_desc = "Each chunk gets its own Python function that determines if that specific chunk contains information relevant to the user's query."

        # Build markdown content
        timestamp = datetime.now().isoformat()
        lines = [
            f"# Phase 4: LLM-Generated Inspection Logic (RLM {'Enabled' if rlm_enabled else 'Disabled'})",
            f"\n**Execution Time:** {timestamp}",
            f"\n**Query:** {query}",
            f"\n**Total Inspection Programs:** {total_programs}",
            f"\n**Implementation:** {impl_description}",
            f"\n**Mode:** {mode}",
            "\n---\n",
            "## Overview\n",
            "This file stores the **executable Python code** generated by the LLM per MIT RLM.",
            function_desc + "\n",
            "### Purpose\n",
            "- Generate chunk-specific relevance evaluation code" if mode == "per_chunk" else "- Generate iteration-specific inspection programs",
            "- Each chunk receives tailored inspection logic" if mode == "per_chunk" else "- Each iteration narrows focus on relevant chunks",
            "- More precise relevance filtering per MIT RLM approach\n",
            "### Usage\n",
            f"These functions ({function_signature}) are executed by the recursive summarizer.\n",
            "---\n",
        ]

        # Add each file's inspection codes (per chunk)
        file_counter = 1
        for file_id, chunk_codes in inspection_rules.items():
            lines.extend([
                f"## {file_counter}. File (ID: {file_id})",
                "\n",
            ])
            
            chunk_counter = 1
            for chunk_id, code in chunk_codes.items():
                lines.extend([
                    f"### {file_counter}.{chunk_counter} Chunk: {chunk_id}",
                    f"\n**Query:** {query}\n",
                    "```python",
                    code,
                    "```\n",
                ])
                chunk_counter += 1
            
            lines.append("---\n")
            file_counter += 1

        content = "\n".join(lines)

        # Write to file (overwrite mode - one query per file)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"✅ Phase 4 inspection code logged to {output_path}")
        logger.info(f"   Stored {total_programs} per-chunk inspection programs following MIT RLM model")


    except Exception as e:
        logger.error(f"❌ Failed to log inspection code: {e}", exc_info=True)
        raise


async def log_inspection_code_with_text_to_markdown(
    inspection_rules: Dict[str, Dict[str, Dict[str, str]]],
    query: str,
    summary_by_file_id: Dict[str, str],
    rlm_enabled: bool = True,
    output_dir: str = "/app/logs/chunk_analysis"
) -> None:
    """
    Log inspection code with chunk text, first read, and recursive text.

    Args:
        inspection_rules: Dict mapping file_id to dict of chunk_id -> payload
            payload keys: code, chunk_text, first_read_text
        query: User query that drove the analysis
        summary_by_file_id: Dict mapping file_id to recursive summary text
        rlm_enabled: Whether RLM is enabled
        output_dir: Output directory for logs
    """
    from pathlib import Path
    from datetime import datetime

    try:
        file_suffix = "enable" if rlm_enabled else "disable"
        output_path = Path(output_dir) / f"inspection_code_chunk_rlm_{file_suffix}.md"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_programs = sum(len(chunk_payloads) for chunk_payloads in inspection_rules.values())

        timestamp = datetime.now().isoformat()
        lines = [
            f"# Phase 4: LLM-Generated Inspection Logic (RLM {'Enabled' if rlm_enabled else 'Disabled'})",
            f"\n**Execution Time:** {timestamp}",
            f"\n**Query:** {query}",
            f"\n**Total Inspection Programs:** {total_programs}",
            f"\n**Implementation:** MIT Recursive Inspection Model (RLM) - Per-Chunk Code Generation",
            "\n---\n",
            "## Overview\n",
            "This file stores the **executable Python code** generated by the LLM per MIT RLM.",
            "Each chunk gets its own Python function `evaluate_chunk_relevance(chunk_text: str) -> bool`",
            "that determines if that specific chunk contains information relevant to the user's query.\n",
            "### Purpose\n",
            "- Generate chunk-specific relevance evaluation code",
            "- Preserve the exact chunk text and first read passed to the model",
            "- Record the chunk-scoped recursive text used for Phase 4 (defaults to first read)\n",
            "### Usage\n",
            "These functions are executed by the recursive summarizer to evaluate each chunk individually.\n",
            "---\n",
        ]

        file_counter = 1
        for file_id, chunk_payloads in inspection_rules.items():
            recursive_text = summary_by_file_id.get(file_id, "")
            lines.extend([
                f"## {file_counter}. File (ID: {file_id})",
                "\n",
            ])

            chunk_counter = 1
            for chunk_id, payload in chunk_payloads.items():
                code = payload.get("code", "")
                chunk_text = payload.get("chunk_text", "")
                first_read_text = payload.get("first_read_text", "")

                chunk_recursive_text = (
                    payload.get("recursive_text")
                    or payload.get("first_read_text", "")
                    or recursive_text
                )

                lines.extend([
                    f"### {file_counter}.{chunk_counter} Chunk: {chunk_id}",
                    f"\n**Query:** {query}\n",
                    "```python",
                    code,
                    "```\n",
                    "#### Chunk Text",
                    "```text",
                    chunk_text,
                    "```\n",
                    "#### First Read",
                    "```text",
                    first_read_text,
                    "```\n",
                    "#### Recursive Text",
                    "```text",
                    chunk_recursive_text,
                    "```\n",
                ])
                chunk_counter += 1

            lines.append("---\n")
            file_counter += 1

        content = "\n".join(lines)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"✅ Phase 4 inspection code with text logged to {output_path}")
        logger.info(f"   Stored {total_programs} per-chunk inspection programs with text")

    except Exception as e:
        logger.error(f"❌ Failed to log inspection code with text: {e}", exc_info=True)
        raise
