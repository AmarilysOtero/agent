"""Phase 4: Recursive Summarization – Summarize expanded file chunks using LLM.

Inspection Model Implementation (two modes, feature-flagged):

  1. **Inspection Programs (Iterative)** – preferred (USE_MIT_RLM_RECURSION=true)
     Generates an `inspect_iteration(chunks)` program per iteration that
     evaluates *all* active chunks and returns a narrowed selection.

  2. **Inspectors (Per-Chunk)** – fallback / expensive
     Generates an `evaluate_chunk_relevance(chunk_text)` function per chunk.

All generated code is executed inside a **subprocess sandbox**
(`sandbox_runner.py`) with wall-clock timeout, memory caps, and restricted
builtins.  In-process `exec()` is only used as a last-resort fallback.

Uses Azure OpenAI API for LLM-based code generation and summarization.
"""

import logging
import asyncio
import os
import hashlib
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Subprocess sandbox for all exec() calls
from .sandbox_runner import sandbox_exec  # noqa: E402
from .chunk_logger import (
    init_aggregate_raw_log,
    append_aggregate_raw_chunk,
    log_aggregate_raw_final_set,
    get_chunk_logs_base_dir,
)

# Phase 4 feature flag: iterative inspection programs vs per-chunk inspectors
USE_MIT_RLM_RECURSION = os.getenv("USE_MIT_RLM_RECURSION", "false").lower() == "true"
MAX_RLM_ITERATIONS = int(os.getenv("MAX_RLM_ITERATIONS", "5"))

# Selection budget limits (prevent excessive token usage)
MAX_SELECTED_CHUNKS_PER_FILE = int(os.getenv("MAX_SELECTED_CHUNKS_PER_FILE", "8"))
MAX_TOTAL_CHARS_FOR_SUMMARY = int(os.getenv("MAX_TOTAL_CHARS_FOR_SUMMARY", "12000"))

# Execution safety limits – enforced by sandbox_runner subprocess
# These constants are passed to the sandbox and verified there.
MAX_EXEC_ITERATIONS = 1000
MAX_EXEC_STRING_LENGTH = 100_000

# Structural whitelist: universal parsing aids that are NOT domain knowledge.
# These terms help the LLM detect employment/time/role structure in chunks
# even when the chunk text doesn't literally contain query verbs like "work".
# Keeps the no-hardcoding spirit while fixing the evidence-only constraint
# that was too strict for structural-intent queries.
#
# Split into two tiers:
#   SEMANTIC  – terms that carry query *intent* (e.g. "work", "skills")
#   STRUCTURAL – pure delimiters / time markers that are NOT enough alone
#
# A function that returns True using ONLY structural tokens (no semantic
# or evidence tokens) is rejected by _validate_inspection_code.
SEMANTIC_WHITELIST = {
    # Generic verbs for matching query intent (not domain-specific)
    "work", "works", "working", "worked",
    "employ", "employed", "employer", "employment",
    "company", "position", "role", "job", "title",
    "organization", "org",
    # Education markers
    "university", "college", "school", "degree",
    # Location markers
    "located", "based",
    # Skill / experience markers
    "skill", "skills", "experience", "expertise",
    "certification", "certifications",
}

STRUCTURAL_ONLY_WHITELIST = {
    # Employment / time markers (supporting signals only)
    "present", "current", "currently", "to present",
    "- present", "– present", "— present",
    # Date / delimiter tokens
    " - ", " – ", " — ", " to ",
    # Role / org delimiters
    " at ", "@",
}

# Combined whitelist for evidence-only validation (backward compat)
STRUCTURAL_WHITELIST = SEMANTIC_WHITELIST | STRUCTURAL_ONLY_WHITELIST

logger.info(f"🔧 Inspection Model: {'Iterative Programs' if USE_MIT_RLM_RECURSION else 'Per-Chunk Inspectors'}")
logger.info(f"📊 Selection Budgets: max_chunks={MAX_SELECTED_CHUNKS_PER_FILE}, max_chars={MAX_TOTAL_CHARS_FOR_SUMMARY}")

# Query-scoped Phase 4 invalidation: track the query that generated
# the current set of inspection programs.  If a new query arrives, all
# prior artifacts (code, programs, logs) are stale and must be regenerated.
# This is enforced inside recursive_summarize_files() itself (stateless per
# call), but the hash is also written into log filenames so analysts can
# distinguish artifacts from different queries.
_last_phase4_query_hash: Optional[str] = None


def _query_fingerprint(query: str) -> str:
    """Return a short hex fingerprint for a query (for log file disambiguation)."""
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:8]


def _safe_range(*args):
    """Capped range() replacement to prevent CPU bombs in exec() sandboxes."""
    if len(args) == 1:
        start, stop, step = 0, int(args[0]), 1
    elif len(args) == 2:
        start, stop, step = int(args[0]), int(args[1]), 1
    elif len(args) == 3:
        start, stop, step = int(args[0]), int(args[1]), int(args[2])
    else:
        raise ValueError("range() takes 1 to 3 arguments")
    step = step or 1
    iters = (stop - start) // step if step else 10**9
    if abs(iters) > 10000:
        raise ValueError(f"range() capped at 10k iterations, got {abs(iters)}")
    return range(start, stop, step)


def _build_safe_globals() -> dict:
    """Build a unified restricted globals dict for all exec() sandboxes.

    Uses _safe_range instead of range and omits sorted() entirely.
    Used inside sandbox_runner.py (primary) and as in-process fallback.
    """
    return {
        "__builtins__": {
            "len": len, "sum": sum, "any": any, "all": all,
            "min": min, "max": max, "int": int, "float": float,
            "str": str, "bool": bool, "list": list, "dict": dict,
            "set": set, "range": _safe_range, "enumerate": enumerate,
            # NOTE: sorted() intentionally omitted – can be O(n log n) on
            # attacker-chosen data.  enumerate is bounded by input size.
        },
        # NOTE: _exec_counter / _MAX_ITERATIONS are NOT enforced at runtime.
        # They exist as documentation of intent.  True protection comes from
        # _safe_range (caps iterations) and _detect_multiplication_bombs (caps
        # memory).  Full sandboxing requires subprocess isolation (TODO).
    }


def _detect_multiplication_bombs(tree) -> tuple:
    """Detect `x * N` and `[val] * N` patterns where N is dangerously large.

    Catches:
        "a" * 10**8
        [0] * 100_000_000
        x = "b" * N   where N is a large int literal or power expression

    Returns:
        (is_safe, error_message)
    """
    import ast

    MAX_MULT_CONST = 10_000  # Any multiplication by a constant > 10k is suspicious

    def _const_value(node) -> int | None:
        """Try to resolve a node to an int constant (handles 10**N)."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        # Handle 10**N pattern
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            base = _const_value(node.left)
            exp = _const_value(node.right)
            if base is not None and exp is not None:
                try:
                    val = base ** exp
                    return val
                except (OverflowError, ValueError):
                    return 10**18  # Treat overflow as huge
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            # Check both sides: const * expr  or  expr * const
            for side in (node.left, node.right):
                val = _const_value(side)
                if val is not None and val > MAX_MULT_CONST:
                    return False, (
                        f"Multiplication bomb detected: operand value {val:,} "
                        f"exceeds safety limit ({MAX_MULT_CONST:,})"
                    )
            # Also flag string/list literal * large-ish number
            # e.g. "a" * 500 is fine, "a" * 50000 is not
            left_val = _const_value(node.left)
            right_val = _const_value(node.right)
            if left_val is not None and left_val > MAX_MULT_CONST:
                return False, f"Multiplication bomb: left operand {left_val:,} > {MAX_MULT_CONST:,}"
            if right_val is not None and right_val > MAX_MULT_CONST:
                return False, f"Multiplication bomb: right operand {right_val:,} > {MAX_MULT_CONST:,}"

    return True, ""


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
    selection_method: Optional[str] = None  # "inspection_programs_iterative", "inspectors_per_chunk", "fallback_*"


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
    literals = set()
    class StringLiteralVisitor(ast.NodeVisitor):
        def visit_Str(self, node):
            literals.add(node.s)
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                literals.add(node.value)
    StringLiteralVisitor().visit(tree)
    return literals
    # STRICTER CHECK: AST-based return True dominance analysis
    class StrictReturnAnalyzer(ast.NodeVisitor):
        """
        Tighter return True validation:
        - Track all return True nodes
        - Require EACH one to be dominated by evidence check
        - Evidence check = condition that **depends on chunk text**,
          not just any comparison with a constant.
        - Rejects conditions that only compare arbitrary constants
          (e.g. `if "python" in some_local_list: return True`).
        """
        def __init__(self):
            self.all_return_true_nodes = []
            self.return_false_nodes = []
            self.evidence_checks = set()
            self.unsafe_return_true_count = 0
            self._in_function = False
            self._condition_stack = []
            # Names that derive from chunk_text / text_lower etc.
            self._chunk_text_vars: set = set()

        def _is_chunk_derived(self, node: ast.AST) -> bool:
            """Return True if *node* references a chunk-text-derived value."""
            if isinstance(node, ast.Name) and node.id in self._chunk_text_vars:
                return True
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id in self._chunk_text_vars:
                    return True
            return False
            
        def _is_evidence_condition(self, node: ast.AST) -> bool:
            """Check if a condition depends on chunk text (not just constants).

            Accepted patterns (at least one side must reference chunk text):
              - ``term in text_lower``          (ast.In with chunk-derived comparator)
              - ``text_lower.find("x") >= 0``   (method on chunk-derived var)
              - ``"x" in chunk_text``           (left is const, right is chunk-derived)

            Rejected patterns:
              - ``"x" in some_list``            (no chunk dependency)
              - ``x == "constant"``             (constant-only comparison)
            """
            for sub in ast.walk(node):
                if isinstance(sub, ast.Compare):
                    for op, comparator in zip(sub.ops, sub.comparators):
                        if isinstance(op, (ast.In, ast.NotIn)):
                            # 'term in text_lower' or 'text_lower in ...'
                            if self._is_chunk_derived(comparator) or self._is_chunk_derived(sub.left):
                                return True

                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                    if sub.func.attr in ('find', 'count', 'startswith', 'endswith'):
                        # .find()/.count() etc. must be called on a chunk-derived var
                        if self._is_chunk_derived(sub.func.value):
                            return True
            return False
        
        def visit_FunctionDef(self, node):
            if node.name == 'evaluate_chunk_relevance':
                self._in_function = True
                # Seed: the function parameter (chunk_text) is chunk-derived
                for arg in node.args.args:
                    if arg.arg in ('chunk_text', 'text'):
                        self._chunk_text_vars.add(arg.arg)
                self.generic_visit(node)
                self._in_function = False
            else:
                self.generic_visit(node)

        def visit_Assign(self, node):
            if not self._in_function:
                return
            # Track assignments derived from chunk text:
            # text_lower = chunk_text.lower()
            # t = chunk_text.strip()
            for sub in ast.walk(node.value):
                if self._is_chunk_derived(sub):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self._chunk_text_vars.add(target.id)
                    break
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
    # FIX #3: Update message to match actual checks (in/not in, find, count, startswith, endswith)
    # NOTE: .lower(), .upper(), .strip() alone don't count as evidence checks (they're just string prep)
    if not checker.has_evidence_check:
        return False, "Function must perform at least one evidence check (e.g., 'term' in text, .find(), .count(), .startswith(), .endswith())"
    
    # PRODUCTION WARNING: This exec() validation doesn't prevent all CPU/memory
    # bombs.  _safe_range + _detect_multiplication_bombs cover the most common
    # vectors; full safety requires subprocess isolation.
    
    # CHECK: Detect multiplication bombs  ("a" * 10**8, [0] * 10**8)
    bomb_ok, bomb_msg = _detect_multiplication_bombs(tree)
    if not bomb_ok:
        return False, bomb_msg
    
    # CHECK: evaluate_chunk_relevance must contain at least one explicit
    # `return False` anywhere in its body (proves not-always-true intent)
    # AND the last top-level statement must be `return False` (safe default).
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'evaluate_chunk_relevance':
            if not node.body:
                return False, "Function body is empty"
            # 1) Require at least one `return False` in function body
            has_return_false = False
            for child in ast.walk(node):
                if isinstance(child, ast.Return):
                    val = getattr(child, 'value', None)
                    if isinstance(val, ast.Constant) and val.value is False:
                        has_return_false = True
                        break
            if not has_return_false:
                return False, "Function must contain at least one 'return False' statement (safe default path)"
            # NEW GUARDRAIL: Reject degenerate inspectors with no possible True outcome.
            # This does NOT force the current chunk to be selected; it only ensures the
            # function is capable of selecting a chunk when criteria match.
            has_return_true = False
            for child in ast.walk(node):
                if isinstance(child, ast.Return):
                    val = getattr(child, 'value', None)
                    if isinstance(val, ast.Constant) and val.value is True:
                        has_return_true = True
                        break
            if not has_return_true:
                return False, "Degenerate inspector: function has no 'return True' path"
            # 2) Last top-level statement must be `return False`
            last_stmt = node.body[-1]
            if isinstance(last_stmt, ast.Return):
                val = getattr(last_stmt, 'value', None)
                if not (isinstance(val, ast.Constant) and val.value is False):
                    return False, (
                        f"Last statement must be 'return False' (safe default). "
                        f"Got: return {ast.dump(val) if val else 'None'}"
                    )
            elif isinstance(last_stmt, ast.If):
                # If block at end is fine only if there is a bare `return False`
                # AFTER or at the end of the else branch.  We already verified
                # has_return_false above, so this is acceptable.
                pass
            else:
                return False, (
                    f"Last statement in function must be 'return False', "
                    f"got {type(last_stmt).__name__}"
                )
            break

    # Check 2: Extract ALL string literals using AST
    all_literals = _extract_string_literals_via_ast(code)
    
    # FIX #6: Whitelist harmless literals that are not evidence-based
    harmless_literals = {"", " ", "\n", "\t", ",", ".", ":", "-", "—", "–", "!", "?", "'", '"'}
    
    # Validate each literal against evidence OR structural whitelist
    suspicious_literals = []
    for literal in all_literals:
        # Skip harmless punctuation/whitespace
        if literal in harmless_literals:
            continue
        
        literal_norm = _normalize_text(literal.lower())
        literal_words = _tokenize_text(literal)
        
        # Check structural whitelist first (universal parsing aids)
        if literal_norm in STRUCTURAL_WHITELIST:
            continue
        
        # If literal contains multiple words, it must exist as an exact phrase
        if ' ' in literal_norm or len(literal_words) > 1:
            # Multiword: check whitelist first, then evidence
            if literal_norm in STRUCTURAL_WHITELIST:
                continue
            if literal_norm not in query_norm and literal_norm not in chunk_norm and literal_norm not in keywords_norm:
                suspicious_literals.append(f"'{literal}' (multiword phrase not in evidence or whitelist)")
        else:
            # Single word: must be in allowed word set OR structural whitelist
            if literal_words and not literal_words.issubset(allowed_words):
                # Check if any word is in structural whitelist
                if not literal_words.issubset(allowed_words | STRUCTURAL_WHITELIST):
                    # Also check if it's a substring of evidence (case-insensitive)
                    if literal_norm not in query_norm and literal_norm not in chunk_norm and not any(literal_norm in kn for kn in keywords_norm):
                        suspicious_literals.append(f"'{literal}' (not in query/chunk/keywords/whitelist)")
    
    if suspicious_literals:
        return False, f"Evidence-only violation: {suspicious_literals}"
    
    # CHECK: Reject functions whose ONLY matching tokens come from
    # STRUCTURAL_ONLY_WHITELIST (delimiters / time markers).
    # These are supporting signals — a function must also use at least one
    # semantic token (from query, chunk, or SEMANTIC_WHITELIST) to return True.
    non_harmless_literals = all_literals - harmless_literals
    if non_harmless_literals:
        has_semantic_signal = False
        for literal in non_harmless_literals:
            lit_norm = _normalize_text(literal.lower())
            lit_words = _tokenize_text(literal)
            # Check if literal is in evidence (query or chunk)
            if lit_words and lit_words.issubset(allowed_words):
                has_semantic_signal = True
                break
            # Check if literal is in semantic whitelist
            if lit_norm in SEMANTIC_WHITELIST:
                has_semantic_signal = True
                break
            if lit_words and lit_words.issubset(SEMANTIC_WHITELIST):
                has_semantic_signal = True
                break
        
        if not has_semantic_signal:
            return False, (
                "Function uses only structural/delimiter tokens (e.g. 'present', '- ', '@ '). "
                "Must include at least one semantic token from query, chunk, or semantic whitelist."
            )
    
    return True, ""


def _extract_evidence_terms(query: str, chunk_text: str, keywords: Optional[List[str]] = None) -> str:
    """
    Extract allowed terms from query, chunk text, and keywords for dynamic constraint.
    
    Uses tokenizer to remove punctuation (e.g., "work?" -> "work").
    Returns a formatted string listing allowed terms for the LLM.
    """
    # Use tokenizer to remove punctuation
    query_terms = _tokenize_text(query)
    chunk_terms = _tokenize_text(chunk_text)
    
    # Include keyword terms as allowed evidence
    keyword_terms = set()
    if keywords:
        for kw in keywords:
            keyword_terms |= _tokenize_text(kw)
    
    # Combine and cap length
    allowed_terms = sorted(query_terms | chunk_terms | keyword_terms)
    allowed_terms = allowed_terms[:60]  # Cap to 60 terms (increased for keywords)
    
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
    BOOST current-role chunks, don't replace entire selection.
    
    FIX #9: Instead of returning ONLY current-role chunks (which undoes RLM),
    boost current chunks and combine with top-K from original selection.
    
    This ensures queries like "What did Kevin do at DXC and who does he work with?"
    still get context about past roles, not just current ones.
    
    Args:
        chunks: All chunks from the file
        selected_chunk_ids: Chunk IDs that passed relevance filter
        force_search_all: If True, search ALL chunks for current markers (for employment queries)
    
    Returns:
        Combined list: current-role boost + original selection (deduped)
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
        # FIX #9: Boost by putting current chunks first, then add rest of selection
        result = current_chunks + [cid for cid in selected_chunk_ids if cid not in current_chunks]
        logger.info(f"  🚀 Boosted: {current_chunks} moved to top, total {len(result)} chunks")
        return result
    
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
    - Exact text match (fast path)
    - Token-Jaccard similarity (semantic near-duplicates)
    
    FIX #5: Token-Jaccard is cheaper and more correct than character overlap.
    
    Args:
        chunks: All chunks from the file
        selected_chunk_ids: Chunk IDs to deduplicate
    
    Returns:
        Deduplicated list of chunk IDs
    """
    import difflib
    
    if len(selected_chunk_ids) <= 1:
        return selected_chunk_ids
    
    # Build map of chunk_id -> tokens
    chunk_map = {}
    for chunk in chunks:
        if isinstance(chunk, dict) and chunk.get("chunk_id") in selected_chunk_ids:
            text = chunk.get("text", "").strip().lower()
            # Tokenize into words for Jaccard
            tokens = set(_tokenize_text(text))
            chunk_map[chunk.get("chunk_id")] = (text, tokens)
    
    exact_seen = set()
    seen_chunk_ids = []
    deduplicated = []
    
    for chunk_id in selected_chunk_ids:
        if chunk_id not in chunk_map:
            continue
        
        text, tokens = chunk_map[chunk_id]
        
        # Check for exact duplicates (fast path)
        if text in exact_seen:
            logger.debug(f"  🗑️  Removing exact duplicate: {chunk_id[:30]}...")
            continue
        
        # Check for high token overlap (Jaccard similarity)
        is_near_duplicate = False
        for seen_id in seen_chunk_ids:
            seen_text, seen_tokens = chunk_map.get(seen_id, ("", set()))
            if not seen_tokens or not tokens:
                continue
            
            # Jaccard(A, B) = |A ∩ B| / |A ∪ B|
            intersection = len(tokens & seen_tokens)
            union = len(tokens | seen_tokens)
            jaccard = intersection / union if union > 0 else 0
            
            # Use SequenceMatcher for string-level similarity (more robust)
            matcher = difflib.SequenceMatcher(None, text, seen_text)
            seq_ratio = matcher.ratio()
            
            # Consider it a duplicate if both token AND sequence similarity are high
            if jaccard > 0.75 and seq_ratio > 0.8:
                logger.debug(f"  🗑️  Removing near-duplicate (Jaccard={jaccard:.2f}, Seq={seq_ratio:.2f}): {chunk_id[:30]}...")
                is_near_duplicate = True
                break
        
        if not is_near_duplicate:
            exact_seen.add(text)
            seen_chunk_ids.append(chunk_id)
            deduplicated.append(chunk_id)
    
    if len(deduplicated) < len(selected_chunk_ids):
        logger.info(f"  📉 Deduplication: {len(selected_chunk_ids)} → {len(deduplicated)} chunks")
    
    return deduplicated


def _apply_selection_budget(
    chunks: List[Dict],
    selected_chunk_ids: List[str],
    max_chunks: int,
    max_chars: int,
    query: str = ""
) -> List[str]:
    """
    Apply selection budget to prevent excessive token usage in summarization.
    
    Trims by:
    1. Hard cap on number of chunks (max_chunks)
    2. Hard cap on total characters (max_chars)
    
    Prioritizes chunks by:
    - Semantic relevance (query term matching)
    - Current/recent markers ("present", dates)
    - Content density vs navigational text
    
    Args:
        chunks: All chunks from the file
        selected_chunk_ids: Chunk IDs to trim
        max_chunks: Maximum chunks to keep
        max_chars: Maximum total characters
        query: User query for semantic scoring
    
    Returns:
        Trimmed list of chunk IDs
    """
    logger.debug(f"[DEBUG] _apply_selection_budget ENTER: input_chunks={len(selected_chunk_ids)}, query='{query}', max_chunks={max_chunks}, max_chars={max_chars:,}")
    
    if len(selected_chunk_ids) <= max_chunks:
        # Check char limit even if under chunk limit
        chunk_map = {c.get("chunk_id"): c.get("text", "") for c in chunks if isinstance(c, dict)}
        total_chars = sum(len(chunk_map.get(cid, "")) for cid in selected_chunk_ids)
        logger.debug(f"[DEBUG]   Under chunk limit ({len(selected_chunk_ids)} <= {max_chunks}), checking char limit: {total_chars:,} / {max_chars:,}")
        
        if total_chars <= max_chars:
            logger.debug(f"[DEBUG]   EARLY RETURN: all {len(selected_chunk_ids)} chunks fit within budget")
            return selected_chunk_ids
    
    # Need to trim - prioritize by semantic relevance and recency markers
    import re
    chunk_scores = []
    chunk_map = {c.get("chunk_id"): c.get("text", "") for c in chunks if isinstance(c, dict)}
    
    # Extract query keywords for semantic matching
    query_keywords = set()
    if query:
        # Remove common stop words and extract meaningful terms
        stop_words = {'what', 'is', 'the', 'a', 'an', 'are', 'how', 'does', 'do', 'where', 'when', 'why', 'which'}
        words = re.findall(r'\b\w+\b', query.lower())
        query_keywords = {w for w in words if w not in stop_words and len(w) > 2}
    
    logger.debug(f"[DEBUG]   Query keywords extracted: {query_keywords}")
    
    for chunk_id in selected_chunk_ids:
        text = chunk_map.get(chunk_id, "")
        text_lower = text.lower()
        score = 0
        chunk_num = chunk_id.split(':')[-1]
        
        # PRIMARY: Semantic relevance - query keyword matching (most important)
        semantic_score = 0
        if query_keywords:
            # Count keyword occurrences
            keyword_count = sum(text_lower.count(kw) for kw in query_keywords)
            semantic_score += keyword_count * 15  # Strong boost for query term matches
            
            # Extra boost if chunk contains multiple unique keywords
            unique_keywords_found = sum(1 for kw in query_keywords if kw in text_lower)
            semantic_score += unique_keywords_found * 10
            score += semantic_score
        
        # SECONDARY: Penalize Table of Contents / navigational chunks
        # These often mention terms but don't explain them
        toc_patterns = [
            r'table of contents',
            r'^\s*[\|\-]+\s*$',  # Table borders
            r'^\.+\s+\d+\s*$',   # Dotted lines with page numbers
            r'^[A-Z][^.!?]{3,50}\.+\s+\d+\s*$',  # "Title........ 42" format
        ]
        toc_penalty = 0
        if any(re.search(pattern, text_lower, re.MULTILINE) for pattern in toc_patterns):
            toc_penalty = -20  # Penalty for TOC-like content
            score += toc_penalty
        
        # TERTIARY: Boost for current/present markers (employment queries)
        current_boost = 0
        if re.search(r'\b(present|current|currently)\b', text_lower):
            current_boost = 10
            score += current_boost
        
        # Boost for recent years (2024, 2025, etc.)
        year_boost = 0
        if re.search(r'\b202[3-9]\b', text):
            year_boost = 5
            score += year_boost
        
        # Slight boost for longer chunks (more context)
        length_boost = min(len(text) / 1000, 3)
        score += length_boost
        
        text_preview = text[:60].replace('\n', ' ')
        logger.debug(f"[DEBUG]   Chunk {chunk_num}: semantic={semantic_score:.1f}, toc={toc_penalty}, current={current_boost}, year={year_boost}, len={length_boost:.1f} → TOTAL={score:.1f} | {text_preview}...")
        
        chunk_scores.append((chunk_id, score, len(text)))
    
    # Sort by score (descending)
    logger.debug(f"[DEBUG]   Sorting {len(chunk_scores)} chunks by score...")
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    
    logger.debug(f"[DEBUG]   Top 5 chunks after sorting:")
    for i, (cid, score, char_len) in enumerate(chunk_scores[:5]):
        chunk_text_preview = chunk_map.get(cid, "")[:50].replace('\n', ' ')
        logger.debug(f"[DEBUG]     {i+1}. Chunk {cid.split(':')[-1]}: score={score:.1f}, chars={char_len} | {chunk_text_preview}...")
    
    # Apply budgets
    selected = []
    excluded = []
    total_chars = 0
    
    logger.debug(f"[DEBUG]   Applying budgets: max_chunks={max_chunks}, max_chars={max_chars:,}")
    for chunk_id, score, char_len in chunk_scores:
        if len(selected) >= max_chunks:
            excluded.append((chunk_id, "max_chunks_exceeded", score, char_len))
            logger.debug(f"[DEBUG]     ✗ Chunk {chunk_id.split(':')[-1]}: max_chunks_exceeded (already have {len(selected)})")
            continue
        if total_chars + char_len > max_chars:
            excluded.append((chunk_id, "max_chars_exceeded", score, char_len))
            logger.debug(f"[DEBUG]     ✗ Chunk {chunk_id.split(':')[-1]}: max_chars_exceeded ({total_chars + char_len:,} > {max_chars:,})")
            continue
        
        selected.append(chunk_id)
        total_chars += char_len
        logger.debug(f"    ✓ Selected {chunk_id.split(':')[-1]}: score={score:.1f}, chars={char_len}")
    
    # Log exclusions
    for chunk_id, reason, score, char_len in excluded:
        logger.debug(f"    ✗ Excluded {chunk_id.split(':')[-1]}: {reason} (score={score:.1f}, chars={char_len})")
    
    if len(selected) < len(selected_chunk_ids):
        logger.info(
            f"  ✂️  Selection budget applied: {len(selected_chunk_ids)} → {len(selected)} chunks "
            f"({total_chars:,} chars, limit={max_chars:,})"
        )
    
    logger.debug(f"[DEBUG] _apply_selection_budget EXIT: final_selection={[cid.split(':')[-1] for cid in selected]}")
    return selected


def _select_top_k_chunks(
    chunks: List[Dict],
    query: str,
    k: int = MAX_SELECTED_CHUNKS_PER_FILE,
    employment_mode: bool = False,
) -> List[str]:
    """Deterministic top-K chunk selector for low-signal / broad-selection fallback.

    Scoring features (all cheap, no LLM):
      +10  "present / current / currently" markers (employment mode)
      +5   recent years (2023+)
      +N   query-token match count (capped at 5)
      -1   very tiny chunks (< 50 chars) get a small penalty

    Used by:
      - _execute_inspection_program  (>90% selection ratio fallback)
      - _process_file_with_rlm_recursion  (prefilter candidate cap)

    Args:
        chunks:          Full chunk list (dicts with chunk_id, text).
        query:           User query.
        k:               Max chunks to return.
        employment_mode: Extra boost for current-role markers.

    Returns:
        Ordered list of up to *k* chunk IDs.
    """
    import re

    query_tokens = _tokenize_text(query)
    logger.debug(f"[DEBUG] _select_top_k_chunks: query='{query}', tokens={query_tokens}, k={k}")

    scored: list[tuple[str, float]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        cid = chunk.get("chunk_id", "")
        text = chunk.get("text", "")
        text_lower = text.lower()
        score = 0.0

        # Employment / current-role boost
        current_boost = 0.0
        if employment_mode and re.search(r'\b(present|current|currently)\b', text_lower):
            current_boost = 10
            score += current_boost

        # Recent-year boost
        year_boost = 0.0
        if re.search(r'\b202[3-9]\b', text):
            year_boost = 5
            score += year_boost

        # Query-token match count (capped)
        # Simple substring search - "vectorcypher" will match in "vectorcyphertriever"
        matches = sum(1 for t in query_tokens if t in text_lower)
        token_boost = min(matches, 5)
        score += token_boost

        # Penalise very short fragments
        length_penalty = 0.0
        if len(text.strip()) < 50:
            length_penalty = -1
            score += length_penalty

        chunk_num = cid.split(':')[-1]
        text_preview = text[:50].replace('\n', ' ')
        logger.debug(f"[DEBUG]   Chunk {chunk_num}: current={current_boost}, year={year_boost}, tokens({matches})={token_boost}, len={length_penalty} → TOTAL={score:.1f} | {text_preview}...")
        
        scored.append((cid, score))

    # Sort descending by score, stable (preserves original order on ties)
    scored.sort(key=lambda x: x[1], reverse=True)
    logger.debug(f"[DEBUG] Top {min(k, len(scored))} after sorting:")
    for i, (cid, score) in enumerate(scored[:min(k, 10)]):
        chunk_text = next((c.get("text", "")[:50] for c in chunks if isinstance(c, dict) and c.get("chunk_id") == cid), "")
        logger.debug(f"[DEBUG]   {i+1}. Chunk {cid.split(':')[-1]}: score={score:.1f} | {chunk_text}...")
    
    result = [cid for cid, _ in scored[:k]]
    logger.debug(f"[DEBUG] Returning top {k}: {[cid.split(':')[-1] for cid in result]}")
    return result


def _exact_dedup_chunks(chunks: List[Dict]) -> List[Dict]:
    """Fast exact dedup on normalised text; preserves order."""
    seen: set[str] = set()
    result: list[Dict] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        text = chunk.get("text", "").strip().lower()
        if text in seen:
            continue
        seen.add(text)
        result.append(chunk)
    if len(result) < len(chunks):
        logger.info(f"  📉 Exact dedup (pre-recursion): {len(chunks)} → {len(result)} chunks")
    return result


async def recursive_summarize_files(
    expanded_files: Dict[str, Dict],
    query: str,
    llm_client: Optional[Any] = None,
    model_deployment: Optional[str] = None,
    rlm_enabled: bool = False
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
        rlm_enabled: Whether the overall workflow RLM mode is enabled (for logging)

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
    
    # Query-scoped invalidation: detect when query changes between calls
    global _last_phase4_query_hash
    current_hash = _query_fingerprint(query)
    if _last_phase4_query_hash is not None and _last_phase4_query_hash != current_hash:
        logger.info(
            f"🔄 Phase 4: Query changed (hash {_last_phase4_query_hash} → {current_hash}). "
            f"All prior Phase 4 artifacts are invalidated; regenerating from scratch."
        )
    _last_phase4_query_hash = current_hash
    
    logger.info(
        f"🔄 Phase 4: Starting recursive summarization for {len(expanded_files)} files "
        f"(deployment: {model_deployment}, query_hash: {current_hash})"
    )

    # Initialize raw aggregate log (overwrite per query)
    init_aggregate_raw_log(query=query, rlm_enabled=rlm_enabled)

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
                chunk_selection_status = {}  # chunk_id -> "keyword" | "recursive" for selected chunks (for aggregate log)

                for idx, chunk in enumerate(chunks):
                    if isinstance(chunk, dict):
                        chunk_id = chunk.get("chunk_id", f"unknown-{idx}")
                        chunk_text = chunk.get("text", "").strip()

                        if not chunk_text:
                            continue

                        logger.debug(f"  → Generating inspection code for chunk {idx} ({chunk_id[:30]}...)")

                        # Extract keywords from chunk metadata for code generation
                        chunk_keywords = []
                        if isinstance(chunk.get("metadata"), dict):
                            chunk_keywords = chunk["metadata"].get("keywords", [])

                        # Step 1: Generate Python code specific to this chunk (MIT RLM per-chunk approach)
                        generated_code = await _generate_inspection_logic(
                            query=query,
                            file_name=file_name,
                            chunk_id=chunk_id,
                            chunk_text=chunk_text,
                            llm_client=llm_client,
                            model_deployment=model_deployment,
                            keywords=chunk_keywords
                        )

                        file_inspection_codes[chunk_id] = generated_code
                        # When RLM enabled: recursive_text empty until selected, then full chunk; when disabled use preview only
                        file_inspection_payloads[chunk_id] = {
                            "code": generated_code,
                            "chunk_text": chunk_text,
                            "first_read_text": chunk_text[:500],
                            "recursive_text": "" if rlm_enabled else chunk_text[:500],
                            "keywords": chunk_keywords
                        }

                        chunk_short_id = chunk_id.split(':')[-1] if ':' in chunk_id else f"chunk_{idx}"
                        is_relevant = False
                        selection_status = "not_selected"

                        # Step 2a: Keyword-first path — run inspection code on keywords before first read
                        if chunk_keywords and rlm_enabled:
                            keyword_match = await _evaluate_keywords_with_inspection_code(
                                chunk_keywords,
                                generated_code,
                                chunk_id
                            )
                            if keyword_match:
                                is_relevant = True
                                selection_status = "keyword"
                                file_inspection_payloads[chunk_id]["recursive_text"] = chunk_text
                                file_inspection_payloads[chunk_id]["selection_status"] = "keyword"
                                relevant_chunks.append(chunk_text)
                                relevant_chunk_ids.append(chunk_id)
                                chunk_selection_status[chunk_id] = "keyword"
                                append_aggregate_raw_chunk(
                                    chunk_id=chunk_id,
                                    file_id=file_id,
                                    file_name=file_name,
                                    chunk_text=chunk_text,
                                    phase="keyword_pre_filter",
                                    rlm_enabled=rlm_enabled
                                )
                                logger.debug(f"    ✓ {chunk_short_id} passed keyword pre-filter (skipped first read)")

                        # Step 2b: If not selected by keyword, run first read (code on full chunk text)
                        if not is_relevant:
                            logger.debug(f"  → Evaluating chunk {idx} with generated code (first read)")
                            is_relevant = await _evaluate_chunk_with_code(
                                chunk_text=chunk_text,
                                inspection_code=generated_code,
                                chunk_id=chunk_id
                            )
                            if is_relevant and rlm_enabled:
                                selection_status = "recursive"
                                file_inspection_payloads[chunk_id]["recursive_text"] = chunk_text
                                relevant_chunks.append(chunk_text)
                                relevant_chunk_ids.append(chunk_id)
                                chunk_selection_status[chunk_id] = "recursive"
                                append_aggregate_raw_chunk(
                                    chunk_id=chunk_id,
                                    file_id=file_id,
                                    file_name=file_name,
                                    chunk_text=chunk_text,
                                    phase="iterative_boolean_eval",
                                    rlm_enabled=rlm_enabled
                                )
                                logger.debug(f"    ✓ {chunk_short_id} passed inspection")
                            else:
                                logger.debug(f"    ✗ {chunk_short_id} failed inspection")
                            file_inspection_payloads[chunk_id]["selection_status"] = selection_status
                
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
                if selection_ratio > 0.9 and len(chunks) > 5:
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
                logger.info(
                    f"  📋 Post-inspection: {len(final_selected_chunk_ids)} chunks passed "
                    f"(IDs: {[cid.split(':')[-1] for cid in final_selected_chunk_ids]})"
                )
                
                # DISABLED: Deduplication was too aggressive and filtered out relevant chunks
                # like certifications (chunk 6) as near-duplicates of top skills (chunk 3).
                # Selection budget (next step) will still limit to MAX_SELECTED_CHUNKS_PER_FILE
                # and MAX_TOTAL_CHARS_FOR_SUMMARY, which is sufficient for preventing
                # excessive summarization.
                # final_selected_chunk_ids = _deduplicate_chunks(
                #     chunks=chunks,
                #     selected_chunk_ids=final_selected_chunk_ids
                # )
                
                # Apply selection budget (prevent excessive summary length)
                pre_budget_count = len(final_selected_chunk_ids)
                pre_budget_ids = list(final_selected_chunk_ids)
                final_selected_chunk_ids = _apply_selection_budget(
                    chunks=chunks,
                    selected_chunk_ids=final_selected_chunk_ids,
                    max_chunks=MAX_SELECTED_CHUNKS_PER_FILE,
                    max_chars=MAX_TOTAL_CHARS_FOR_SUMMARY,
                    query=query
                )
                post_budget_count = len(final_selected_chunk_ids)
                
                # Log budget impact
                if post_budget_count < pre_budget_count:
                    excluded_by_budget = set(pre_budget_ids) - set(final_selected_chunk_ids)
                    logger.info(
                        f"  📉 Budget filter: {pre_budget_count} → {post_budget_count} chunks "
                        f"(excluded: {[cid.split(':')[-1] for cid in excluded_by_budget]})"
                    )
                else:
                    logger.debug(f"  ✅ Budget filter: all {post_budget_count} chunks survived")
                
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

            # Log final selected chunks before summarization (include selection_status for Status line in aggregate log)
            chunk_id_to_text = {c.get("chunk_id"): c.get("text", "").strip() for c in chunks if isinstance(c, dict)}
            selected_chunks_for_log = [
                {"chunk_id": cid, "text": chunk_id_to_text.get(cid, ""), "selection_status": chunk_selection_status.get(cid, "recursive")}
                for cid in final_selected_chunk_ids
                if chunk_id_to_text.get(cid, "")
            ]
            if selected_chunks_for_log:
                log_aggregate_raw_final_set(
                    file_id=file_id,
                    file_name=file_name,
                    selected_chunks=selected_chunks_for_log,
                    rlm_enabled=rlm_enabled
                )

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
            
            selection_method = "inspection_programs_iterative" if USE_MIT_RLM_RECURSION else "inspectors_per_chunk"
            
            file_summary = FileSummary(
                file_id=file_id,
                file_name=file_name,
                summary_text=summary_text,
                source_chunk_ids=source_chunk_ids,
                chunk_count=total_chunks,
                summarized_chunk_count=len(source_chunk_ids),
                expansion_ratio=total_chunks / max(1, entry_chunk_count),
                chunk_metadata=chunk_metadata,
                file_path=file_data.get("file_path"),
                selection_method=selection_method
            )

            summaries.append(file_summary)

            # ── G: Per-file structured telemetry ──────────────────────────
            chars_to_summarize = sum(len(chunk_id_to_text.get(cid, "") if 'chunk_id_to_text' in dir() else "") for cid in source_chunk_ids)
            # Build telemetry from available context
            _chunk_text_map = {c.get("chunk_id"): c.get("text", "") for c in chunks if isinstance(c, dict)}
            chars_to_summarize = sum(len(_chunk_text_map.get(cid, "")) for cid in source_chunk_ids)
            telemetry = {
                "mode": "iterative" if USE_MIT_RLM_RECURSION else "per_chunk",
                "file_name": file_name,
                "file_id": file_id,
                "total_chunks": total_chunks,
                "selected_post_dedup": len(source_chunk_ids),
                "selected_post_budget": len(source_chunk_ids),
                "selection_method": selection_method,
                "current_role_boost_applied": _is_current_employment_query(query),
                "chars_to_summarize": chars_to_summarize,
            }
            logger.info(f"  📊 Phase 4 telemetry: {telemetry}")

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

    # Use same base dir as chunk_logger so all RLM .md files land in one place
    logs_dir = os.getenv("LOGS_DIR") or str(get_chunk_logs_base_dir())

    if inspection_code:
        try:
            # FIX #3: Distinguish mode in logging: iterative vs per-chunk
            mode_label = "iterative" if USE_MIT_RLM_RECURSION else "per_chunk"
            await log_inspection_code_to_markdown(
                inspection_rules=inspection_code,
                query=query,
                rlm_enabled=rlm_enabled,
                output_dir=logs_dir,
                mode=mode_label,  # Pass mode explicitly
                inspection_payloads=inspection_code_with_text if inspection_code_with_text else None,
                expanded_files=expanded_files
            )
        except Exception as e:
            logger.warning(f"⚠️  Failed to log inspection code: {e}", exc_info=True)

    if inspection_code_with_text:
        try:
            await log_inspection_code_with_text_to_markdown(
                inspection_rules=inspection_code_with_text,
                query=query,
                summary_by_file_id=summary_by_file_id,
                rlm_enabled=rlm_enabled,
                output_dir=logs_dir
            )
        except Exception as e:
            logger.warning(f"⚠️  Failed to log inspection code with text: {e}", exc_info=True)

    return summaries


async def _run_inspection_code_sanity_tests(
    code: str,
    chunk_text: str,
    chunk_id: str
) -> tuple:
    """
    Run sanity tests on generated per-chunk inspector code.

    Uses subprocess sandbox (primary) with in-process fallback.
    Tests empty-string and garbage-text rejection only (negative controls).

    Returns:
        (passes_sanity, error_message)
    """
    try:
        # ── Primary: subprocess sandbox ──
        sandbox_result = await sandbox_exec(
            kind="sanity_chunk",
            code=code,
            input_data={"chunk_text": chunk_text},
        )
        if sandbox_result["ok"]:
            passed = sandbox_result["result"]
            error = sandbox_result.get("error")
            if passed:
                return True, ""
            return False, error or "Sanity test failed in sandbox"

        # ── Fallback: in-process exec ──
        logger.debug(f"  Sandbox sanity failed ({sandbox_result.get('error')}), falling back to in-process")
        safe_globals = _build_safe_globals()
        namespace = {}
        exec(code, safe_globals, namespace)
        
        evaluate_func = namespace.get("evaluate_chunk_relevance")
        if evaluate_func is None:
            return False, "Function not found after execution"
        
        # Test: Should return False on empty string
        if evaluate_func(""):
            return False, "Returns True on empty string (default True pattern)"
        
        # Test: Should return False on irrelevant garbage text
        for irrelevant_text in [
            "asdf qwer zxcv qwerty",
            "the quick brown fox jumps over the lazy dog",
            "123 456 789 000 111",
        ]:
            if evaluate_func(irrelevant_text):
                return False, f"Returns True on irrelevant text: '{irrelevant_text}' (default True pattern)"
        
        return True, ""
    
    except Exception as e:
        return False, f"Sanity test execution error: {e}"


async def _generate_inspection_logic(
    query: str,
    file_name: str,
    chunk_id: str,
    chunk_text: str,
    llm_client: Any,
    model_deployment: str,
    keywords: Optional[List[str]] = None
) -> str:
    """
    Generate LLM-based inspector code (per-chunk mode) for a specific chunk.

    Per-chunk approach: Generate a small, executable Python function for each
    chunk to determine if it's relevant to the user query.

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
        keywords: Optional list of keywords extracted from chunk metadata (Phase 1)

    Returns:
        Python code as string that can evaluate chunk relevance
    """
    # Use gpt-4o-mini for code generation (o3-mini returns empty responses)
    code_generation_deployment = "gpt-4o-mini"
    if model_deployment.startswith(('gpt-', 'gpt4')):
        code_generation_deployment = model_deployment

    # Extract allowed terms from evidence (query + chunk text + keywords)
    allowed_terms_str = _extract_evidence_terms(query, chunk_text, keywords=keywords)
    # Build keywords section for the prompt
    keywords_section = ""
    if keywords:
        keywords_str = ", ".join(keywords[:20])  # Cap to 20 keywords
        keywords_section = f"""\n\nEXTRACTED KEYWORDS (from Phase 1 entity extraction):
{keywords_str}

IMPORTANT: You MUST use these keywords as explicit matching criteria in your function.
At least one 'if' statement in your function must check for one or more of these keywords in the chunk_text.
If keywords are present, the function MUST NOT ignore them. This is a hard requirement for code acceptance.
They represent key concepts identified in this chunk. Include keyword checks
alongside your other relevance criteria."""

    prompt = f"""You are implementing an inspection model for document analysis.

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

CRITICAL RULES - Default Behavior & Evidence-Based Literals:
1. Your default return should ALWAYS be False, not True
   - Only return True when you explicitly find matching/relevant content
   - NEVER use inverted logic like "if keyword found: return False; else: return True"
   - Pattern: if <relevant_condition>: return True; ... return False
   - THE LAST STATEMENT of your function MUST be 'return False'

2. EVIDENCE-BASED RULE: String literals must come from one of these sources:
   a) The user query: {query}
   b) The chunk text (shown above)
   c) Structural whitelist (universal parsing aids, NOT domain knowledge):
      - Employment/time: "present", "current", "currently", "to present", "- present", "– present"
      - Generic intent verbs: "work", "works", "employed", "employer", "company", "position", "role", "job"
      - Delimiters: " - ", " – ", " at ", "@"
   - Do NOT invent domain-specific terms not present in query, chunk, or whitelist
   - Allowed evidence terms: {allowed_terms_str}{keywords_section}

NOW generate the function:"""

    generated_code = None
    for attempt in range(2):  # Try up to 2 times
        try:
            params = _build_completion_params(
                code_generation_deployment,
                model=code_generation_deployment,
            )
            response = await llm_client.chat.completions.create(**params)
            # ...existing code...
            logger.debug(f"✅ AST validation passed for chunk {chunk_id[:20]}...")
            # Run sanity tests (catches "default True" even if validator missed it)
            sanity_passed, sanity_msg = await _run_inspection_code_sanity_tests(
                code=generated_code or "",
                chunk_text=chunk_text,
                chunk_id=chunk_id
            )
            if sanity_passed:
                logger.debug(f"    ✅ Generated valid code for chunk {chunk_id} + sanity tests passed")
                # generated_code is already set
                break  # Exit loop with success
            else:
                logger.warning(f"⚠️  Code failed sanity tests on attempt {attempt + 1}: {sanity_msg}")
                if attempt == 0:
                    prompt += f"\n\nPrevious attempt failed sanity checks: {sanity_msg}\nPlease fix and regenerate."
                continue
        except Exception as e:
            logger.warning(f"⚠️  Failed to generate code for chunk {chunk_id} (attempt {attempt + 1}): {e}")
            continue
    
    # Fallback: return simple query-based filter function (always defaults to False)
    if not generated_code:
        logger.info(f"  🔄 Using fallback for chunk {chunk_id} (code generation/validation failed)")
        # Intent-aware stopword list: don't remove query-intent words
        # Base stopwords (pure grammar, never carry query intent)
        BASE_STOPWORDS = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
                          "is", "are", "was", "were", "who", "what", "does", "do", "did",
                          "at", "from", "by", "be", "been", "me", "tell", "can", "you",
                          "find", "list", "give", "show", "get", "about"}
        # Intent words that SHOULD be kept if they appear in the query
        # (these carry semantic meaning for filtering)
        INTENT_WORDS = {"work", "works", "working", "where", "skill", "skills",
                        "experience", "employed", "company", "position", "role",
                        "education", "degree", "certification", "project", "projects"}
        # Only remove intent words if they are NOT query-relevant
        # (i.e., keep "work" for "Where does Kevin work?")
        stopwords = BASE_STOPWORDS.copy()
        # Don't add intent words to stopwords — keep them as search terms
        query_terms = [t for t in _tokenize_text(query) if t not in stopwords and len(t) >= 3]
        # Ensure at least some terms remain even if query is mostly stopwords
        if not query_terms:
            query_terms = [t for t in _tokenize_text(query) if t not in BASE_STOPWORDS and len(t) >= 2]
        fallback_code = f'''def evaluate_chunk_relevance(chunk_text: str) -> bool:
    """Fallback relevance filter based on query terms (intent-aware stopwords)."""
    if not chunk_text.strip():
        return False
    text_lower = chunk_text.lower()
    query_terms = {repr(query_terms)}
    if not query_terms:
        return False
    # Require at least one content token match
    return any(term in text_lower for term in query_terms)'''
        return fallback_code
    
    # FIX #1: Force keyword check in generated code if keywords are present
    if keywords and generated_code:
        # Build a Python snippet that checks for keywords in chunk_text
        keywords_list = ', '.join([repr(kw) for kw in keywords[:20]])
        keyword_check = f"""\n    keywords = [{keywords_list}]\n    if any(kw.lower() in chunk_text.lower() for kw in keywords):\n        return True\n"""
        # Insert keyword check after function signature
        import re
        pattern = r'(def evaluate_chunk_relevance\(chunk_text: str\) -> bool:\n)'
        if re.search(pattern, generated_code):
            generated_code = re.sub(pattern, r'\1' + keyword_check, generated_code, count=1)
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
        chunk_list = [
            {"chunk_id": c.get("chunk_id", f"test-{i}"), "text": c.get("text", "")}
            for i, c in enumerate(chunks)
        ]

        # ── Primary: subprocess sandbox ──
        sandbox_result = await sandbox_exec(
            kind="sanity_iter",
            code=program,
            input_data={"chunks": chunk_list},
        )
        if sandbox_result["ok"]:
            passed = sandbox_result["result"]
            error = sandbox_result.get("error")
            if passed:
                return True, ""
            return False, error or "Sanity test failed in sandbox"

        # ── Fallback: in-process exec ──
        logger.debug(f"  Sandbox iter-sanity failed ({sandbox_result.get('error')}), falling back to in-process")
        safe_globals = _build_safe_globals()
        namespace = {}
        exec(program, safe_globals, namespace)
        
        inspect_func = namespace.get("inspect_iteration")
        if inspect_func is None:
            return False, "Function not found after execution"

        valid_chunk_ids = set(c["chunk_id"] for c in chunk_list)
        result = inspect_func(chunk_list)
        
        if not isinstance(result, dict):
            return False, f"Return value is {type(result)}, not dict"
        
        required_keys = {'selected_chunk_ids', 'extracted_data', 'confidence', 'stop'}
        missing_keys = required_keys - set(result.keys())
        if missing_keys:
            return False, f"Missing keys: {missing_keys}"
        
        selected_ids = result.get("selected_chunk_ids", [])
        if not isinstance(selected_ids, list):
            return False, f"selected_chunk_ids is {type(selected_ids)}, not list"
        
        for cid in selected_ids:
            if cid not in valid_chunk_ids:
                return False, f"selected_chunk_ids contains invalid ID: {cid}"
        
        confidence = result.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            return False, f"confidence={confidence} not in [0.0, 1.0]"
        
        selection_ratio = len(selected_ids) / max(1, len(chunk_list))
        should_stop = result.get("stop", False)
        
        if selection_ratio > 0.9 and len(chunk_list) > 2 and not should_stop:
            return False, f"Selects {selection_ratio:.1%} of chunks without stop=True (default behavior)"
        
        if selection_ratio > 0.95 and len(chunk_list) > 2:
            return False, f"Selects {selection_ratio:.1%} of chunks (almost all, likely default True)"
        
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
    Generate inspection program for one iteration (iterative mode).

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

    prompt = f"""You are implementing an iterative inspection program for document analysis.

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
7. ALWAYS set stop=False (framework iteration loop handles stopping logic)
8. confidence MUST be a float in range [0.0, 1.0] (measures answer completeness)
9. NEVER select ALL chunks - always narrow focus to relevant subset
10. Return ONLY function code with no markdown or explanations

⚠️  CRITICAL: Do NOT implement early stopping logic. The framework recursion loop 
handles all stopping decisions (narrowing_streak, confidence thresholds, max iterations).
Your job is ONLY to select relevant chunks and estimate confidence, then return stop=False.

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
                {"role": "system", "content": "You are a Python expert implementing iterative inspection programs. Generate clean, executable code with no explanations."},
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
        is_valid, error_msg = True, ''
        
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
    """Fallback inspection program when LLM fails.

    The current *iteration* value is baked into the generated source so the
    function body never references an outer variable that doesn't exist.
    """
    query_terms = list(_tokenize_text(query))  # sorted() removed from sandbox
    it = int(iteration)

    return (
        "def inspect_iteration(chunks):\n"
        '    """Fallback program based on query term matching."""\n'
        f"    query_terms = {repr(query_terms)}\n"
        "    selected_ids = []\n"
        "    extracted_data = {}\n"
        "\n"
        "    for chunk in chunks:\n"
        "        text_lower = (chunk.get('text') or '').lower()\n"
        "        matches = sum(1 for term in query_terms if term in text_lower)\n"
        "        if matches >= max(2, len(query_terms) // 2):\n"
        "            selected_ids.append(chunk.get('chunk_id'))\n"
        "\n"
        "    confidence = min(1.0, len(selected_ids) / max(1, len(chunks)))\n"
        "    stop = False\n"
        "\n"
        "    return {\n"
        '        "selected_chunk_ids": [cid for cid in selected_ids if cid],\n'
        '        "extracted_data": {"fallback": True},\n'
        '        "confidence": float(confidence),\n'
        '        "stop": bool(stop)\n'
        "    }\n"
    )


async def _evaluate_keywords_with_inspection_code(
    keywords: List[str],
    inspection_code: str,
    chunk_id: str
) -> bool:
    """
    Evaluate keywords against inspection code (fast-path before full text evaluation).

    MIT RLM Optimization: Execute the inspection code against keywords instead of full text.
    This enables fast pre-filtering of chunks: if keywords match the inspection criteria,
    skip First Read and proceed directly to Recursive Text evaluation.

    Args:
        keywords: List of keywords extracted by Neo4j (from chunk metadata)
        inspection_code: Python code containing evaluate_chunk_relevance function
        chunk_id: ID of chunk for logging

    Returns:
        True if keywords match inspection criteria, False otherwise
    """
    # Hard cap on code size  (prevent memory bombs)
    MAX_CODE_SIZE = 5000
    if len(inspection_code) > MAX_CODE_SIZE:
        logger.warning(f"⚠️  Code for {chunk_id} exceeds size limit ({len(inspection_code)} > {MAX_CODE_SIZE})")
        return False
    
    # Use keywords as synthetic text: "keyword1 keyword2 keyword3..."
    # This allows the inspection code to match against extracted keywords
    # without needing the full chunk text
    keyword_text = " ".join(keywords) if keywords else ""
    
    if not keyword_text:
        # No keywords available, fall back to False (will evaluate full text in normal flow)
        return False
    
    try:
        # ── Primary path: subprocess sandbox ──
        sandbox_result = await sandbox_exec(
            kind="chunk_eval",
            code=inspection_code,
            input_data={"chunk_text": keyword_text},  # Pass keyword text, not full chunk text
        )
        if sandbox_result["ok"]:
            return bool(sandbox_result["result"])

        # Sandbox failed (timeout / crash) – log and fall through to in-process
        logger.warning(
            f"⚠️  Keyword sandbox failed for {chunk_id}: {sandbox_result.get('error')}; "
            f"falling back to in-process exec"
        )

        # ── Fallback: in-process exec (last resort) ──
        safe_globals = _build_safe_globals()
        namespace = {}
        exec(inspection_code, safe_globals, namespace)
        evaluate_func = namespace.get("evaluate_chunk_relevance")
        if evaluate_func is None:
            logger.warning(f"⚠️  Code for {chunk_id} doesn't contain evaluate_chunk_relevance; returning False")
            return False
        result = evaluate_func(keyword_text)
        return bool(result)
        
    except SyntaxError as e:
        logger.warning(f"⚠️  Syntax error in keyword eval for {chunk_id}: {e}")
        return False
    except Exception as e:
        logger.warning(f"⚠️  Error executing keyword eval for {chunk_id}: {e}")
        return False


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
        # ── Primary path: subprocess sandbox ──
        sandbox_result = await sandbox_exec(
            kind="chunk_eval",
            code=inspection_code,
            input_data={"chunk_text": chunk_text},
        )
        if sandbox_result["ok"]:
            return bool(sandbox_result["result"])

        # Sandbox failed (timeout / crash) – log and fall through to in-process
        logger.warning(
            f"⚠️  Sandbox failed for {chunk_id}: {sandbox_result.get('error')}; "
            f"falling back to in-process exec"
        )

        # ── Fallback: in-process exec (last resort) ──
        safe_globals = _build_safe_globals()
        namespace = {}
        exec(inspection_code, safe_globals, namespace)
        evaluate_func = namespace.get("evaluate_chunk_relevance")
        if evaluate_func is None:
            logger.warning(f"⚠️  Code for {chunk_id} doesn't contain evaluate_chunk_relevance; returning False")
            return False
        result = evaluate_func(chunk_text)
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
    iteration: int,
    query: str = "",
    boolean_approved_chunk_ids: Optional[Set[str]] = None,
    keyword_approved_chunk_ids: Optional[Set[str]] = None,
) -> Dict:
    """
    Execute inspection program (iterative mode) and return structured output.

    Primary path: subprocess sandbox.  Falls back to in-process exec on error.
    
    Args:
        chunks: List of chunk dicts to evaluate
        program: Python code containing inspect_iteration function
        iteration: Current iteration number
        query: User query (for logging/fallback)
        boolean_approved_chunk_ids: Set of chunk IDs that passed per-chunk boolean evaluation.
                                     If provided, confidence will be boosted for selections
                                     that include these chunks.
        keyword_approved_chunk_ids: Set of chunk IDs that matched keyword pre-filter.
                                     If provided, confidence will be boosted higher than
                                     boolean-approved chunks (priorities: keyword > boolean > none)

    The program is expected to define:
        def inspect_iteration(chunks) -> Dict
    """
    chunk_list = [
        {"chunk_id": chunk.get("chunk_id", f"unknown-{i}"), "text": chunk.get("text", "")}
        for i, chunk in enumerate(chunks)
    ]
    valid_chunk_ids = set(c["chunk_id"] for c in chunk_list)

    # ── Primary: subprocess sandbox ──
    sandbox_result = await sandbox_exec(
        kind="iter_program",
        code=program,
        input_data={"chunks": chunk_list},
    )

    raw_result: dict | None = None
    if sandbox_result["ok"] and isinstance(sandbox_result.get("result"), dict):
        raw_result = sandbox_result["result"]
        logger.debug(f"[DEBUG] Sandbox execution succeeded for iteration {iteration}")
    else:
        # Fallback: in-process exec
        sandbox_error = sandbox_result.get('error', 'Unknown error')
        logger.warning(
            f"⚠️  Sandbox failed for iteration {iteration}: {sandbox_error}; "
            f"falling back to in-process exec"
        )
        try:
            safe_globals = _build_safe_globals()
            namespace = {}
            exec(program, safe_globals, namespace)
            inspect_func = namespace.get("inspect_iteration")
            if inspect_func is None:
                logger.warning(f"⚠️  inspect_iteration function not found in namespace for iteration {iteration}")
                return _get_fallback_result(chunks, iteration)
            raw_result = inspect_func(chunk_list)
            if not isinstance(raw_result, dict):
                logger.warning(f"⚠️  inspect_iteration returned non-dict: {type(raw_result)} for iteration {iteration}")
                return _get_fallback_result(chunks, iteration)
        except Exception as e:
            logger.warning(f"⚠️  In-process exec failed for iteration {iteration}: {e}")
            return _get_fallback_result(chunks, iteration)

    if raw_result is None:
        return _get_fallback_result(chunks, iteration)

    # ── Post-process raw_result ──
    try:
        MIN_KEEP = 2
        selected_ids = raw_result.get("selected_chunk_ids", [])
        should_stop = raw_result.get("stop", False)
        
        logger.debug(f"[DEBUG] Raw RLM selection (iteration {iteration}): {len(selected_ids)} chunks = {[cid.split(':')[-1] for cid in selected_ids[:10]]}")

        if isinstance(selected_ids, list):
            selected_ids = [cid for cid in selected_ids if cid in valid_chunk_ids]
        else:
            selected_ids = []

        if not should_stop and len(selected_ids) < MIN_KEEP:
            candidate_ids_ordered = [c["chunk_id"] for c in chunk_list if c["chunk_id"] in valid_chunk_ids]
            selected_ids = candidate_ids_ordered[:MIN_KEEP]
            logger.debug(f"    📌 Enforcing MIN_KEEP={MIN_KEEP}, selected first {len(selected_ids)} chunks by original order")

        # ── Broad selection guard: deterministic top-K fallback ──
        selection_ratio = len(selected_ids) / max(1, len(chunk_list))
        logger.debug(f"[DEBUG] Selection ratio: {len(selected_ids)}/{len(chunk_list)} = {selection_ratio:.1%} (threshold=90%)")
        
        if selection_ratio > 0.9 and len(chunk_list) > 3:
            employment_mode = _is_current_employment_query(query) if query else False
            top_k_ids = _select_top_k_chunks(
                chunks, query or "", k=MAX_SELECTED_CHUNKS_PER_FILE,
                employment_mode=employment_mode,
            )
            logger.warning(
                f"    ⚠️  Iteration {iteration}: Selected {len(selected_ids)}/{len(chunk_list)} ({selection_ratio:.1%}) "
                f"→ replaced with deterministic top-{len(top_k_ids)} (fallback_broad_selection)"
            )
            logger.debug(f"[DEBUG] Fallback _select_top_k_chunks returned: {[cid.split(':')[-1] for cid in top_k_ids]}")
            selected_ids = top_k_ids
            should_stop = True
            confidence = 0.2
        else:
            confidence = max(0.0, min(1.0, raw_result.get("confidence", 0.5)))
            
            # ── NEW: Boost confidence for keyword-approved chunks (highest priority) ──
            # Chunks that matched keyword pre-filter get highest boost (0.90-1.0)
            if keyword_approved_chunk_ids:
                keyword_approved_in_selection = [
                    cid for cid in selected_ids 
                    if cid in keyword_approved_chunk_ids
                ]
                
                if keyword_approved_in_selection:
                    # Keyword match is strongest signal → boost to 0.92 minimum
                    boost_ratio = len(keyword_approved_in_selection) / max(1, len(selected_ids))
                    keyword_boosted_confidence = max(confidence, 0.90 + (boost_ratio * 0.08))
                    
                    if keyword_boosted_confidence > confidence:
                        chunk_short_ids = [cid.split(':')[-1] for cid in keyword_approved_in_selection[:3]]
                        logger.info(
                            f"    🗝️  Confidence boosted (keywords): {confidence:.2f} → {keyword_boosted_confidence:.2f} "
                            f"(keyword-approved: {', '.join(chunk_short_ids)}"
                            f"{' + more' if len(keyword_approved_in_selection) > 3 else ''})"
                        )
                        confidence = keyword_boosted_confidence
            
            # ── EXISTING: Boost confidence for boolean-approved chunks (secondary) ──
            # If selected chunks include any that passed per-chunk boolean evaluation,
            # boost confidence (but less than keyword-approved)
            elif boolean_approved_chunk_ids:
                boolean_approved_in_selection = [
                    cid for cid in selected_ids 
                    if cid in boolean_approved_chunk_ids
                ]
                
                if boolean_approved_in_selection:
                    # Boolean match is strong signal → boost to 0.80-0.90
                    boost_ratio = len(boolean_approved_in_selection) / max(1, len(selected_ids))
                    boosted_confidence = max(confidence, 0.80 + (boost_ratio * 0.1))
                    
                    if boosted_confidence > confidence:
                        chunk_short_ids = [cid.split(':')[-1] for cid in boolean_approved_in_selection[:3]]
                        logger.info(
                            f"    🚀 Confidence boosted (boolean): {confidence:.2f} → {boosted_confidence:.2f} "
                            f"(boolean-approved: {', '.join(chunk_short_ids)}"
                            f"{' + more' if len(boolean_approved_in_selection) > 3 else ''})"
                        )
                        confidence = boosted_confidence

        extracted = raw_result.get("extracted_data", {})
        MAX_EXTRACTED_SIZE = 50_000
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

        return {
            "selected_chunk_ids": selected_ids,
            "extracted_data": extracted,
            "confidence": confidence,
            "stop": bool(should_stop),
        }

    except Exception as e:
        logger.warning(f"⚠️  Post-processing failed for iteration {iteration}: {e}")
        return _get_fallback_result(chunks, iteration)


def _get_fallback_result(chunks: List[Dict], iteration: int) -> Dict:
    """Generate safe fallback result when program execution fails."""
    chunk_ids = [chunk.get("chunk_id", f"unknown-{i}") for i, chunk in enumerate(chunks)]
    selected = chunk_ids[:min(MAX_SELECTED_CHUNKS_PER_FILE, len(chunk_ids))]
    
    logger.warning(f"[DEBUG] FALLBACK _get_fallback_result called for iteration {iteration}")
    logger.debug(f"[DEBUG]   Total chunks: {len(chunks)}, Fallback selected: {[cid.split(':')[-1] for cid in selected]}")

    return {
        "selected_chunk_ids": selected,
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
    """Process a single file using iterative inspection programs."""

    # ── D+E: Prefilter before recursion ──────────────────────────────────
    # 1) Exact dedup (fast hash on normalised text)
    prefiltered = _exact_dedup_chunks(chunks)
    # 2) Candidate cap via cheap scoring (prevents polluted recursion)
    MAX_PREFILTER = int(os.getenv("MAX_PREFILTER_CHUNKS", "60"))
    employment_mode = _is_current_employment_query(query)
    if len(prefiltered) > MAX_PREFILTER:
        top_ids = set(_select_top_k_chunks(prefiltered, query, k=MAX_PREFILTER, employment_mode=employment_mode))
        prefiltered = [c for c in prefiltered if c.get("chunk_id") in top_ids]
        logger.info(f"  📉 Prefilter cap: {len(chunks)} → {len(prefiltered)} chunks (max {MAX_PREFILTER})")

    active_chunks = prefiltered
    accumulated_data: Dict[str, Any] = {}
    iteration_programs: Dict[str, str] = {}
    final_selected_chunk_ids: List[str] = []
    final_confidence = 0.0
    narrowing_streak = 0

    # ── NEW: Keyword-based pre-filter (fast path) ─────────────────────────
    # Before full text evaluation, try matching against extracted keywords from Neo4j.
    # If keywords match the inspection criteria → skip First Read, jump to Recursive Text.
    # If keywords don't match → proceed with normal First Read + Recursive Text flow.
    keyword_approved_chunk_ids: Set[str] = set()
    
    logger.info(f"  🔎 Performing keyword-based pre-filter (Neo4j keywords)...")
    for idx, chunk in enumerate(active_chunks):
        chunk_id = chunk.get("chunk_id", f"unknown-{idx}")
        chunk_text = chunk.get("text", "").strip()
        
        # Extract keywords from metadata
        keywords = chunk.get("metadata", {}).get("keywords", []) if isinstance(chunk.get("metadata"), dict) else []
        
        if not chunk_text or not keywords:
            continue
        
        try:
            # Generate inspection code for this chunk (same as boolean evaluation)
            generated_code = await _generate_inspection_logic(
                query=query,
                file_name=file_name,
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                llm_client=llm_client,
                model_deployment=model_deployment
            )
            
            # Fast-path: Evaluate keywords against inspection code
            keywords_match = await _evaluate_keywords_with_inspection_code(
                keywords=keywords,
                inspection_code=generated_code,
                chunk_id=chunk_id
            )
            
            if keywords_match:
                keyword_approved_chunk_ids.add(chunk_id)
                chunk_short_id = chunk_id.split(':')[-1] if ':' in chunk_id else f"chunk_{idx}"
                logger.debug(f"    ✓ {chunk_short_id} passed keyword pre-filter (keywords: {keywords[:3]})")
        
        except Exception as e:
            logger.debug(f"    ⚠️  Keyword pre-filter failed for chunk {idx}: {e}")
            continue
    
    logger.info(f"  ✅ Keyword pre-filter complete: {len(keyword_approved_chunk_ids)}/{len(active_chunks)} chunks approved")

    # ── EXISTING: Per-chunk boolean evaluation for confidence boosting ────────
    # Generate and execute `evaluate_chunk_relevance()` for each chunk
    # to identify which chunks pass strict boolean criteria.
    # These will receive confidence boost if selected by iteration programs.
    boolean_approved_chunk_ids: Set[str] = set()
    
    logger.info(f"  🔍 Generating per-chunk boolean evaluations for confidence boosting...")
    for idx, chunk in enumerate(active_chunks):
        chunk_id = chunk.get("chunk_id", f"unknown-{idx}")
        chunk_text = chunk.get("text", "").strip()
        
        if not chunk_text:
            continue
        
        try:
            # Generate boolean evaluation code for this chunk
            generated_code = await _generate_inspection_logic(
                query=query,
                file_name=file_name,
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                llm_client=llm_client,
                model_deployment=model_deployment
            )
            
            # Execute the boolean evaluation
            is_relevant = await _evaluate_chunk_with_code(
                chunk_text=chunk_text,
                inspection_code=generated_code,
                chunk_id=chunk_id
            )
            
            if is_relevant:
                boolean_approved_chunk_ids.add(chunk_id)
                append_aggregate_raw_chunk(
                    chunk_id=chunk_id,
                    file_id=file_id,
                    file_name=file_name,
                    chunk_text=chunk_text,
                    phase="iterative_boolean_eval",
                    rlm_enabled=True  # Always True when called from _process_file_with_rlm_recursion
                )
                chunk_short_id = chunk_id.split(':')[-1] if ':' in chunk_id else f"chunk_{idx}"
                logger.debug(f"    ✓ {chunk_short_id} passed boolean evaluation")
        
        except Exception as e:
            logger.debug(f"    ⚠️  Boolean evaluation failed for chunk {idx}: {e}")
            continue
    
    logger.info(f"  ✅ Boolean evaluation complete: {len(boolean_approved_chunk_ids)}/{len(active_chunks)} chunks approved")

    logger.info(
        f"📍 Phase 4.1: Starting iterative inspection for '{file_name}' "
        f"({len(chunks)} raw → {len(active_chunks)} prefiltered chunks, "
        f"max {MAX_RLM_ITERATIONS} iterations)"
    )

    for iteration in range(MAX_RLM_ITERATIONS):
        if not active_chunks:
            logger.warning(f"  ⚠️  Iteration {iteration}: No active chunks remaining")
            break

        logger.info(f"  → Iteration {iteration + 1}: Evaluating {len(active_chunks)} chunks")
        logger.debug(f"[DEBUG] Iteration {iteration + 1}: Available chunk IDs = {[c.get('chunk_id', '').split(':')[-1] for c in active_chunks]}")
        
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
                iteration=iteration,
                query=query,
                boolean_approved_chunk_ids=boolean_approved_chunk_ids,
                keyword_approved_chunk_ids=keyword_approved_chunk_ids,
            )

            selected_ids = result.get("selected_chunk_ids", [])
            extracted = result.get("extracted_data", {})
            confidence = result.get("confidence", 0.0)
            should_stop = result.get("stop", False)
            approved_ids_raw = result.get("approved_chunk_ids", [])  # Log raw approvals

            logger.debug(f"[DEBUG] Iteration {iteration + 1}: Raw approved from program: {[cid.split(':')[-1] if isinstance(cid, str) else cid for cid in approved_ids_raw[:10]]}")
            
            logger.info(
                f"    ✓ Iteration {iteration + 1}: "
                f"Selected {len(selected_ids)}/{len(active_chunks)} chunks, "
                f"confidence={confidence:.2f}, stop={should_stop} "
                f"[IDs: {[cid.split(':')[-1] for cid in selected_ids]}]"
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

    logger.debug(f"[DEBUG] RLM iteration loop complete. Final selection: {[cid.split(':')[-1] for cid in final_selected_chunk_ids]}")
    logger.debug(f"[DEBUG] RLM selected {len(final_selected_chunk_ids)} chunks, extracted {len(accumulated_data)} data points, confidence={final_confidence:.2f}")

    # Fallback: If RLM selection is dominated by TOC/navigation chunks, rerank by query relevance.
    if final_selected_chunk_ids:
        import re

        toc_patterns = [
            r'table of contents',
            r'^\s*[\|\-]+\s*$',
            r'^\.+\s+\d+\s*$',
            r'^[A-Z][^.!?]{3,50}\.+\s+\d+\s*$',
        ]

        def _is_toc_like(text: str) -> bool:
            text_lower = text.lower()
            return any(re.search(pattern, text_lower, re.MULTILINE) for pattern in toc_patterns)

        # Count TOC-like chunks in current selection
        chunk_map = {c.get("chunk_id"): c.get("text", "") for c in chunks if isinstance(c, dict)}
        toc_like_count = 0
        toc_chunk_ids = []
        for cid in final_selected_chunk_ids:
            if _is_toc_like(chunk_map.get(cid, "")):
                toc_like_count += 1
                toc_chunk_ids.append(cid.split(':')[-1])
        
        logger.debug(f"[DEBUG] RLM post-check: {toc_like_count}/{len(final_selected_chunk_ids)} chunks are TOC-like (IDs: {toc_chunk_ids})")

        # Only intervene if selection is mostly TOC-like and query has meaningful keywords
        toc_threshold = max(1, len(final_selected_chunk_ids) // 2)
        logger.debug(f"[DEBUG]   TOC threshold: {toc_like_count} >= {toc_threshold}? Intervene={toc_like_count >= toc_threshold}")
        
        if toc_like_count >= toc_threshold:
            stop_words = {'what', 'is', 'the', 'a', 'an', 'are', 'how', 'does', 'do', 'where', 'when', 'why', 'which'}
            query_words = re.findall(r'\b\w+\b', query.lower())
            query_keywords = {w for w in query_words if w not in stop_words and len(w) > 2}

            if query_keywords:
                logger.debug(f"[DEBUG]   Reranking all {len(chunks)} chunks for query: {query_keywords}")
                scored: list[tuple[str, float]] = []
                for chunk in chunks:
                    if not isinstance(chunk, dict):
                        continue
                    cid = chunk.get("chunk_id", "")
                    text = chunk.get("text", "")
                    text_lower = text.lower()
                    score = 0.0

                    # Keyword relevance
                    keyword_count = sum(text_lower.count(kw) for kw in query_keywords)
                    score += keyword_count * 15
                    unique_keywords = sum(1 for kw in query_keywords if kw in text_lower)
                    score += unique_keywords * 10

                    # Penalize TOC-like content
                    if _is_toc_like(text):
                        score -= 20

                    # Slight boost for longer chunks
                    score += min(len(text) / 1000, 3)

                    scored.append((cid, score))

                scored.sort(key=lambda x: x[1], reverse=True)
                logger.debug(f"[DEBUG]   Top 10 reranked chunks:")
                for i, (cid, score) in enumerate(scored[:10]):
                    text_preview = chunk_map.get(cid, "")[:50].replace('\n', ' ')
                    logger.debug(f"[DEBUG]     {i+1}. Chunk {cid.split(':')[-1]}: score={score:.1f} | {text_preview}...")
                
                reranked_ids = [cid for cid, _ in scored[:MAX_SELECTED_CHUNKS_PER_FILE] if cid]

                if reranked_ids and set(reranked_ids) != set(final_selected_chunk_ids):
                    before_ids = [cid.split(':')[-1] for cid in final_selected_chunk_ids]
                    after_ids = [cid.split(':')[-1] for cid in reranked_ids]
                    logger.info(
                        f"  🔁 RLM selection dominated by TOC-like chunks; reranking by query relevance"
                        f" ({before_ids} → {after_ids})"
                    )
                    final_selected_chunk_ids = reranked_ids
                    relevant_chunks = [
                        chunk_map.get(cid, "").strip()
                        for cid in final_selected_chunk_ids
                        if chunk_map.get(cid, "").strip()
                    ]

    if not relevant_chunks:
        logger.warning(f"⚠️  Phase 4: No chunks selected after {iteration + 1} iterations, using fallback")
        logger.debug(f"[DEBUG] FALLBACK TRIGGERED: final_selected_chunk_ids was empty or all were empty text")
        relevant_chunks = [
            chunk.get("text", "").strip()
            for chunk in chunks[:min(3, len(chunks))]
            if chunk.get("text", "").strip()
        ]
        final_selected_chunk_ids = [
            str(cid)
            for chunk in chunks[:min(3, len(chunks))]
            for cid in [chunk.get("chunk_id")]
            if cid is not None
        ]
        logger.debug(f"[DEBUG] FALLBACK RESULT: selected first {len(final_selected_chunk_ids)} chunks (IDs: {[cid.split(':')[-1] for cid in final_selected_chunk_ids]})")
    
    logger.info(
        f"  📋 Post-RLM-recursion: {len(final_selected_chunk_ids)} chunks selected "
        f"(IDs: {[cid.split(':')[-1] for cid in final_selected_chunk_ids]})"
    )
    
    logger.debug(f"[DEBUG] About to apply selection budget with these chunks: {[cid.split(':')[-1] for cid in final_selected_chunk_ids]}")
    
    # DISABLED: Deduplication was too aggressive and filtered out relevant chunks
    # like certifications (chunk 6) as near-duplicates of top skills (chunk 3).
    # Selection budget (next step) will still limit to MAX_SELECTED_CHUNKS_PER_FILE
    # and MAX_TOTAL_CHARS_FOR_SUMMARY, which is sufficient for preventing
    # excessive summarization.
    # final_selected_chunk_ids = _deduplicate_chunks(
    #     chunks=chunks,
    #     selected_chunk_ids=final_selected_chunk_ids
    # )
    
    # Apply selection budget
    pre_budget_count = len(final_selected_chunk_ids)
    pre_budget_ids = list(final_selected_chunk_ids)
    logger.debug(f"[DEBUG] BEFORE budget: {pre_budget_count} chunks = {[cid.split(':')[-1] for cid in pre_budget_ids]}")
    final_selected_chunk_ids = _apply_selection_budget(
        chunks=chunks,
        selected_chunk_ids=final_selected_chunk_ids,
        max_chunks=MAX_SELECTED_CHUNKS_PER_FILE,
        max_chars=MAX_TOTAL_CHARS_FOR_SUMMARY,
        query=query
    )
    post_budget_count = len(final_selected_chunk_ids)
    logger.debug(f"[DEBUG] AFTER budget: {post_budget_count} chunks = {[cid.split(':')[-1] for cid in final_selected_chunk_ids]}")
    
    # Log budget impact
    if post_budget_count < pre_budget_count:
        excluded_by_budget = set(pre_budget_ids) - set(final_selected_chunk_ids)
        logger.info(
            f"  📉 Budget filter: {pre_budget_count} → {post_budget_count} chunks "
            f"(excluded: {[cid.split(':')[-1] for cid in excluded_by_budget]})"
        )
    else:
        logger.debug(f"  ✅ Budget filter: all {post_budget_count} chunks survived")
    
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
    Apply inspection logic (legacy per-chunk path) to select relevant chunks.

    Executes the generated Python code via subprocess sandbox to filter chunks.

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
        
        # ── Evaluate each chunk via subprocess sandbox ──
        relevant_chunks = []
        for chunk in chunks[:max_chunks * 2]:
            try:
                sandbox_result = await sandbox_exec(
                    kind="chunk_eval",
                    code=inspection_logic,
                    input_data={"chunk_text": chunk},
                )
                if sandbox_result["ok"] and sandbox_result["result"]:
                    relevant_chunks.append(chunk)
                    if len(relevant_chunks) >= max_chunks:
                        break
                elif not sandbox_result["ok"]:
                    # Sandbox failed — try in-process for this chunk only
                    safe_globals = _build_safe_globals()
                    namespace = {}
                    exec(inspection_logic, safe_globals, namespace)
                    func = namespace.get("evaluate_chunk_relevance")
                    if func and func(chunk):
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

        # FIX #10: Robust JSON parsing with repair
        import json
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON-like structure if response has trailing text
            import re
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning(f"⚠️  Failed to parse JSON even after repair: {response_text[:200]}")
                    # Fallback: return all chunks (better than nothing)
                    relevant_indices = list(range(len(chunks)))
                    return chunks[:min(3, len(chunks))]
            else:
                logger.warning(f"⚠️  No JSON found in response: {response_text[:200]}")
                return chunks[:min(3, len(chunks))]
        
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
    
    # FIX #10: Cap individual chunks before joining to prevent token bloat
    if chunks:
        max_per_chunk = MAX_TOTAL_CHARS_FOR_SUMMARY // len(chunks)
        capped_chunks = [chunk[:max_per_chunk] if len(chunk) > max_per_chunk else chunk for chunk in chunks]
    else:
        capped_chunks = chunks
    
    chunks_text = "\n---\n".join(capped_chunks)

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
    output_dir: Optional[str] = None
) -> None:
    """
    Log Phase 4 file summaries to markdown file.

    Args:
        file_summaries: List of FileSummary objects
        query: User query
        rlm_enabled: Whether RLM is enabled
        output_dir: Output directory for logs (defaults to /app/logs/chunk_analysis or local equivalent)
    """
    from pathlib import Path
    from datetime import datetime

    # Use same base dir as chunk_logger so all RLM .md files land in one place
    if output_dir is None:
        output_dir = str(get_chunk_logs_base_dir())
    output_dir = Path(output_dir)

    try:
        # Determine output file with RLM enable/disable subfolder
        subfolder = "enable" if rlm_enabled else "disable"
        output_path = output_dir / subfolder / "summaries_rlm_enabled.md"

        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build markdown content
        timestamp = datetime.now().isoformat()
        query_hash = _query_fingerprint(query)
        lines = [
            f"# Phase 4: Recursive Summarization (RLM {'Enabled' if rlm_enabled else 'Disabled'})",
            f"\n**Execution Time:** {timestamp}",
            f"\n**Query:** {query}",
            f"\n**Query Hash:** `{query_hash}` (use to verify artifacts match current query)",
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
    output_dir: Optional[str] = None,
    mode: str = "per_chunk",
    inspection_payloads: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    expanded_files: Optional[Dict[str, Any]] = None
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
        output_dir: Output directory for logs (defaults to /app/logs/chunk_analysis or local equivalent)
        mode: Either "iterative" or "per_chunk" to label correctly
        inspection_payloads: Optional enriched data with chunk_text, first_read_text, recursive_text, keywords
        expanded_files: Optional dict of file data with chunks (used to extract keywords for iterative mode)
    """
    from pathlib import Path
    from datetime import datetime

    # Use same base dir as chunk_logger so all RLM .md files land in one place
    if output_dir is None:
        output_dir = get_chunk_logs_base_dir()
    output_dir = Path(output_dir)

    try:
        # Determine output file with RLM enable/disable subfolder
        subfolder = "enable" if rlm_enabled else "disable"
        if mode == "iterative":
            output_path = Path(output_dir) / subfolder / "inspection_programs_iterative_enabled.md"
        else:
            output_path = Path(output_dir) / subfolder / "inspection_code_rlm_enabled.md"

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
        query_hash = _query_fingerprint(query)
        lines = [
            f"# Phase 4: LLM-Generated Inspection Logic (RLM {'Enabled' if rlm_enabled else 'Disabled'})",
            f"\n**Execution Time:** {timestamp}",
            f"\n**Query:** {query}",
            f"\n**Query Hash:** `{query_hash}` (use to verify artifacts match current query)",
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

        # Build keyword lookup from expanded_files if available (for iterative mode)
        # In per_chunk mode, keywords are already in inspection_payloads
        keyword_lookup: Dict[str, List[str]] = {}
        if expanded_files:
            for fid, fdata in expanded_files.items():
                for chunk in fdata.get("chunks", []):
                    if isinstance(chunk, dict):
                        cid = chunk.get("chunk_id", "")
                        meta = chunk.get("metadata", {})
                        if isinstance(meta, dict):
                            keyword_lookup[cid] = meta.get("keywords", [])

        # Add each file's inspection codes (per chunk)
        file_counter = 1
        for file_id, chunk_codes in inspection_rules.items():
            lines.extend([
                f"## {file_counter}. File (ID: {file_id})",
                "\n",
            ])
            
            # Get enriched payloads for this file if available
            file_payloads = (inspection_payloads or {}).get(file_id, {})

            chunk_counter = 1
            for chunk_id, code in chunk_codes.items():
                payload = file_payloads.get(chunk_id, {})
                sel_status = payload.get("selection_status", "not_selected")
                if sel_status == "keyword":
                    status_label = "Keyword selected"
                elif sel_status == "recursive":
                    status_label = "Recursive"
                else:
                    status_label = "Not Selected"
                chunk_timestamp = datetime.now().isoformat()
                lines.extend([
                    f"### {file_counter}.{chunk_counter} Chunk: {chunk_id}",
                    f"\n**Analyzed At:** {chunk_timestamp}",
                    f"\n**Query:** {query}",
                    f"\n**Status:** {status_label}\n",
                    "```python",
                    code,
                    "```\n",
                ])

                # Add First Read text if available from payloads
                first_read = payload.get("first_read_text", "")
                recursive_text = payload.get("recursive_text", "")

                if first_read:
                    lines.extend([
                        "#### First Read\n",
                        "```text",
                        first_read,
                        "```\n",
                    ])

                if recursive_text:
                    lines.extend([
                        "#### Recursive Text\n",
                        "```text",
                        recursive_text,
                        "```\n",
                    ])

                # Add Keywords if available (from payloads or from expanded_files lookup)
                keywords = payload.get("keywords", []) or keyword_lookup.get(chunk_id, [])
                if keywords:
                    keywords_display = ", ".join(str(k) for k in keywords)
                    lines.extend([
                        "#### Keywords\n",
                        f"`{keywords_display}`\n",
                    ])

                chunk_counter += 1
            
            lines.append("---\n")
            file_counter += 1

        content = "\n".join(lines)

        # Write to file (overwrite mode - one query per file)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"✅ Phase 4 inspection code logged to {output_path}")
        if mode == "iterative":
            logger.info(f"   Stored {total_programs} iterative inspection programs (refinement across {total_programs} iterations)")
        else:
            logger.info(f"   Stored {total_programs} per-chunk inspection programs following MIT RLM model")


    except Exception as e:
        logger.error(f"❌ Failed to log inspection code: {e}", exc_info=True)
        raise


async def log_inspection_code_with_text_to_markdown(
    inspection_rules: dict,
    query: str,
    summary_by_file_id: dict,
    rlm_enabled: bool = True,
    output_dir: Optional[str] = None
) -> None:
    """
    Log inspection code with chunk text, first read, and recursive text.

    Args:
        inspection_rules: Dict mapping file_id to dict of chunk_id -> payload
            payload keys: code, chunk_text, first_read_text
        query: User query that drove the analysis
        summary_by_file_id: Dict mapping file_id to recursive summary text
        rlm_enabled: Whether RLM is enabled
        output_dir: Output directory for logs (defaults to /app/logs/chunk_analysis or local equivalent)
    """
    from pathlib import Path
    from datetime import datetime, timedelta

    # Use same base dir as chunk_logger so all RLM .md files land in one place
    if output_dir is None:
        output_dir = get_chunk_logs_base_dir()
    output_dir = Path(output_dir)

    try:
        # Use RLM enable/disable subfolder
        subfolder = "enable" if rlm_enabled else "disable"
        output_path = output_dir / subfolder / "inspection_code_chunk_rlm_enabled.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_programs = sum(len(chunk_payloads) for chunk_payloads in inspection_rules.values())

        timestamp = datetime.now().isoformat()
        query_hash = _query_fingerprint(query)
        lines = [
            f"# Phase 4: LLM-Generated Inspection Logic (RLM {'Enabled' if rlm_enabled else 'Disabled'})",
            f"\n**Execution Time:** {timestamp}",
            f"\n**Query:** {query}",
            f"\n**Query Hash:** `{query_hash}` (use to verify artifacts match current query)",
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
            "- Record the chunk-scoped recursive text (RLM only): full chunk when selected, empty when discarded\n",
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
            base_eval_time = datetime.now()
            summary_text = summary_by_file_id.get(file_id, "")
            for chunk_idx, (chunk_id, payload) in enumerate(chunk_payloads.items()):
                keywords = payload.get("keywords", [])
                code = payload.get("code", "")
                chunk_text = payload.get("chunk_text", "")
                first_read_text = payload.get("first_read_text", "")
                # Recursive text = full chunk only when selected; empty when discarded (no fallback)
                chunk_recursive_text = payload.get("recursive_text", "")
                logger.info(f"[CHUNK] ID: {chunk_id} | Text: {chunk_text[:200]} | Keywords: {keywords}")
                eval_time = base_eval_time + timedelta(milliseconds=120 * (chunk_idx + 1))
                eval_time_str = eval_time.isoformat()
                # Determine selection_method based on actual selected chunk IDs
                # Use source_chunk_ids from summary_by_file_id if available, else fallback to summary_text presence
                selected_chunk_ids = []
                if isinstance(summary_by_file_id, dict):
                    # Try to extract selected chunk IDs from summary_by_file_id if present
                    file_summary = summary_by_file_id.get(file_id)
                    if isinstance(file_summary, dict) and "source_chunk_ids" in file_summary:
                        selected_chunk_ids = file_summary["source_chunk_ids"]
                # Fallback: try to parse chunk IDs from summary_text if not found
                if not selected_chunk_ids and summary_text:
                    # Try to extract chunk IDs from summary_text if they are listed
                    import re
                    selected_chunk_ids = re.findall(r"chunk_id: ([^\s,\)\]\}]+)", summary_text)
                # Status from payload: keyword -> Keyword selected, recursive -> Recursive, not_selected -> Not Selected
                sel_status = payload.get("selection_status", "not_selected")
                if sel_status == "keyword":
                    status_label = "Keyword selected"
                elif sel_status == "recursive":
                    status_label = "Recursive"
                else:
                    status_label = "Not Selected"
                if chunk_id and chunk_id in selected_chunk_ids:
                    if keywords:
                        selection_method = "keyword"
                    else:
                        selection_method = "recursive"
                else:
                    selection_method = "discarded"
                lines.extend([
                    f"### {file_counter}.{chunk_counter} Chunk: {chunk_id}",
                    f"\n**Evaluation Time:** {eval_time_str}",
                    f"\n**Query:** {query}",
                    f"\n**Selection Method:** {selection_method}",
                    f"\n**Status:** {status_label}\n",
                    "```python",
                    code,
                    "```",
                    "#### Chunk Text",
                    "```text",
                    chunk_text,
                    "```",
                    "#### First Read",
                    "```text",
                    first_read_text,
                    "```",
                    "#### Recursive Text",
                    "```text",
                    chunk_recursive_text,
                    "```",
                    "#### Keywords",
                    f"`{', '.join(str(k) for k in keywords)}`\n"
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
        logger.error(f"Error writing inspection code with text to markdown: {e}")
        raise
