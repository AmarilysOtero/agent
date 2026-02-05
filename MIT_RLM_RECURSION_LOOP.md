# MIT Recursive Language Model (RLM) — Recursion Loop Pseudo-Code

**Reference:** MIT Dec-2025 RLM Paper  
**Status:** Corrected Architecture  
**Date:** 2026-02-05

---

## Core Principle

The recursion loop is **NOT** a query-string matcher.

It is a **hypothesis-driven recursive refinement** where:
- Each recursive level narrows uncertainty
- Intent is normalized once, then fixed
- LLM evaluates relevance *in context of the current hypothesis*
- Recursion stops when confidence threshold is met

---

## Pseudo-Code: MIT RLM Recursion Loop

```python
# ============================================================================
# MIT RECURSIVE LANGUAGE MODEL (RLM) — CORE RECURSION LOOP
# ============================================================================

def rlm_recursive_search(
    query_text: str,
    corpus: List[Chunk],
    max_depth: int = 5,
    confidence_threshold: float = 0.85,
    llm: LanguageModel = gpt4
) -> RLMResult:
    """
    MIT-style recursive language model refinement loop.
    
    Key invariant:
    - Intent is extracted once and FROZEN
    - Recursion operates on LLM-generated hypotheses, not raw query terms
    - Each level refines or rejects the prior hypothesis
    """
    
    # ========================================================================
    # PHASE 0: INTENT NORMALIZATION (Happens once, before recursion)
    # ========================================================================
    
    intent = extract_semantic_intent(query_text, llm)
    # Returns something like:
    # {
    #   "entity": "Kevin",
    #   "attribute": "skills",
    #   "output_type": "list",
    #   "semantic_signature": embedding(query_text)
    # }
    
    # Store this as the FROZEN RECURSION TARGET
    target_intent = intent
    
    # ========================================================================
    # PHASE 1: INITIAL RETRIEVAL (High recall, pre-recursion)
    # ========================================================================
    
    # Get initial candidate set (cheap retrieval)
    initial_candidates = hybrid_retrieve(
        query_text,
        corpus,
        k=200,  # Over-retrieve; RLM will narrow
        strategy="vector+keyword"
    )
    
    # ========================================================================
    # PHASE 2: RECURSIVE RELEVANCE FILTERING LOOP
    # ========================================================================
    
    current_hypothesis = None
    relevant_chunks = initial_candidates
    recursion_depth = 0
    confidence_scores = {}
    
    while recursion_depth < max_depth:
        
        # ====================================================================
        # RECURSIVE STEP 1: Generate hypothesis from current chunk set
        # ====================================================================
        
        hypothesis = llm.generate_hypothesis(
            intent=target_intent,
            chunks=relevant_chunks,
            depth=recursion_depth
        )
        # Returns:
        # {
        #   "statement": "Kevin has skills in: [Java, Frontend, ...]",
        #   "confidence": 0.72,
        #   "gaps": ["Cloud platforms", "Leadership experience"],
        #   "semantic_embedding": embedding(statement)
        # }
        
        # ====================================================================
        # RECURSIVE STEP 2: Evaluate each chunk against current hypothesis
        # ====================================================================
        
        next_relevant = []
        
        for chunk in relevant_chunks:
            # The core RLM question: does this chunk *reduce uncertainty*
            # in the current hypothesis?
            
            relevance_score = llm.evaluate_recursive_relevance(
                chunk=chunk,
                hypothesis=hypothesis,
                target_intent=target_intent,
                prior_context=current_hypothesis
            )
            # Outputs a score 0.0-1.0:
            # - 1.0 = directly confirms/refines hypothesis
            # - 0.5 = weakly related
            # - 0.0 = contradicts or irrelevant
            
            confidence_scores[chunk.id] = relevance_score
            
            # Only keep chunks that improve the hypothesis
            if relevance_score >= 0.5:  # Threshold
                next_relevant.append(chunk)
        
        # ====================================================================
        # RECURSIVE STEP 3: Check stopping condition
        # ====================================================================
        
        avg_confidence = mean(confidence_scores.values())
        
        # Stopping criteria (any one triggers):
        stop_conditions = [
            avg_confidence >= confidence_threshold,  # High confidence
            len(next_relevant) == len(relevant_chunks),  # No new filtering
            recursion_depth >= max_depth - 1,  # Depth limit
            len(next_relevant) < 5  # Minimum viable chunk set
        ]
        
        if any(stop_conditions):
            current_hypothesis = hypothesis
            break
        
        # ====================================================================
        # RECURSIVE STEP 4: Prepare for next recursion
        # ====================================================================
        
        relevant_chunks = next_relevant
        current_hypothesis = hypothesis
        recursion_depth += 1
        
        # Log recursion state (optional)
        log_recursion_step(
            depth=recursion_depth,
            chunk_count=len(relevant_chunks),
            avg_confidence=avg_confidence,
            hypothesis=hypothesis
        )
    
    # ========================================================================
    # PHASE 3: RECURSIVE AGGREGATION (Synthesis)
    # ========================================================================
    
    # Final summary is built from recursively filtered chunks
    final_summary = llm.aggregate_recursive(
        chunks=relevant_chunks,
        hypothesis=current_hypothesis,
        intent=target_intent,
        depth=recursion_depth
    )
    
    # ========================================================================
    # PHASE 4: RETURN WITH RECURSION METADATA
    # ========================================================================
    
    return RLMResult(
        answer=final_summary,
        relevant_chunks=relevant_chunks,
        intent=target_intent,
        hypothesis=current_hypothesis,
        recursion_depth=recursion_depth,
        confidence=avg_confidence,
        confidence_per_chunk=confidence_scores
    )


# ============================================================================
# HELPER FUNCTIONS (RLM-Specific)
# ============================================================================

def extract_semantic_intent(query_text: str, llm) -> dict:
    """
    Normalize user query into query-agnostic semantic intent.
    
    This is the FROZEN recursion target.
    
    Example:
        Input: "Give me the list of Kevin's skills"
        Output: {
            "entity": "Kevin",
            "attribute": "skills",
            "output_type": "list",
            "semantic_intent": "Extract all technical and non-technical capabilities"
        }
    """
    prompt = f"""
    Extract the semantic intent from this query in a structured way.
    Focus on: WHAT entity, WHAT attribute, WHAT output format.
    Ignore: filler words, examples, phrasing variations.
    
    Query: {query_text}
    
    Return JSON:
    {{
        "entity": <string>,
        "attribute": <string>,
        "output_type": <string>,
        "semantic_intent": <string>
    }}
    """
    
    intent_json = llm.completion(prompt)
    return json.loads(intent_json)


def hybrid_retrieve(query_text: str, corpus: List[Chunk], k: int, strategy: str) -> List[Chunk]:
    """
    Initial retrieval: combines vector + keyword matching.
    Goal: High recall, before RLM narrows.
    
    Strategy: "vector+keyword"
        - BM25 for keyword hits
        - Vector embedding similarity
        - Combined ranking (diversity)
    """
    vector_results = corpus.search_vector(query_text, k=k)
    keyword_results = corpus.search_bm25(query_text, k=k)
    
    # Merge, deduplicate, re-rank
    combined = deduplicate_and_rerank(
        vector_results + keyword_results,
        k=k
    )
    return combined


def log_recursion_step(depth: int, chunk_count: int, avg_confidence: float, hypothesis: dict):
    """
    Log each recursion step for transparency and debugging.
    """
    print(f"""
    [RLM Recursion Step {depth}]
    - Active chunks: {chunk_count}
    - Avg confidence: {avg_confidence:.2f}
    - Hypothesis: {hypothesis['statement'][:80]}...
    """)


# ============================================================================
# KEY DIFFERENCES FROM NAIVE QUERY MATCHING
# ============================================================================

"""
NAIVE (What was in inspection_code_rlm_enabled.md):
────────────────────────────────────────────────────────────────────────────
def evaluate_chunk_relevance(chunk_text: str) -> bool:
    text_lower = chunk_text.lower()
    query_terms = ['give', 'the', 'list', 'of', 'kevin', 'skills']
    return sum(1 for term in query_terms if term in text_lower) >= 2
    
❌ Hard-codes query phrasing
❌ Treats stopwords as signals
❌ NOT recursive (static filter)
❌ Violates query-agnostic principle


MIT RLM (This pseudo-code):
────────────────────────────────────────────────────────────────────────────
def rlm_recursive_search(...):
    intent = extract_semantic_intent(query_text, llm)  # Normalize once
    
    for depth in range(max_depth):
        hypothesis = llm.generate_hypothesis(intent, chunks)
        for chunk in chunks:
            score = llm.evaluate_recursive_relevance(chunk, hypothesis)
        if confidence >= threshold:
            break
        chunks = filter_by_new_scores(chunks)

✅ Intent extraction (once, frozen)
✅ LLM-driven relevance (recursively re-evaluated)
✅ Hypothesis-driven (not query-driven)
✅ Stops when uncertainty is reduced below threshold
✅ Aligns with MIT Dec-2025 paper
"""
```

---

## Recursion Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 0: Normalize Intent (Once - FROZEN)                              │
│ ───────────────────────────────────────────────────────────────────────│
│ Input: "Give me the list of Kevin's skills"                            │
│ Output: {entity: "Kevin", attribute: "skills", ...}                    │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Initial Retrieval (High Recall)                               │
│ ───────────────────────────────────────────────────────────────────────│
│ Hybrid search: vector + BM25 → 200 candidates                          │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────────────────┐
         │ RECURSION LOOP (Depth = 0, 1, 2, ...)          │
         │─────────────────────────────────────────────────│
         │                                                 │
         │ Step 1: LLM generates hypothesis from chunks   │
         │         (Does not reference raw query)         │
         │                                                 │
         │ Step 2: Evaluate each chunk:                   │
         │         "Does this reduce uncertainty in       │
         │          the current hypothesis?"              │
         │                                                 │
         │ Step 3: Check stopping condition:              │
         │         - Confidence >= threshold?             │
         │         - No new filtering?                    │
         │         - Max depth reached?                   │
         │                                                 │
         │ Step 4: Filter chunks, recurse                 │
         │         (IF stopping condition NOT met)        │
         │                                                 │
         └────────┬────────────────────────────────────────┘
                  │
            Yes   │ (Stop?)
         ◄────────┘
              │
              No
              │
              ▼
         [Iterate: depth++]
         [chunks = filtered_chunks]
         [Return to Step 1]

                  │ (Stop)
                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Aggregate Final Answer                                        │
│ ───────────────────────────────────────────────────────────────────────│
│ LLM synthesizes final answer from recursively filtered chunks          │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 4: Return with Metadata                                          │
│ ───────────────────────────────────────────────────────────────────────│
│ Answer + Relevant Chunks + Recursion Depth + Confidence Scores         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Invariants

| Invariant | Why | Example |
|-----------|-----|---------|
| **Intent is frozen** | Ensures recursion converges to the right target | Extract once: `{entity: "Kevin", attribute: "skills"}` |
| **No raw query matching** | Query phrasing is irrelevant to recursion | "Give list of Kevin skills" ≠ "What are Kevin's abilities" |
| **LLM-driven relevance** | Only the LLM understands semantic context | A chunk about "career path" matters for "skills" context |
| **Confidence-based stopping** | Recursion stops when uncertainty is reduced | avg_confidence >= 0.85 → stop recursing |
| **Hypothesis-driven filtering** | Chunks are evaluated against the *current best guess*, not the query | Prior hypothesis shapes evaluation of next chunk |

---

## Comparison: Your Current Code vs. MIT RLM

### Current (inspection_code_rlm_enabled.md)
```python
def evaluate_chunk_relevance(chunk_text: str) -> bool:
    text_lower = chunk_text.lower()
    query_terms = ['give', 'the', 'list', 'of', 'kevin', 'skills']
    return sum(1 for term in query_terms if term in text_lower) >= 2
```

**Issues:**
- ❌ Hard-codes query phrasing → query-dependent
- ❌ Stopwords are noise → low signal-to-noise
- ❌ Static filter → not recursive
- ❌ No confidence scoring → can't know when to stop

### MIT RLM Corrected
```python
def rlm_recursive_search(...):
    intent = extract_semantic_intent(query_text, llm)  # Frozen
    chunks = high_recall_retrieve(query_text, corpus)  # Over-retrieve
    
    while depth < max_depth:
        hypothesis = llm.generate_hypothesis(intent, chunks)
        scores = [llm.evaluate_vs_hypothesis(c, hypothesis) for c in chunks]
        chunks = [c for c in chunks if scores[c] >= 0.5]
        
        if mean(scores) >= confidence_threshold:
            break
        depth += 1
    
    return aggregate_answer(chunks, hypothesis, intent)
```

**Strengths:**
- ✅ Intent-agnostic → works with query paraphrases
- ✅ LLM-driven scoring → high semantic fidelity
- ✅ Recursive refinement → reduces uncertainty
- ✅ Confidence-based → automated stopping

---

## Where to Implement This

**In your architecture (neo4j_backend + Agent):**

```
User Query
    ↓
[Stage 1] Vector/Hybrid Retrieval (200 chunks) ← hybrid_retrieve()
    ↓
[Stage 2] (Optional) Graph Expansion
    ↓
[Stage 3] ⭐ MIT RLM Recursion Loop ← rlm_recursive_search()
    ├─ Intent extraction
    ├─ Recursive hypothesis generation
    ├─ Recursive relevance scoring
    └─ Confidence-based stopping
    ↓
[Stage 4] Final Answer Synthesis
    ↓
User Answer + Citations
```

Current placement: Your `evaluate_chunk_relevance()` is Stage 1–2 guardrail.  
Correct placement: Replace with full `rlm_recursive_search()` in Stage 3.

---

## Next Steps

1. **Replace** `inspect_code_rlm_enabled.md` with this corrected pseudo-code
2. **Implement** `extract_semantic_intent()` in your Agent backend
3. **Wire** `rlm_recursive_search()` into the phase 4 summarizer
4. **Test** against Kevin's resume query (should still work, but with semantic correctness)
5. **Validate** that recursion depth and confidence scores are logged

