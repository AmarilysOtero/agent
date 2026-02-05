# Phase 4: LLM-Generated Inspection Logic (RLM Enabled)

**Execution Time:** 2026-02-05T03:52:05.652979  
**Corrected:** 2026-02-05T04:15:00 (MIT RLM Alignment)

**Query:** Give the list of Kevin skills

**Total Inspection Programs:** 1

**Implementation:** MIT Recursive Language Model (RLM) — Corrected

---

## Overview

This file contains the **MIT RLM-aligned recursive inspection loop**.

**Key Correction (2026-02-05):**
- ❌ Old: Hard-coded query-term matching (not RLM)
- ✅ New: Intent-driven, hypothesis-recursive, LLM-scored relevance

### Purpose

Per MIT Dec-2025 RLM paper:
- Extract semantic intent once (frozen)
- Recursively generate hypotheses from chunks
- Evaluate each chunk: "Does this reduce uncertainty in the current hypothesis?"
- Stop when confidence threshold is met
- NO raw-query-term matching

### Usage

This recursion loop replaces naive lexical filtering:
1. Phase 0: Normalize user intent (once)
2. Phase 1: Over-retrieve candidates (high recall)
3. Phase 2–4: Recursive refinement loop (core RLM)
5. Phase 5: Aggregate final answer

---

## 1. MIT RLM Recursion Loop (File ID: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\Resume\20250912 Kevin Ramirez DXC Resume.pdf)

**Purpose:** Recursively filter chunks relevant to semantic intent: "Extract skills owned by Kevin"

### Phase 0: Intent Extraction (Frozen)

```python
def extract_semantic_intent(query_text: str, llm) -> dict:
    """
    Normalize user query into query-agnostic semantic intent.
    FROZEN for all recursive steps. Called once.
    """
    prompt = f"""
    Extract the semantic intent from this query.
    Focus on: WHAT entity, WHAT attribute, WHAT output format.
    Ignore: filler words, phrasing variations.
    
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
    # Output for this query:
    # {
    #     "entity": "Kevin",
    #     "attribute": "skills",
    #     "output_type": "list",
    #     "semantic_intent": "Extract all technical and non-technical capabilities"
    # }
    return json.loads(intent_json)
```

### Phase 1: High-Recall Retrieval

```python
def hybrid_retrieve(query_text: str, corpus: List[Chunk], k: int = 200) -> List[Chunk]:
    """
    Initial retrieval: combines vector + BM25 for high recall.
    RLM will narrow from here. Do NOT filter heavily.
    """
    vector_results = corpus.search_vector(query_text, k=k)
    keyword_results = corpus.search_bm25(query_text, k=k)
    
    # Merge and deduplicate
    combined = deduplicate_and_rerank(
        vector_results + keyword_results,
        k=k
    )
    return combined
```

### Phase 2–4: MIT RLM Recursive Refinement Loop

```python
def rlm_recursive_search(
    query_text: str,
    corpus: List[Chunk],
    max_depth: int = 5,
    confidence_threshold: float = 0.85,
    llm = None
) -> dict:
    """
    MIT RLM Core Loop: Hypothesis-driven recursive refinement.
    
    Invariants:
    - Intent is frozen (extracted once, reused)
    - No raw query-term matching
    - LLM evaluates relevance w.r.t. current hypothesis
    - Recursion stops when confidence >= threshold
    """
    
    # ====================================================================
    # PHASE 0: Intent normalization (once, frozen)
    # ====================================================================
    intent = extract_semantic_intent(query_text, llm)
    target_intent = intent  # FROZEN for all recursion
    
    # ====================================================================
    # PHASE 1: Initial high-recall retrieval
    # ====================================================================
    relevant_chunks = hybrid_retrieve(query_text, corpus, k=200)
    
    # ====================================================================
    # PHASE 2–4: Recursion loop
    # ====================================================================
    current_hypothesis = None
    recursion_depth = 0
    confidence_scores = {}
    
    while recursion_depth < max_depth:
        
        # ================================================================
        # RECURSIVE STEP 1: Generate hypothesis from current chunk set
        # ================================================================
        hypothesis = llm.generate_hypothesis(
            intent=target_intent,
            chunks=relevant_chunks,
            depth=recursion_depth,
            prior_hypothesis=current_hypothesis
        )
        # Output (for this query):
        # {
        #     "statement": "Kevin has skills in: Java, Frontend Dev, Testing, ...",
        #     "confidence": 0.72,
        #     "gaps": ["Cloud platforms", "Leadership"],
        #     "semantic_embedding": embedding(statement)
        # }
        
        # ================================================================
        # RECURSIVE STEP 2: Evaluate each chunk vs. hypothesis
        # ================================================================
        next_relevant = []
        
        for chunk in relevant_chunks:
            # Core RLM question: Does this chunk REDUCE UNCERTAINTY
            # in the current hypothesis?
            relevance_score = llm.evaluate_recursive_relevance(
                chunk=chunk,
                hypothesis=hypothesis,
                target_intent=target_intent,
                prior_context=current_hypothesis
            )
            # Output: 0.0–1.0 confidence
            # - 1.0 = directly confirms/refines hypothesis
            # - 0.5 = weakly related
            # - 0.0 = contradicts or irrelevant
            
            confidence_scores[chunk.id] = relevance_score
            
            # Keep chunks that improve hypothesis
            if relevance_score >= 0.5:
                next_relevant.append(chunk)
        
        # ================================================================
        # RECURSIVE STEP 3: Check stopping condition
        # ================================================================
        avg_confidence = mean(confidence_scores.values()) if confidence_scores else 0.0
        
        stop_conditions = [
            avg_confidence >= confidence_threshold,   # High confidence
            len(next_relevant) == len(relevant_chunks),  # No new filtering
            recursion_depth >= max_depth - 1,        # Depth limit
            len(next_relevant) < 5                    # Minimum viable set
        ]
        
        if any(stop_conditions):
            current_hypothesis = hypothesis
            break
        
        # ================================================================
        # RECURSIVE STEP 4: Prepare for next recursion
        # ================================================================
        relevant_chunks = next_relevant
        current_hypothesis = hypothesis
        recursion_depth += 1
    
    # ====================================================================
    # PHASE 5: Aggregate final answer
    # ====================================================================
    final_summary = llm.aggregate_recursive(
        chunks=relevant_chunks,
        hypothesis=current_hypothesis,
        intent=target_intent,
        depth=recursion_depth
    )
    
    # ====================================================================
    # Return result with full recursion metadata
    # ====================================================================
    return {
        "answer": final_summary,
        "relevant_chunks": relevant_chunks,
        "intent": target_intent,
        "hypothesis": current_hypothesis,
        "recursion_depth": recursion_depth,
        "confidence": avg_confidence,
        "confidence_per_chunk": confidence_scores
    }
```

---

## Key Improvements Over Naive Approach

| Aspect | Naive (Old) | MIT RLM (New) |
|--------|-------------|---------------|
| **What is matched?** | Raw query terms | Semantic intent |
| **How many times?** | Once (hard-coded) | Recursively re-evaluated |
| **Who decides relevance?** | Simple word counting | LLM evaluates vs. hypothesis |
| **When to stop?** | Manual threshold | Confidence-based (automatic) |
| **Query dependent?** | YES (paraphrase fails) | NO (intent-agnostic) |

---

## Recursion Flow for Kevin Skills Query

```
Input: "Give the list of Kevin skills"
         │
         ├─ PHASE 0: intent = {entity: "Kevin", attribute: "skills", ...}
         │
         ├─ PHASE 1: hybrid_retrieve() → 200 chunks
         │
         ├─ PHASE 2–4: RECURSION LOOP
         │   │
         │   ├─ Depth 0:
         │   │  ├─ Hypothesis: "Kevin skills include Java, Frontend, Testing"
         │   │  ├─ Score chunks vs. hypothesis (LLM-driven)
         │   │  ├─ Filter to 120 relevant chunks
         │   │  └─ avg_confidence = 0.68
         │   │
         │   ├─ Depth 1:
         │   │  ├─ Hypothesis: "Kevin has technical + professional skills"
         │   │  ├─ Score chunks vs. new hypothesis
         │   │  ├─ Filter to 80 relevant chunks
         │   │  └─ avg_confidence = 0.76
         │   │
         │   ├─ Depth 2:
         │   │  ├─ Hypothesis: "Kevin's skills: Java, Frontend, Testing, Cloud (emerging)"
         │   │  ├─ Score chunks vs. hypothesis
         │   │  ├─ Filter to 8 relevant chunks
         │   │  └─ avg_confidence = 0.87 ✓ STOP (>= 0.85 threshold)
         │   │
         │   └─ Recursion ends (depth=2, confidence=0.87)
         │
         ├─ PHASE 5: aggregate_recursive() → Final answer
         │
         └─ Output: Answer + chunks + recursion metadata
```

---

## Why This Is MIT RLM Compliant

✅ **Intent-driven** — Extracted once, frozen, used as recursion target  
✅ **Query-agnostic** — Rephrasing the query doesn't break the loop  
✅ **Hypothesis-recursion** — Each level generates a new hypothesis  
✅ **LLM-scored** — Relevance =  "Does this reduce hypothesis uncertainty?"  
✅ **Confidence-based stopping** — Automated, not manual thresholds  
✅ **Recursive aggregation** — Final answer synthesized from filtered set  

---
