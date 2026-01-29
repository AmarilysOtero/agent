# Plan: Complete Log Analysis & Two-Stage Architecture Implementation

## Executive Summary

Your logs show that semantic detection is disabled (missing config), person query detection fails for single names, and graph traversal never executes (code exists but is unreachable). This plan fixes these three root causes, integrates header vocabulary boosts, and restructures the retrieval pipeline into proper Stage 1 (hybrid) and Stage 2 (graph) components with deterministic entity resolution and relationship filtering.

## Steps

1. **Fix person query detection** in `src/agents.py` (lines 40–95)
   - Update name detection to allow single-token person candidates when supported by intent keywords (e.g., “who does X work with”, “X skills”), rather than requiring multi-token names.
   - Current logic requires 2+ names; single-token person queries fail

2. **Enable graph traversal** in `neo4j_backend/src/services/graphrag_retrieval.py` (line ~290)
   - Add call to `_graph_expand()` after vector seed returns
   - Pass `max_hops` parameter (currently ignored)
   - Currently `hop_count=0` always because expansion code is unreachable

3. **Restore semantic detection** by adding to `.env`
   - Add `OPENAI_API_VERSION=2024-08-01-preview`
   - Restores compatibility with Azure OpenAI API and fixes EmbeddingsProvider warnings

4. **Integrate header vocabulary boost** in `Agent/src/agents.py` (AiSearchAgent.run(), line ~355)
   - Call header vocab logic before `graphrag_search()`
   - Boosts chunks under "Professional Experience", "Work History" sections

5. **Enhance chunk metadata structure** in `neo4j_backend/main.py` (lines 210–240)
   - Add to returned chunks: `header_text`, `header_parents`, `graph_path_length`
   - Currently missing these fields needed for proper Stage 1 → Stage 2 handoff

6. **Extend graph traversal queries** in `_graph_expand()` method
   - Add `WORKS_WITH` relationship patterns to coworker lookup
   - Currently only supports project/org traversal

---

## Further Considerations

### 1. Stage 1 Output Validation — Keyword Score Always 0.0

**Problem:** If `keyword_score=0.0` for every hit, then:

- The keyword retrieval path is disabled (even if flag says `use_keyword_search=True`), OR
- Keyword results are retrieved but not scored/surfaced/fused, OR
- Keyword query is empty (`keywords=None`) so the scorer never runs, OR
- Backend uses keyword matching only as a filter gate (pass/fail), not as a scorer

**Recommendation:** Make Stage 1 pass a real keyword payload and require observable hybrid behavior.

**Validation Checklist (fast):**

Run 3 test queries where keyword should dominate:

- `"BrightStart"` (rare token in resume chunk)
- `"Kraft Heinz"` (two-token phrase)
- `"UPRM Industrial Affiliates Program"` (longer phrase)

For each query, log:

- `keyword_query_string` actually sent
- Count of pure keyword hits
- Top 5 hits' `bm25_score` and `vector_score`

Assert at least one of these is true:

- Some returned hits have `keyword_score > 0`
- Top hits differ when keyword is on vs. off
- Fused ranking changes when you vary `keyword_boost`

**Hard Rule for Your Pipeline:**
If `keyword_score` is missing or always 0 after the fix, don't call it "hybrid" in Stage 1—treat it as vector-only and fix the keyword path before relying on it.

---

### 2. Entity Resolution Strategy ("Kevin" → Graph Entity ID)

Stage 2 traversal must be deterministic, so entity resolution must be reliable and explainable.

**Recommendation: 3-Tier Resolution (in this order)**

#### **Tier A (Best): Resolve Using Entities Already Attached to Top Chunks**

If Stage 1 returns `entity_ids_mentioned` (or you can extract entities from those chunks):

- Look at top N chunks (N = 5–10)
- Collect person entities mentioned
- Pick the one that best matches the query token ("Kevin") using:
  - Exact/alias match on first name, full name, normalized name
  - Frequency across top chunks (appears in multiple top chunks)
  - Proximity to `header_text` like "Introduction" / "Contact" / "Employment history" (usually identifies the resume owner)
- This avoids fuzzy guessing across the whole graph

#### **Tier B: Exact Match Against Person.name_normalized + Alias Table**

Maintain:

- `Person.name_normalized` (lowercase, stripped punctuation/diacritics)
- `Person.aliases` (["Kevin", "K. Ramirez", "Kevin Ramirez", etc.])

Resolution:

- Exact match on normalized name/aliases
- If multiple matches, prefer those connected to the retrieved file(s) (same resume file)

#### **Tier C (Last Resort): Fuzzy Match, But Bounded by Context**

If you must fuzzy match:

- Restrict candidates to people mentioned in the same files returned by Stage 1
- Or within the same org/project nodes surfaced from Stage 1
- Avoid global fuzzy search across all People; it causes "Kevin" collisions

#### **Resolution Output (Important)**

Always return:

- `resolved_person_id`
- `resolution_method` (tier A/B/C)
- `resolution_confidence` (simple numeric)
- `alternate_candidates` (top 3) if confidence < threshold

So the system can decide whether to proceed, ask for clarification, or fall back to "I don't have enough data."

---

### 3. Coworker Relationship Confidence (WORKS_WITH Based on Co-mentions)

If `WORKS_WITH` is computed from co-mention count, you need to avoid returning noisy "coworkers" that are just incidental mentions.

**Recommendation: Two-Phase Approach (Filter + Rank), Not "Return Everything"**

#### **Phase 1: Graph-Side Filtering (Minimum Thresholds)**

Set thresholds that depend on data size, but start with something simple:

- Use distribution-based or configurable thresholds (e.g., defaults equivalent to co_mention_count >= 2, adjustable per corpus or intent).

So coworker candidates must satisfy either:

- Strong single signal: `shared_project_count >= 1`
- OR repeated weak signal: `co_mention_count >= 2`

This keeps garbage out early.

#### **Phase 2: Graph-Native Ranking (Deterministic)**

Rank coworker candidates by a composite that you can explain.

**Example Scoring (Conceptual):**

- +3 per shared project
- +2 per shared org/team relation
- +1 per co-mention chunk
- +bonus if co-mentions occur in "Work Experience" headers vs "Skills/Intro"
- +bonus for recency (if you can timestamp roles/projects/chunks)

Then return:

- Top K coworkers (K = 5–10)
- Plus the evidence paths and supporting chunk IDs

#### **What the LLM Should Do**

The LLM should:

- Summarize the top ranked results
- Cite evidence (chunks + headers)
- Handle natural language

The LLM should **not** be the primary filter against noise. That's the graph's job.

#### **Short Policy (For Consistency)**

- **Stage 1 must prove it's hybrid.** If keyword scoring is always zero, treat as vector-only and fix the keyword path.
- **Entity resolution should be local-first.** Use entities/chunks/files returned by Stage 1 before any global fuzzy match.
- **Stage 2 should filter + rank deterministically.** Apply minimum thresholds and graph-native ranking; the LLM only narrates.

---

## Related Files & Locations

| Component              | File                                               | Lines     | Status                                |
| ---------------------- | -------------------------------------------------- | --------- | ------------------------------------- |
| Person query detection | `Agent/src/agents.py`                              | 40–95     | ✅ Fixed (Step 1 complete)            |
| Graph traversal entry  | `neo4j_backend/src/services/graphrag_retrieval.py` | ~290      | ✅ API integration complete           |
| Graph expansion logic  | `neo4j_backend/src/services/graphrag_retrieval.py` | 818–900   | ✅ Implemented (awaiting backend restart) |
| Chunk metadata output  | `neo4j_backend/main.py`                            | 210–240   | ⚠️ Incomplete (missing header fields) |
| Header vocab logic     | `Agent/build_header_vocab.py`                      | all       | ❌ Not integrated                     |
| Coworker relationships | `neo4j_backend/database/operations.py`             | 2831–2930 | ✅ Created, not used in traversal     |
| OPENAI_API_VERSION     | `RAG_Infra/.env`                                   | —         | ❌ Missing                            |
| EmbeddingsProvider     | `neo4j_backend/src/embeddings.py`                  | 12–100    | ✅ Working                            |

---

## Summary of Changes

This plan addresses the three root causes that prevented the Kevin query from working:

1. **Person detection** now recognizes single-token names in context-rich queries
2. **Graph traversal** is actually called and produces multi-hop results
3. **Semantic layer** is enabled with proper API versioning
4. **Keyword search** is validated to produce observable hybrid ranking
5. **Entity resolution** is deterministic and context-aware
6. **Coworker relationships** are pre-filtered and ranked, not dumped to the LLM

The goal remains clear: **hybrid search finds the evidence; graph traversal finds the relationships.**
