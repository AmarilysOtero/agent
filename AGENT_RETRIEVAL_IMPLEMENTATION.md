# Agent Retrieval Index Implementation File

**Date:** January 30, 2026  
**Status:** Planning - Ready to Implement  
**Target:** Implement structural index routing in Agent for section-aware retrieval

---

## Overview

This file documents the implementation tasks needed to add query routing and section-aware retrieval to the Agent layer. The backend index (Phases 1-2) is complete; now we need to implement the retrieval layer (Phases 3-5).

---

## Files to Modify

### 1. Agent/src/news_reporter/agents/agents.py

**Location:** AiSearchAgent class (~line 520)

**Add Methods:**

- _classify_query_intent(query: str, person_names: List[str]) -> dict
  - Detects: section_based_scoped, section_based_cross_document, semantic
  - Input: query text, extracted person names
  - Output: routing decision + section_query parameter

- _extract_attribute_phrase(query: str, attribute_keywords: List[str]) -> str
  - Extracts section name from query around attribute keyword
  - Example: "Kevin's industry experience" → "industry experience"
  - Used by classification to generate section_query

**Modify Method:**

- search_hybrid(query, person_names, **kwargs)
  - Add classification call after person name extraction
  - Add conditional routing based on intent type
  - Add result filtering by intent type
  - Add logging for routing decisions

**Add Class Variable:**

- ATTRIBUTE_KEYWORDS = [list of keywords]
  - skill, experience, education, qualification, role, position, project, certification, background, expertise, training, achievement

---

## Implementation Checklist

### Phase 3: Query Classification

- [ ] Add _classify_query_intent() method
- [ ] Add _extract_attribute_phrase() helper
- [ ] Define ATTRIBUTE_KEYWORDS list
- [ ] Test classification with 10 sample queries
- [ ] Verify 90%+ accuracy on: person+attribute, attribute-only, general

### Phase 4: Routing Logic

- [ ] Update search_hybrid() to call classification
- [ ] Add hard routing for section_based_scoped
- [ ] Add hard routing for section_based_cross_document
- [ ] Add soft routing for semantic
- [ ] Add logging for each routing decision
- [ ] Test routing with all 3 query types

### Phase 5: Result Processing

- [ ] Add filtering by routing type
- [ ] Filter hard-routed results to include section_name
- [ ] Ensure section context in findings
- [ ] Test result quality and relevance
- [ ] Verify precision for structural queries

### Phase 6: Testing & Validation

- [ ] Test 1: Person + Attribute (hard routing)
  - Query: "What are Kevin's Industry Experience?"
  - Expected: Only chunks from Industry Experience section

- [ ] Test 2: Cross-Document (hard routing)
  - Query: "All candidates with Python skills"
  - Expected: Skills sections from multiple files

- [ ] Test 3: Semantic (soft routing)
  - Query: "Tell me about machine learning"
  - Expected: Ranked results with section boosting

- [ ] Test 4: Edge Cases
  - Query with no person names
  - Query with no attribute keywords
  - Query with multiple people
  - Empty results handling

### Phase 7: Deployment

- [ ] Update docstrings in methods
- [ ] Add examples to method comments
- [ ] Add logging explanations
- [ ] Test in Docker container
- [ ] Commit: "Implement structural index routing for Agent retrieval"

---

## Code Change Locations

| Item | File | Method/Line | Type | Effort |
|------|------|-------------|------|--------|
| Add classification | agents.py | ~line 1100 | New Method | 30 lines |
| Add extraction | agents.py | ~line 1140 | New Method | 20 lines |
| Add keywords | agents.py | ~line 100 | Class Var | 5 lines |
| Update search_hybrid | agents.py | ~line 540 | Modify | 50 lines |
| Add result filter | agents.py | ~line 600 | New Method | 15 lines |

**Total:** ~120 lines of code, 3-4 hours implementation

---

## Key Decision Points

1. **Classification Accuracy**
   - How strict should person name matching be? (exact vs partial)
   - How many keywords before classifying as attribute? (any vs multiple)
   - Decision: Conservative (exact name, any keyword)

2. **Routing Decision Point**
   - When should routing happen? Before or after person extraction?
   - Decision: After person extraction, before graphrag_search() call

3. **Result Filtering**
   - Strict filter (section queries only return section results)?
   - Or soft filter (boost section results but include others)?
   - Decision: Strict for hard routing, soft for semantic

4. **Error Handling**
   - What if section clustering not run? (SectionType nodes missing)
   - Decision: Log warning, fall back to semantic search

---

## Testing Queries by Type

### Hard Routing - Person + Attribute
`
"What are Kevin's Industry Experience?"
"Tell me about Sarah's technical skills"
"Show me John's certification background"
`

### Hard Routing - Cross-Document Attribute
`
"All candidates with Python skills"
"Find people with machine learning experience"
"Show me all project management expertise"
`

### Soft Routing - General Semantic
`
"Tell me about deep learning"
"What's the difference between SQL and NoSQL?"
"Explain neural networks"
`

---

## Success Metrics

- ✅ Classification accuracy: 90%+ on mixed query types
- ✅ Section query precision: 95%+ (only relevant sections returned)
- ✅ Semantic query relevance: Top 3 results are relevant
- ✅ No regressions: Existing semantic queries still work
- ✅ Logging: Clear routing decisions in logs

---

## Next Steps

1. Implement _classify_query_intent() method
2. Implement _extract_attribute_phrase() helper
3. Update search_hybrid() routing logic
4. Add result filtering
5. Run test cases
6. Commit and push

---

## Dependencies

- Backend: Section index already implemented (Phases 1-2)
- Backend: Retrieval methods already implemented (Phase 3)
- Agent: Person name extraction already working
- Agent: graphrag_search() already works

No external dependencies needed - all backend support is ready.

---

## Rollback Plan

If issues arise:

1. **Remove classification:** Comment out _classify_query_intent() call
2. **Disable hard routing:** Force all queries to soft routing (semantic)
3. **Disable section boosting:** Set use_section_routing=False
4. **Git revert:** git revert HEAD~1 to previous working state

---

## Notes

- No changes to ingestion or backend needed
- No new dependencies required
- Backward compatible (existing semantic search still works)
- Optional: Can gradually enable for subset of users
