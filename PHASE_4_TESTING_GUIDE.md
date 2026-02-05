# Phase 4 Testing Guide

## Pre-Test Checklist

- [ ] OpenAI API key configured: `export OPENAI_API_KEY="sk-..."`
- [ ] Model configured: `export OPENAI_MODEL_NAME="gpt-4"`
- [ ] Neo4j running with Phase 3 data (expanded chunks)
- [ ] Docker-compose updated with latest code
- [ ] All dependencies installed (openai, neo4j, etc.)

## Test Scenarios

### 1. Manual End-to-End Test

**Objective:** Verify Phase 4 executes in RLM mode and generates summaries

**Setup:**
```bash
cd c:\Alexis\Projects\neo4j_backend
export OPENAI_API_KEY="sk-your-key-here"
export OPENAI_MODEL_NAME="gpt-4"
export RLM_ENABLED=true
```

**Steps:**
1. Start Docker containers:
   ```bash
   docker-compose -f docker-compose.dev.yml up -d
   ```

2. Make a test query (e.g., to chat endpoint):
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{
       "messages": [{"role": "user", "content": "Explain the key findings from the documents"}],
       "rlm_enabled": true
     }'
   ```

3. Monitor logs for Phase 4 execution:
   ```bash
   docker-compose logs -f agent | grep -E "Phase 4|recursive|summary"
   ```

4. Check output files:
   ```bash
   docker exec agent ls -la /app/logs/chunk_analysis/
   docker exec agent cat /app/logs/chunk_analysis/summaries_rlm_enabled.md
   ```

**Expected Results:**
- Logs show: "🔄 Phase 4: Attempting recursive summarization..."
- File `summaries_rlm_enabled.md` created with content
- Summary includes file names, chunk counts, and summaries
- No errors in workflow (graceful handling if Phase 4 fails)

### 2. Verify Graceful Degradation (No OpenAI)

**Objective:** Ensure workflow completes even without OpenAI API

**Setup:**
```bash
# Unset OpenAI key
unset OPENAI_API_KEY
export RLM_ENABLED=true
```

**Steps:**
1. Make a query with RLM enabled
2. Check logs for: "⚠️ Phase 4: OpenAI not available..."
3. Verify workflow completes and returns answer
4. Confirm answer uses Phase 3 context (not Phase 4 summaries)

**Expected Results:**
- Phase 4 skips with warning
- Phase 3 context used instead
- Final answer still generated
- No workflow failure

### 3. Verify RLM Disabled Path

**Objective:** Ensure Phase 4 doesn't execute when RLM disabled

**Setup:**
```bash
export RLM_ENABLED=false
export OPENAI_API_KEY="sk-..."
```

**Steps:**
1. Make a query
2. Check logs for Phase 4 references
3. Verify no summaries file created

**Expected Results:**
- Logs don't show Phase 4 execution
- No `summaries_rlm_enabled.md` file created
- Workflow uses default retrieval (Phase 1-2)

### 4. Verify Markdown Logging

**Objective:** Check markdown file format and content

**Steps:**
1. Make query with RLM enabled
2. Check file: `Agent/logs/chunk_analysis/summaries_rlm_enabled.md`

**Verify:**
```markdown
# Phase 4: Recursive Summarization (RLM Enabled)

**Execution Time:** 2024-...
**Query:** [user query]
**Total Summaries:** [number]

---

## 1. [Filename]

**File ID:** [uuid]
**Chunks:** [X]/[Y] summarized
**Expansion Ratio:** [ratio]x

### Summary
[LLM-generated summary text]

### Source Chunks
[comma-separated chunk IDs]

---
```

**Expected Results:**
- Proper markdown formatting
- All sections present
- Correct metadata
- Summaries contain specific details (not generic)

## Expected LLM Behavior

### Inspection Logic Generation
- Should identify 3-5 specific criteria related to query
- Examples: "mentions company X", "contains financial data", "discusses risks"
- Temperature: 0.3 (deterministic)

### Chunk Filtering
- Should return JSON: `{"relevant_indices": [0, 1, 3, ...]}`
- Max 10 chunks per file
- Fallback to first N chunks if fewer match

### Summarization
- 3-5 sentence summaries
- Specific details from chunks
- Query-focused content
- Ready for final answer generation

## Performance Benchmarks

| Metric | Expected | Acceptable |
|--------|----------|------------|
| Time per file | 2-3 sec | <10 sec |
| Time for 5 files | 10-15 sec | <50 sec |
| LLM API calls | 3 per file | N/A |
| Cost per file | $0.01-0.05 | <$0.10 |

## Error Scenarios to Test

### 1. OpenAI API Timeout
```python
# Mock slow API response
# Expected: Logs warning, uses Phase 3 context
```

### 2. Invalid LLM Response Format
```python
# LLM returns invalid JSON
# Expected: Falls back to chunk selection, continues
```

### 3. Empty Expanded Files
```python
# Phase 3 returns no files
# Expected: Phase 4 skipped, uses search context
```

### 4. Missing Source Chunks
```python
# Chunk IDs don't exist in expanded_files
# Expected: Logs warning, uses available data
```

### 5. Malformed Query
```python
# Query is empty or very long
# Expected: Phase 4 handles gracefully
```

## Log Monitoring

### Key Log Messages to Track

**Success Path:**
```
✅ Phase 4: Attempting recursive summarization...
📍 Phase 4.1: Analyzing file '...' (X entry → Y total chunks)
  → Step 1: Generating inspection logic
  → Step 2: Identifying relevant chunks
  → Step 3: Summarizing ... relevant chunks
  ✅ File '...': Summary generated from X chunks
✅ Phase 4: Successfully assembled N file summaries
```

**Graceful Degradation:**
```
⚠️ Phase 4: OpenAI not available; skipping
⚠️ Phase 4: LLM-based summarization failed
⚠️ Failed to apply inspection logic: [error]
⚠️ Phase 4: Recursive summarization skipped
```

**Failures (that should be caught):**
```
❌ Failed to log Phase 4 summaries: [error]
❌ Phase 4: Error during analysis: [error]
```

## Test Data Preparation

### Recommended Test Queries

1. **Financial Query:** "What are the key financial metrics?"
   - Expected: Summaries should include numbers, ratios, performance indicators

2. **Procedural Query:** "How does the process work?"
   - Expected: Summaries should include steps, workflows, dependencies

3. **Comparative Query:** "Compare the two approaches"
   - Expected: Summaries should highlight differences and similarities

4. **Complex Query:** "What are the implications of X on Y?"
   - Expected: Summaries should show relationships and impacts

### Test Document Characteristics

Ideal test scenario:
- 2-3 expanded files (not too many LLM calls)
- Each file 20-30 chunks (reasonable size for summarization)
- Mix of relevant and irrelevant chunks (tests filtering)
- Variety of content types (tests rule generation)

## Regression Testing

### Check These Don't Break

1. **Non-RLM Mode:** Should work exactly as before
2. **Phase 1-3:** Should still work independently
3. **Citation Format:** Should be preserved from Phase 3
4. **Context Assembly:** Should maintain ordering

### Run These Tests Before Deployment

```bash
# Test 1: Non-RLM mode still works
RLM_ENABLED=false python -m pytest tests/test_workflow.py -k "test_non_rlm_workflow"

# Test 2: Phase 3 still works without Phase 4
RLM_ENABLED=true OPENAI_API_KEY="" python -m pytest tests/test_workflow.py -k "test_phase3"

# Test 3: Citations preserved
python -m pytest tests/test_citations.py -k "test_phase4_citations"
```

## Success Criteria for Phase 4

✅ **Functionality:**
- [ ] Phase 4 executes after Phase 3 when RLM enabled
- [ ] Generates per-file summaries using LLM
- [ ] Summaries include 3-5 sentences
- [ ] Citations reference real chunk IDs

✅ **Robustness:**
- [ ] Handles missing OpenAI gracefully
- [ ] Handles empty chunk sets
- [ ] Handles LLM API errors
- [ ] Handles malformed JSON responses

✅ **Integration:**
- [ ] Context properly assembled before assistant
- [ ] Markdown logging works correctly
- [ ] No workflow failures due to Phase 4

✅ **Performance:**
- [ ] <3 sec per file typical case
- [ ] <50 sec for 10 files
- [ ] Acceptable API costs

## Known Issues / Future Improvements

### Current Limitations
1. Sequential LLM calls (can parallelize)
2. No caching of inspection logic
3. No rate limiting on LLM calls
4. Token limits not checked before calling LLM

### Potential Optimizations
1. Batch multiple files in single LLM call
2. Cache inspection logic per query
3. Implement exponential backoff retry
4. Add token counting before LLM calls
5. Fallback to rule-based summarization

### Testing These Later
- [ ] Load testing with 50+ files
- [ ] Stress testing with long queries
- [ ] Concurrency testing (multiple simultaneous queries)
- [ ] Cost analysis over time

---

**Next Phase:** Phase 5 - Cross-File Merge + Final Answer + Citations
