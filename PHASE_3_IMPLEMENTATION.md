# Phase 3 Implementation: Full File Expansion API

**Status:** ✅ **Implemented & Committed**  
**Date:** February 4, 2026  
**Branch:** RLM  
**Commit:** b83446d

---

## Overview

Phase 3 implements the **Full File Expansion API** - a critical component of the RLM (High-Recall Learner) system that expands retrieval entry chunks to complete file contexts.

**Goal:** When RLM is enabled and retrieval returns entry chunks, Phase 3 automatically expands these entry points to fetch ALL chunks from the source files, providing broader context for LLM-based processing in subsequent phases.

---

## Architecture

### Phase Sequence (RLM Pipeline)

```
Phase 1: RLM Enable/Disable
    ↓
Phase 2: High-Recall Retrieval (lower thresholds, more chunks)
    ↓
Phase 3: Full File Expansion (entry chunks → all chunks per file) ← YOU ARE HERE
    ↓
Phase 4: Recursive Summarization (MIT RLM - LLM inspection programs)
    ↓
Phase 5: Cross-File Merge + Citations
    ↓
Phase 6: Final Answer Generation
```

---

## Implementation Details

### New Module: `src/news_reporter/retrieval/file_expansion.py`

**Key Components:**

1. **`expand_to_full_files(entry_chunk_ids, neo4j_driver)`**
   - Takes list of entry chunk UUIDs from Phase 2 retrieval
   - Queries Neo4j for ALL chunks in files containing these entries
   - Returns chunks grouped by file_id, ordered by chunk_index

2. **`expand_with_chunks_only(entry_chunks, neo4j_driver)`**
   - Alternative: directly use chunk objects instead of IDs
   - Convenience wrapper around expand_to_full_files

3. **`filter_chunks_by_relevance(expanded_files, entry_chunk_ids, context_window=3)`**
   - Optional: maintain context around entry chunks if full expansion too large
   - Preserves entry chunk + surrounding chunks (configurable window)
   - Useful for context window size limits

### Integration: `src/news_reporter/workflows/workflow_factory.py`

```python
# Import Phase 3 API
from ..retrieval.file_expansion import expand_to_full_files, filter_chunks_by_relevance

# In run_one() function, after Phase 2 retrieval:
if high_recall_mode and hasattr(cfg, 'neo4j_driver') and cfg.neo4j_driver:
    try:
        expanded_context = await expand_to_full_files(
            entry_chunk_ids=extracted_chunk_ids,
            neo4j_driver=cfg.neo4j_driver
        )
        # Use expanded_context instead of entry context
    except Exception as e:
        logger.warning(f"Phase 3 expansion failed: {e}")
        # Fallback to original context
```

---

## Database Query Strategy

### Step 1: Identify Source Files
```cypher
MATCH (c:Chunk)-[:BELONGS_TO]->(f:File)
WHERE c.chunk_id IN $entry_chunk_ids
RETURN DISTINCT f.file_id, f.file_name
```

### Step 2: Fetch All Chunks Per File
```cypher
MATCH (c:Chunk)-[:BELONGS_TO]->(f:File)
WHERE f.file_id = $file_id
RETURN c
ORDER BY c.chunk_index ASC
```

**Execution:** Parallel queries for multiple files using async/await

---

## Logging & Diagnostics

### Log Pattern: `🔄 Phase 3`

All Phase 3 operations log with prefix `🔄 Phase 3` for easy filtering:

```bash
docker logs rag-agent | grep "Phase 3"
```

### Typical Log Output

```
2026-02-04 12:42:51,234 INFO 🔄 Phase 3: Starting file expansion for 15 entry chunks
2026-02-04 12:42:51,235 INFO 📍 Phase 3.1: Identifying source files from entry chunks...
2026-02-04 12:42:51,240 INFO ✅ Phase 3.1: Found 3 source files
2026-02-04 12:42:51,241 INFO 📍 Phase 3.2: Fetching full chunk sets per file...
2026-02-04 12:42:51,242 INFO   → Expanding file: Alexis Torres - DXC Resume.pdf (ID: file-123)
2026-02-04 12:42:51,250 INFO   ✅ File Alexis Torres - DXC Resume.pdf: Expanded 5 entry → 48 total chunks
2026-02-04 12:42:51,251 INFO   → Expanding file: project-doc.md (ID: file-456)
2026-02-04 12:42:51,258 INFO   ✅ File project-doc.md: Expanded 7 entry → 62 total chunks
2026-02-04 12:42:51,259 INFO   → Expanding file: technical-report.pdf (ID: file-789)
2026-02-04 12:42:51,265 INFO   ✅ File technical-report.pdf: Expanded 3 entry → 35 total chunks
2026-02-04 12:42:51,266 INFO ✅ Phase 3: Expansion complete - 15 entry chunks → 145 chunks across 3 files
```

---

## Return Data Structure

```python
{
    "file_id_1": {
        "chunks": [
            {
                "chunk_id": "uuid-1",
                "chunk_index": 0,
                "text": "Content...",
                "embedding_id": "emb-1",
                "chunk_type": "paragraph",
                "metadata": {...}
            },
            # ... more chunks ordered by index
        ],
        "file_name": "resume.pdf",
        "total_chunks": 48,
        "entry_chunk_count": 5
    },
    # ... more files
}
```

---

## Testing

### Test File: `test_phase3.py`

```bash
python test_phase3.py
```

**Test Queries:**
1. "What technical skills and certifications does Kelvin have?"
2. "List all projects Alexis has worked on and describe each one"

**Expected Behavior:**
- RLM enabled (high_recall_mode=True)
- Phase 2 retrieves entry chunks with low similarity threshold
- Phase 3 expands to full files
- Logs show expansion statistics

**Sample Log Check:**
```bash
docker logs rag-agent | grep -E "Phase 3|Expanded|expansion"
```

---

## Integration Checklist

- [x] Create `file_expansion.py` module
- [x] Implement `expand_to_full_files()` function
- [x] Implement `expand_with_chunks_only()` wrapper
- [x] Implement `filter_chunks_by_relevance()` optional filter
- [x] Import and integrate in `workflow_factory.py`
- [x] Add logging with 🔄 Phase 3 prefix
- [x] Create test file `test_phase3.py`
- [x] Commit and push changes

---

## Next Steps: Phase 4

Phase 4 (Recursive Summarization) will use Phase 3's expanded chunks:

1. For each file returned by Phase 3:
   - Generate LLM inspection program based on user query
   - Filter chunks using generated program
   - Summarize filtered chunks

2. Collect file-level summaries for Phase 5 merge

---

## Performance Considerations

### Chunk Expansion Overhead
- **Best case:** 5 entry chunks → 50 total chunks (minimal overhead)
- **Worst case:** 15 entry chunks → 200+ total chunks (larger context)

### Optimization Options (Future)
1. **Lazy Expansion:** Expand only top-K files
2. **Chunk Sampling:** Return every Nth chunk if file too large
3. **Caching:** Cache file chunk structures across queries

### Database Load
- Parallel async queries per file
- Index on `file_id` and `chunk_index` for fast retrieval
- No write operations (read-only)

---

## Error Handling

### Error Cases Handled

1. **No Entry Chunks:** Returns empty dict
2. **File Not Found:** Skips file, logs warning
3. **Neo4j Connection Error:** Falls back to original context
4. **Metadata Missing:** Continues with available metadata

### Error Recovery

```python
try:
    expanded_context = await expand_to_full_files(entry_ids, driver)
except Exception as e:
    logger.warning(f"Phase 3 expansion failed: {e}")
    # Use original Phase 2 context as fallback
    expanded_context = original_context
```

---

## Configuration

### Required Configuration

```python
# settings.py
neo4j_driver: Driver  # Active Neo4j connection
rlm_enabled: bool     # Enable RLM mode
```

### Optional Configuration

```python
# file_expansion.py defaults
include_metadata: bool = True  # Include chunk metadata
context_window: int = 3        # For filter_chunks_by_relevance()
```

---

## File Manifest

**New Files:**
- `src/news_reporter/retrieval/file_expansion.py` (180+ lines)
- `test_phase3.py` (120+ lines)

**Modified Files:**
- `src/news_reporter/workflows/workflow_factory.py`
  - Added import: `expand_to_full_files`, `filter_chunks_by_relevance`
  - Added Phase 3 expansion logic in `run_one()` function

---

## Success Criteria

✅ Phase 3 is considered successful when:

1. **Logs show expansion:** "Expanded X entry chunks → Y total chunks"
2. **Multiple files:** Processes multiple source files
3. **Chunk ordering:** Chunks returned in order by index
4. **Fallback works:** Graceful handling if expansion fails
5. **Performance:** Expansion completes in <2 seconds for typical queries

---

## Related Documentation

- `PHASE_2_THROUGH_6_DETAILED_SPEC.md` - Full RLM pipeline specification
- `AGENT_EXECUTION_FLOW.md` - Agent workflow details
- `RLM_IMPLEMENTATION_PLAN.md` - RLM implementation roadmap

---

## Contact

For questions about Phase 3 implementation:
- Check logs with: `docker logs rag-agent | grep "Phase 3"`
- Review code: [file_expansion.py](src/news_reporter/retrieval/file_expansion.py)
- Run test: `python test_phase3.py`
