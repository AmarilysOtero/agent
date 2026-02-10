# Chunk #6 Execution Trace - Where Did It Get Lost?

**Query Date:** February 7, 2026, 04:22:53  
**Query:** "tell me the list of skill you can find for Kevin"

---

## 1. Inspection Function Evaluation

### Chunk #6 Inspection Code
```python
def evaluate_chunk_relevance(chunk_text: str) -> bool:
    if "skill" in chunk_text.lower():
        return True
    if "kevin" in chunk_text.lower():
        return True
    if "certifications" in chunk_text.lower():
        return True
    return False
```

### Chunk #6 Text
```
Certifications (Professional Activities, Certifications, and Training Attended)
Azure AI Fundamentals, granted June 2025
Solumina T1610 Config Basics, granted in February 2023
...
- Backend Development (3 years)
- PostgreSQL - (3 years)
- Node.js/Express.js - (2 years)
...
```

### Evaluation Result
✅ **Should return TRUE**
- Contains "Certifications" → matches condition 3: `if "certifications" in chunk_text.lower()`
- Also contains "Backend Development" which has "skills" meaning

---

## 2. Pipeline Stages (with New Debug Logging)

### Stage A: Inspection Execution (lines ~1406-1413)
**Expected log output:**
```
✓ chunk:6 passed inspection
```

**Current status:** UNKNOWN (debug logging not in this execution yet)

---

### Stage B: Post-Inspection Summary (line 1448)
**Expected log output:**
```
📋 Post-inspection: N chunks passed (IDs: [chunk:0, chunk:1, chunk:2, chunk:6, ...])
```

**Actual result from logs:** Only shows 3 chunks in final answer
- chunk:0 = Introduction (should have failed - just blank)
- chunk:1 = Kevin intro paragraph (✓ correct)
- chunk:2 = Skills header (should have failed - just a header)
- ❌ chunk:6 = Actual skills list (MISSING - this should be here!)

### Analysis: Post-Inspection Chunk Selection

Looking at the inspection codes:
- **chunk:0**: Code looks for "skill" OR "kevin" OR "list" → Text is just "Introduction" → **Should be FALSE**
  - But it made the final answer!
- **chunk:1**: Code looks for "skill" OR "list" OR "kevin" OR "full-stack" → Contains all of these → **Should be TRUE** ✓
- **chunk:2**: Code looks for "skills" OR "skill" OR "list" → Text is "Skills ## Skills" → **Should be TRUE** ✓
- **chunk:6**: Code looks for "skill" OR "kevin" OR "certifications" → Contains "certifications" → **Should be TRUE** ✓

❌ **Discrepancy:** chunk:0 passed but shouldn't have. chunk:6 should have passed but didn't.

---

### Stage C: De-duplication Filter (DISABLED)
```python
# DISABLED: Deduplication was too aggressive and filtered out relevant chunks
# like certifications (chunk 6) as near-duplicates of top skills (chunk 3).
# final_selected_chunk_ids = _deduplicate_chunks(
#     chunks=chunks,
#     selected_chunk_ids=final_selected_chunk_ids
# )
```

**Decision:** De-duplication is now DISABLED, so it can't be the culprit.

**Expected log output (if it were enabled):**
```
📉 Budget filter: pre_dedup → post_dedup chunks
```

**Status:** This filter is disabled, so chunk #6 must be lost in a different stage.

---

### Stage D: Selection Budget Filter (lines 1161-1177, 1472-1479, 2476-2483)

The budget limits are:
- `MAX_SELECTED_CHUNKS_PER_FILE = 8`
- `MAX_TOTAL_CHARS_FOR_SUMMARY = 12000`

**Expected log output:**
```
✓ Selected chunk:0: score=..., chars=...
✓ Selected chunk:1: score=..., chars=...
✓ Selected chunk:2: score=..., chars=...
✓ Selected chunk:6: score=..., chars=...
✗ Excluded chunk:X: max_chars_exceeded (score=..., chars=...)

📉 Budget filter: N → M chunks (excluded: [...])
```

**Current status:** 
- Only 3 chunks made it through
- Budget allows 8 chunks, so capacity wasn't exceeded
- This suggests chunk #6 either:
  1. Never reached the budget filter (failed inspection or intermediate filter)
  2. Was scored lower than other chunks and didn't make the cut-off

**Chunk Scoring in Budget Filter:**
```python
# Boost for current/present markers → +10 points
# Boost for recent years (2024, 2025) → +5 points
# Text length bonus → up to +3 points
```

### Score Analysis:

| Chunk | Contains "present" | Year 2024+ | Chars | Expected Score |
|-------|-------------------|----------|-------|-----------------|
| chunk:0 | ❌ | ❌ | ~27 | 0 |
| chunk:1 | ❌ | ❌ | ~576 | 0.5 |
| chunk:2 | ❌ | ❌ | ~16 | 0 |
| chunk:6 | ❌ | ✅ (2025) | ~600 | 5.6 |

**Problem:** chunk:6 has a higher score than chunk:0 and chunk:2, yet they were selected and chunk:6 wasn't!

This suggests chunk:6 may not have been in the post-inspection selection at all.

---

## 3. Root Cause - THE GUARDRAIL!

### Code at Lines 1431-1451 in recursive_summarizer.py

```python
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
```

### What This Means

If **> 70% of chunks pass inspection**, the system automatically:
1. **Resets** all selected chunks to empty
2. **Falls back** to ONLY the first 3 chunks (by original order)
3. **Ignores** everything the inspection functions found

### The Smoking Gun

This guardrail triggers when the LLM-generated inspection code is **too permissive** (too many chunks marked as relevant). When it triggers:
- ✅ chunk:0 is selected (first chunk by order)
- ✅ chunk:1 is selected (second chunk by order)  
- ✅ chunk:2 is selected (third chunk by order)
- ❌ chunk:6 is NEVER EVALUATED (outside the fallback range)

**This explains exactly what happened!**

---

## Root Cause Hypothesis (REVISED)

---

## 4. How to Verify This With The New Debug Logging

Run the query and look for these log messages:

```
⚠️  Phase 4: Per-chunk mode selected N/20 chunks (X%) - too many, treating as low-signal filter. Using fallback.
⚠️  Phase 4: No relevant chunks identified, using fallback (first 3 chunks)

✓ chunk:0 passed inspection  (from fallback, not inspection result)
✓ chunk:1 passed inspection  (from fallback, not inspection result)
✓ chunk:2 passed inspection  (from fallback, not inspection result)
```

If you see the "too many... treating as low-signal" warning, that's the smoking gun!

---

## 5. The Fix

The guardrail threshold of **0.7 (70%)** or **0.35 (35%)** is too aggressive for the per-chunk inspection model. Options:

### Option 1: Increase the guardrail threshold
```python
if selection_ratio > 0.9 and len(chunks) > 5:  # Changed from 0.7 to 0.9
    # Only reset if >90% of chunks pass
```

### Option 2: Improve inspection code generation
- Make the LLM generate stricter inspection functions
- Add constraints like "must check for BOTH query terms"
- Don't use OR logic, require AND for relevance

### Option 3: Remove the guardrail for per-chunk mode
- If using per-chunk mode, trust the LLM-generated code
- Only apply guardrail to iterative mode

### Recommended: Option 2

Make the LLM generate stricter code. For example, instead of:
```python
# Current (too permissive):
if "skill" in text.lower():
    return True
if "kevin" in text.lower():
    return True
```

Ask for (more strict):
```python
# Stricter:
if "skill" in text.lower() and ("kevin" in text.lower() or "list" in text.lower()):
    return True
```

---

## 6. Summary of The Real Issue

| Item | Finding |
|------|---------|
| **Inspection Code** | ✅ Looks reasonable |
| **Chunk #6 Logic** | ✅ Correct (contains "certifications") |
| **Why chunk #6 Excluded** | ❌ Guardrail reset all selections to fallback (first 3 chunks) |
| **Root Cause** | LLM generated inspection functions that are TOO PERMISSIVE (>70% of chunks pass) |
| **Trigger Event** | Guardrail at line 1431: `if selection_ratio > 0.7 and len(chunks) > 5` |
| **Fix** | Increase threshold to 0.9 OR improve inspection code generation |
