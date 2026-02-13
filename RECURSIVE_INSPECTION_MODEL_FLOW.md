# Recursive Inspection Model (RLM) Flow Documentation

## Overview

This document describes the complete flow of the Recursive Inspection Model (RLM) from the moment a file UUID is identified through full chunk evaluation and summarization. The process operates in **Phase 4: Recursive Summarization** of the document retrieval pipeline.

## High-Level Architecture

```
File UUID Identified
       ↓
    ┌─────────────────────────────┐
    │ Phase 4: RLM Processing     │
    │ (recursive_summarizer.py)   │
    └──────────────┬──────────────┘
                   ↓
        ┌──────────────────────┐
        │ Chunks Retrieved     │
        │ (N chunks per file)  │
        └──────────┬───────────┘
                   ↓
        ┌──────────────────────────────┐
        │ Two Evaluation Modes:        │
        ├──────────────────────────────┤
        │ USE_MIT_RLM_RECURSION=false: │
        │ Per-Chunk Mode               │ (Legacy)
        │                              │
        │ USE_MIT_RLM_RECURSION=true:  │
        │ Iterative Mode               │ (Preferred)
        └──────────────────────────────┘
```

---

## Phase 4: Entry Point

**Function**: `recursively_retrieve_and_summarize()`  
**File**: `recursive_summarizer.py` (lines 1270-1700+)

### Step 0: Initialization

```python
for file_id, file_data in expanded_files.items():
    file_name = file_data.get("file_name", "unknown")
    chunks = file_data.get("chunks", [])
    entry_chunk_count = file_data.get("entry_chunk_count", 0)
    total_chunks = len(chunks)
```

**Current State**:

- `file_id`: Unique identifier for the file (e.g., "file_abc123")
- `chunks`: List of Dict objects, each containing: `{chunk_id, text, page, offset, section, ...}`
- `total_chunks`: Number of chunks expanded by Phase 3
- `query`: User query passed through entire pipeline

---

## Mode 1: Per-Chunk Inspection Mode (Legacy)

**Enabled When**: `USE_MIT_RLM_RECURSION = false` (default)  
**Performance**: Expensive (N LLM calls for N chunks)

### Flow Diagram

```
                    For Each Chunk (1 to N):
                            ↓
        ┌───────────────────────────────────────┐
        │ STEP 1: CODE GENERATION (LLM)         │
        │─────────────────────────────────────── │
        │ Function: _generate_inspection_logic()│
        │ Input: Query, chunk text, chunk_id    │
        │ Output: Python code (evaluate_chunk_  │
        │         relevance function)           │
        └───────────────┬───────────────────────┘
                        ↓
        ┌───────────────────────────────────────┐
        │ STEP 2: CODE EXECUTION (Sandbox)      │
        │─────────────────────────────────────── │
        │ Function: _evaluate_chunk_with_code() │
        │ Input: Generated code, chunk text     │
        │ Output: Boolean (True = relevant)     │
        │ Execution: subprocess_exec()          │
        │  Primary: Sandbox (subprocess)        │
        │  Fallback: In-process exec()          │
        └───────────────┬───────────────────────┘
                        ↓
        ┌───────────────────────────────────────┐
        │ STEP 3: COLLECT RESULTS               │
        │─────────────────────────────────────── │
        │ Add to relevant_chunks if True        │
        │ Continue to next chunk                │
        └───────────────┬───────────────────────┘
                        ↓
                  All Chunks Done
                        ↓
        ┌───────────────────────────────────────┐
        │ STEP 4: GUARDRAILS                    │
        │─────────────────────────────────────── │
        │ Check selection_ratio > 0.9           │
        │ If True: Too many selected, use       │
        │          deterministic fallback       │
        │ Check if no chunks selected:          │
        │ If True: Use first 3 chunks fallback  │
        └───────────────┬───────────────────────┘
                        ↓
        ┌───────────────────────────────────────┐
        │ STEP 5: BUDGET FILTER & POST-PROCESS  │
        │─────────────────────────────────────── │
        │ Apply selection budget:               │
        │  MAX_SELECTED_CHUNKS_PER_FILE (8)     │
        │  MAX_TOTAL_CHARS_FOR_SUMMARY (12000)  │
        │ Apply current-role prioritization     │
        │ Deduplicate (disabled)                │
        └───────────────┬───────────────────────┘
                        ↓
                Final Relevant Chunks
```

### Detailed Per-Chunk Flow

#### Phase 4.1: Per-Chunk Code Generation

**Function**: `_generate_inspection_logic()`  
**Input Parameters**:

- `query`: User query (e.g., "where does Kevin work?")
- `file_name`: Document name (e.g., "resume.pdf")
- `chunk_id`: Unique chunk identifier (e.g., "chunk_5:offset_2400")
- `chunk_text`: Full text of the chunk (e.g., "Kevin worked at Google...")
- `llm_client`: Azure OpenAI async client
- `model_deployment`: Azure deployment name (e.g., "o3-mini")

**LLM Call Details**:

```
System Prompt: "You are a Python expert implementing iterative inspection programs..."
Temperature: 0.3 (low randomness, deterministic)
Max Tokens: 300
Deployment: gpt-4o-mini (forced for code gen, even if o3-mini configured)
```

**Generated Code**:

```python
def evaluate_chunk_relevance(chunk_text: str) -> bool:
    """
    Evaluate if this specific chunk is relevant to the query.
    Returns True if relevant, False otherwise.
    """
    # LLM-generated criteria based on chunk content
    if 'google' in chunk_text.lower():
        return True
    if 'company' in chunk_text.lower() and 'kevin' in chunk_text.lower():
        return True
    return False  # IMPORTANT: Must default to False
```

**Validation** (`_validate_inspection_code()`):

- Checks for FunctionDef AST node
- Validates return statements (defaults to False, not True)
- Checks evidence-only rule: string literals come from query or chunk text
- Catches inverted logic patterns

**Sanity Tests** (`_run_inspection_code_sanity_tests()`):

- Tests with empty string → must return False
- Tests with garbage text → must return False
- Tests with actual chunk text → returns deterministic result

**Result**: Valid Python code string (500-1000 chars)

#### Phase 4.2: Per-Chunk Code Execution

**Function**: `_evaluate_chunk_with_code()`  
**Input**: Generated code + chunk text

**Execution Path Priority**:

```
1. PRIMARY: Subprocess Sandbox
   └─ Function: sandbox_exec(kind="chunk_eval", code=..., input_data=...)
   └─ Timeout: 2 seconds (configurable)
   └─ Memory Limit: 256 MB (configurable)
   └─ Restricted Builtins: No imports, limited iteration
   └─ Returns: {ok: bool, result: bool, error: str}

   IF sandbox succeeds:
      Return result boolean

   IF sandbox fails (timeout/crash):
      Fall through to fallback

2. FALLBACK: In-Process exec()
   └─ Function: exec(code, safe_globals, namespace)
   └─ safe_globals: Restricted dictionary (no dangerous modules)
   └─ Execute: evaluate_chunk_relevance(chunk_text)
   └─ Returns: func(chunk_text) → boolean
```

**Result**: Binary decision - chunk is relevant (True) or not (False)

#### Phase 4.3: Selection and Post-Processing

**Guardrails**:

```python
selection_ratio = len(relevant_chunk_ids) / max(1, len(chunks))
if selection_ratio > 0.9 and len(chunks) > 5:
    # REJECT: Too many chunks selected (>90%)
    # Use deterministic top-K fallback instead
    # Set confidence = 0.2
```

**Budget Filter**:

```python
final_selected_chunk_ids = _apply_selection_budget(
    chunks=chunks,
    selected_chunk_ids=final_selected_chunk_ids,
    max_chunks=MAX_SELECTED_CHUNKS_PER_FILE,     # Default: 8
    max_chars=MAX_TOTAL_CHARS_FOR_SUMMARY        # Default: 12000
)
```

**Output**: List of selected chunk IDs and corresponding chunk texts

---

## Mode 2: Iterative Inspection Mode (Preferred RLM)

**Enabled When**: `USE_MIT_RLM_RECURSION = true`  
**Performance**: Efficient (1 per-chunk eval + N iterations)  
**Advantages**:

- Smarter narrowing each iteration
- One LLM call per iteration (not per chunk)
- Handles 100+ chunks efficiently

### Flow Diagram

```
All Chunks from File
        ↓
┌───────────────────────────────────────────────┐
│ PREFILTER STAGE                               │
├───────────────────────────────────────────────┤
│ 1. Exact dedup (hash-based)                   │
│ 2. Candidate cap (cheap scoring)              │
│    MAX_PREFILTER = 60 chunks max              │
└──────────────┬────────────────────────────────┘
               ↓
         Active Chunks
               ↓
┌───────────────────────────────────────────────┐
│ PHASE A: BOOLEAN EVALUATION (NEW)             │
├───────────────────────────────────────────────┤
│ For each chunk:                               │
│   1. Generate evaluate_chunk_relevance()      │
│      (LLM call per chunk)                     │
│   2. Execute via sandbox + fallback           │
│   3. Track approved chunk IDs                 │
│        ↓                                       │
│   Result: boolean_approved_chunk_ids set     │
└──────────────┬────────────────────────────────┘
               ↓
┌───────────────────────────────────────────────┐
│ ITERATION LOOP (up to 5 iterations)           │
├───────────────────────────────────────────────┤
│                                               │
│  For iteration 1 to MAX_RLM_ITERATIONS:      │
│                                               │
│  ┌─────────────────────────────────────────┐  │
│  │ STEP 1A: Generate Iteration Program    │  │
│  │─────────────────────────────────────── │  │
│  │ Function: _generate_recursive_         │  │
│  │           inspection_program()         │  │
│  │ Input:                                 │  │
│  │   - Query                              │  │
│  │   - current_active_chunks              │  │
│  │   - iteration number                   │  │
│  │   - previous_extracted_data            │  │
│  │ Output:                                │  │
│  │   Python code for                      │  │
│  │   inspect_iteration(chunks) function   │  │
│  │ LLM Call: 1 per iteration              │  │
│  └──────────┬────────────────────────────┘  │
│             ↓                                 │
│  ┌─────────────────────────────────────────┐  │
│  │ STEP 1B: Validate Generated Program    │  │
│  │─────────────────────────────────────── │  │
│  │ - AST validation                       │  │
│  │ - Sanity tests (no "select all")       │  │
│  │ - If invalid: use fallback program     │  │
│  └──────────┬────────────────────────────┘  │
│             ↓                                 │
│  ┌─────────────────────────────────────────┐  │
│  │ STEP 2A: Execute Iteration Program     │  │
│  │─────────────────────────────────────── │  │
│  │ Function: _execute_inspection_program()│  │
│  │ Primary: Sandbox exec()                │  │
│  │ Fallback: In-process exec()            │  │
│  │ Input:                                 │  │
│  │   - inspect_iteration(chunks)          │  │
│  │   - active_chunks with full text       │  │
│  │ Output:                                │  │
│  │   {                                    │  │
│  │     "selected_chunk_ids": [...],       │  │
│  │     "extracted_data": {...},           │  │
│  │     "confidence": 0.0-1.0,             │  │
│  │     "stop": True/False                 │  │
│  │   }                                    │  │
│  └──────────┬────────────────────────────┘  │
│             ↓                                 │
│  ┌─────────────────────────────────────────┐  │
│  │ STEP 2B: Confidence Boosting (NEW)     │  │
│  │─────────────────────────────────────── │  │
│  │ If selected chunks overlap with        │  │
│  │ boolean_approved_chunk_ids:            │  │
│  │   boost_ratio = approved / selected    │  │
│  │   confidence = max(conf, 0.85 + (      │  │
│  │              boost_ratio * 0.1))       │  │
│  │ Ensures approved chunks pass 0.9       │  │
│  │ confidence threshold                   │  │
│  └──────────┬────────────────────────────┘  │
│             ↓                                 │
│  ┌─────────────────────────────────────────┐  │
│  │ STEP 2C: Check Stopping Condition     │  │
│  │─────────────────────────────────────── │  │
│  │ If:                                    │  │
│  │   stop flag = True                     │  │
│  │   OR confidence > 0.9                  │  │
│  │   OR iteration >= MAX_RLM_ITERATIONS   │  │
│  │   OR no narrowing for 2+ iterations    │  │
│  │  THEN: Break loop (stop iterations)   │  │
│  └──────────┬────────────────────────────┘  │
│             ↓                                 │
│  ┌─────────────────────────────────────────┐  │
│  │ STEP 3: Prepare Next Iteration (or End)│  │
│  │─────────────────────────────────────── │  │
│  │ If continuing:                         │  │
│  │   active_chunks = selected chunks      │  │
│  │   accumulated_data += extracted_data   │  │
│  │   iteration += 1                       │  │
│  │                                        │  │
│  │ Else if stopping:                      │  │
│  │   final_selected_chunk_ids = selected  │  │
│  │   break                                │  │
│  └──────────┬────────────────────────────┘  │
│             ↓                                 │
└──────────────────────────────────────────────┘
               ↓
        Final Selected Chunks
               ↓
┌───────────────────────────────────────────────┐
│ POST-ITERATION (same as per-chunk mode)       │
│ - Apply selection budget                      │
│ - Apply current-role prioritization           │
│ - Deduplicate (disabled)                      │
└───────────────────────────────────────────────┘
               ↓
        Final Relevant Chunks
```

### Detailed Iterative Flow

#### Phase A: Per-Chunk Boolean Evaluation

**New in RLM**: Before iterations begin, all chunks are evaluated with their own `evaluate_chunk_relevance()` function.

**Loop**:

```python
for idx, chunk in enumerate(active_chunks):
    # STEP 1: Generate code for this chunk (per-chunk, Phase 4.1 above)
    generated_code = await _generate_inspection_logic(...)

    # STEP 2: Execute code with sandbox (per-chunk, Phase 4.2 above)
    is_relevant = await _evaluate_chunk_with_code(...)

    # STEP 3: Track approved chunks
    if is_relevant:
        boolean_approved_chunk_ids.add(chunk_id)
```

**Result**: Set of chunk IDs that definitively pass the boolean relevance check

#### Phase B.1: Iterative Program Generation

**Function**: `_generate_recursive_inspection_program()`  
**Iteration**: 0, 1, 2, ... (up to 5)

**LLM Call Details**:

```
System Prompt: "You are a Python expert implementing iterative inspection programs..."
Temperature: 0.3
Max Tokens: 600
Deployment: gpt-4o-mini (code generation)

User Prompt includes:
  - Current iteration number (1/5, 2/5, etc.)
  - User query
  - File name
  - Summary of active chunks (first 10 chunks, 100 chars each)
  - Previous iteration extracted data
```

**Generated Function Signature**:

```python
def inspect_iteration(chunks):
    """
    Evaluate chunks for this iteration and return structured output.

    Args:
        chunks: List of dicts with keys: chunk_id, text

    Returns:
        {
            "selected_chunk_ids": [...],    # IDs to keep
            "extracted_data": {...},        # Any extracted data
            "confidence": 0.0-1.0,          # Confidence score
            "stop": True/False              # Should stop?
        }
    """
```

**Logical Requirements**:

1. Evaluate ALL chunks
2. Select FEWER chunks than input (narrow focus)
3. Return 2+ selected IDs (unless stopping)
4. Confidence must be 0.0-1.0
5. Must not select >90% of chunks
6. Use simple string operations only (no imports)

**Validation** (`_validate_inspection_program()`):

- Checks for correct function definition
- Validates return structure
- Prevents "select all" patterns
- AST-based validation

**Sanity Tests** (`_run_inspection_program_sanity_tests()`):

- Test with limited chunk set
- Verify narrowing occurs
- Check output schema validity
- Ensure no infinite loops

**Result**: Validated Python code string (500-1000 chars)

#### Phase B.2: Iterative Program Execution

**Function**: `_execute_inspection_program()`

**Execution Path**:

```
1. PRIMARY: Subprocess Sandbox
   timeout: 5 seconds (longer than per-chunk)
   memory: 512 MB (more for full set)

2. FALLBACK: In-process exec()
   Same safe_globals as per-chunk
```

**Input Data Structure**:

```python
chunk_list = [
    {
        "chunk_id": "chunk_id_abc",
        "text": "Full chunk text..."
    },
    ...
]
# Note: Full text passed, not summaries
```

**Program Execution**:

```python
# Inside sandbox or in-process:
raw_result = inspect_iteration(chunk_list)

# Result validation:
{
    "selected_chunk_ids": [...],     # List must be subset of input
    "extracted_data": {...},         # Dict of any structured data
    "confidence": 0.75,              # Float 0.0-1.0
    "stop": False                    # Boolean
}
```

#### Phase B.3: Post-Execution Processing

**Steps in order**:

1. **Validate Selection**:

   ```python
   selected_ids = raw_result.get("selected_chunk_ids", [])
   # Filter to valid IDs only
   selected_ids = [cid for cid in selected_ids if cid in valid_chunk_ids]
   ```

2. **Enforce Minimum**:

   ```python
   MIN_KEEP = 2
   if not should_stop and len(selected_ids) < MIN_KEEP:
       # Add first M chunks by original order
       selected_ids = all_chunk_ids[:MIN_KEEP]
   ```

3. **Broad Selection Guard**:

   ```python
   selection_ratio = len(selected_ids) / max(1, len(chunk_list))
   if selection_ratio > 0.9 and len(chunk_list) > 3:
       # Too many! Use deterministic fallback
       selected_ids = _select_top_k_chunks(...)
       confidence = 0.2
       should_stop = True
   ```

4. **Confidence Boosting** (NEW):

   ```python
   boolean_approved_in_selection = [
       cid for cid in selected_ids
       if cid in boolean_approved_chunk_ids
   ]

   if boolean_approved_in_selection:
       boost_ratio = len(boolean_approved_in_selection) / len(selected_ids)
       boosted_confidence = max(confidence, 0.85 + (boost_ratio * 0.1))
       # Ensures approved chunks pass 0.9 threshold
   ```

5. **Extract Limit**:
   ```python
   if len(extracted_data) > 50_000 bytes:
       # Truncate large extracted data
       truncated = {}
       for k, v in extracted_data.items():
           if isinstance(v, str):
               truncated[k] = v[:500]  # Cap at 500 chars
       extracted_data = truncated
   ```

#### Phase B.4: Stopping Condition

```python
if should_stop or confidence > 0.9:
    logger.info("🛑 Stopping: high confidence reached")
    final_selected_chunk_ids = selected_ids
    break  # Exit iteration loop
else:
    # Continue to next iteration
    active_chunks = [
        chunk for chunk in active_chunks
        if chunk.get("chunk_id") in selected_ids
    ]
    # Narrowing validation...
    iteration += 1
```

#### Phase B.5: Narrowing Validation

**Enforces that each iteration meaningfully narrows**:

```python
prev_active_ids = set(...)
active_chunks = [c for c in active_chunks if c['chunk_id'] in selected_ids]
new_active_ids = set(...)

if new_active_ids == prev_active_ids:
    narrowing_streak += 1
    if narrowing_streak >= 2:
        logger.warning("No narrowing for 2 iterations, stopping")
        break

shrink_ratio = len(new_active_ids) / max(1, len(prev_active_ids))
if shrink_ratio > 0.9 and not should_stop:
    narrowing_streak += 0.5  # Partial strike
```

---

## Phase 5: Summarization

**After chunk selection completes** (either mode), proceed to summarization:

**Function**: `_summarize_chunks()`

**Input**:

- `relevant_chunks`: List of selected chunk texts
- `query`: Original user query
- `file_name`: File name for context

**Process**:

```
1. Concatenate chunk texts
2. Call LLM summarization:
   "Summarize these chunks relevant to: {query}"
3. Return summary text
```

**Result**: Single summary string per file

---

## Complete Data Flow Example: "Skills for Alexis"

### Per-Chunk Mode (Hypothetical Trace)

```
Query: "tell me the list of skill you can find for Alexis"
File: "alexis_torres_resume.pdf"
Total Chunks: 36

CHUNK 0 (Work History):
  Code Gen: "if 'work' in text or 'company' in text: return True"
  Execute: ✓ True → Added to relevant_chunks

CHUNK 5 (Skills):
  Code Gen: "if 'skills' in text or 'list' in text: return True"
  Execute: ✓ True → Added to relevant_chunks

CHUNK 10 (Education):
  Code Gen: "if 'education' in text or 'degree' in text: return True"
  Execute: ✗ False → Not added

... (30 more chunks evaluated)

Final: 16 relevant chunks selected
Selection ratio: 16/36 = 44% ✓ Below 90% threshold
Result: Use all 16 chunks for summarization
```

### Iterative Mode (Hypothetical Trace)

```
Query: "tell me the list of skill you can find for Alexis"
File: "alexis_torres_resume.pdf"
Prefilter: 36 chunks → 36 chunks (under 60 limit)

PHASE A: BOOLEAN EVALUATION
  Chunk 0: ✓ Approved (has job details)
  Chunk 5: ✓ Approved (has "Skills" keyword)
  Chunk 10: ✓ Approved (has degree info)
  ... (evaluate all 36 chunks)
  Result: 12 chunks approved

ITERATION 1:
  Program Gen: "For skills query, select chunks with skill keywords"
  Execute: Returns {
    "selected_chunk_ids": [0, 5, 10, 12, 15, 18, 22],  // 7 chunks
    "confidence": 0.72,
    "stop": false
  }
  Boosting: Chunks 0, 5, 10 are approved → boost_ratio = 3/7 = 0.43
    Boosted confidence: max(0.72, 0.85 + 0.043) = 0.893
  Continue? confidence 0.893 < 0.9 → continue

ITERATION 2:
  Active: [0, 5, 10, 12, 15, 18, 22]
  Program Gen: "Given previous selections, narrow to most relevant"
  Execute: Returns {
    "selected_chunk_ids": [5, 10, 15],  // 3 chunks
    "confidence": 0.91,
    "stop": false
  }
  Boosting: Chunks 5, 10 approved → boost_ratio = 2/3 = 0.67
    Boosted confidence: max(0.91, 0.85 + 0.067) = 0.917
  Continue? confidence 0.917 > 0.9 → STOP

Final: Chunks [5, 10, 15] selected
  Chunk 5 = Skills ✓
  Chunk 10 = Education ✓
  Chunk 15 = Certifications ✓
Result: Summarize with actual skill data included
```

---

## Key Code Locations

| Component           | Function                                      | File                    | Lines     |
| ------------------- | --------------------------------------------- | ----------------------- | --------- |
| Entry point         | `recursively_retrieve_and_summarize()`        | recursive_summarizer.py | 1270-1700 |
| Per-chunk code gen  | `_generate_inspection_logic()`                | recursive_summarizer.py | 1685-1850 |
| Per-chunk execution | `_evaluate_chunk_with_code()`                 | recursive_summarizer.py | 2140-2180 |
| Iterative setup     | `_process_file_with_rlm_recursion()`          | recursive_summarizer.py | 2360-2550 |
| Boolean eval loop   | (inline in \_process_file_with_rlm_recursion) | recursive_summarizer.py | 2385-2425 |
| Program generation  | `_generate_recursive_inspection_program()`    | recursive_summarizer.py | 1960-2090 |
| Program execution   | `_execute_inspection_program()`               | recursive_summarizer.py | 2210-2340 |
| Confidence boosting | (inline in \_execute_inspection_program)      | recursive_summarizer.py | 2295-2312 |
| Summarization       | `_summarize_chunks()`                         | recursive_summarizer.py | 1630-1660 |

---

## Execution Environment

### Sandbox Execution (`sandbox_exec()`)

- **Location**: `sandbox_runner.py`
- **Security**: Subprocess isolation
- **Restrictions**:
  - No imports allowed
  - No file system access
  - No network access
  - Wall-clock timeout (2-5 seconds)
  - Memory limit (256 MB per-chunk, 512 MB per iteration)
  - Restricted builtins (no `exec`, `eval`, `__import__`, etc.)

### In-Process Fallback (`exec()`)

- **Trigger**: When sandbox fails (timeout, crash, permission denied)
- **Safety**: `safe_globals` environment only
- **Restrictions**: Same as sandbox but enforced by Python scope

---

## LLM Models Used

| Stage             | Model                 | Config   | Purpose                             |
| ----------------- | --------------------- | -------- | ----------------------------------- |
| Code Generation   | gpt-4o-mini (forced)  | temp=0.3 | Reliable code output                |
| Per-Chunk Eval    | gpt-4o-mini           | temp=0.3 | Generate evaluate_chunk_relevance() |
| Iteration Program | gpt-4o-mini           | temp=0.3 | Generate inspect_iteration()        |
| Summarization     | o3-mini (or override) | temp=0.2 | Final text summarization            |

---

## Feature Flags

```python
USE_MIT_RLM_RECURSION = os.getenv("USE_MIT_RLM_RECURSION", "false").lower() == "true"
MAX_RLM_ITERATIONS = int(os.getenv("MAX_RLM_ITERATIONS", "5"))
MAX_PREFILTER_CHUNKS = int(os.getenv("MAX_PREFILTER_CHUNKS", "60"))
MAX_SELECTED_CHUNKS_PER_FILE = int(os.getenv("MAX_SELECTED_CHUNKS_PER_FILE", "8"))
MAX_TOTAL_CHARS_FOR_SUMMARY = int(os.getenv("MAX_TOTAL_CHARS_FOR_SUMMARY", "12000"))
```

---

## Error Handling

### Per-Chunk Mode

- **LLM Call Fails**: Log warning, skip chunk
- **Code Generation Invalid**: Log warning, skip chunk
- **Code Execution Fails**: Default to False (not relevant)
- **Multiple Generators Exceed Limit**: Reject entire file

### Iterative Mode

- **LLM Program Gen Fails**: Use fallback program (query-term based)
- **Program Execution Fails**: Use fallback result
- **Program Returns Invalid Data**: Repair and continue
- **All Iterations Fail**: Use first 3 chunks fallback

---

## Performance Characteristics

| Metric                 | Per-Chunk | Iterative       |
| ---------------------- | --------- | --------------- |
| LLM Calls (36 Chunks)  | 36        | 1-5             |
| Code Executions        | 36        | 36 + iterations |
| Typical Duration       | 2-5 min   | 30-60 sec       |
| Recommended Max Chunks | <50       | 100+            |

---

## Confidence Boosting (Latest Addition)

**Problem**: Chunks passing boolean evaluation were filtered by 0.9 confidence threshold

**Solution**: Track boolean-approved chunks, boost confidence when they're selected

**Formula**:

```python
boosted_confidence = max(current_confidence, 0.85 + (approved_ratio * 0.1))
```

**Effect**:

- 100% approved: 0.95 (passes 0.9 threshold)
- 50% approved: 0.90 (passes 0.9 threshold)
- 0% approved: No boost (maintain current confidence)

---

## References

- **File**: [recursive_summarizer.py](src/news_reporter/retrieval/recursive_summarizer.py)
- **Sandbox Runner**: `src/news_reporter/retrieval/sandbox_runner.py`
- **Type Definitions**: `src/news_reporter/retrieval/retrieval.py` (FileSummary dataclass)
- **Feature Documentation**: [CONFIDENCE_BOOST_FIX.md](CONFIDENCE_BOOST_FIX.md)
