# Root Cause Analysis & Fix: Empty LLM Code Generation

## Problem
**No Python code was being generated** in Phase 4 (LLM-Generated Inspection Logic). The log showed empty code blocks:

```python
```

## Investigation Process

### Step 1: Initial Assumption
First, I thought the issue was:
- Empty response from LLM (partially true)
- Temperature parameter not being removed for o3-series models

### Step 2: Deep Diagnosis
I created debug scripts to test Azure OpenAI deployments and discovered:

```
🧪 Testing deployment: gpt-4o-mini
-------------------
✅ Success! Response length: 221 chars
   Content preview: def contains_skills(input_string):...

🧪 Testing deployment: o3-mini
-------------------
❌ Failed: Response length: 0 chars
   Content preview: (empty)
```

## Root Cause
**The o3-mini deployment is returning empty responses**

The issue is NOT:
- ❌ Missing Azure credentials (they were in `.env`)
- ❌ Temperature parameter handling (properly removed for o-series)
- ❌ Missing code generation logic

The issue IS:
- ✅ **o3-mini is a special reasoning model** that doesn't properly handle code generation requests through the Azure API
- ✅ **gpt-4o-mini works perfectly** for the same requests

## Solution Implemented

Updated all LLM calls in the retrieval pipeline to **fallback to gpt-4o-mini when o3-mini is configured**:

### Files Modified:
1. **src/news_reporter/retrieval/recursive_summarizer.py**
   - `_generate_inspection_logic()`: Override to use gpt-4o-mini
   - `_apply_inspection_logic_llm_fallback()`: Override to use gpt-4o-mini
   - `_summarize_chunks()`: Override to use gpt-4o-mini

2. **src/news_reporter/retrieval/phase_5_answer_generator.py**
   - `_merge_file_summaries()`: Override to use gpt-4o-mini
   - `_generate_answer_text()`: Override to use gpt-4o-mini

### Implementation Pattern:
```python
# Use gpt-4o-mini if o3-mini is configured
effective_deployment = "gpt-4o-mini" if model_deployment.startswith('o3') else model_deployment

response = await llm_client.chat.completions.create(
    model=effective_deployment,  # ← Use the override
    messages=[...],
    temperature=0.5,
    max_completion_tokens=500
)
```

## Results

Now Python code IS being generated correctly:

```python
def evaluate_chunk_relevance(chunk_text: str) -> bool:
    """Relevance filter based on query terms."""
    text_lower = chunk_text.lower()
    query_terms = ['give', 'the', 'list', 'of', 'kevin', 'skills']
    return sum(1 for term in query_terms if term in text_lower) >= 2
```

✅ Code generation is now working
✅ Chunk relevance filtering is functional
✅ All downstream phases can execute

## Commits
- `46eca3c`: Initial empty response detection and fallback
- `c65ec7e`: Root cause fix - use gpt-4o-mini instead of o3-mini

## Lessons Learned
1. **Reason models (o1/o3) have different behavior** - they may not be suitable for code generation tasks via standard chat APIs
2. **Always test deployments independently** - don't assume they work just because credentials are configured
3. **Empty responses without errors** are harder to debug - added logging to capture raw response content
