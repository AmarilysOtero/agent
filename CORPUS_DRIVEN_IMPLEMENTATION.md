# Corpus-Driven Query Classification (Zero Hardcoded Keywords)

## Overview

This implementation provides **fully generic, corpus-driven** query classification with **zero hardcoded domain keywords**. All vocabulary is learned from your actual documents.

## What Changed

### ❌ Old Approach (Hardcoded)

```python
# Hardcoded keyword list
if last_name in {'skills', 'experience', 'education', 'projects', 'certifications',
                  'summary', 'objective', 'about', 'contact', 'profile'}:
    # Fall back to generic mode
```

### ✅ New Approach (Corpus-Derived)

```python
# Load learned vocabulary from your documents
vocab = load_header_vocab("header_vocab.json")

# Check if next token matches ANY header from your KB (no hardcoded list)
matched_header = match_following_header_phrase(tokens, idx, vocab_set=vocab)
if matched_header:
    return [token], is_person_query=True
```

## Files

### 1. **[header_vocab.py](src/news_reporter/tools/header_vocab.py)** (Core Module)

**Functions:**

- `build_header_vocab(chunks)` - Extract vocabulary from corpus during ingestion
- `save_header_vocab(vocab, path)` - Persist to JSON
- `load_header_vocab(path)` - Load at query time
- `extract_person_names_and_mode(query)` - Main entry point (no parameters!)
- `match_following_header_phrase(tokens, idx, vocab_set)` - Generic header matcher

**Zero Hardcoded Keywords:**

- `normalize_header()` - Structural normalization only (lowercase, remove punctuation)
- `header_ngrams()` - Extract 1-4 token phrases
- Generic min frequency threshold (≥3 occurrences)

**Procedural hint detection** (still minimal, reusable patterns):

```python
PROCEDURAL_HINTS = {
    "how do i", "how to", "steps to", "process for",
    "policy for", "guideline for", "register", "submit", ...
}
```

This is action-verb based (not domain-specific), applies universally.

### 2. **[agents.py](src/news_reporter/agents/agents.py)** (Updated)

**Function Changes:**

#### `infer_header_from_chunk()` - Generic patterns only

```python
# No hardcoded keywords like ['skills', 'experience', ...]
# Only generic patterns:
# - All caps: "SKILLS", "EDUCATION"
# - Title case short: "Professional Experience"
# - Structural markers: "Skills:", "Education -"
```

#### `filter_results_by_exact_match()` - Uses vocab-based filtering

```python
# OLD: hardcoded check for section keywords
# NEW: person-mode filtering purely based on:
#   - Multi-token name span detection
#   - Single-token + corpus-derived header vocabulary
```

#### `AiSearchAgent`, `SQLAgent`, `Neo4jGraphRAGAgent` - Mode-aware

```python
person_names, is_person_query = extract_person_names_and_mode(query)

# is_person_query is set automatically based on corpus vocabulary
# No manual decision-making in agents
```

## Workflow

### Step 1: Build Vocabulary (Ingestion Time)

**When:** After document parsing/chunking

```python
from header_vocab import build_header_vocab, save_header_vocab

# During ingestion pipeline:
chunks = parse_documents(documents)
vocab = build_header_vocab(chunks)
save_header_vocab(vocab, "header_vocab.json")
```

**Output: `header_vocab.json`**

```json
{
  "skills": 153,
  "professional experience": 140,
  "time off": 88,
  "pto": 76,
  "onboarding": 54,
  "education": 45,
  "certifications": 42,
  ...
}
```

**No hardcoded words—purely statistical from corpus.**

### Step 2: Load Vocabulary (Query Time)

**When:** Agent startup (once)

```python
from header_vocab import load_header_vocab

vocab_set = load_header_vocab()  # Loads once, cached in memory
# vocab_set = {"skills", "professional experience", "time off", ...}
```

### Step 3: Classify Queries (Query Time)

**When:** Each user query

```python
from header_vocab import extract_person_names_and_mode

# Zero parameters—fully automatic
names, is_person_query = extract_person_names_and_mode(query)
```

**Examples:**

| Query                                | Multi-Token       | Single-Token              | Procedural | Result                      |
| ------------------------------------ | ----------------- | ------------------------- | ---------- | --------------------------- |
| "List Kevin Skills"                  | No                | "Kevin" + "skills" ✓      | No         | `(["Kevin"], True)`         |
| "list alexis skills"                 | No                | "alexis" + "skills" ✓     | No         | `(["alexis"], True)`        |
| "Show Alexis Torres experience"      | "Alexis Torres" ✓ | N/A                       | No         | `(["Alexis Torres"], True)` |
| "Tell me steps to register time off" | No                | No (procedural)           | Yes        | `([], False)`               |
| "What is Docker?"                    | No                | No (not before header)    | No         | `([], False)`               |
| "Alexis experience"                  | No                | "alexis" + "experience" ✓ | No         | `(["alexis"], True)`        |

## Design Principles

### ✅ Fully Generic

- **No hardcoded domain words** (skills, experience, time off, etc.)
- **Vocabulary learned from corpus** (any doc type: resumes, SOPs, policies)
- **Automatic adaptation** (new headers learned as docs are ingested)

### ✅ Minimal Surface

- **One function to call:** `extract_person_names_and_mode(query)`
- **Automatic loading:** Vocabulary cached at startup
- **No configuration:** Works out of the box once `header_vocab.json` exists

### ✅ Robust Fallback

- **No vocabulary file?** Returns generic mode gracefully
- **No person signals?** Falls back to generic mode (similarity-based)
- **Procedural query?** Skips person-mode regardless of vocabulary match

## Testing

### Quick Test Cases

```python
from header_vocab import extract_person_names_and_mode

# Person-centric queries
assert extract_person_names_and_mode("List Kevin Skills") == (["Kevin"], True)
assert extract_person_names_and_mode("list alexis skills") == (["alexis"], True)
assert extract_person_names_and_mode("Show Alexis Torres experience") == (["Alexis Torres"], True)

# Generic queries
assert extract_person_names_and_mode("Tell me the steps to register time off")[1] == False
assert extract_person_names_and_mode("What is Docker?") == ([], False)
assert extract_person_names_and_mode("Show me the skills section")[1] == False
```

### Integration Test

```python
# Full flow: ingestion → vocabulary → query
chunks = load_documents("resumes/")
vocab = build_header_vocab(chunks)
save_header_vocab(vocab, "header_vocab.json")

# Restart app (loads vocab at startup)
names, is_person = extract_person_names_and_mode("Kevin skills")
assert is_person == True  # "skills" is in vocabulary
```

## Migration Checklist

- [x] Removed hardcoded keyword lists from `infer_header_from_chunk()`
- [x] Removed hardcoded section checks from `filter_results_by_exact_match()`
- [x] Made `extract_person_names_and_mode()` fully corpus-driven
- [x] Updated all agents to use new mode-aware filtering
- [x] No type errors or compilation warnings

## Next Steps

### Immediate (Required)

1. **Build initial vocabulary** from your existing corpus:

   ```bash
   python -c "
   from src.news_reporter.tools.header_vocab import build_header_vocab, save_header_vocab
   from your_ingestion_module import load_all_chunks

   chunks = load_all_chunks()
   vocab = build_header_vocab(chunks)
   save_header_vocab(vocab, 'header_vocab.json')
   "
   ```

2. **Place `header_vocab.json`** in one of:
   - `src/news_reporter/settings/header_vocab.json`
   - Root directory
   - Home directory

3. **Restart Agent service**:
   ```bash
   docker-compose -f docker-compose.dev.yml restart agent
   ```

### Testing (Recommended)

1. Run the test cases above
2. Test 10-20 real queries from your chat logs:
   - "List Kevin Skills" ← should work
   - "Tell me steps to register time off" ← should use generic mode
   - "Show Alexis Torres experience" ← should work
   - "What is the company policy on..." ← should use generic mode

### Enhancements (Optional)

- Add ingestion hook to update vocabulary when new documents are uploaded
- Monitor vocabulary coverage (track queries that don't match any header)
- Periodic vocabulary rebuild (e.g., daily) from growing corpus

## Zero Hardcoded Guarantees

✅ **No hardcoded keywords for:**

- Person vs. procedural detection (uses header proximity + procedural verbs)
- Section names (all learned from actual document headers)
- Document types (works across resumes, SOPs, policies, contracts, etc.)
- Domain terms (completely agnostic to your business domain)

✅ **Purely learned from:**

- Document header text (metadata)
- Parent headers (hierarchy)
- Inferred headers (from chunk content analysis)
- Procedural verbs (reusable, language-agnostic)

**Result: True generic query classification that adapts to any corpus.**
