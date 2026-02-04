# RLM Phases 2-6: Detailed Implementation Spec

## Phase 2 — High-Recall Retrieval Mode

**Status:** Ready for implementation

**Goal:** When RLM is enabled, Stage 1 returns more entry chunks (lower score threshold).

**Backend Changes Required:**

**File:** `src/news_reporter/retrieval/semantic_search.py`

- Add parameter: `high_recall_mode: bool = False`
- When enabled:
  - Reduce similarity threshold from (e.g.) 0.7 → 0.5
  - Increase `top_k` from (e.g.) 5 → 15
  - Log: `logger.info("High recall mode enabled: threshold=0.5, top_k=15")`

**File:** `src/news_reporter/workflows/workflow_factory.py`

- In RLM branch: pass `high_recall_mode=True` to retrieval services
- Log retrieved chunk count: `logger.info(f"Retrieved {len(chunks)} chunks in high-recall mode")`

**Test Checklist:**

- ✅ Query: "What are the main skills?"
- ✅ Default flow: 5 chunks, all high-scoring
- ✅ RLM-enabled: 15 chunks, includes lower-scoring but relevant
- ✅ Logs show mode switch
- ✅ No behavior change in downstream processing

---

## Phase 3 — Full File Expansion API

**Status:** Requires implementation

**Goal:** Expand entry chunks → all chunks per file (Neo4j query).

**New File:** `src/news_reporter/retrieval/file_expansion.py`

```python
from typing import List, Dict
from neo4j import Driver
from src.types.chunk import Chunk

async def expand_to_full_files(
    entry_chunk_ids: List[str],
    neo4j_driver: Driver
) -> Dict[str, List[Chunk]]:
    """
    Given entry chunks, fetch ALL chunks per file.

    Args:
        entry_chunk_ids: List of chunk UUIDs from retrieval
        neo4j_driver: Neo4j connection

    Returns:
        {file_id: [Chunk1, Chunk2, ...]} ordered by chunk index

    Steps:
    1. Query Neo4j: MATCH (c:Chunk)-[:BELONGS_TO]->(f:File)
       WHERE c.chunk_id IN entry_chunk_ids
       RETURN DISTINCT f.file_id

    2. For each file_id, fetch all chunks:
       MATCH (c:Chunk)-[:BELONGS_TO]->(f:File)
       WHERE f.file_id = $file_id
       RETURN c ORDER BY c.chunk_index

    3. Return grouped result
    """
    pass
```

**File:** `src/news_reporter/routers/workflows.py`

- Add to `WorkflowRequest`: `expand_files: Optional[bool] = None`
- In RLM branch, after retrieval:
  - Call `expand_to_full_files(entry_chunk_ids, neo4j_driver)`
  - Log: `logger.info(f"Expanded {len(entry_chunks)} → {len(expanded_chunks)} chunks across {len(files)} files")`

**Test Checklist:**

- ✅ Enable RLM, send query
- ✅ Logs show: "Expanded 15 → 48 chunks across 3 files"
- ✅ Response includes file-grouped chunks
- ✅ Chunks are ordered by index per file

---

## Phase 4 — Recursive Summarization (MIT RLM)

**Status:** Requires implementation (most complex)

**Goal:** Apply LLM-generated inspection programs per file for selective summarization.

**New File:** `src/news_reporter/rlm/inspector.py`

```python
from typing import List, Dict
from src.types.chunk import Chunk
from src.services.llm import LLMClient

class RLMInspector:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def generate_inspection_program(
        self,
        user_query: str,
        file_chunks: List[Chunk]
    ) -> str:
        """
        LLM generates small Python/regex program to identify
        relevant chunks based on user query.

        Returns: Python code as string (executed with ast.literal_eval safety)
        """
        prompt = f"""Given this query: {user_query}

        And these chunks from a file, generate a Python filter function
        that returns True for chunks containing relevant information.

        Chunks: {[c.text[:100] for c in file_chunks]}

        Return ONLY valid Python function code."""
        pass

    async def execute_inspection(
        self,
        program: str,
        chunks: List[Chunk]
    ) -> List[Chunk]:
        """
        Safely execute generated program over chunks.
        Returns filtered/prioritized chunks.
        """
        # Use RestrictedPython or similar for safety
        pass

    async def summarize_file(
        self,
        file_id: str,
        filtered_chunks: List[Chunk],
        user_query: str
    ) -> str:
        """
        Generate file-level summary from filtered chunks.
        """
        pass
```

**File:** `src/news_reporter/workflows/workflow_factory.py`

- When RLM enabled, for each file:
  ```python
  inspector = RLMInspector(llm_client)
  program = await inspector.generate_inspection_program(query, file_chunks)
  filtered = await inspector.execute_inspection(program, file_chunks)
  summary = await inspector.summarize_file(file_id, filtered, query)
  file_summaries.append({
      "file_id": file_id,
      "summary": summary,
      "chunks_used": len(filtered),
      "total_chunks": len(file_chunks)
  })
  ```

**Test Checklist:**

- ✅ Query: "What technical skills does Alexis have?"
- ✅ Logs per-file: "File 1: Generated program, matched 8/20 chunks, summary=..."
- ✅ Response includes `file_summaries` array
- ✅ No errors in program execution

---

## Phase 5 — Cross-File Merge + Citations

**Status:** Requires implementation

**Goal:** Merge summaries into final answer with chunk-level citations.

**New File:** `src/news_reporter/rlm/merge_and_answer.py`

```python
from typing import List, Dict
from src.services.llm import LLMClient

async def merge_and_answer(
    file_summaries: List[Dict],
    user_query: str,
    llm_client: LLMClient,
    citation_policy: str = "strict"  # strict | best_effort
) -> Dict:
    """
    Merge file summaries → global context
    Generate final answer with citations
    """
    # 1. Compile global context from all summaries
    # 2. LLM generates answer citing file summaries
    # 3. Map citations back to original chunks
    # 4. Validate against citation_policy
    pass
```

**Response Format:**

```json
{
	"answer": "Alexis has strong experience in Python, Neo4j, and RAG systems...",
	"citations": [
		{
			"chunk_id": "uuid:path:chunk:5",
			"file_id": "uuid:resume",
			"text": "Python expertise demonstrated across 5+ projects",
			"relevance": 0.95,
			"source_file": "Alexis_Torres_Resume.pdf"
		}
	],
	"metadata": {
		"files_processed": 3,
		"chunks_used": 12,
		"mode": "rlm_recursive",
		"citation_policy": "strict"
	}
}
```

**Test Checklist:**

- ✅ RLM query with multiple files
- ✅ Answer synthesizes across files coherently
- ✅ Each fact has chunk citation
- ✅ Citations reference real chunks
- ✅ Metadata shows RLM mode used

---

## Phase 6 — Agent Integration + End-to-End UI

**Status:** Final integration phase

**Goal:** Connect RLM output to agents and display in chat UI.

**Backend Changes:**

**File:** `src/agents/search_agent.py`

- When RLM enabled:
  - Return both raw chunks AND file_summaries
  - Pass summaries to Assistant agent for final answer generation
  - Log: `logger.info("RLM mode: passing file summaries to Assistant")`

**Frontend Changes:**

**File:** `src/services/chatApi.ts`

- Read `rlmEnabled` from Redux before calling API
- Pass in request: `rlm_enabled: rlmEnabled`
- Handle new response shape with citations

**File:** `src/app/chat/page.tsx`

- Display "🔴 RLM Mode" badge when active
- Render citations as footnotes/expandable links
- Show chunk source on click

**Test Checklist:**

- ✅ Chat: Enable RLM in Settings
- ✅ Chat: Ask question
- ✅ UI shows "RLM Mode" badge
- ✅ Answer includes inline citations
- ✅ Click citation → see source chunk + file
- ✅ Without RLM: no badge, no citations (backward compatible)

---

## Implementation Timeline

| Phase                    | Duration       | Complexity | Dependencies        |
| ------------------------ | -------------- | ---------- | ------------------- |
| 2 - High-recall          | 2-3 days       | Low        | Neo4j tuning        |
| 3 - File expansion       | 1-2 days       | Low        | Phase 2 complete    |
| 4 - Recursive inspection | 3-5 days       | **High**   | LLM safety, Phase 3 |
| 5 - Merge + citations    | 2-3 days       | Medium     | Phase 4 complete    |
| 6 - Agent integration    | 1-2 days       | Medium     | All phases          |
| **Total**                | **~2-3 weeks** | -          | Sequential          |

**Critical Path:** Phase 2 → Phase 3 → Phase 4 (longest) → Phase 5 → Phase 6

---

## Safety & Risk Mitigations

- **Phase 4 Risk:** LLM-generated code execution
  - **Mitigation:** Use RestrictedPython, sandbox execution, max timeout
- **Phase 5 Risk:** Hallucinated citations
  - **Mitigation:** `strict` policy validates citations exist in source chunks
- **Phase 6 Risk:** UI complexity
  - **Mitigation:** Feature flag to disable RLM rendering if issues arise
