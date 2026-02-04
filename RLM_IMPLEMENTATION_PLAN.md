# RLM Optional Answering Flow — Phased Implementation Plan

## Goal

Add an optional RLM-based answering flow that uses existing Stage 1 retrieval as entry points, expands to full files, performs recursive summarization, and returns answers with chunk-level citations—while leaving the current flow unchanged when disabled.

When enabled, the RLM flow runs after Search Agent execution and before Assistant execution. It post-processes retrieved chunks into file-level summaries before final answering.

---

## Phased Plan (Each Phase Has a Manual Test)

### Enablement Controls (Choose One or Combine)

RLM execution should be controlled by one of the following (implementation chooses one):

- **Settings-based** (recommended): add `rlm_enabled: bool` to Settings
- **Per-request option**: add a request flag (e.g., `use_rlm=true`)
- **Triage-based routing**: triage emits `use_rlm: bool` to route internally
- **Combination**: default setting with per-request override

### UI Enablement (Orchestration Tab)

The `RLM_ENABLED` flag is exposed as a toggle in the **Orchestration side panel**.

- Location: Orchestration tab → Advanced / Experimental options
- Control type: On/Off toggle
- Default state: Off

When the toggle is enabled:

- `RLM_ENABLED` is set to true for the current orchestration run
- The sequential workflow routes through the RLM branch

When disabled:

- The workflow follows the existing default path with no behavior changes

This toggle applies per orchestration execution and does not modify global defaults unless explicitly saved.

**Execution precedence note:** Graph workflows run when an active workflow is present. The sequential path (and thus the RLM branch) is the fallback/default when graph execution is not selected.

### Phase 0 — Decision Gate (Must Pass First)

Goal: Confirm the data model supports file expansion.

Verify in Neo4j:

- Every chunk links to a stable file identifier (UUID or equivalent).
- One file → many chunks is consistent and ordered.

Manual test result:

- Pick a chunk_id → resolve its file_id.
- Pick a file_id → list all chunks in correct order.

If it fails: implement file UUID linking first, then proceed.

---

### Phase 0.5 — File UUID Linking (If Needed)

**Status: ✅ COMPLETE**

Goal: Ensure every chunk has a stable file identifier.

Actions:

- If `file_id` is missing or inconsistent: add UUID to File nodes.
- Update all chunks to reference their parent file UUID.
- Verify consistency: every chunk → one file, every file → many chunks.

Manual test result:

- Sample chunk → file UUID resolves correctly.
- Sample file UUID → all chunks listed in order.

✅ **Validation PASSED** (all 4 Neo4j queries returned expected results):
- Query 1: 0 chunks missing file_id ✅
- Query 2: 0 orphan chunks ✅
- Query 3: File→chunks relationships confirmed ✅
- Query 4: Chunk ordering verified ✅

Backfill script: `neo4j_backend/backfill_file_uuid_links.py`

---

### Phase 1 — Feature Flag + Routing (No New Behavior Yet)

**Status: ✅ IMPLEMENTED**

Goal: Add `RLM_ENABLED` routing without changing output.

Actions:

- Add `RLM_ENABLED` to config, default false.
- Add branching in [src/news_reporter/workflows/workflow_factory.py](src/news_reporter/workflows/workflow_factory.py) where both paths still call the existing flow.
- Add logging: “RLM branch selected”.
- Ensure config requirements are explicit:
  - When `RLM_ENABLED=true`, Neo4j configuration must be present (e.g., `use_neo4j_search` and API URL).
  - When `RLM_ENABLED=false`, existing defaults apply.
- Document triage routing nuance: `preferred_agent=csv` currently falls back to default search/Neo4j because selection only branches on SQL or Neo4j.

### Implementation Complete ✅

**Config Changes** (`src/news_reporter/config.py`):
- Added `rlm_enabled: bool = False` to Settings dataclass
- Added environment variable parsing: `RLM_ENABLED` reads from .env

**Workflow Changes** (`src/news_reporter/workflows/workflow_factory.py`):
- Added RLM routing branch after Triage step
- When enabled: logs "RLM branch selected" (INFO level)
- When disabled: logs "Default sequential branch selected (RLM not enabled)" (DEBUG level)
- Currently both paths execute identical downstream logic (Phase 1 = no behavior change)

Manual test result:

- ✅ `RLM_ENABLED=false`: behavior identical to current (check logs: should show "Default sequential branch")
- ✅ `RLM_ENABLED=true`: behavior still identical, but logs confirm RLM branch (check logs: should show "RLM branch selected")

---

### Phase 2 — High-Recall Stage 1 Retrieval Toggle (Still No Stage 2)

Goal: When RLM is enabled, Stage 1 returns more entry chunks.

Actions:

- Add `RLM_LOW_RECALL_MODE`.
- Only adjust retrieval parameters when `RLM_ENABLED=true`.

Manual test result:

- Same query:
  - Default flow returns N chunks.
  - RLM-enabled flow returns more chunks or lower-score chunks.
- No other differences.

---

### Phase 3 — Full File Expansion API (No Recursion Yet)

Goal: Expand entry chunks → all chunks per file, return expanded sets.

Actions:

- Backend API option: `expand_files=true`.
- Implement: entry chunks → unique file_ids → fetch all chunks per file.
- Return output shape:
  - files: [{file_id, chunks:[...]}]
  - chunk metadata for citations.

Manual test result:

- In RLM mode, response includes file-grouped full chunk sets.
- Confirm all chunks per file_id are retrieved in order.
- Answering behavior still unchanged (or return a placeholder message).

---

### Phase 4 — Stage 2 RLM Recursive Inspection (MIT RLM Behavior)

Goal: Implement recursive summarization per file.

Actions:

- Apply LLM-generated inspection logic (e.g., regex or rules) to chunks → selectively summarize matched content → file summary.
- Return `file_summaries[]` (plus a temporary response if needed).

### MIT RLM Recursive Inspection Model

In this phase, recursion is not limited to static summarization.

Instead, the LLM is allowed to generate a small executable inspection program
(e.g., Python logic with regex or filtering rules) based on the user query.

This generated program is executed over the chunk environment to:

- evaluate which chunks contain relevant information,
- extract specific signals or matches,
- decide which subsets require deeper inspection.

The results of program execution are fed back to the LLM, which may:

- refine the program,
- recurse on a smaller chunk subset,
- summarize selected content,
- or terminate when sufficient information is gathered.

This loop may repeat multiple times and represents the core Recursive Language Model behavior described in the MIT paper.

Manual test result:

- Each file has a relevant summary.
- No cross-file merge yet.

---

### Phase 5 — Cross-File Merge + Final Answer + Citations

Goal: Complete the RLM flow.

Actions:

- Merge file summaries into a global understanding.
- Generate final answer.
- Enforce citation policy (`strict` | `best_effort`).
- Respect safety caps (`RLM_MAX_FILES`, `RLM_MAX_CHUNKS`).

Manual test result:

- Final answer produced in RLM mode.
- Citations reference real chunk ids and files.
- Caps are respected.

---

### Phase 6 — Agent-Side Integration + Docs

Goal: Pass flags and render RLM outputs correctly end-to-end.

Actions:

- Update agent-side retrieval calls to pass new options.
- Handle new output shapes in the agent.
- Update documentation.

Manual test result:

- UI end-to-end: enabling RLM changes behavior; disabling restores default.
- Docs reflect the final design.

---

## Configuration Flags

- `RLM_ENABLED` (default: false)
- Optional enablement fields (choose one):
  - Settings: `rlm_enabled: bool`
  - Request flag: `use_rlm=true`
  - Triage output: `use_rlm: bool`
- `RLM_LOW_RECALL_MODE` (default: false)
- `RLM_MAX_FILES`
- `RLM_MAX_CHUNKS` (optional)
- `RLM_CITATION_POLICY` (`strict` | `best_effort`)
- API toggles split cleanly:
  - `high_recall=true`
  - `expand_files=true`
  - `rlm_summarize=true`

---

## Logging and Traceability (Add Early)

- “RLM branch selected”
- “High recall mode on”
- “Files expanded: X, total chunks: Y”

---

## Acceptance Criteria

### When `RLM_ENABLED=false`

- System behaves exactly as it does today.

### When `RLM_ENABLED=true`

- Stage 1 runs Semantic + GraphRAG retrieval.
- High-recall mode applies when configured.
- Entry chunks identify relevant files.
- Full chunks per file are retrieved.
- Recursive summarization runs (file-level then cross-file).
- Final answer includes chunk-level citations.
- Safety limits are respected.
