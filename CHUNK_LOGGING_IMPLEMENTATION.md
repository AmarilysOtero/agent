# Chunk Logging Addition to Phase 3

## Overview

Added functionality to automatically log all chunks retrieved/expanded during Phase 3 operations to reusable markdown files for analysis and debugging.

## Files Created

### 1. `src/news_reporter/retrieval/chunk_logger.py`

Utility module for chunk logging with the following features:

- **Two reusable markdown files**:
  - `chunks_rlm_disabled.md`: Chunks retrieved with RLM mode disabled (standard retrieval)
  - `chunks_rlm_enabled.md`: Chunks retrieved and expanded with RLM mode enabled (Phase 3)
- **Key Functions**:
  - `log_chunks_to_markdown()`: Log chunk information to markdown with metadata
  - `log_phase3_expansion()`: Specialized logging for Phase 3 file expansion results
  - `get_chunk_log_file_path()`: Get path to chunk log files
  - `clear_chunk_logs()`: Clear logs for debugging

- **Output Format**: Markdown tables with:
  - Execution timestamp
  - Original query
  - Metadata (entry chunks, expanded chunks, source files, expansion ratio)
  - Chunk table with ID, file, text preview, and size

### 2. Modified Files

#### `src/news_reporter/retrieval/file_expansion.py`

- Added import: `from .chunk_logger import log_chunks_to_markdown, log_phase3_expansion`
- Added new function: `log_expanded_chunks()` wrapper for logging Phase 3 expansion

#### `src/news_reporter/workflows/workflow_factory.py`

- Added imports: `from ..retrieval.chunk_logger import log_chunks_to_markdown`
- **Non-RLM logging**: After standard search, logs chunks if RLM is disabled
- **RLM logging**: After Phase 3 expansion, logs all expanded chunks with metadata

## Logging Points

### 1. Standard Retrieval (RLM Disabled)

```python
# After search_agent.run() in workflow_factory.py
if not high_recall_mode and raw_results:
    await log_chunks_to_markdown(
        chunks=raw_results,
        rlm_enabled=False,
        query=goal
    )
```

### 2. Phase 3 Expansion (RLM Enabled)

```python
# After expand_to_full_files() in workflow_factory.py
await log_expanded_chunks(
    entry_chunks=raw_results or [],
    expanded_files=expanded_files,
    query=goal
)
```

## Output Location

```
/app/logs/chunk_analysis/
├── chunks_rlm_disabled.md   (appended on each non-RLM query)
└── chunks_rlm_enabled.md    (appended on each RLM query)
```

## Usage Examples

### Viewing Logged Chunks

```bash
# Check non-RLM chunks
docker exec rag-agent cat /app/logs/chunk_analysis/chunks_rlm_disabled.md

# Check RLM expanded chunks
docker exec rag-agent cat /app/logs/chunk_analysis/chunks_rlm_enabled.md
```

### Clear Logs

```python
from src.news_reporter.retrieval.chunk_logger import clear_chunk_logs

# Clear both files
clear_chunk_logs()

# Clear only RLM disabled
clear_chunk_logs(rlm_enabled=False)

# Clear only RLM enabled
clear_chunk_logs(rlm_enabled=True)
```

## Log File Format

### Header (first write only)

```markdown
# Chunk Analysis - RLM ENABLED/DISABLED

This file contains chunk information logged from Phase 3 retrieval operations.
File is reused and updated with each query execution.

---
```

### Per-Query Section

```markdown
## Execution: 2026-02-04T20:03:08.205123

**Query**: What technical skills does Kelvin have?

**Metadata**:

- Entry Chunks: 6
- Expanded Chunks: 36
- Source Files: 2
- Expansion Ratio: 36 / 6
- Files Expanded: Alexis Torres - DXC Resume.pdf (2/18); 20250912 Kevin Ramirez DXC Resume.pdf (4/18)

**Total Chunks**: 36

### Chunks

| #   | Chunk ID    | File                           | Text Preview                         | Size      |
| --- | ----------- | ------------------------------ | ------------------------------------ | --------- |
| 1   | `chunk-001` | Alexis Torres - DXC Resume.pdf | Skills: Java, Python, AWS, Docker... | 234 bytes |
| 2   | `chunk-002` | Alexis Torres - DXC Resume.pdf | Experience: Senior Engineer, DXC...  | 456 bytes |

...
```

## Benefits

- 📊 **Analysis**: Compare chunks retrieved with/without RLM
- 🔍 **Debugging**: Understand chunk expansion ratios and file coverage
- 📈 **Optimization**: Identify which files are being expanded and how
- 🔄 **Reusable**: Single file per mode accumulates data across multiple queries
- 📝 **Automatic**: No manual intervention needed

## Testing Status

✅ Container built successfully  
✅ Application started successfully  
✅ Chunk logging directory created: `/app/logs/chunk_analysis`  
✅ Ready for query execution to generate logs
