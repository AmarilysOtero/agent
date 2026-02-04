# Phase 3 File Expansion Fix Summary

## Issues Found

### 1. Incorrect Neo4j Relationship Type
- **Problem**: Query used non-existent relationship `BELONGS_TO` between Chunk and File
- **Schema Check**: Docker logs showed available relationships: `HAS_PATH`, `CONTAINS`, `HAS_CHUNK`, `CONTAINS_CHUNK`, `MENTIONS`, etc.
- **Correct Relationship**: `File --[HAS_CHUNK]--> Chunk` (File to Chunk direction)

### 2. Incorrect Neo4j Property Names
- **Problem**: Query referenced properties that don't exist
  - Used: `c.chunk_id`, `c.chunk_index`, `f.file_id`, `f.file_name`
  - Actual: `c.id`, `c.index`, `f.id`, `f.name`

## Fixes Applied

### File: `src/news_reporter/retrieval/file_expansion.py`

#### Fix 1: Query to find source files (Phase 3.1)

**Before:**
```cypher
MATCH (c:Chunk)-[:BELONGS_TO]->(f:File)
WHERE c.chunk_id IN $entry_chunk_ids
RETURN DISTINCT f.file_id as file_id, f.file_name as file_name
```

**After:**
```cypher
MATCH (f:File)-[:HAS_CHUNK]->(c:Chunk)
WHERE c.id IN $entry_chunk_ids
RETURN DISTINCT f.id as file_id, f.name as file_name
```

#### Fix 2: Query to fetch chunks per file (Phase 3.2)

**Before:**
```cypher
MATCH (c:Chunk)-[:BELONGS_TO]->(f:File)
WHERE f.file_id = $file_id
RETURN {
    chunk_id: c.chunk_id,
    chunk_index: c.chunk_index,
    text: c.text,
    embedding_id: c.embedding_id,
    chunk_type: c.chunk_type,
    metadata: properties(c)
} as chunk_data
ORDER BY c.chunk_index ASC
```

**After:**
```cypher
MATCH (f:File)-[:HAS_CHUNK]->(c:Chunk)
WHERE f.id = $file_id
RETURN {
    chunk_id: c.id,
    chunk_index: c.index,
    text: c.text,
    embedding_id: c.embedding_id,
    chunk_type: c.chunk_type,
    metadata: properties(c)
} as chunk_data
ORDER BY c.index ASC
```

## Neo4j Schema Reference

### Graph Structure
```
Machine
  └─[:CONTAINS]→ Directory
          ├─[:CONTAINS]→ File
          │       └─[:HAS_CHUNK]→ Chunk (CORRECTED)
          │               ├─[:SEMANTICALLY_SIMILAR]→ Chunk
          │               └─[:SEMANTICALLY_SIMILAR]→ Chunk
          │
          └─[:CONTAINS_CHUNK]→ Chunk
```

### Relationship Types Confirmed
- **HAS_CHUNK**: 36 edges (File → Chunk)
- **CONTAINS_CHUNK**: 36 edges (Directory → Chunk)

### Chunk Properties
- `id`: Unique chunk identifier
- `index`: Chunk sequence order within file
- `text`: Chunk content
- `embedding_id`: Vector embedding reference
- `chunk_type`: Type classification

### File Properties
- `id`: Unique file identifier
- `name`: File name/path

## Testing Status

### Test Execution
- ✅ Docker image rebuilt with fixed code
- ✅ Agent container restarted successfully
- ✅ No syntax errors in updated code
- ⏳ Full E2E test pending (requires API authentication)

### Expected Behavior When RLM is Enabled
1. Phase 2 retrieves entry chunks with `high_recall_mode`
2. Phase 3 now correctly:
   - Uses `File -[HAS_CHUNK]-> Chunk` relationship
   - Identifies source files from entry chunks using corrected properties
   - Fetches all chunks per file using corrected properties
   - Orders chunks by actual index property
3. Returns expanded context for LLM processing

### Docker Logs Verification
Latest startup: `2026-02-04 19:52:51`
- ✅ Application startup complete
- ✅ All routers mounted successfully
- ✅ Chat sessions router initialized
- ✅ Ready for RLM queries

## Impact
Phase 3 file expansion will now correctly:
- Query the Neo4j graph using actual schema relationships
- Expand entry chunks to all chunks per file
- Provide complete file context for high-recall retrieval mode (RLM)
- Support broader LLM reasoning over complete document context
