# Neo4j GraphRAG Integration Guide

This guide explains how to integrate the Neo4j GraphRAG backend with the main News Reporter application.

## Overview

The integration enables:
1. Entity extraction from document chunks
2. Entity-based graph search and retrieval
3. Typed relationships between entities
4. Enhanced context through graph traversal

## Setup Steps

### 1. Install Neo4j Database

**Option A: Docker (Recommended)**
```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:latest
```

**Option B: Neo4j Desktop**
- Download from https://neo4j.com/download/
- Create a new database
- Note the connection credentials

### 2. Configure Environment

Add to your `.env` file:
```bash
# Neo4j Database Connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Neo4j Backend API (already configured)
NEO4J_API_URL=http://localhost:8000
USE_NEO4J_SEARCH=false  # Set to true to use Neo4j search
```

### 3. Install Dependencies

```bash
# Install Neo4j driver and backend dependencies
pip install -r requirements.txt
pip install -r neo4j_backend/requirements.txt
```

### 4. Start the Neo4j Backend

```bash
# Start the backend API server
python -m neo4j_backend.main
```

The backend will start on `http://localhost:8000` by default.

Verify it's running:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "neo4j": "connected",
  "openai": "configured"
}
```

### 5. Set Up Database Constraints

One-time setup to create Neo4j constraints:
```bash
curl -X POST http://localhost:8000/api/graph/setup-constraints
```

### 6. Process Documents and Extract Entities

After ingesting documents (chunks exist in Neo4j), extract entities:

**Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/graph/extract-entities",
    json={
        "file_path": "/path/to/your/document.pdf",
        "extract_relationships": True
    }
)

result = response.json()
print(f"Processed {result['chunks_processed']} chunks")
print(f"Created {result['entities_created']} entities")
print(f"Created {result['relationships_created']} relationships")
```

**cURL:**
```bash
curl -X POST http://localhost:8000/api/graph/extract-entities \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/your/document.pdf",
    "extract_relationships": true
  }'
```

### 7. View Entity Statistics

```bash
curl http://localhost:8000/api/graph/entity-stats
```

## Integration Points

### 1. Document Ingestion Flow

When documents are ingested, chunks are created in Neo4j. After ingestion:
```python
# Extract entities from the newly ingested document
import requests

requests.post(
    "http://localhost:8000/api/graph/extract-entities",
    json={"file_path": file_path, "extract_relationships": True}
)
```

### 2. Enhanced Search

The existing `neo4j_graphrag.py` tool uses the backend API for search:
```python
from src.news_reporter.tools.neo4j_graphrag import graphrag_search

results = graphrag_search(
    query="Find information about AI companies",
    top_k=10,
    similarity_threshold=0.7
)
```

With entities extracted, the search can:
- Find entities mentioned in chunks
- Traverse relationships between entities
- Expand context through multi-hop graph queries

### 3. Agent Integration

The News Reporter agents can use entity-aware search:
```python
# In agent code
if settings.use_neo4j_search:
    from ..tools.neo4j_graphrag import graphrag_search
    
    results = graphrag_search(
        query=user_query,
        top_k=12,
        similarity_threshold=0.75
    )
```

## Workflow

```
1. Document Upload
   ↓
2. Chunk Creation (existing)
   ↓
3. Entity Extraction (NEW)
   - Extract entities from each chunk
   - Create Entity nodes
   - Create MENTIONS edges
   - Extract typed relationships
   ↓
4. Search & Retrieval
   - Vector search finds relevant chunks
   - Graph expansion finds related entities
   - Relationship traversal adds context
   ↓
5. Response Generation
   - LLM generates response from enhanced context
```

## Testing

Run the test suite:
```bash
python neo4j_backend/test_entity_extraction.py
```

Expected output:
```
Testing Neo4j connection...
✓ Neo4j connection successful

Setting up Neo4j constraints...
✓ Constraints set up successfully

Testing entity extraction...
✓ Extracted 5 entities:
  - Dr. Jane Smith (Person) [confidence: 0.95]
  - Microsoft Corporation (Organization) [confidence: 0.92]
  ...

✓ All tests passed!
```

## Monitoring

Monitor entity extraction progress:
```bash
# Watch backend logs
tail -f neo4j_backend.log

# Check entity statistics
curl http://localhost:8000/api/graph/entity-stats
```

## Troubleshooting

### Backend won't start
- Check Neo4j is running: `docker ps` or Neo4j Desktop
- Verify .env has NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
- Check port 8000 is not in use: `lsof -i :8000`

### No entities extracted
- Verify Azure OpenAI credentials in .env
- Check AZURE_OPENAI_CHAT_DEPLOYMENT exists in your Azure resource
- Review backend logs for API errors

### Chunks not found
- Verify documents have been ingested first
- Check file_path matches exactly (case-sensitive)
- Query Neo4j: `MATCH (c:Chunk) RETURN c.file_path LIMIT 10`

## Performance Considerations

Entity extraction is LLM-intensive:
- ~2 API calls per chunk (entities + relationships)
- Processing time: ~1-2 seconds per chunk
- Cost: ~$0.002-0.004 per chunk (GPT-4o-mini)

For a 100-chunk document:
- Processing time: ~2-3 minutes
- Cost: ~$0.20-0.40

Optimize by:
- Batching extraction during off-peak hours
- Using cheaper models for initial extraction
- Extracting relationships only when needed

## Next Steps

After Phase 1 is working:
- **Phase 2**: Entity Canonicalization (merge duplicate entities)
- **Phase 3**: Multi-hop Entity Graph Retrieval
- **Phase 4**: Community Detection and Summarization

See `neo4j_backend/README.md` for detailed API documentation.
