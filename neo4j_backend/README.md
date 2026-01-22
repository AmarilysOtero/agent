# Neo4j GraphRAG Backend

Backend API for Neo4j GraphRAG operations including entity extraction, typed relationship creation, and graph-based retrieval.

## Features

### Phase 1: Entity Extraction (Implemented)
- ✅ Entity extraction from document chunks using Azure OpenAI
- ✅ Entity node creation in Neo4j with type labels (Person, Organization, Location, Concept, Event, Product)
- ✅ Chunk-to-entity mention edges (`MENTIONS` relationships)
- ✅ Typed relationships between entities (WORKS_FOR, LOCATED_IN, PART_OF, etc.)
- ✅ Confidence scoring and provenance tracking
- ✅ Entity deduplication via stable ID generation

### Planned Features
- Phase 2: Entity Canonicalization (fuzzy matching, embedding similarity)
- Phase 3: Multi-hop Graph Retrieval
- Phase 4: Community Detection and Summarization

## Architecture

```
neo4j_backend/
├── database/
│   └── operations.py      # Neo4j database operations
├── routers/
│   └── graph.py          # FastAPI endpoints
├── utils/
│   ├── llm_client.py     # Azure OpenAI client
│   └── prompts.py        # Entity/relationship extraction prompts
├── models/
│   └── schemas.py        # Pydantic request/response models
└── main.py               # FastAPI application
```

## Setup

### 1. Install Dependencies

```bash
cd neo4j_backend
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the repository root with:

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
***REMOVED***
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini
AZURE_AI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002

# Server Configuration (optional)
HOST=0.0.0.0
PORT=8000
```

### 3. Set Up Neo4j Database

Ensure Neo4j is running and create constraints:

```bash
# Start the backend server first
python -m neo4j_backend.main

# Then call the setup endpoint
curl -X POST http://localhost:8000/api/graph/setup-constraints
```

This creates:
- Unique constraint on Entity.id
- Unique constraint on Chunk.id

### 4. Start the Server

```bash
# From repository root
python -m neo4j_backend.main

# Or with uvicorn directly
uvicorn neo4j_backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
```bash
GET /health
```

Returns Neo4j connection status and OpenAI configuration.

### Extract Entities
```bash
POST /api/graph/extract-entities
Content-Type: application/json

{
  "file_path": "/path/to/file.pdf",
  "entity_types": ["Person", "Organization", "Location"],
  "extract_relationships": true
}
```

Extracts entities from all chunks of a file and creates:
- Entity nodes in Neo4j
- MENTIONS edges from chunks to entities
- Typed relationships between entities (if extract_relationships=true)

**Response:**
```json
{
  "success": true,
  "file_path": "/path/to/file.pdf",
  "chunks_processed": 45,
  "entities_created": 123,
  "mention_edges_created": 156,
  "relationships_created": 34,
  "processing_time_seconds": 12.5,
  "errors": []
}
```

### Get Entity Statistics
```bash
GET /api/graph/entity-stats
```

Returns statistics about entities in the graph:
```json
{
  "total_entities": 523,
  "entities_by_type": {
    "Person": 234,
    "Organization": 156,
    "Location": 89,
    "Concept": 44
  },
  "total_mentions": 1247,
  "total_relationships": 89
}
```

### Get Chunk Entities
```bash
GET /api/graph/chunk-entities/{chunk_id}
```

Returns all entities mentioned in a specific chunk.

## Neo4j Schema

### Node Types

**Entity Nodes:**
```cypher
(e:Entity:Person {
  id: "stable_hash",
  name: "John Doe",
  type: "Person",
  confidence: 0.95,
  extraction_method: "llm",
  mention_count: 5,
  created_at: datetime(),
  updated_at: datetime()
})
```

Supported entity types:
- `Entity:Person`
- `Entity:Organization`
- `Entity:Location`
- `Entity:Concept`
- `Entity:Event`
- `Entity:Product`

**Chunk Nodes** (created by document ingestion):
```cypher
(c:Chunk {
  id: "chunk_id",
  text: "chunk content",
  file_path: "/path/to/file",
  file_name: "file.pdf",
  index: 0
})
```

### Relationship Types

**MENTIONS** (Chunk → Entity):
```cypher
(c:Chunk)-[m:MENTIONS {
  confidence: 0.9,
  context: "text snippet...",
  created_at: datetime()
}]->(e:Entity)
```

**Typed Relationships** (Entity → Entity):
- `WORKS_FOR` - Person works for Organization
- `LOCATED_IN` - Entity is located in Location
- `PART_OF` - Entity is part of another Entity
- `COLLABORATES_WITH` - Entity collaborates with another
- `CREATES` - Entity creates another (e.g., Person creates Product)
- `MENTIONS` - Entity mentions another
- `RELATED_TO` - Generic relationship
- `CAUSES` - Entity causes another Entity or Event
- `PARTICIPATES_IN` - Entity participates in Event
- `OWNS` - Entity owns another Entity

```cypher
(e1:Entity:Person)-[r:WORKS_FOR {
  confidence: 0.88,
  source_chunk_id: "chunk_123",
  extraction_method: "llm",
  created_at: datetime()
}]->(e2:Entity:Organization)
```

## Usage Example

```python
import requests

# Extract entities from a file
response = requests.post(
    "http://localhost:8000/api/graph/extract-entities",
    json={
        "file_path": "/documents/research_paper.pdf",
        "entity_types": ["Person", "Organization", "Concept"],
        "extract_relationships": True
    }
)

result = response.json()
print(f"Extracted {result['entities_created']} entities")
print(f"Created {result['relationships_created']} relationships")

# Get statistics
stats = requests.get("http://localhost:8000/api/graph/entity-stats").json()
print(f"Total entities in graph: {stats['total_entities']}")
print(f"Entities by type: {stats['entities_by_type']}")
```

## Integration with News Reporter

The main News Reporter application uses the `neo4j_graphrag.py` tool which calls this backend API:

```python
# In src/news_reporter/tools/neo4j_graphrag.py
from src.news_reporter.tools.neo4j_graphrag import graphrag_search

results = graphrag_search(
    query="Find information about AI",
    top_k=10,
    similarity_threshold=0.7
)
```

To enable entity-enhanced search, the backend needs to:
1. Extract entities from ingested documents
2. Use entity relationships during graph expansion
3. Return entity-aware results

## Development

### Testing Entity Extraction

```bash
# 1. Ensure Neo4j and the backend are running
python -m neo4j_backend.main

# 2. In another terminal, test extraction
curl -X POST http://localhost:8000/api/graph/extract-entities \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/test/file.pdf",
    "extract_relationships": true
  }'
```

### Debugging

Enable debug logging:
```bash
# In .env
LOG_LEVEL=DEBUG
```

View logs in the terminal where the server is running.

## Cost Considerations

Entity extraction uses Azure OpenAI API calls:
- ~1 call per chunk for entity extraction
- ~1 call per chunk for relationship extraction (if enabled)
- ~1 call per entity for embedding generation (future)

For a 100-chunk document with relationship extraction:
- ~200 API calls
- Estimated cost: $0.20 - $0.40 (depending on model and pricing)

To reduce costs:
- Use smaller/cheaper models (gpt-4o-mini instead of gpt-4)
- Extract relationships only when needed
- Batch process during off-peak hours
- Cache entity extractions

## Troubleshooting

### Neo4j Connection Issues
- Verify Neo4j is running: `docker ps` or check Neo4j Desktop
- Check NEO4J_URI, username, and password in .env
- Test connection: `curl http://localhost:8000/health`

### Azure OpenAI Issues
- Verify AZURE_OPENAI_ENDPOINT and API key are correct
- Check deployment names match your Azure OpenAI resource
- Ensure sufficient quota for API calls

### No Chunks Found Error
- Verify documents have been ingested and chunks exist in Neo4j
- Check file_path matches exactly (case-sensitive)
- Query Neo4j directly: `MATCH (c:Chunk) RETURN c.file_path LIMIT 10`

## License

See repository LICENSE file.
