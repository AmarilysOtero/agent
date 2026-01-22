# GraphRAG Quick Reference

## Start the Backend

```bash
# 1. Make sure Neo4j is running
docker ps | grep neo4j

# 2. Start the backend
python -m neo4j_backend.main
```

Backend runs at: `http://localhost:8000`

## API Endpoints

### Extract Entities
```bash
curl -X POST http://localhost:8000/api/graph/extract-entities \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/document.pdf",
    "extract_relationships": true
  }'
```

### Get Statistics
```bash
curl http://localhost:8000/api/graph/entity-stats
```

### Health Check
```bash
curl http://localhost:8000/health
```

### Setup Database
```bash
# Run once to create constraints
curl -X POST http://localhost:8000/api/graph/setup-constraints
```

## Python Usage

```python
import requests

# Extract entities
response = requests.post(
    "http://localhost:8000/api/graph/extract-entities",
    json={
        "file_path": "/documents/report.pdf",
        "extract_relationships": True
    }
)

print(f"Created {response.json()['entities_created']} entities")

# Get statistics
stats = requests.get("http://localhost:8000/api/graph/entity-stats").json()
print(f"Total entities: {stats['total_entities']}")
print(f"By type: {stats['entities_by_type']}")
```

## Entity Types Supported

- **Person** - Individual people
- **Organization** - Companies, institutions, groups
- **Location** - Cities, countries, places
- **Concept** - Ideas, technologies, methodologies
- **Event** - Occurrences, happenings
- **Product** - Products, tools, services

## Relationship Types Supported

- `WORKS_FOR` - Person works for Organization
- `LOCATED_IN` - Entity is located in Location
- `PART_OF` - Entity is part of another Entity
- `COLLABORATES_WITH` - Entity collaborates with another
- `CREATES` - Entity creates another
- `MENTIONS` - Entity mentions another
- `RELATED_TO` - Generic relationship
- `CAUSES` - Entity causes another
- `PARTICIPATES_IN` - Entity participates in Event
- `OWNS` - Entity owns another

## Neo4j Queries

```cypher
// View all entities
MATCH (e:Entity) RETURN e LIMIT 25

// Count entities by type
MATCH (e:Entity)
RETURN e.type as type, count(*) as count
ORDER BY count DESC

// Find entities mentioned in a chunk
MATCH (c:Chunk {id: 'chunk_id'})-[m:MENTIONS]->(e:Entity)
RETURN e.name, e.type, m.confidence

// Find relationships between entities
MATCH (e1:Entity)-[r]->(e2:Entity)
WHERE type(r) <> 'MENTIONS'
RETURN e1.name, type(r), e2.name, r.confidence
LIMIT 25

// Find most mentioned entities
MATCH (e:Entity)
RETURN e.name, e.type, e.mention_count
ORDER BY e.mention_count DESC
LIMIT 10
```

## Environment Variables

Required in `.env`:

```bash
# Neo4j Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Azure OpenAI (reuses existing config)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
***REMOVED***
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini
```

## Troubleshooting

**Backend won't start**
```bash
# Check Neo4j is running
docker ps | grep neo4j

# Check port 8000 is free
lsof -i :8000

# Check logs
python -m neo4j_backend.main
```

**No entities extracted**
- Verify Azure OpenAI credentials
- Check deployment name matches
- Review backend logs for errors

**Chunks not found**
- Verify documents are ingested first
- Check file_path is exact match
- Query Neo4j: `MATCH (c:Chunk) RETURN c.file_path LIMIT 10`

## Testing

```bash
# Run test suite
python neo4j_backend/test_entity_extraction.py

# Test connection only
python -c "from neo4j_backend.database.operations import Neo4jOperations; db = Neo4jOperations(); print('✓ Connected'); db.close()"
```

## Performance

**Processing Time:**
- ~1-2 seconds per chunk
- 100 chunks = ~2-3 minutes

**API Costs:**
- Entity extraction: ~$0.001 per chunk
- Relationship extraction: ~$0.001 per chunk
- Total: ~$0.002 per chunk (GPT-4o-mini)

**Optimization:**
- Batch process during off-peak hours
- Use `extract_relationships: false` to skip relationship extraction
- Process only new/updated documents

## Documentation

- **[neo4j_backend/README.md](neo4j_backend/README.md)** - Full API docs
- **[NEO4J_INTEGRATION.md](NEO4J_INTEGRATION.md)** - Integration guide
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Implementation details
