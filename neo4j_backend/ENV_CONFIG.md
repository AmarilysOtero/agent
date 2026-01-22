# Neo4j GraphRAG Backend - Environment Configuration

Add the following environment variables to your `.env` file to use the Neo4j GraphRAG backend:

## Neo4j Database Connection (Required for Backend)

```bash
# Neo4j database connection (bolt protocol)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

## Neo4j Backend API (Already Configured)

The main application already has these configured:
```bash
# Neo4j Backend API URL
NEO4J_API_URL=http://localhost:8000

# Use Neo4j search instead of Azure Search (true/false)
USE_NEO4J_SEARCH=false
```

## Azure OpenAI (Already Configured)

The Neo4j backend uses the existing Azure OpenAI configuration:
```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
***REMOVED***
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini
AZURE_AI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002
```

## Quick Setup

1. Install Neo4j:
   - Download from https://neo4j.com/download/
   - Or use Docker: `docker run -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/your_password neo4j:latest`

2. Add the Neo4j connection variables to your `.env` file

3. Start the Neo4j backend:
   ```bash
   python -m neo4j_backend.main
   ```

4. The backend API will be available at `http://localhost:8000`

See `neo4j_backend/README.md` for detailed setup instructions.
