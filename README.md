# News Reporter — Microsoft Agent Framework (Python)

This project uses **Microsoft Agent Framework** (Python) with:

- `WorkflowBuilder` for orchestration
- `AzureChatClient` for LLM-backed agents
- LLM-based **intent triage** → sequential or concurrent routing
- Keeps `print()` for quick local feedback + adds structured logging

## Quick start

```bash
# 1) Create venv & install deps
python -m venv .venv
# Windows
.venv\Scripts\python -m pip install -r requirements.txt
# macOS/Linux
. .venv/bin/activate && python -m pip install -r requirements.txt

# 2) Copy env and fill values
cp .env.example .env

# 3) (optional) pre-commit
pre-commit install

# 4) Run
python src/news_reporter/app.py
# or hit F5 in VS Code

python -m pip install --upgrade --pre agent-framework agent-framework-azure-ai
python -m src.news_reporter.app



```

## Environment

See `.env.example` for required variables:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY` (or use Azure CLI auth)
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_CHAT_DEPLOYMENT` (default chat model)
- `ROUTING_DEPLOYMENTS` (semicolon-separated for fan-out)
- `MULTI_ROUTE_ALWAYS` (`true`/`false`)

## Notes

- This build uses the **Agent Framework** (`agent-framework` + `agent-framework-azure-ai`).
- We _don’t_ rely on `azure-ai-projects` here; we run locally using chat clients.
- If you want Azure Agent Service (Foundry) threads/runs, we can add that as a separate path.
  # RAG

## Additional instruction after make an agent connection in Azure

.venv\Scripts\Activate

pip install "azure-ai-projects>=1.1.0b4" "azure-ai-agents>=1.2.0b5"

az login --use-device-code
az account show --output table
python -m src.news_reporter.tools.new_01_create_agent

az login --use-device-code
az account show --output table
python -m src.news_reporter.tools.check_agent_reachability

az login --use-device-code
az account show --output table
python -m src.news_reporter.foundry_runner

az login --use-device-code
az account show --output table
python -m src.news_reporter.app

python -m src.news_reporter.tools.debug_env

#for upload and vectorized PDF
pip install fastapi uvicorn python-dotenv pydantic==2.\* \
 azure-search-documents azure-storage-blob azure-cosmos \
 PyMuPDF openai

# If you’ll use Foundry Projects SDK:

pip install azure-ai-projects azure-identity

az login --use-device-code
az account show --output table
python -m src.news_reporter.api

pip install azure-search-documents azure-storage-blob requests

# (only for vector querying with your own embedding:)

# pip install azure-ai-projects azure-identity

## Docker Setup

For running the Agent application in Docker (recommended for Windows to avoid MongoDB authentication issues):

### Quick Start

1. **Configure .env for Docker:**
   - Update MongoDB connection strings to use `mongo` hostname instead of `127.0.0.1`
   - Add Azure service principal credentials (see below)

2. **Start with Docker Compose:**

   ```powershell
   cd C:\Alexis\Projects\RAG_Infra
   docker-compose -f docker-compose.dev.yml up -d --build agent
   ```

3. **View logs:**
   ```powershell
   docker logs -f rag-agent
   ```

### Azure Authentication for Docker

Since `az login` doesn't work in Docker containers, you need to use service principal authentication:

1. **Create a service principal:**

   ```powershell
   az login
   az ad sp create-for-rbac --name "rag-agent-sp" --role contributor --scopes /subscriptions/<your-subscription-id>
   ```

2. **Add to `.env` file:**

   ```env
   AZURE_CLIENT_ID=<appId-from-output>
   ***REMOVED***
   AZURE_TENANT_ID=<tenant-from-output>
   ```

3. **Assign Foundry permissions:**
   ```powershell
   az role assignment create --assignee <appId> --role "Cognitive Services User" --scope /subscriptions/<subscription-id>/resourceGroups/<resource-group>/providers/Microsoft.CognitiveServices/accounts/<account-name>
   ```

For detailed Docker setup instructions, see [DOCKER_SETUP.md](DOCKER_SETUP.md).

## Neo4j GraphRAG Backend (New)

The repository now includes a Neo4j GraphRAG backend for entity extraction and graph-based retrieval:

- **Entity extraction** from document chunks using Azure OpenAI
- **Typed relationships** between entities (WORKS_FOR, LOCATED_IN, etc.)
- **Graph-based search** with multi-hop traversal
- **Entity deduplication** and canonicalization

### Quick Start

```bash
# 1. Start Neo4j database
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password neo4j:latest

# 2. Configure .env (add Neo4j connection details)
# See neo4j_backend/ENV_CONFIG.md

# 3. Start the backend
python -m neo4j_backend.main

# 4. Extract entities from documents
curl -X POST http://localhost:8000/api/graph/extract-entities \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/document.pdf", "extract_relationships": true}'
```

For detailed setup and integration instructions, see:
- **[neo4j_backend/README.md](neo4j_backend/README.md)** - Backend API documentation
- **[NEO4J_INTEGRATION.md](NEO4J_INTEGRATION.md)** - Integration guide

## Test
