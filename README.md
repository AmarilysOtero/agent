# News Reporter — Microsoft Agent Framework (Python)

This project uses **Microsoft Agent Framework** (Python) with:

- `WorkflowBuilder` for orchestration
- `AzureChatClient` for LLM-backed agents
- LLM-based **intent triage** → sequential or concurrent routing
- Keeps `print()` for quick local feedback + adds structured logging

## Python Requirements

**Python 3.9+ is required** (Python 3.11 recommended to match project target).

The Azure AI Projects SDK requires Python 3.9+ due to subscriptable generics support. The Docker container uses Python 3.11, and local development should match this version.

### Creating/Upgrading Virtual Environment

**Windows (using Python Launcher):**
```bash
# Check available Python versions
py -0

# Create venv with Python 3.11 (recommended)
py -3.11 -m venv .venv

# Or use Python 3.9+ if 3.11 not available
py -3.9 -m venv .venv
```

**macOS/Linux:**
```bash
# Ensure Python 3.9+ is installed
python3 --version

# Create venv
python3 -m venv .venv
```

## Quick start

```bash
# 1) Create venv & install deps (see Python Requirements above)
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

## Tool Registration

The Agent service uses **registered Foundry tools** for SQL generation and execution. Tools are registered with Foundry agents to enable configuration-based tool assignment.

### Available Tools

1. **`query_database`** - SQL generation + execution (used by `SQLAgent`)
2. **`generate_sql_query`** - SQL generation only (no execution)
3. **`get_file_schema`** - CSV/Excel schema retrieval

### Registering Tools

**Programmatic Registration (Recommended):**

```bash
# Activate venv (Python 3.9+ required)
.venv\Scripts\activate  # Windows
# or
. .venv/bin/activate    # macOS/Linux

# Run registration script
python -m src.news_reporter.tools.register_foundry_tools
```

**Prerequisites:**
- Python 3.9+ (required for Azure SDK compatibility)
- `AI_PROJECT_CONNECTION_STRING` set in `.env`
- Agent IDs configured in `.env`:
  - `AGENT_ID_TRIAGE`
  - `AGENT_ID_AISEARCH`
  - `AGENT_ID_NEO4J_SEARCH` (optional)

**Manual Registration (if programmatic fails):**

If the SDK doesn't support programmatic registration, register tools manually in Azure AI Foundry Studio:

1. Go to Azure AI Foundry Studio → Your Hub → Project → Agents
2. Edit each agent (AiSearchAgent, TriageAgent, etc.)
3. Add tools using definitions from:
   ```bash
   python -m src.news_reporter.tools.generate_tool_definitions
   ```

### Code Usage

**SQLAgent** uses registered tool functions directly:

```python
from ..tools_sql.text_to_sql_tool import query_database

# Tool function returns JSON string
result_json = query_database(
    natural_language_query=query,
    database_id=database_id
)
# Parse to dict
sql_result = json.loads(result_json)
```

This enables:
- **Configuration-based assignment**: Tools can be assigned to agents via Foundry configuration
- **Consistency**: Same tools used by Foundry agents and Python code
- **Flexibility**: Change tool assignments without code changes

**Note**: Tool registration is a **one-time setup**. Once registered, tools work in both Foundry chat and Python code.

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

**Note**: The Docker container uses **Python 3.11** (see `Dockerfile`), which is compatible with all Azure SDK features including tool registration. The container is already configured correctly - no changes needed.

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
