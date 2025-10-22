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
