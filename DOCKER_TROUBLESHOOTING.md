# Docker Troubleshooting Guide

## Issues Fixed

### 1. Neo4j Connection (Fixed ✅)

**Problem:** Neo4j API at `localhost:8000` not accessible from Docker container.

**Solution:**

- Added `extra_hosts: host.docker.internal:host-gateway` to docker-compose
- Updated `neo4j_graphrag.py` to automatically replace `localhost` with `host.docker.internal` when running in Docker

**Result:** Neo4j connection now works from Docker container.

### 2. Foundry Authentication (Needs Configuration)

**Problem:** Foundry connection fails with "Operation failed after retries" because Azure CLI is not available in Docker.

**Current Status:**

- Azure CLI is not installed in the Docker container (expected)
- Code falls back to `DefaultAzureCredential()` which requires environment variables

**Solutions:**

#### Option A: Use Environment Variables (Recommended for Docker)

Add these to your `Agent/.env` file:

```env
# Azure Service Principal (for DefaultAzureCredential)
AZURE_CLIENT_ID=your-client-id
***REMOVED***
AZURE_TENANT_ID=your-tenant-id
```

#### Option B: Install Azure CLI in Docker (Not Recommended)

You could modify the Dockerfile to install Azure CLI, but this is not recommended for production.

#### Option C: Use Managed Identity (If running on Azure)

If running on Azure infrastructure, you can use Managed Identity which DefaultAzureCredential supports automatically.

## Rebuild Container After Changes

After updating code or .env:

```powershell
cd C:\Alexis\Projects\RAG_Infra
docker-compose -f docker-compose.dev.yml up -d --build agent
```

## View Logs

```powershell
# Follow logs in real-time
docker logs -f rag-agent

# View last 50 lines
docker logs rag-agent --tail 50

# Filter for specific errors
docker logs rag-agent | Select-String -Pattern "Foundry|Neo4j|Error" -Context 2
```

## Test Connections

**Test Neo4j:**

```powershell
docker exec rag-agent python -c "import requests; r = requests.get('http://host.docker.internal:8000/api/health'); print(r.status_code)"
```

**Test Foundry:**
Check logs for Foundry connection errors after making a chat request.








