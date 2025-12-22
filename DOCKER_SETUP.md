# Docker Setup for Agent Application

## Overview

This setup allows you to run the Agent application in a Docker container, which solves the MongoDB authentication issues on Windows.

## Prerequisites

- Docker and Docker Compose installed
- MongoDB already running in Docker (via RAG_Infra)

## Setup Steps

### 1. Update .env File for Docker

In `C:\Alexis\Projects\Agent\.env`, update the MongoDB connection strings to use the container name `mongo` instead of `127.0.0.1`:

```env
# Change from:
MONGO_AUTH_URL=mongodb://user_rw:BestRAG2026@127.0.0.1:27017/auth_db?authSource=auth_db
MONGO_AGENT_URL=mongodb://user_rw:BestRAG2026@127.0.0.1:27017/agent_db?authSource=agent_db

# To:
MONGO_AUTH_URL=mongodb://user_rw:BestRAG2026@mongo:27017/auth_db?authSource=auth_db
MONGO_AGENT_URL=mongodb://user_rw:BestRAG2026@mongo:27017/agent_db?authSource=agent_db
```

**Note:** The hostname `mongo` is the container name from docker-compose, which allows containers to communicate on the same Docker network.

### 1.5. Configure Azure Authentication for Docker

Since `az login` doesn't work inside Docker containers, you need to use service principal authentication via environment variables. Add these to your `.env` file:

```env
# Azure Service Principal Authentication (required for Docker)
AZURE_CLIENT_ID=<your-service-principal-client-id>
***REMOVED***
AZURE_TENANT_ID=<your-azure-tenant-id>
```

**To create a service principal:**

1. Install Azure CLI on your host machine (if not already installed)
2. Run:
   ```powershell
   az login
   az ad sp create-for-rbac --name "rag-agent-sp" --role contributor --scopes /subscriptions/<your-subscription-id>
   ```
3. Copy the output values:

   - `appId` → `AZURE_CLIENT_ID`
   - `password` → `AZURE_CLIENT_SECRET`
   - `tenant` → `AZURE_TENANT_ID`

4. **Assign Foundry project permissions** (required for Foundry access):

   The service principal needs access to your Azure AI Foundry project. You can assign it via Azure Portal or CLI:

   **Option A: Via Azure Portal**

   - Go to your Azure AI Foundry project in Azure Portal
   - Navigate to "Access control (IAM)" → "Add role assignment"
   - Select role: **"Cognitive Services User"** or **"AI Developer"**
   - Assign to: Service principal (search for "rag-agent-sp")

   **Option B: Via Azure CLI** (if you know your AI Services account name):

   ```powershell
   # Find your AI Services account resource ID
   az cognitiveservices account list --query "[].{name:name, id:id}" -o table

   # Assign Cognitive Services User role (replace <account-name> and <resource-group>)
   az role assignment create \
     --assignee <appId-from-step-3> \
     --role "Cognitive Services User" \
     --scope /subscriptions/<subscription-id>/resourceGroups/<resource-group>/providers/Microsoft.CognitiveServices/accounts/<account-name>
   ```

**Alternative:** If running on Azure (e.g., Azure Container Instances, Azure Kubernetes Service), you can use Managed Identity instead by not setting these variables. The code will automatically use the managed identity.

### 2. Build and Start the Agent Container

From the `RAG_Infra` directory:

```powershell
cd C:\Alexis\Projects\RAG_Infra
docker-compose -f docker-compose.dev.yml up -d --build agent
```

This will:

- Build the Agent Docker image
- Start the Agent container
- Connect it to the same network as MongoDB

### 3. View Logs

```powershell
docker logs -f rag-agent
```

### 4. Stop the Container

```powershell
docker-compose -f docker-compose.dev.yml stop agent
```

Or to stop and remove:

```powershell
docker-compose -f docker-compose.dev.yml down agent
```

## Running Both MongoDB and Agent Together

To start both services:

```powershell
cd C:\Alexis\Projects\RAG_Infra
docker-compose -f docker-compose.dev.yml up -d
```

## Accessing the Application

Once running, the Agent API will be available at:

- **From host machine:** http://localhost:8787
- **From other containers:** http://rag-agent:8787

## Troubleshooting

### Container won't start

- Check logs: `docker logs rag-agent`
- Verify MongoDB is running: `docker ps | findstr rag-mongo`
- Check .env file has correct MongoDB connection strings with `mongo` hostname

### Still can't connect to MongoDB

- Verify both containers are on the same network: `docker network inspect rag-infra_default`
- Check MongoDB is accessible: `docker exec rag-mongo mongosh -u root -p rootpassword --authenticationDatabase admin --eval "db.runCommand({ping: 1})"`

### Azure Authentication Errors

If you see errors like "Please run 'az login' to set up an account" or "Failed to authenticate with Azure":

1. **Verify environment variables are set:**

   ```powershell
   docker exec rag-agent env | findstr AZURE
   ```

   You should see `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, and `AZURE_TENANT_ID`.

2. **Check .env file:** Ensure the Azure credentials are in `C:\Alexis\Projects\Agent\.env` and the file is being loaded by docker-compose.

3. **Verify service principal has correct permissions:**

   - The service principal needs access to your Azure AI Foundry project
   - Check that `AZURE_AI_PROJECT_ENDPOINT` is correct

4. **For local development (outside Docker):** You can use `az login` instead of service principal credentials.

### Rebuild after code changes

```powershell
docker-compose -f docker-compose.dev.yml up -d --build agent
```

## Development vs Production

- **Development (local):** Run directly with `python -m src.news_reporter.api` (uses 127.0.0.1)
- **Docker:** Run via docker-compose (uses mongo hostname, binds to 0.0.0.0)

The code automatically detects the environment and adjusts the host binding.
