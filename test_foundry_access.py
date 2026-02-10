
import os
from dotenv import load_dotenv
load_dotenv()
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

# Load environment variables
def require_env(name):
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing environment variable: {name}")
    return v

try:
    endpoint = require_env("AZURE_AI_PROJECT_ENDPOINT")
    client_id = require_env("AZURE_CLIENT_ID")
    client_secret = require_env("AZURE_CLIENT_SECRET")
    tenant_id = require_env("AZURE_TENANT_ID")
except Exception as e:
    print(f"Environment error: {e}")
    exit(1)

try:
    credential = DefaultAzureCredential()
    client = AIProjectClient(endpoint=endpoint, credential=credential)
    print("[SUCCESS] Connected to Foundry endpoint.")
    # List available agents
    agents = getattr(client, "agents", None)
    if agents:
        print("Available agent methods:", dir(agents))
    else:
        print("No agents object found in client.")
except Exception as e:
    print(f"[FAIL] Could not connect to Foundry: {e}")
