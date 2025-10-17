"""
Check reachability of all configured Azure AI Foundry agents, and list all agents in the project.

Run with:
    python -m src.news_reporter.tools.check_agent_reachability
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from azure.identity import AzureCliCredential, DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError


# ========== ENV LOADING ==========
def load_env():
    """Load .env from repo root (…/repo/.env)."""
    env_path = Path(__file__).resolve().parents[3] / ".env"
    print(f"[check_agent] Loading .env from: {env_path}")
    load_dotenv(dotenv_path=env_path, override=True)


# ========== AUTH ==========
def get_credential():
    """Prefer Azure CLI credential; fall back to DefaultAzureCredential."""
    try:
        return AzureCliCredential()
    except Exception:
        return DefaultAzureCredential()


def _require_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise SystemExit(f"❌ Missing required env var: {name}")
    return v


# ========== CLIENT ==========
def build_client() -> AIProjectClient:
    """
    Build a project-scoped client for Azure AI Foundry.
    Requires:
        AZURE_AI_PROJECT_ENDPOINT=https://<account>.services.ai.azure.com/api/projects/<ProjectName>
    """
    endpoint = _require_env("AZURE_AI_PROJECT_ENDPOINT").rstrip("/")
    if not endpoint.startswith("https://"):
        raise SystemExit("❌ AZURE_AI_PROJECT_ENDPOINT must start with https://")
    if "/api/projects/" not in endpoint:
        raise SystemExit(
            "❌ AZURE_AI_PROJECT_ENDPOINT must be the *project endpoint*, e.g.\n"
            "   https://<account>.services.ai.azure.com/api/projects/<ProjectName>"
        )

    print(f"[check_agent] Using project endpoint:\n  {endpoint}")
    return AIProjectClient(endpoint=endpoint, credential=get_credential())


# ========== ACTIONS ==========
def check_agent(client: AIProjectClient, agent_id: str, label: str | None = None):
    """Check if a specific agent ID is reachable."""
    display_name = f"{label or 'Unnamed'} ({agent_id})"
    try:
        ag = client.agents.get_agent(agent_id=agent_id)
        print(f"✅ {display_name} reachable! → Name: {getattr(ag, 'name', '—')}")
    except ResourceNotFoundError:
        print(f"❌ 404: {display_name} not found in this project.")
    except HttpResponseError as e:
        code = getattr(e, 'status_code', 'unknown')
        print(f"⚠️ HTTP {code} while checking {display_name}: {e.message}")
    except Exception as e:
        print(f"❌ Unexpected error for {display_name}: {repr(e)}")


def list_all_agents(client: AIProjectClient):
    """List all agents available in the Foundry project."""
    print("\n🔎 Fetching all agents in this Foundry project...\n")
    try:
        agents_iter = client.agents.list_agents()
        found = False
        for ag in agents_iter:
            found = True
            print(f"🧠 Name: {getattr(ag, 'name', '—')}")
            print(f"   ID: {getattr(ag, 'id', '—')}")
            print(f"   Model: {getattr(ag, 'model', '—')}\n")

        if not found:
            print("⚠️ No agents found in this project.")
    except HttpResponseError as e:
        code = getattr(e, "status_code", "unknown")
        msg = getattr(e, "message", str(e))
        print(f"❌ HTTP error listing agents ({code}): {msg}")
        if code == 404:
            print("   → The endpoint may not be a project endpoint, or you lack RBAC permissions.")
    except Exception as e:
        print(f"❌ Unexpected error while listing agents: {repr(e)}")


# ========== MAIN ==========
if __name__ == "__main__":
    load_env()
    client = build_client()

    print("\n================ CHECKING ALL KNOWN AGENTS ================\n")

    # Agents defined in .env (like yours)
    known_agents = {
        "TRIAGE": os.getenv("AGENT_ID_TRIAGE"),
        "WEBSEARCH": os.getenv("AGENT_ID_WEBSEARCH"),
        "REPORTER_LIST": os.getenv("AGENT_ID_REPORTER_LIST"),
        "REVIEWER": os.getenv("AGENT_ID_REVIEWER"),
    }

    any_found = False
    for label, agent_id in known_agents.items():
        if agent_id:
            any_found = True
            check_agent(client, agent_id, label)
        else:
            print(f"⚠️ No {label} agent ID found in .env")

    if not any_found:
        print("⚠️ No agent IDs defined in .env; skipping checks.")

    print("\n================ LISTING ALL PROJECT AGENTS ================\n")
    list_all_agents(client)
