"""
Agent management utilities for Azure AI Foundry.
Handles listing and creating agents using the Azure AI Projects SDK.
"""
from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.core.exceptions import HttpResponseError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_env():
    """Load .env from the project root"""
    try:
        from dotenv import load_dotenv
        here = Path(__file__).resolve()
        candidates = [
            here.parents[2] / ".env",  # repo root
            here.parents[1] / ".env",
            Path.cwd() / ".env",
        ]
        for p in candidates:
            if p.exists():
                load_dotenv(p)
                logger.info(f"Loaded .env from: {p}")
                break
    except Exception as e:
        logger.warning(f"Failed to load .env: {e}")


_load_env()


def parse_connection_string(conn: str) -> Dict[str, str]:
    """Parse Azure AI Project connection string into components"""
    parts: Dict[str, str] = {}
    for chunk in conn.split(";"):
        if "=" in chunk:
            k, v = chunk.split("=", 1)
            parts[k.strip().lower()] = v.strip()
    
    required = ["endpoint", "project", "subscription_id", "resource_group", "account"]
    missing = [k for k in required if not parts.get(k)]
    if missing:
        raise ValueError(f"Connection string missing: {', '.join(missing)}")
    
    endpoint = parts["endpoint"].rstrip("/")
    if not endpoint.startswith("https://"):
        raise ValueError(f"Endpoint must start with https://. Got: {endpoint}")
    parts["endpoint"] = endpoint
    
    return parts

def get_ai_project_client() -> AIProjectClient:
    conn = os.getenv("AI_PROJECT_CONNECTION_STRING")
    if not conn:
        raise ValueError("AI_PROJECT_CONNECTION_STRING is not set")

    parts = parse_connection_string(conn)
    credential = DefaultAzureCredential()

    return AIProjectClient(
        endpoint=parts["endpoint"],
        project=parts["project"],
        subscription_id=parts["subscription_id"],
        resource_group_name=parts["resource_group"],
        account_name=parts["account"],
        credential=credential,
    )

def list_agents() -> List[Dict[str, Any]]:
    """
    List all available agents.
    First tries graph-based registry (workflow-native agents), 
    falls back to Foundry if graph query fails.
    
    Returns:
        List of agent dictionaries with id, name, model, description, etc.
    """
    try:
        return list_agents_from_foundry()
    except Exception as foundry_error:
        logger.error(f"Foundry queries failed")
        # Return empty list rather than crashing
        return []

def list_agents_from_foundry() -> List[Dict[str, Any]]:
    """
    List all available agents from Azure AI Foundry.
    
    Returns:
        List of agent dictionaries with id, name, model, description, etc.
    """
    try:
        # Ensure OPENAI_API_VERSION is set for the SDK
        if "OPENAI_API_VERSION" not in os.environ:
            os.environ["OPENAI_API_VERSION"] = "2024-05-01-preview"
            logger.info("Set default OPENAI_API_VERSION to 2024-05-01-preview for agents")

        client = get_ai_project_client()
        listing = client.agents.list_agents()

        agents: List[Dict[str, Any]] = []

        # In the current SDK, `listing` is an iterable of Agent objects.
        # We still keep a fallback for any response shape that has `.data`/`.value`.
        iterable = None

        # If the SDK ever returns a wrapper with `.data` or `.value`, use it
        maybe_data = getattr(listing, "data", None) or getattr(listing, "value", None)
        if maybe_data is not None:
            iterable = maybe_data
        else:
            # Normal case: pageable iterator
            iterable = listing

        for agent in iterable:
            agent_dict = {
                "id": getattr(agent, "id", None) or getattr(agent, "value", None) or "",
                "name": getattr(agent, "name", "Unknown"),
                "model": getattr(agent, "model", ""),
                "description": getattr(agent, "description", ""),
                "created_at": getattr(agent, "created_at", None),
                "instructions": getattr(agent, "instructions", ""),
            }
            agents.append(agent_dict)

        logger.info(
            "Successfully listed %d agents from Foundry: %s",
            len(agents),
            [a["name"] for a in agents],
        )
        return agents

    except HttpResponseError as e:
        logger.error(f"HTTP error listing agents from Foundry: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to list agents from Foundry: {e}")
        raise



def list_agents_from_local_graph() -> List[Dict[str, Any]]:
    """
    List all workflow-created agents from Neo4j graph.
    
    Returns:
        List of agent dictionaries from graph registry
    """
    try:
        import requests
        
        neo4j_api_url = os.getenv("NEO4J_API_URL", "http://localhost:8000")
        url = f"{neo4j_api_url.rstrip('/')}/api/agents/list"
        
        logger.info(f"Fetching agents from Neo4j graph: {url}")
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        agents = response.json()
        logger.info(f"Successfully fetched {len(agents)} agents from graph")
        
        return agents
    except Exception as e:
        logger.error(f"Failed to fetch agents from graph: {e}")
        raise

def create_agent(
    name: str,
    model: Optional[str] = None,
    instructions: Optional[str] = None,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a new agent in Azure AI Foundry.
    
    Args:
        name: Agent name (required)
        model: Model deployment name (defaults to FOUNDRY_DEFAULT_MODEL env var)
        instructions: Agent instructions
        description: Agent description
        
    Returns:
        Created agent dictionary with id, name, etc.
    """
    try:
        client = get_ai_project_client()
        
        # Use default model if not provided
        if not model:
            model = (
                os.getenv("FOUNDRY_DEFAULT_MODEL") 
                or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
                or "gpt-4o-mini"
            )
        
        # Set default instructions if not provided
        if not instructions:
            instructions = f"You are {name}, a helpful AI assistant."
        
        # Prepare agent creation params
        body = {
            "model": model,
            "name": name,
            "description": description or name,
            "instructions": instructions,
        }
        
        # Try different SDK method signatures
        agent = None
        try:
            agent = client.agents.create_agent(**body)
        except (AttributeError, TypeError):
            try:
                agent = client.agents.create(**body)
            except (AttributeError, TypeError):
                try:
                    agent = client.agents.create_agent(body=body)
                except (AttributeError, TypeError):
                    agent = client.agents.create(body=body)
        
        if not agent:
            raise RuntimeError("Could not create agent with available SDK methods")
        
        # Extract agent data
        agent_dict = {
            "id": getattr(agent, "id", None) or getattr(agent, "value", None) or "",
            "name": getattr(agent, "name", name),
            "model": getattr(agent, "model", model),
            "description": getattr(agent, "description", description or ""),
            "created_at": getattr(agent, "created_at", None),
            "instructions": getattr(agent, "instructions", instructions),
        }
        
        logger.info(f"Successfully created agent: {agent_dict['name']} ({agent_dict['id']})")
        return agent_dict
        
    except HttpResponseError as e:
        logger.error(f"HTTP error creating agent: {e}")
        if getattr(e, "status_code", None) == 404:
            raise ValueError(
                "Agent creation not available (404). "
                "Programmatic creation may be disabled for this project/region. "
                "Please create agents via Azure AI Foundry Studio UI."
            )
        raise
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        raise

