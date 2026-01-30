"""
Agent Service - Functions for managing Foundry agents
Provides create, update, delete, and list operations for agents
Uses robust method discovery from create_foundry_agents.py
"""
from __future__ import annotations
import logging
import inspect
from typing import Dict, List, Optional, Any
from fastapi import HTTPException

from azure.ai.projects import AIProjectClient
from azure.core.exceptions import HttpResponseError

from ..foundry_runner import get_foundry_client

logger = logging.getLogger(__name__)


def _get_id(obj: Any) -> str:
    """Extract ID from agent object (handles SDK models and dicts)"""
    if obj is None:
        return ""
    # Attribute access (SDK model)
    for attr in ("id", "assistant_id", "value"):
        v = getattr(obj, attr, None)
        if v and isinstance(v, str):
            return v
    # Dict access (REST response)
    if isinstance(obj, dict):
        return obj.get("id") or obj.get("assistant_id") or obj.get("value") or ""
    return ""


def _get_attr(obj: Any, *attrs: str, default: str = "") -> str:
    """Extract first non-empty string from obj (attr or dict key)"""
    for a in attrs:
        v = obj.get(a) if isinstance(obj, dict) else getattr(obj, a, None)
        if v is not None and isinstance(v, str) and v.strip():
            return v.strip()
    return default


def _explain_http(e: HttpResponseError, ctx: str) -> str:
    """Provide user-friendly error messages for HTTP errors"""
    sc = getattr(e, "status_code", None)
    if sc == 404:
        return f"{ctx}: 404 Not Found — programmatic creation may be disabled for this hub/project/region. Create via Azure AI Studio UI."
    if sc in (401, 403):
        return f"{ctx}: {sc} Authentication/Authorization — ensure you're logged in and have RBAC permissions on the hub/project."
    if sc in (429, 500, 502, 503, 504):
        return f"{ctx}: {sc} Transient error — retry later."
    return f"{ctx}: HTTP {sc or 'unknown'} — {str(e)}"


def _try_create_methods(agents_ops: Any, name: str, model: str, instructions: str, description: Optional[str] = None, tools: Optional[List[str]] = None) -> Any:
    """
    Try multiple SDK method signatures to create an agent.
    
    Newer versions of the Azure Agents API require `tool_resources` to be
    present in the JSON body (even if empty). We always include an empty
    object plus a tools array (empty by default) to satisfy the contract.
    """
    body = {
        "model": model,
        "name": name,
        "instructions": instructions,
        "description": description or name,
        # Required by current Foundry / Agents API even when you don't
        # attach any external resources.
        "tool_resources": {},
        # Ensure tools is always an array (the service expects a list)
        "tools": tools or [],
    }

    # Try different method signatures (SDK compatibility)
    trials = [
        ("agents.create_agent(model=..., name=..., instructions=...)", lambda: agents_ops.create_agent(**body)),
        ("agents.create(model=..., name=..., instructions=...)", lambda: agents_ops.create(**body)),
        ("agents.create_agent(body={...})", lambda: agents_ops.create_agent(body=body)),
        ("agents.create(body={...})", lambda: agents_ops.create(body=body)),
    ]
    
    errors = []
    for label, fn in trials:
        try:
            return fn()
        except AttributeError as e:
            errors.append(f"{label} -> AttributeError: {e}")
        except TypeError as e:
            errors.append(f"{label} -> TypeError: {e}")
        except HttpResponseError as e:
            # Bubble up HTTP errors for caller to handle
            raise e
    
    # No working method found - provide diagnostic info
    members = sorted([m for m in dir(agents_ops) if not m.startswith("_")])
    sigs = []
    for m in members:
        try:
            obj = getattr(agents_ops, m)
            if callable(obj):
                sigs.append(f"{m}{inspect.signature(obj)}")
        except Exception:
            pass
    
    diag = "\n".join(errors)
    sigdump = "\n  ".join(sigs[:50])
    raise HTTPException(
        status_code=501,
        detail=(
            "Could not find a working create method on client.agents.\n"
            "Tried:\n" + diag + "\n\n"
            "Available callables on client.agents (first 50):\n  " + sigdump
        )
    )


def create_foundry_agent(
    name: str,
    model: str,
    instructions: str,
    description: Optional[str] = None,
    tools: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create a new agent in Foundry
    Uses robust method discovery from create_foundry_agents.py
    """
    try:
        client = get_foundry_client()
        agents_ops = client.agents
        
        # Use robust method trying logic
        try:
            agent = _try_create_methods(agents_ops, name, model, instructions, description, tools)
        except HttpResponseError as e:
            # Provide user-friendly error messages
            error_msg = _explain_http(e, f"Creating agent '{name}'")
            status_code = getattr(e, "status_code", 500)
            logger.error(f"HTTP error creating agent: {error_msg}")
            raise HTTPException(
                status_code=status_code,
                detail=error_msg
            )
        
        agent_id = _get_id(agent)
        logger.info(f"Successfully created agent: {name} (ID: {agent_id})")
        
        return {
            "id": agent_id,
            "name": name,
            "model": model,
            "instructions": instructions,
            "description": description or name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error creating agent")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error creating agent: {str(e)}"
        )


def update_foundry_agent(
    agent_id: str,
    name: Optional[str] = None,
    model: Optional[str] = None,
    instructions: Optional[str] = None,
    description: Optional[str] = None,
    tools: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Update an existing agent in Foundry"""
    try:
        client = get_foundry_client()
        agents_ops = client.agents
        
        updates = {}
        if name is not None:
            updates["name"] = name
        if model is not None:
            updates["model"] = model
        if instructions is not None:
            updates["instructions"] = instructions
        if description is not None:
            updates["description"] = description
        if tools is not None:
            updates["tools"] = tools
        
        if not updates:
            raise HTTPException(
                status_code=400,
                detail="No updates provided"
            )
        
        # Try different update methods
        agent = None
        errors = []
        
        if hasattr(agents_ops, "update_agent"):
            try:
                agent = agents_ops.update_agent(agent_id, **updates)
                logger.info(f"Updated agent using update_agent(agent_id, **updates)")
            except (TypeError, AttributeError) as e:
                errors.append(f"update_agent(agent_id, **updates): {e}")
        
        if agent is None and hasattr(agents_ops, "update"):
            try:
                agent = agents_ops.update(agent_id, **updates)
                logger.info(f"Updated agent using update(agent_id, **updates)")
            except (TypeError, AttributeError) as e:
                errors.append(f"update(agent_id, **updates): {e}")
        
        if agent is None and hasattr(agents_ops, "update_agent"):
            try:
                agent = agents_ops.update_agent(agent_id, body=updates)
                logger.info(f"Updated agent using update_agent(agent_id, body=updates)")
            except (TypeError, AttributeError) as e:
                errors.append(f"update_agent(agent_id, body=updates): {e}")
        
        if agent is None:
            raise HTTPException(
                status_code=501,
                detail=f"Could not update agent. Tried methods: {', '.join(errors)}"
            )
        
        logger.info(f"Successfully updated agent: {agent_id}")
        
        return {
            "id": agent_id,
            "name": getattr(agent, "name", name) or name,
            "model": getattr(agent, "model", model) or model,
            "instructions": getattr(agent, "instructions", instructions) or instructions,
            "description": getattr(agent, "description", description) or description
        }
        
    except HTTPException:
        raise
    except HttpResponseError as e:
        status_code = getattr(e, "status_code", 500)
        error_msg = str(e)
        logger.error(f"HTTP error updating agent: {status_code} - {error_msg}")
        raise HTTPException(
            status_code=status_code,
            detail=f"Failed to update agent in Foundry: {error_msg}"
        )
    except Exception as e:
        logger.exception("Unexpected error updating agent")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error updating agent: {str(e)}"
        )


def delete_foundry_agent(agent_id: str) -> bool:
    """Delete an agent from Foundry"""
    try:
        client = get_foundry_client()
        agents_ops = client.agents
        
        # Try different delete methods
        deleted = False
        errors = []
        
        if hasattr(agents_ops, "delete_agent"):
            try:
                agents_ops.delete_agent(agent_id)
                deleted = True
                logger.info(f"Deleted agent using delete_agent(agent_id)")
            except (TypeError, AttributeError) as e:
                errors.append(f"delete_agent(agent_id): {e}")
        
        if not deleted and hasattr(agents_ops, "delete"):
            try:
                agents_ops.delete(agent_id)
                deleted = True
                logger.info(f"Deleted agent using delete(agent_id)")
            except (TypeError, AttributeError) as e:
                errors.append(f"delete(agent_id): {e}")
        
        if not deleted:
            raise HTTPException(
                status_code=501,
                detail=f"Could not delete agent. Tried methods: {', '.join(errors)}"
            )
        
        logger.info(f"Successfully deleted agent: {agent_id}")
        return True
        
    except HTTPException:
        raise
    except HttpResponseError as e:
        status_code = getattr(e, "status_code", 500)
        error_msg = str(e)
        logger.error(f"HTTP error deleting agent: {status_code} - {error_msg}")
        raise HTTPException(
            status_code=status_code,
            detail=f"Failed to delete agent in Foundry: {error_msg}"
        )
    except Exception as e:
        logger.exception("Unexpected error deleting agent")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error deleting agent: {str(e)}"
        )


def list_all_foundry_agents() -> List[Dict[str, Any]]:
    """List all agents in Foundry"""
    try:
        client = get_foundry_client()
        agents_ops = client.agents

        # Try to list agents
        result = None
        if hasattr(agents_ops, "list_agents"):
            result = agents_ops.list_agents()
        elif hasattr(agents_ops, "list"):
            result = agents_ops.list()
        else:
            logger.warning("No list method found on agents_ops")
            return []

        # Extract agents from result (handle data, value, or ItemPaged)
        raw = None
        if hasattr(result, "data"):
            raw = result.data
        elif hasattr(result, "value"):
            raw = result.value
        elif hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
            raw = result
        if raw is None:
            logger.warning("Could not extract agents from result: type=%s", type(result))
            return []

        # Ensure we have a list (ItemPaged/iterator -> full list)
        agents = list(raw) if hasattr(raw, "__iter__") and not isinstance(raw, (str, bytes)) else [raw]

        # Convert to dict format
        agent_list = []
        for i, agent in enumerate(agents):
            agent_id = _get_id(agent)
            if not agent_id:
                if i == 0:
                    logger.warning(
                        "Agent object has no id (tried id, assistant_id, value). "
                        "Sample: %s",
                        repr(agent)[:200] if agent else "None",
                    )
                continue
            agent_list.append({
                "id": agent_id,
                "name": _get_attr(agent, "name", "display_name"),
                "model": _get_attr(agent, "model"),
                "instructions": _get_attr(agent, "instructions"),
                "description": _get_attr(agent, "description"),
            })

        logger.info("Listed %d agents from Foundry", len(agent_list))
        return agent_list
        
    except HttpResponseError as e:
        status_code = getattr(e, "status_code", 500)
        error_msg = str(e)
        logger.error(f"HTTP error listing agents: {status_code} - {error_msg}")
        raise HTTPException(
            status_code=status_code,
            detail=f"Failed to list agents from Foundry: {error_msg}"
        )
    except Exception as e:
        logger.exception("Unexpected error listing agents")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error listing agents: {str(e)}"
        )


def get_foundry_agent(agent_id: str) -> Dict[str, Any]:
    """Get details of a specific agent"""
    try:
        client = get_foundry_client()
        agents_ops = client.agents
        
        # Try different get methods
        agent = None
        errors = []
        
        if hasattr(agents_ops, "get_agent"):
            try:
                agent = agents_ops.get_agent(agent_id)
                logger.info(f"Retrieved agent using get_agent(agent_id)")
            except (TypeError, AttributeError) as e:
                errors.append(f"get_agent(agent_id): {e}")
        
        if agent is None and hasattr(agents_ops, "get"):
            try:
                agent = agents_ops.get(agent_id)
                logger.info(f"Retrieved agent using get(agent_id)")
            except (TypeError, AttributeError) as e:
                errors.append(f"get(agent_id): {e}")
        
        # Fallback: search in list
        if agent is None:
            logger.info(f"Trying to find agent in list (fallback)")
            all_agents = list_all_foundry_agents()
            for agent_info in all_agents:
                if agent_info["id"] == agent_id:
                    return agent_info
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_id} not found"
            )
        
        agent_id_found = _get_id(agent)
        return {
            "id": agent_id_found or agent_id,
            "name": getattr(agent, "name", "") or getattr(agent, "display_name", ""),
            "model": getattr(agent, "model", ""),
            "instructions": getattr(agent, "instructions", ""),
            "description": getattr(agent, "description", ""),
            "tools": getattr(agent, "tools", [])
        }
        
    except HTTPException:
        raise
    except HttpResponseError as e:
        status_code = getattr(e, "status_code", 500)
        if status_code == 404:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        error_msg = str(e)
        logger.error(f"HTTP error getting agent: {status_code} - {error_msg}")
        raise HTTPException(
            status_code=status_code,
            detail=f"Failed to get agent from Foundry: {error_msg}"
        )
    except Exception as e:
        logger.exception("Unexpected error getting agent")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error getting agent: {str(e)}"
        )
