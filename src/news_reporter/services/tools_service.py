"""Tools service - orchestration for tools registry, Foundry sync, and agent-tool relations."""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from .agent_service import get_foundry_agent, update_foundry_agent
from ..workflows.tools_storage import get_tools_storage

logger = logging.getLogger(__name__)


def _foundry_tool_to_ref(t: Any) -> str:
    """Build stable foundry_ref from Foundry tool object."""
    tt = getattr(t, "type", None) or (t.get("type") if isinstance(t, dict) else None) or "function"
    name = ""
    if hasattr(t, "function") and t.function:
        name = getattr(t.function, "name", "") or ""
    elif isinstance(t, dict) and "function" in t:
        name = (t["function"] or {}).get("name", "")
    if not name:
        name = getattr(t, "name", "") or (t.get("name") if isinstance(t, dict) else "") or "unknown"
    return f"{tt}:{name}"


def _foundry_tool_to_spec(t: Any) -> tuple[str, str, str, dict]:
    """Extract (name, description, type, spec) from Foundry tool."""
    tt = getattr(t, "type", None) or (t.get("type") if isinstance(t, dict) else None) or "function"
    name = desc = ""
    params: dict = {}
    if hasattr(t, "function") and t.function:
        fn = t.function
        name = getattr(fn, "name", "") or ""
        desc = getattr(fn, "description", "") or ""
        params = getattr(fn, "parameters", None) or {}
        if hasattr(params, "model_dump"):
            params = params.model_dump()
        elif not isinstance(params, dict):
            params = {}
    elif isinstance(t, dict) and "function" in t:
        fn = t["function"] or {}
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = fn.get("parameters") or {}
    spec = {"parameters": params}
    return name, desc, tt, spec


def _tool_def_from_db(t: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build Foundry ToolDefinition dict from our DB tool.
    For code-defined tools (source="code"), includes webhook endpoint for execution.
    """
    tt = t.get("type") or "function"
    name = t.get("name") or ""
    desc = t.get("description") or ""
    spec = t.get("spec") or {}
    params = spec.get("parameters")
    if params is None:
        params = {"type": "object", "properties": {}}
    
    # Build base tool definition
    tool_def = {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": params
        }
    }
    
    # Check if this is a code-defined tool that needs webhook
    source = t.get("source")
    if source == "code":
        # For code-defined tools, add webhook endpoint configuration
        import os
        
        # Get backend URL from environment
        backend_url = (
            os.getenv("AGENT_BACKEND_URL") or
            os.getenv("NEXT_PUBLIC_AGENT_URL") or
            os.getenv("BACKEND_URL") or
            "http://localhost:8787"  # Default fallback
        )
        backend_url = backend_url.rstrip("/")
        webhook_url = f"{backend_url}/api/tools/webhook/execute"
        
        logger.info(f"[TOOLS_SERVICE] Registering code tool '{name}' with webhook: {webhook_url}")
        
        # Add endpoint to tool definition
        # Foundry may recognize this in different formats depending on SDK version
        # Try adding it as endpoint in the function definition
        tool_def["function"]["endpoint"] = webhook_url
        
        # Also try adding at top level (some Foundry versions may look here)
        tool_def["endpoint"] = webhook_url
    
    return tool_def


def sync_foundry_agent_tools(agent_id: str) -> List[str]:
    """
    Fetch agent from Foundry, sync its tools to DB (create missing), replace agent-tool relations.
    Returns list of our tool ids assigned to this agent.
    """
    storage = get_tools_storage()
    if not storage._ensure():
        return []
    agent = get_foundry_agent(agent_id)
    raw = agent.get("tools") or []
    if not isinstance(raw, list):
        raw = []
    raw = list(raw)
    assigned: List[str] = []
    for r in raw:
        try:
            ref = _foundry_tool_to_ref(r)
            name, desc, tt, spec = _foundry_tool_to_spec(r)
            existing = storage.find_tool_by_foundry_ref(ref)
            if existing:
                tid = existing["id"]
            else:
                t = storage.upsert_tool_by_foundry_ref(ref, name, desc, tt, spec)
                tid = t["id"]
            assigned.append(tid)
        except Exception as e:
            logger.warning("Failed to sync Foundry tool to DB: %s", e)
    storage.set_agent_tools(agent_id, assigned)
    return assigned


def list_tools_for_agent(agent_id: str) -> Dict[str, Any]:
    """
    Sync Foundry -> DB, merge duplicates, then return { tools: [...], assigned_ids: [...] }.
    tools = all tools from DB (deduplicated); assigned_ids = ids assigned to this agent.
    Uses list_agent_tool_ids AFTER merge so assigned_ids never references deleted duplicates.
    """
    storage = get_tools_storage()
    if not storage._ensure():
        return {"tools": [], "assigned_ids": []}
    sync_foundry_agent_tools(agent_id)
    storage.merge_duplicate_tools()
    tools = storage.list_tools()
    assigned = storage.list_agent_tool_ids(agent_id)
    return {"tools": tools, "assigned_ids": assigned}


def assign_tools_to_agent(agent_id: str, tool_ids: List[str]) -> None:
    """
    Update agent in Foundry with tool definitions from DB, then set agent-tool relations in DB.
    """
    storage = get_tools_storage()
    if not storage._ensure():
        raise HTTPException(status_code=503, detail="Tools storage not available")
    defs: List[Dict[str, Any]] = []
    for tid in tool_ids:
        t = storage.get_tool(tid)
        if not t:
            continue
        defs.append(_tool_def_from_db(t))
    update_foundry_agent(agent_id, tools=[d for d in defs])
    storage.set_agent_tools(agent_id, tool_ids)


def delete_tool_and_detach(tool_id: str) -> bool:
    """
    Remove tool from DB and from all agents that had it.
    For each affected agent, update Foundry to drop this tool, then delete agent_tool rows and the tool.
    """
    storage = get_tools_storage()
    if not storage._ensure():
        raise HTTPException(status_code=503, detail="Tools storage not available")
    if not storage.get_tool(tool_id):
        return False
    agent_ids = storage.list_agent_ids_for_tool(tool_id)
    for aid in agent_ids:
        try:
            current = storage.list_agent_tool_ids(aid)
            rest = [tid for tid in current if tid != tool_id]
            assign_tools_to_agent(aid, rest)
        except Exception as e:
            logger.warning("Failed to detach tool %s from agent %s: %s", tool_id, aid, e)
    return storage.delete_tool(tool_id)
