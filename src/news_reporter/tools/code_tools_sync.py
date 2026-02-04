"""Code Tools Synchronization - Sync code-defined tools to MongoDB on startup"""

from __future__ import annotations
import logging
import hashlib
import json
from typing import Dict, Any, List

from ..workflows.tools_storage import get_tools_storage
from .code_tools_registry import get_registered_code_tools

logger = logging.getLogger(__name__)


def _generate_code_tool_id(name: str, code_location: str) -> str:
    """Generate a stable ID for a code tool based on its name and location."""
    # Use hash of name + location for stable ID
    content = f"code:{name}:{code_location}"
    return hashlib.md5(content.encode()).hexdigest()


def _build_tool_spec_from_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Build tool spec dict from code tool metadata."""
    return {
        "parameters": metadata.get("parameters", {}),
        "return_type": metadata.get("return_type", {}),
    }


def _compute_tool_hash(metadata: Dict[str, Any]) -> str:
    """Compute hash of tool metadata to detect changes."""
    # Hash name, description, and parameters schema
    content = json.dumps({
        "name": metadata.get("name"),
        "description": metadata.get("description"),
        "parameters": metadata.get("parameters", {}),
        "code_location": metadata.get("code_location"),
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def sync_code_tools_to_storage() -> Dict[str, Any]:
    """
    Synchronize all registered code tools to MongoDB storage.
    
    Returns:
        Dict with:
        - created: List of tool names that were created
        - updated: List of tool names that were updated
        - skipped: List of tool names that were skipped (already up to date)
        - errors: List of (tool_name, error) tuples
    """
    storage = get_tools_storage()
    if not storage._ensure():
        logger.warning("Tools storage not available, skipping code tools sync")
        return {
            "created": [],
            "updated": [],
            "skipped": [],
            "errors": [("all", "Tools storage not available")],
        }
    
    registered_tools = get_registered_code_tools()
    if not registered_tools:
        logger.info("No code tools registered, nothing to sync")
        return {
            "created": [],
            "updated": [],
            "skipped": [],
            "errors": [],
        }
    
    logger.info(f"Syncing {len(registered_tools)} code tools to storage...")
    
    created = []
    updated = []
    skipped = []
    errors = []
    
    for tool_name, metadata in registered_tools.items():
        try:
            code_location = metadata.get("code_location", f"unknown.{tool_name}")
            current_hash = _compute_tool_hash(metadata)
            
            # Check if tool already exists by name and source
            existing = storage.find_tool_by_name_and_type(tool_name, "function")
            
            # Filter to only code tools
            if existing and existing.get("source") == "code":
                # Check if tool has changed by comparing hash
                existing_hash = existing.get("spec", {}).get("_code_hash")
                if existing_hash == current_hash:
                    # Tool unchanged, skip
                    skipped.append(tool_name)
                    continue
                
                # Tool exists but changed, update it
                spec = _build_tool_spec_from_metadata(metadata)
                spec["_code_hash"] = current_hash  # Store hash for change detection
                spec["code_location"] = code_location  # Store location for tracking
                
                updated_tool = storage.update_tool(
                    tool_id=existing["id"],
                    name=tool_name,
                    description=metadata.get("description", ""),
                    tool_type="function",
                    spec=spec,
                )
                
                if updated_tool:
                    logger.info(f"Updated code tool: {tool_name} ({existing['id']})")
                    updated.append(tool_name)
                else:
                    logger.warning(f"Failed to update code tool: {tool_name}")
                    errors.append((tool_name, "Update failed"))
            elif existing and existing.get("source") != "code":
                # Tool exists but from different source, skip to avoid conflicts
                logger.warning(
                    f"Tool '{tool_name}' exists with source '{existing.get('source')}', "
                    f"skipping code tool registration"
                )
                skipped.append(tool_name)
                continue
            else:
                # New tool, create it
                spec = _build_tool_spec_from_metadata(metadata)
                spec["_code_hash"] = current_hash  # Store hash for change detection
                spec["code_location"] = code_location  # Store location for tracking
                
                new_tool = storage.create_tool(
                    name=tool_name,
                    description=metadata.get("description", ""),
                    tool_type="function",
                    spec=spec,
                    source="code",
                    foundry_ref=None,  # Code tools don't have Foundry ref initially
                )
                
                if new_tool:
                    logger.info(f"Created code tool: {tool_name} ({new_tool['id']})")
                    created.append(tool_name)
                else:
                    errors.append((tool_name, "Create failed"))
                    
        except Exception as e:
            logger.exception(f"Error syncing code tool '{tool_name}': {e}")
            errors.append((tool_name, str(e)))
    
    logger.info(
        f"Code tools sync complete: {len(created)} created, {len(updated)} updated, "
        f"{len(skipped)} skipped, {len(errors)} errors"
    )
    
    return {
        "created": created,
        "updated": updated,
        "skipped": skipped,
        "errors": errors,
    }


def sync_code_tools_on_startup():
    """Convenience function to sync code tools on server startup."""
    try:
        result = sync_code_tools_to_storage()
        if result["errors"]:
            logger.warning(f"Code tools sync completed with {len(result['errors'])} errors")
        return result
    except Exception as e:
        logger.exception(f"Failed to sync code tools on startup: {e}")
        return {
            "created": [],
            "updated": [],
            "skipped": [],
            "errors": [("startup", str(e))],
        }
