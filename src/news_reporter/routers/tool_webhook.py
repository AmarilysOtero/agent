"""Webhook handler for executing code-defined tools from Foundry

When Foundry agents call code-defined tools, they need a webhook endpoint
to execute the Python function and return the result.
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, Optional
import logging
import json

from ..tools.code_tools_registry import get_code_tool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tools/webhook", tags=["tools"])


@router.post("/execute")
async def execute_tool_webhook(
    request: Dict[str, Any] = Body(...)
):
    """
    Webhook endpoint for Foundry to execute code-defined tools.
    
    Expected request format from Foundry:
    {
        "tool_name": "translate_text",
        "parameters": {
            "text": "Hello",
            "target_language": "es"
        }
    }
    
    Returns:
    {
        "result": "<tool execution result>",
        "success": true/false,
        "error": "<error message if failed>"
    }
    """
    try:
        # Extract tool name and parameters from request
        tool_name = request.get("tool_name") or request.get("name") or request.get("function_name")
        if not tool_name:
            raise HTTPException(
                status_code=400,
                detail="Missing 'tool_name', 'name', or 'function_name' in request"
            )
        
        # Get parameters - Foundry may send them in different formats
        parameters = request.get("parameters") or request.get("arguments") or request.get("args") or {}
        if not isinstance(parameters, dict):
            raise HTTPException(
                status_code=400,
                detail="Parameters must be a dictionary/object"
            )
        
        logger.info(f"[TOOL_WEBHOOK] Executing tool '{tool_name}' with parameters: {parameters}")
        
        # Get the code tool from registry
        code_tool_metadata = get_code_tool(tool_name)
        if not code_tool_metadata:
            from ..tools.code_tools_registry import get_registered_code_tools
            available_tools = list(get_registered_code_tools().keys())
            logger.error(f"[TOOL_WEBHOOK] Tool '{tool_name}' not found in code tools registry. Available: {available_tools}")
            raise HTTPException(
                status_code=404,
                detail=f"Tool '{tool_name}' not found in code tools registry. Available tools: {available_tools}"
            )
        
        # Get the function
        func = code_tool_metadata.get("function")
        if not func or not callable(func):
            logger.error(f"[TOOL_WEBHOOK] Tool '{tool_name}' function is not callable")
            raise HTTPException(
                status_code=500,
                detail=f"Tool '{tool_name}' function is not callable"
            )
        
        # Execute the function
        try:
            logger.info(f"[TOOL_WEBHOOK] Calling function '{tool_name}' with args: {parameters}")
            result = func(**parameters)
            
            # If result is already a string (JSON), return it as-is
            # Otherwise, convert to JSON string
            if isinstance(result, str):
                result_str = result
            else:
                result_str = json.dumps(result, indent=2)
            
            logger.info(f"[TOOL_WEBHOOK] Tool '{tool_name}' executed successfully")
            
            return {
                "result": result_str,
                "success": True,
                "tool_name": tool_name
            }
            
        except TypeError as e:
            # Function signature mismatch
            logger.exception(f"[TOOL_WEBHOOK] Error calling tool '{tool_name}': {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid parameters for tool '{tool_name}': {str(e)}"
            )
        except Exception as e:
            logger.exception(f"[TOOL_WEBHOOK] Error executing tool '{tool_name}': {e}")
            return {
                "result": "",
                "success": False,
                "error": str(e),
                "tool_name": tool_name
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[TOOL_WEBHOOK] Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/health")
async def webhook_health():
    """Health check endpoint for the webhook"""
    return {
        "status": "healthy",
        "endpoint": "/api/tools/webhook/execute"
    }
