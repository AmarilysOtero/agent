"""Code Tools Registry - Register Python functions as Foundry tools

This module provides a decorator and registration system for tools defined directly in code.
Tools registered here will be automatically synchronized to MongoDB on server startup.
"""

from __future__ import annotations
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional
import json

logger = logging.getLogger(__name__)

# Global registry of code-defined tools
_CODE_TOOLS_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _extract_json_schema_from_type(annotation: Any) -> Dict[str, Any]:
    """Convert Python type annotation to JSON Schema type."""
    if annotation is None or annotation == inspect.Parameter.empty:
        return {"type": "string"}  # Default to string
    
    # Handle typing types
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", None)
    
    # str
    if annotation is str or annotation == str:
        return {"type": "string"}
    # int
    if annotation is int or annotation == int:
        return {"type": "integer"}
    # float
    if annotation is float or annotation == float:
        return {"type": "number"}
    # bool
    if annotation is bool or annotation == bool:
        return {"type": "boolean"}
    # list/List
    if origin is list or (hasattr(annotation, "__origin__") and annotation.__origin__ is list):
        item_type = args[0] if args else str
        return {"type": "array", "items": _extract_json_schema_from_type(item_type)}
    # dict/Dict
    if origin is dict or (hasattr(annotation, "__origin__") and annotation.__origin__ is dict):
        return {"type": "object"}
    # Optional/Union
    if origin is type(None) or (hasattr(annotation, "__origin__") and annotation.__origin__ is type(None)):
        # Extract the non-None type
        non_none_types = [arg for arg in args if arg is not type(None)] if args else [str]
        if non_none_types:
            return _extract_json_schema_from_type(non_none_types[0])
    
    # Default to string for unknown types
    return {"type": "string"}


def _extract_function_metadata(func: Callable) -> Dict[str, Any]:
    """Extract metadata from a function for tool registration."""
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    
    # Extract description from docstring (first paragraph)
    description = doc.split("\n\n")[0].strip() if doc else ""
    
    # Extract parameters
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue  # Skip self parameter
        
        param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
        param_schema = _extract_json_schema_from_type(param_type)
        
        # Check if parameter has a default value
        has_default = param.default != inspect.Parameter.empty
        
        properties[param_name] = param_schema
        
        if not has_default:
            required.append(param_name)
    
    # Build JSON Schema for parameters
    parameters_schema = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters_schema["required"] = required
    
    # Get return type
    return_type = sig.return_annotation
    if return_type != inspect.Signature.empty:
        return_schema = _extract_json_schema_from_type(return_type)
    else:
        return_schema = {"type": "string"}
    
    # Get module and function name for tracking
    module_name = func.__module__ if hasattr(func, "__module__") else "unknown"
    function_name = func.__name__
    code_location = f"{module_name}.{function_name}"
    
    return {
        "name": function_name,
        "description": description,
        "parameters": parameters_schema,
        "return_type": return_schema,
        "code_location": code_location,
        "function": func,  # Store reference to actual function
    }


def register_code_tool(func: Optional[Callable] = None, name: Optional[str] = None, description: Optional[str] = None):
    """
    Decorator to register a Python function as a code-defined tool.
    
    Usage:
        @register_code_tool
        def my_tool(param1: str, param2: int) -> str:
            \"\"\"Tool description\"\"\"
            return result
        
        # Or with custom name/description:
        @register_code_tool(name="custom_name", description="Custom description")
        def my_tool(...):
            ...
    """
    def decorator(f: Callable) -> Callable:
        # Extract metadata
        metadata = _extract_function_metadata(f)
        
        # Override with custom values if provided
        if name:
            metadata["name"] = name
        if description:
            metadata["description"] = description
        
        tool_name = metadata["name"]
        
        # Register in global registry
        _CODE_TOOLS_REGISTRY[tool_name] = metadata
        
        logger.info(f"Registered code tool: {tool_name} from {metadata['code_location']}")
        
        return f
    
    if func is None:
        # Called as @register_code_tool() or @register_code_tool(name=..., description=...)
        return decorator
    else:
        # Called as @register_code_tool directly
        return decorator(func)


def get_registered_code_tools() -> Dict[str, Dict[str, Any]]:
    """Get all registered code tools."""
    return _CODE_TOOLS_REGISTRY.copy()


def get_code_tool(name: str) -> Optional[Dict[str, Any]]:
    """Get a specific code tool by name."""
    return _CODE_TOOLS_REGISTRY.get(name)


def list_code_tools() -> List[Dict[str, Any]]:
    """List all registered code tools as a list."""
    return list(_CODE_TOOLS_REGISTRY.values())


def clear_registry():
    """Clear the registry (mainly for testing)."""
    global _CODE_TOOLS_REGISTRY
    _CODE_TOOLS_REGISTRY.clear()
