"""Generate tool definitions for manual registration in Azure AI Studio

This script generates the exact tool definitions needed to manually register
tools in Azure AI Foundry Studio.
"""

import json
import inspect
from typing import Dict, Any

try:
    from ..tools_sql.text_to_sql_tool import query_database
    from .csv_schema_tool import get_file_schema
except ImportError:
    import sys
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root.parent))
    from src.news_reporter.tools_sql.text_to_sql_tool import query_database
    from src.news_reporter.tools.csv_schema_tool import get_file_schema


def get_function_schema(func) -> Dict[str, Any]:
    """Extract function schema for tool definition"""
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    
    # Parse parameters
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    for param_name, param in sig.parameters.items():
        param_type = "string"  # Default
        if param.annotation != inspect.Parameter.empty:
            ann_str = str(param.annotation)
            if "str" in ann_str:
                param_type = "string"
            elif "int" in ann_str:
                param_type = "integer"
            elif "float" in ann_str or "number" in ann_str:
                param_type = "number"
            elif "bool" in ann_str:
                param_type = "boolean"
        
        parameters["properties"][param_name] = {
            "type": param_type,
            "description": f"Parameter {param_name}"
        }
        
        if param.default == inspect.Parameter.empty:
            parameters["required"].append(param_name)
    
    return {
        "name": func.__name__,
        "description": doc.split("\n")[0] if doc else f"Tool function: {func.__name__}",
        "parameters": parameters
    }


def main():
    """Generate tool definitions"""
    print("=" * 60)
    print("Tool Definitions for Azure AI Foundry Studio")
    print("=" * 60)
    
    tools = [
        ("query_database", query_database),
        ("get_file_schema", get_file_schema),
    ]
    
    print("\n## Tool 1: query_database\n")
    schema1 = get_function_schema(query_database)
    print(json.dumps(schema1, indent=2))
    
    print("\n## Tool 2: get_file_schema\n")
    schema2 = get_function_schema(get_file_schema)
    print(json.dumps(schema2, indent=2))
    
    print("\n" + "=" * 60)
    print("Manual Registration Instructions")
    print("=" * 60)
    print("""
1. Go to Azure AI Foundry Studio
2. Navigate to: Your Hub → AgentFrameworkProject → Agents
3. For each agent (AiSearchAgent, TriageAgent, etc.):
   a. Click on the agent name
   b. Go to "Tools" or "Functions" section
   c. Click "Add Tool" or "Add Function"
   d. Add each tool using the definitions above

Alternative: Use the JSON definitions above to register via API or SDK
if your SDK version supports it.
    """)


if __name__ == "__main__":
    main()









