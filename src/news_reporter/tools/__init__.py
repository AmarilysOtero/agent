"""Tools package - Import all code-defined tools to register them"""

# Import code tools registry first
from .code_tools_registry import (
    register_code_tool,
    get_registered_code_tools,
    get_code_tool,
    list_code_tools,
)

# Import tool modules to trigger registration
# This ensures all @register_code_tool decorated functions are registered
try:
    # Import SQL tools
    from ..tools_sql.text_to_sql_tool import query_database  # noqa: F401
except ImportError:
    pass

try:
    # Import CSV schema tools
    from .csv_schema_tool import get_file_schema  # noqa: F401
except ImportError:
    pass

try:
    # Import translation tools
    from .translation_tool import translate_text  # noqa: F401
except ImportError:
    pass

__all__ = [
    "register_code_tool",
    "get_registered_code_tools",
    "get_code_tool",
    "list_code_tools",
]
