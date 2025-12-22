"""CSV/Excel Schema Tool for Foundry agent registration

Provides a Foundry tool function to get schema information (columns, types, sample values)
from CSV or Excel files.
"""

from __future__ import annotations
from typing import Dict, Any
import logging
import json

try:
    from .csv_query import CSVQueryTool
except ImportError:
    import sys
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root.parent))
    from src.news_reporter.tools.csv_query import CSVQueryTool

logger = logging.getLogger(__name__)


def get_file_schema(file_path: str) -> str:
    """
    Foundry tool function for getting CSV/Excel file schema.
    
    Gets schema information (columns, types, sample values) from a CSV or Excel file.
    This function is registered with Foundry agents as a callable tool.
    
    Args:
        file_path: Path to the CSV or Excel file (.csv, .xlsx, .xls)
    
    Returns:
        JSON string with column information including:
        - columns: Dictionary mapping column names to their metadata (dtype, sample_values, etc.)
        - file_path: Path to the file
        - error: Error message if operation failed (optional)
    """
    try:
        tool = CSVQueryTool()
        result = tool.get_column_info(file_path)
        
        # Format as JSON string for Foundry
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.exception(f"Error getting file schema for {file_path}: {e}")
        error_result = {
            "columns": {},
            "file_path": file_path,
            "error": f"Failed to get schema: {str(e)}"
        }
        return json.dumps(error_result, indent=2)






