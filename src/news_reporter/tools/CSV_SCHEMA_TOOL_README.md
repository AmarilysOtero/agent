# CSV/Excel Schema Tool Documentation

## Overview

The CSV/Excel Schema tool retrieves schema information (column names, data types, sample values) from CSV or Excel files. This tool is designed to be used as a Foundry agent tool, enabling ChatGPT and other Foundry agents to understand the structure of uploaded data files.

## Features

- **Multi-Format Support**: Works with CSV (`.csv`), Excel (`.xlsx`, `.xls`) files
- **Schema Extraction**: Retrieves column names, data types, and sample values
- **Automatic Detection**: Automatically detects file format based on extension
- **Error Handling**: Provides detailed error messages for invalid files or missing files

## Architecture

```
File Path → Format Detection → Backend API Call → Schema Extraction → Formatted Results
```

### Components

1. **get_file_schema()**: Foundry tool function wrapper
2. **CSVQueryTool**: Core tool that handles schema extraction
3. **Backend API**: Processes files via `/api/v1/csv/columns` endpoint

## Usage

### As a Foundry Tool

The tool is registered with Foundry agents and can be called automatically:

```python
from src.news_reporter.tools.csv_schema_tool import get_file_schema

# This function is automatically called by Foundry agents
result = get_file_schema("path/to/file.csv")
```

### Direct Usage (Python)

```python
from src.news_reporter.tools.csv_query import CSVQueryTool

tool = CSVQueryTool()
result = tool.get_column_info("path/to/inventory.xlsx")

print(result)
# {
#     "columns": {
#         "id": {"dtype": "int64", "sample_values": [1, 2, 3]},
#         "name": {"dtype": "object", "sample_values": ["Item A", "Item B"]},
#         "quantity": {"dtype": "int64", "sample_values": [10, 20]},
#         "price": {"dtype": "float64", "sample_values": [9.99, 19.99]}
#     },
#     "file_path": "path/to/inventory.xlsx"
# }
```

## Function Signature

### `get_file_schema(file_path: str) -> str`

**Parameters:**

- `file_path` (str, required): Path to the CSV or Excel file

**Returns:**

- JSON string containing:
  - `columns`: Dictionary mapping column names to metadata:
    - `dtype`: Data type (e.g., "int64", "object", "float64")
    - `sample_values`: Sample values from the column
  - `file_path`: Path to the file
  - `error`: Error message (if operation failed)

## Example Usage

### CSV File

**Input:**

```python
get_file_schema("data/employees.csv")
```

**Output:**

```json
{
	"columns": {
		"id": {
			"dtype": "int64",
			"sample_values": [1, 2, 3]
		},
		"name": {
			"dtype": "object",
			"sample_values": ["John Doe", "Jane Smith"]
		},
		"email": {
			"dtype": "object",
			"sample_values": ["john@example.com", "jane@example.com"]
		}
	},
	"file_path": "data/employees.csv"
}
```

### Excel File

**Input:**

```python
get_file_schema("data/inventory.xlsx")
```

**Output:**

```json
{
	"columns": {
		"product_id": {
			"dtype": "int64",
			"sample_values": [1001, 1002, 1003]
		},
		"product_name": {
			"dtype": "object",
			"sample_values": ["Widget A", "Widget B"]
		},
		"quantity": {
			"dtype": "int64",
			"sample_values": [50, 75, 100]
		},
		"price": {
			"dtype": "float64",
			"sample_values": [9.99, 19.99, 29.99]
		}
	},
	"file_path": "data/inventory.xlsx"
}
```

### Error Handling

**Input:**

```python
get_file_schema("nonexistent.csv")
```

**Output:**

```json
{
	"columns": {},
	"file_path": "nonexistent.csv",
	"error": "Failed to get schema: File not found: nonexistent.csv"
}
```

## Supported File Formats

- **CSV**: `.csv` files (comma-separated values)
- **Excel**: `.xlsx` (Excel 2007+), `.xls` (Excel 97-2003)

## Tool Registration

### Manual Registration via Azure AI Studio

1. Go to Azure AI Foundry Studio → Your Hub → Project → Agents
2. Select the agent (e.g., `AiSearchAgent`, `TriageAgent`)
3. Click "Edit" or go to "Tools" section
4. Click "Add Tool" or "Add Function"
5. Add the following function:

**Function Definition:**

```json
{
	"name": "get_file_schema",
	"description": "Gets schema information (columns, types, sample values) from CSV or Excel files",
	"parameters": {
		"type": "object",
		"properties": {
			"file_path": {
				"type": "string",
				"description": "Path to the CSV or Excel file"
			}
		},
		"required": ["file_path"]
	}
}
```

6. Enable "Automatic function calling" or "Auto tool selection"
7. Save the agent

### Programmatic Registration

Use the registration script:

```bash
python -m src.news_reporter.tools.register_foundry_tools
```

Or use the helper script to generate tool definitions:

```bash
python -m src.news_reporter.tools.generate_tool_definitions
```

## Integration with Chat

Once registered, the tool is automatically available in Chat conversations:

**User:** "What columns are in the inventory.csv file?"

**Agent Response:**

- Agent recognizes the need for file schema information
- Automatically calls `get_file_schema("inventory.csv")`
- Returns formatted results: "The inventory.csv file has the following columns: product_id (integer), product_name (text), quantity (integer), price (decimal)..."

## Backend API

The tool uses the neo4j_backend API endpoint:

- `POST /api/v1/csv/columns`: Retrieves column information from CSV/Excel files

**Request:**

```json
{
	"file_path": "path/to/file.csv"
}
```

**Response:**

```json
{
  "columns": {
    "column_name": {
      "dtype": "data_type",
      "sample_values": [value1, value2, ...]
    }
  },
  "file_path": "path/to/file.csv"
}
```

## Error Handling

The tool handles various error scenarios:

1. **File Not Found**: Returns error message with file path
2. **Invalid File Format**: Handles unsupported file types
3. **Parsing Errors**: Handles corrupted or malformed files
4. **Backend API Errors**: Handles connection errors and API failures

## Dependencies

- `pandas`: For CSV/Excel file processing
- `openpyxl`: For Excel file support (`.xlsx`)
- `xlrd`: For legacy Excel file support (`.xls`)
- Neo4j backend API: For file processing

## Testing

Run unit tests:

```bash
pytest tests/unit/tools/test_csv_schema_tool.py
```

Run integration tests:

```bash
pytest tests/integration/tools/test_csv_schema_integration.py
```

## Troubleshooting

### "File not found"

- Verify the file path is correct
- Check file permissions
- Ensure the file exists in the expected location

### "Unsupported file format"

- Verify file extension is `.csv`, `.xlsx`, or `.xls`
- Check if file is corrupted
- Try opening the file in a spreadsheet application

### "Backend API error"

- Ensure neo4j_backend service is running
- Check backend API logs for errors
- Verify file path is accessible from backend

## Use Cases

1. **Data Exploration**: Understand structure of uploaded files before querying
2. **Schema Validation**: Verify file structure matches expectations
3. **Query Planning**: Use schema information to generate better SQL queries
4. **Data Quality Checks**: Identify data types and sample values for validation

## Related Documentation

- [Text-to-SQL Tool README](../tools_sql/TEXT_TO_SQL_README.md)
- [Foundry Tools README](./FOUNDRY_TOOLS_README.md)
- [CSV Query Tool Documentation](./csv_query.py)






