# Text-to-SQL Tool Documentation

## Overview

The Text-to-SQL tool converts natural language queries into SQL statements, executes them against a database, and returns structured results. This tool is designed to be used as a Foundry agent tool, enabling ChatGPT and other Foundry agents to query databases using natural language.

## Features

- **Natural Language to SQL**: Converts user queries like "list the names" into executable SQL
- **Automatic SQL Execution**: Executes generated SQL queries against the target database
- **Schema-Aware**: Retrieves relevant database schema from Neo4j to generate accurate SQL
- **Error Handling**: Provides detailed error messages for debugging
- **Confidence Scoring**: Returns confidence scores for generated SQL queries

## Architecture

```
User Query → SQL Generation → Schema Retrieval (Neo4j) → LLM SQL Generation → SQL Execution → Results
```

### Components

1. **TextToSQLTool**: Main tool class that orchestrates SQL generation and execution
2. **SQLGenerator**: Handles natural language to SQL conversion using LLM
3. **Backend API**: Executes SQL queries via `/api/databases/{db_id}/execute` endpoint

## Usage

### As a Foundry Tool

The tool is registered with Foundry agents and can be called automatically:

```python
from src.news_reporter.tools_sql.text_to_sql_tool import query_database

# This function is automatically called by Foundry agents
result = query_database(
    natural_language_query="list the names",
    database_id="your-database-id"
)
```

### Direct Usage (Python)

```python
from src.news_reporter.tools_sql.text_to_sql_tool import TextToSQLTool

tool = TextToSQLTool()
result = tool.query_database(
    natural_language_query="show me all employees",
    database_id="db-123",
    top_k=10,
    similarity_threshold=0.7
)

print(result)
# {
#     "success": True,
#     "generated_sql": "SELECT \"name\" FROM \"Employee\"",
#     "explanation": "This query retrieves all names from the Employee table",
#     "confidence": 0.9,
#     "results": {
#         "columns": ["name"],
#         "rows": [{"name": "Kevin"}, {"name": "Anthony"}],
#         "row_count": 2
#     },
#     "error": None
# }
```

## Function Signature

### `query_database(natural_language_query: str, database_id: str) -> str`

**Parameters:**

- `natural_language_query` (str, required): Natural language query (e.g., "list the names", "show me all employees")
- `database_id` (str, required): Database configuration ID stored in Neo4j

**Returns:**

- JSON string containing:
  - `success`: Boolean indicating if operation succeeded
  - `generated_sql`: The generated SQL query
  - `explanation`: Human-readable explanation of the SQL query
  - `confidence`: Confidence score (0.0 to 1.0)
  - `results`: Execution results (if successful):
    - `columns`: List of column names
    - `rows`: List of row dictionaries
    - `row_count`: Number of rows returned
  - `error`: Error message (if operation failed)

## Example Queries

### Simple SELECT Query

**Input:**

```python
query_database("list the names", "db-123")
```

**Generated SQL:**

```sql
SELECT "name" FROM "Employee"
```

**Output:**

```json
{
	"success": true,
	"generated_sql": "SELECT \"name\" FROM \"Employee\"",
	"explanation": "This query retrieves all names from the Employee table",
	"confidence": 0.9,
	"results": {
		"columns": ["name"],
		"rows": [{ "name": "Kevin" }, { "name": "Anthony" }],
		"row_count": 2
	},
	"error": null
}
```

### Query with Conditions

**Input:**

```python
query_database("show employees born after 1990", "db-123")
```

**Generated SQL:**

```sql
SELECT * FROM "Employee" WHERE "date_of_birth" > '1990-01-01'
```

### Error Handling

**Input:**

```python
query_database("list invalid table", "db-123")
```

**Output:**

```json
{
	"success": false,
	"generated_sql": null,
	"explanation": null,
	"confidence": 0.0,
	"results": null,
	"error": "No relevant schema elements found. Could not find relevant tables or columns for the query"
}
```

## Configuration

### Environment Variables

- `NEO4J_BACKEND_URL`: URL of the neo4j_backend API (default: `http://localhost:8000`)
- `AI_PROJECT_CONNECTION_STRING`: Azure AI Foundry connection string (for LLM calls)
- `AGENT_ID_AISEARCH`: Foundry agent ID for SQL generation (optional)

### Backend Requirements

The tool requires the neo4j_backend API to be running with the following endpoint:

- `POST /api/databases/{database_id}/execute`: Executes SQL queries
- `GET /api/v1/database/schema/{database_id}`: Retrieves database schema (used by SQLGenerator)

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
	"name": "query_database",
	"description": "Converts natural language to SQL, executes it, and returns results",
	"parameters": {
		"type": "object",
		"properties": {
			"natural_language_query": {
				"type": "string",
				"description": "Natural language query (e.g., 'list the names')"
			},
			"database_id": {
				"type": "string",
				"description": "Database configuration ID"
			}
		},
		"required": ["natural_language_query", "database_id"]
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

**User:** "list the names"

**Agent Response:**

- Agent recognizes the need for database query
- Automatically calls `query_database("list the names", database_id)`
- Returns formatted results: "Here are the names: Kevin, Anthony..."

## Error Handling

The tool handles various error scenarios:

1. **SQL Generation Failures**: Returns error message with details
2. **Backend API Errors**: Handles connection errors, timeouts, HTTP errors
3. **Database Errors**: Returns SQL execution errors from the database
4. **Schema Retrieval Failures**: Handles cases where schema cannot be retrieved

## Dependencies

- `requests`: For HTTP calls to backend API
- `azure-ai-projects`: For Foundry agent integration
- `azure-identity`: For Azure authentication
- Neo4j backend API: For schema retrieval and SQL execution

## Testing

Run unit tests:

```bash
pytest tests/unit/tools_sql/test_text_to_sql_tool.py
```

Run integration tests:

```bash
pytest tests/integration/tools_sql/test_text_to_sql_integration.py
```

## Troubleshooting

### "Backend API request timed out"

- Check if neo4j_backend is running
- Verify `NEO4J_BACKEND_URL` is correct
- Check network connectivity

### "No relevant schema elements found"

- Verify database_id exists in Neo4j
- Check if schema metadata is properly stored
- Try adjusting `similarity_threshold` parameter

### "Could not connect to backend API"

- Ensure neo4j_backend service is running
- Check firewall/network settings
- Verify backend URL in configuration

## Related Documentation

- [CSV Schema Tool README](../tools/CSV_SCHEMA_TOOL_README.md)
- [Foundry Tools README](../tools/FOUNDRY_TOOLS_README.md)
- [SQL Generator Documentation](./sql_generator.py)














