# Foundry Tools Registration Guide

This guide explains how to register text-to-SQL and CSV/Excel schema tools with Foundry agents.

## Available Tools

### 1. `query_database(natural_language_query: str, database_id: str) -> str`

Converts natural language to SQL, executes it, and returns results.

**Example usage in Chat:**

- User: "list the names"
- Agent calls: `query_database("list the names", "database_id_123")`
- Returns: JSON with generated SQL, execution results, and metadata

### 2. `get_file_schema(file_path: str) -> str`

Gets schema information (columns, types, sample values) from CSV or Excel files.

**Example usage in Chat:**

- User: "What columns are in inventory.csv?"
- Agent calls: `get_file_schema("inventory.csv")`
- Returns: JSON with column information

## Registration Methods

### Method 1: Automated Script (Recommended)

Run the registration script:

```bash
cd Agent
python -m src.news_reporter.tools.register_foundry_tools
```

**Prerequisites:**

- `AI_PROJECT_CONNECTION_STRING` set in `.env`
- Agent IDs set in `.env`:
  - `AGENT_ID_TRIAGE`
  - `AGENT_ID_AISEARCH`
  - `AGENT_ID_NEO4J_SEARCH` (optional)

**What it does:**

- Creates `FunctionTool` with both tools
- Creates `ToolSet` and adds the tools
- Updates specified agents with the toolset
- Enables automatic function calling

### Method 2: Manual Registration via Azure AI Studio

If the automated script doesn't work (SDK limitations), register tools manually:

1. **Go to Azure AI Foundry Studio**

   - Navigate to your Hub → Project → Agents

2. **Edit each agent** (AiSearchAgent, TriageAgent, etc.)

   - Click on the agent name
   - Go to "Tools" or "Functions" section
   - Click "Add Tool" or "Add Function"

3. **Add the following functions:**

   **Function 1: query_database**

   - Name: `query_database`
   - Description: "Converts natural language to SQL, executes it, and returns results"
   - Parameters:
     - `natural_language_query` (string, required): Natural language query (e.g., "list the names")
     - `database_id` (string, required): Database configuration ID
   - Returns: JSON string with SQL query, execution results, and metadata

   **Function 2: get_file_schema**

   - Name: `get_file_schema`
   - Description: "Gets schema information from CSV or Excel files"
   - Parameters:
     - `file_path` (string, required): Path to the CSV or Excel file
   - Returns: JSON string with column information

4. **Enable automatic function calling**
   - In agent settings, enable "Auto function calling" or "Automatic tool selection"

## Verification

After registration, test the tools in Chat:

1. **Test text-to-SQL:**

   ```
   User: "list the names from the Employee table"
   Expected: Agent calls query_database() and returns results
   ```

2. **Test CSV schema:**
   ```
   User: "What columns are in the inventory.csv file?"
   Expected: Agent calls get_file_schema() and returns column info
   ```

## Troubleshooting

### Tools not available in Chat

1. **Check agent configuration:**

   - Verify tools are registered with the agent
   - Ensure automatic function calling is enabled

2. **Check function signatures:**

   - Function names must match exactly: `query_database`, `get_file_schema`
   - Parameter names and types must match

3. **Check backend connectivity:**
   - Ensure `NEO4J_BACKEND_URL` is set correctly
   - Verify backend API is accessible

### SDK version issues

If `FunctionTool` or `ToolSet` are not available:

- Check Azure AI Projects SDK version: `pip show azure-ai-projects`
- Update if needed: `pip install --upgrade azure-ai-projects`
- Use manual registration method if SDK doesn't support tools yet

### Agent update fails

If automated registration fails:

- Check agent IDs are correct in `.env`
- Verify you have permissions to update agents
- Use manual registration via Azure AI Studio

## Tool Implementation Details

### Text-to-SQL Tool (`query_database`)

**Implementation:** `src/news_reporter/tools_sql/text_to_sql_tool.py`

**Flow:**

1. Receives natural language query
2. Uses `SQLGenerator` to generate SQL
3. Executes SQL via neo4j_backend API
4. Returns results with SQL query and data

**Dependencies:**

- `SQLGenerator` (for SQL generation)
- neo4j_backend API (for SQL execution)
- Neo4j (for schema metadata)

### CSV/Excel Schema Tool (`get_file_schema`)

**Implementation:** `src/news_reporter/tools/csv_schema_tool.py`

**Flow:**

1. Receives file path
2. Calls `CSVQueryTool.get_column_info()`
3. Returns column information as JSON

**Dependencies:**

- `CSVQueryTool` (for column info retrieval)
- neo4j_backend API `/api/v1/csv/columns`

## Support

For issues or questions:

1. Check logs in `register_foundry_tools.py` output
2. Verify environment variables are set correctly
3. Test backend API endpoints directly
4. Check Azure AI Studio for agent configuration
