# Agent Flow - Code Implementation Trace

This document traces the actual code flow for chat sessions and agent interactions in the Agent service, based on the real implementation.

## Table of Contents

1. [Entry Point](#entry-point)
2. [Authentication](#authentication)
3. [Chat Session Management](#chat-session-management)
4. [Message Processing](#message-processing)
5. [Agent Workflow Execution](#agent-workflow-execution)
   - [Graph-Based Workflow Executor](#graph-based-workflow-executor)
     - [Graph Definition](#graph-definition)
     - [Node Types](#node-types)
     - [Execution Flow](#execution-flow)
   - [Sequential Workflow (Legacy)](#sequential-workflow-legacy)
     - [Triage Agent](#triage-agent)
     - [Search Agent Selection](#search-agent-selection)
     - [Search Execution](#search-execution)
     - [Reporter Agent](#reporter-agent)
     - [Review Agent](#review-agent)
6. [Neo4j GraphRAG Search](#neo4j-graphrag-search)
   - [Keyword Search Details](#keyword-search-details)
7. [Response Generation](#response-generation)
8. [Complete Code Flow Diagram](#complete-code-flow-diagram)

---

## Entry Point

### File: `routers/chat_sessions.py`

**Function**: `add_message(session_id: str, message: dict, user: dict)` (lines 373-545)

**Route**: `POST /api/chat/sessions/{session_id}/messages`

**Code Flow**:

```python
@router.post("/sessions/{session_id}/messages")
async def add_message(
    session_id: str,
    message: dict,
    user: dict = Depends(get_current_user)
):
    # Step 1: Authenticate user (via get_current_user dependency)
    # Step 2: Validate session exists and belongs to user
    # Step 3: Validate message content
    # Step 4: Insert user message into MongoDB
    # Step 5: Get sources from Neo4j (if enabled)
    # Step 6: Run agent workflow
    # Step 7: Insert assistant message with sources
    # Step 8: Update session timestamp
    # Step 9: Return response with sources
```

**Imports Used**:

- `from .auth import get_current_user`
- `from ..config import Settings`
- `from ..workflows.workflow_factory import run_graph_workflow, run_sequential_goal`
- `from ..tools.neo4j_graphrag import graphrag_search`

---

## Authentication

### Location: `routers/chat_sessions.py` → `add_message()` → `get_current_user` dependency

**File**: `routers/auth.py` (lines 278-295)

**Function**: `get_current_user(token: str = Depends(oauth2_scheme))`

**Process**:

1. **Extract Token** (from Authorization header):

   ```python
   # FastAPI automatically extracts Bearer token from Authorization header
   token: str = Depends(oauth2_scheme)
   ```

2. **Verify Token**:

   ```python
   # Decode and verify JWT token
   payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
   user_id: str = payload.get("sub")
   ```

3. **Get User from MongoDB**:

   ```python
   user = users_collection.find_one({"_id": ObjectId(user_id)})
   if not user:
       raise HTTPException(status_code=401, detail="User not found")
   ```

4. **Return User Object**:
   - Returns user dictionary with `_id`, `username`, `email`, etc.
   - Used throughout the request to identify the authenticated user

**Error Handling**:

- Invalid token → 401 Unauthorized
- User not found → 401 Unauthorized
- Missing token → 401 Unauthorized

---

## Chat Session Management

### Location: `routers/chat_sessions.py`

**MongoDB Collections**:

- `chat_sessions`: Stores chat session metadata
- `chat_messages`: Stores individual messages within sessions

**Session Structure**:

```python
{
    "_id": ObjectId("..."),
    "userId": "user_id_string",
    "title": "New Chat",
    "createdAt": datetime.utcnow(),
    "updatedAt": datetime.utcnow()
}
```

**Message Structure**:

```python
{
    "_id": ObjectId("..."),
    "sessionId": "session_id_string",
    "userId": "user_id_string",
    "role": "user" | "assistant",
    "content": "message text",
    "sources": [...],  # Optional, only for assistant messages
    "createdAt": datetime.utcnow()
}
```

**Step 1: Validate Session** (lines 387-394)

```python
# Verify session exists and belongs to user
session = sessions_collection.find_one({"_id": ObjectId(session_id), "userId": user_id})
if not session:
    raise HTTPException(status_code=404, detail="Session not found")
```

**Step 2: Validate Message Content** (lines 397-399)

```python
user_message_content = message.get("content", "")
if not user_message_content or not user_message_content.strip():
    raise HTTPException(status_code=400, detail="Message content required")
```

**Step 3: Insert User Message** (lines 403-411)

```python
user_message = {
    "sessionId": session_id,
    "userId": user_id,
    "role": "user",
    "content": user_message_content,
    "createdAt": now,
}
messages_collection.insert_one(user_message)
```

---

## Message Processing

### Location: `routers/chat_sessions.py` → `add_message()` (lines 413-472)

**Step 1: Load Settings**

```python
from ..config import Settings
cfg = Settings.load()
```

**Step 2: Get Sources from Neo4j** (if enabled)

```python
sources = []
if cfg.use_neo4j_search:
    # Extract person names from query for keyword filtering
    person_names = extract_person_names(user_message_content)

    # Search Neo4j GraphRAG
    search_results = graphrag_search(
        query=user_message_content,
        top_k=12,
        similarity_threshold=0.75,
        keywords=person_names if person_names else None,
        keyword_match_type="any",
        keyword_boost=0.4
    )

    # Filter results to require exact name match or very high similarity
    filtered_results = filter_results_by_exact_match(
        search_results,
        user_message_content,
        min_similarity=0.7
    )

    # Limit to top 8 after filtering
    filtered_results = filtered_results[:8]

    # Format sources
    sources = [
        {
            "file_name": res.get("file_name"),
            "file_path": res.get("file_path"),
            "directory_name": res.get("directory_name"),
            "text": res.get("text", "")[:500],
            "similarity": float(res.get("similarity", 0.0)),
            "hybrid_score": float(res.get("hybrid_score", 0.0)),
            "metadata": {...}
        }
        for res in filtered_results
    ]
```

**Person Name Extraction** (lines 23-39):

```python
def extract_person_names(query: str) -> List[str]:
    """Extract potential person names from query"""
    words = query.split()
    # Extract capitalized words (length > 2, starts with capital)
    names = [w.strip('.,!?;:') for w in words if w and w[0].isupper() and len(w.strip('.,!?;:')) > 2]
    # Remove common words
    common_words = {'The', 'This', 'That', 'What', 'When', 'Where', 'Who', 'Why', 'How', 'Tell', 'Show', 'Give', 'Find', 'Search', 'Get'}
    names = [n for n in names if n not in common_words]
    return names
```

**Result Filtering** (lines 42-102):

```python
def filter_results_by_exact_match(results: List[dict], query: str, min_similarity: float = 0.9) -> List[dict]:
    """Filter search results to require query name appears in chunk text or very high similarity"""
    # Extract person names
    names = extract_person_names(query)
    query_words = [n.lower() for n in names]

    # Get first name and last name
    first_name = query_words[0] if query_words else None
    last_name = query_words[-1] if len(query_words) > 1 else None

    filtered = []
    for res in results:
        text = res.get("text", "").lower()
        similarity = res.get("similarity", 0.0)

        # Apply absolute minimum similarity threshold
        if similarity < 0.3:
            continue

        # Check if first name appears in text
        first_name_found = first_name in text if first_name else True

        # If we have both first and last name, require both to match
        if first_name and last_name:
            last_name_found = last_name in text
            name_match = first_name_found and last_name_found
        else:
            name_match = first_name_found

        # Also check if file name contains the person's name
        file_name_lower = res.get("file_name", "").lower()
        file_contains_name = False
        if first_name and last_name:
            file_contains_name = (first_name in file_name_lower and last_name in file_name_lower) or (last_name in file_name_lower)
        elif first_name:
            file_contains_name = first_name in file_name_lower

        # Keep if: (name matches AND similarity >= 0.3) OR (file contains name AND similarity >= 0.4) OR similarity >= min_similarity
        if (name_match and similarity >= 0.3) or (file_contains_name and similarity >= 0.4) or similarity >= min_similarity:
            filtered.append(res)

    return filtered
```

---

## Agent Workflow Execution

### Location: `routers/chat_sessions.py` → `add_message()` → `workflows/workflow_factory.py`

The Agent service supports two workflow execution modes:

1. **Graph-Based Workflow** (Primary) - `run_graph_workflow()` - Uses a graph executor with nodes and edges
2. **Sequential Workflow** (Legacy/Fallback) - `run_sequential_goal()` - Original sequential execution

**Note**: Currently, `chat_sessions.py` uses `run_sequential_goal()`, but the graph executor is available and can be enabled.

---

## Graph-Based Workflow Executor

### Overview

The graph-based workflow executor (`GraphExecutor`) provides a flexible, declarative way to define and execute agent workflows using a graph structure with nodes and edges.

**File**: `workflows/graph_executor.py`

**Function**: `run_graph_workflow(cfg: Settings, goal: str, graph_path: str | None = None)` (lines 13-49)

### Graph Definition

Workflows are defined as JSON graphs with:

- **Nodes**: Represent agents, conditionals, loops, fanouts, and merges
- **Edges**: Define the flow between nodes with optional conditions
- **Entry Node**: Explicit entry point (`entry_node_id`)

**Default Graph**: `workflows/default_workflow.json`

**Graph Structure**:

```json
{
  "name": "Default News Reporter Workflow",
  "entry_node_id": "triage",
  "nodes": [...],
  "edges": [...],
  "limits": {
    "max_steps": 100,
    "timeout_ms": 300000,
    "max_iters": 3,
    "max_parallel": 10
  }
}
```

### Node Types

1. **Agent Node** (`type: "agent"`)
   - Executes a single agent (TriageAgent, SQLAgent, NewsReporterAgent, etc.)
   - Maps inputs/outputs to workflow state
   - Example: `triage`, `search_sql`, `report_branch`, `review`

2. **Conditional Node** (`type: "conditional"`)
   - Routes execution based on condition evaluation
   - Uses safe expression evaluator (no `eval()`)
   - Example: `select_search`, `should_search`

3. **Fanout Node** (`type: "fanout"`)
   - Executes multiple branches in parallel
   - Creates isolated execution contexts per branch
   - Example: `report_fanout` - runs reporter for each reporter_id

4. **Loop Node** (`type: "loop"`)
   - Iterative execution with max iterations
   - Re-enqueues body node until termination condition
   - Example: `review_loop` - reviews until accepted (max 3 passes)

5. **Merge Node** (`type: "merge"`)
   - Combines multiple inputs with explicit strategy
   - Acts as join barrier (waits for all branches)
   - Example: `stitch` - merges final outputs from all reporters

### Execution Flow

**Graph Workflow Flow** (from `default_workflow.json`):

```
TRIAGE → SELECT_SEARCH → [SQL | NEO4J | AISEARCH] → SHOULD_SEARCH →
REPORT_FANOUT → REVIEW_LOOP → [REVIEW → REPORTER_IMPROVE] → STITCH
```

**Step-by-Step**:

1. **TRIAGE Node**: Runs `TriageAgent`, writes to `state.triage`, `state.selected_search`, `state.database_id`

2. **SELECT_SEARCH Node** (Conditional): Routes based on `triage.preferred_agent`:
   - If `"sql"` → `search_sql` node
   - If Neo4j enabled → `search_neo4j` node
   - Otherwise → `search_aisearch` node

3. **SEARCH Nodes**: Execute selected search agent, write to `state.latest`

4. **SHOULD_SEARCH Node** (Conditional): Checks if search should run:
   - Condition: `"ai_search" in triage.intents or ...`

5. **REPORT_FANOUT Node**: Fan-out execution:
   - Creates branch per `reporter_id` from config
   - Each branch runs `report_branch` node in parallel
   - Writes to `state.drafts[reporter_id]`

6. **REVIEW_LOOP Node**: Loop with max 3 iterations:
   - Body: `review` → `reporter_improve` (if not accepted)
   - Terminates when: `verdicts[reporter_id][-1].decision == "accept"` OR `max_iters` reached
   - Writes to `state.verdicts[reporter_id]` and `state.final[reporter_id]`

7. **STITCH Node**: Merges all `state.final[reporter_id]` values using "stitch" strategy (markdown concatenation)

**Key Features**:

- **Queue-Based Execution**: Uses token/queue system instead of topological sort (supports cycles, conditionals, dynamic fanout)
- **Branch Isolation**: Each fanout branch gets isolated `ExecutionContext` to prevent output collisions
- **State Management**: `WorkflowState` tracks goal, triage, drafts, final, verdicts, logs, execution_trace
- **Condition Evaluation**: Safe parser-based evaluator (no `eval()`) for routing conditions
- **Metrics Collection**: Tracks performance metrics per node and overall workflow

**Code Location**: `workflows/graph_executor.py` → `GraphExecutor.execute(goal: str)`

**Fallback**: If graph execution fails, automatically falls back to `run_sequential_goal()`

---

## Sequential Workflow (Legacy)

### Location: `routers/chat_sessions.py` → `add_message()` → `workflows/workflow_factory.py`

**Function**: `run_sequential_goal(cfg: Settings, goal: str)` (lines 52-161)

**Note**: This is the legacy sequential workflow. The graph-based workflow (`run_graph_workflow`) is the primary method, but `chat_sessions.py` currently still uses this for backward compatibility.

**Workflow Flow**:

```
TRIAGE → AISEARCH → REPORTER → REVIEWER (≤3 passes)
```

**Step 1: Triage Agent** (lines 18-21)

```python
triage = TriageAgent(cfg.agent_id_triage)
tri = await triage.run(goal)
print("Triage:", tri.model_dump())
```

**TriageAgent Output** (`IntentResult` model):

```python
{
    "intents": List[str],  # e.g., ["ai_search", "news_script"]
    "preferred_agent": Optional[str],  # e.g., "sql", "neo4j", "azure_search"
    "database_id": Optional[str],  # Database ID if SQL query detected
    "database_type": Optional[str]  # Database type if detected
}
```

**Step 2: Determine Multi-Route** (lines 23-25)

```python
do_multi = ("multi" in tri.intents) or cfg.multi_route_always
targets: List[str] = cfg.reporter_ids if do_multi else [cfg.reporter_ids[0]]
```

**Step 3: Select Search Agent** (lines 27-53)

```python
# Check if TriageAgent detected SQL query
if tri.preferred_agent == "sql" and hasattr(cfg, 'agent_id_aisearch_sql') and cfg.agent_id_aisearch_sql:
    print(f"✅ Using SQL Agent (PostgreSQL → CSV → Vector) for database_id: {tri.database_id}")
    search_agent = SQLAgent(cfg.agent_id_aisearch_sql)
    search_database_id = tri.database_id
elif cfg.use_neo4j_search and cfg.agent_id_neo4j_search:
    print("Using Neo4j GraphRAG Agent (cost-efficient)")
    search_agent = Neo4jGraphRAGAgent(cfg.agent_id_neo4j_search)
else:
    print("Using Azure Search Agent (production)")
    search_agent = AiSearchAgent(cfg.agent_id_aisearch)
```

**Agent Selection Priority**:

1. **SQL Agent**: If `preferred_agent == "sql"` AND `agent_id_aisearch_sql` is configured
2. **Neo4j GraphRAG Agent**: If `use_neo4j_search == True` AND `agent_id_neo4j_search` is configured
3. **Azure Search Agent**: Default fallback

**Step 4: Initialize Reviewer** (line 54)

```python
reviewer = ReviewAgent(cfg.agent_id_reviewer)
```

**Step 5: Execute Workflow for Each Reporter** (lines 57-120)

```python
async def run_one(reporter_id: str) -> str:
    reporter = NewsReporterAgent(reporter_id)

    # AI Search step
    should_search = "ai_search" in tri.intents or (
        "unknown" in tri.intents and tri.preferred_agent and tri.database_id
    )
    if should_search:
        if isinstance(search_agent, SQLAgent):
            latest = await search_agent.run(goal, database_id=search_database_id)
        else:
            latest = await search_agent.run(goal)
    else:
        latest = ""

    # Reporter step
    script = (
        await reporter.run(goal, latest or "No ai-search content")
        if ("news_script" in tri.intents)
        else latest
    )

    if not script:
        return "No action taken."

    # Review step (max 3 passes)
    max_iters = 3
    for i in range(1, max_iters + 1):
        verdict = await reviewer.run(goal, script)
        decision = (verdict.get("decision") or "revise").lower()
        reason = verdict.get("reason", "")
        suggested = verdict.get("suggested_changes", "")
        revised = verdict.get("revised_script", script)

        if decision == "accept":
            return revised or script

        # Ask reporter to improve using reviewer notes
        improve_context = (
            f"Apply these review notes strictly:\n{suggested or reason}\n\n"
            f"Original draft:\n{script}"
        )
        script = await reporter.run(goal, improve_context)

    return f"[After {max_iters} review passes]\n{script}"
```

**Multi-Reporter Handling** (lines 113-118):

```python
if len(targets) > 1:
    results = await asyncio.gather(*[run_one(rid) for rid in targets])
    stitched = []
    for rid, out in zip(targets, results):
        stitched.append(f"### ReporterAgent={rid}\n{out}")
    return "\n\n---\n\n".join(stitched)
```

---

## Triage Agent

### Location: `agents/agents.py` → `TriageAgent` (lines 148-200)

**Class**: `TriageAgent`

**Function**: `run(self, goal: str) -> IntentResult`

**Process**:

1. **Call Foundry Agent**:

   ```python
   result = run_foundry_agent_json(
       agent_id=self.agent_id,
       goal=goal,
       thread_id=thread_id
   )
   ```

2. **Parse JSON Response**:
   ```python
   parsed = json.loads(result)
   return IntentResult(
       intents=parsed.get("intents", []),
       preferred_agent=parsed.get("preferred_agent"),
       database_id=parsed.get("database_id"),
       database_type=parsed.get("database_type")
   )
   ```

**IntentResult Model**:

```python
class IntentResult(BaseModel):
    intents: List[str] = Field(description="List of detected intents")
    preferred_agent: Optional[str] = Field(None, description="Preferred agent type: 'sql', 'neo4j', 'azure_search'")
    database_id: Optional[str] = Field(None, description="Database ID if SQL query detected")
    database_type: Optional[str] = Field(None, description="Database type if detected")
```

**Common Intents**:

- `"ai_search"`: Query requires search/retrieval
- `"news_script"`: Query requires script generation
- `"sql"`: Query is a SQL-related question
- `"unknown"`: Intent unclear (may still route to search if database detected)

---

## Search Agent Selection

### Location: `workflows/workflow_factory.py` (lines 27-53)

**Agent Types**:

1. **SQLAgent** (`agents/agents.py`, lines 800-900):
   - Used when `preferred_agent == "sql"` AND `agent_id_aisearch_sql` is configured
   - Executes SQL queries against PostgreSQL databases
   - Converts results to CSV format
   - Uses vector search on CSV data
   - **Function**: `run(self, goal: str, database_id: Optional[str] = None) -> str`

2. **Neo4jGraphRAGAgent** (`agents/agents.py`, lines 600-700):
   - Used when `use_neo4j_search == True` AND `agent_id_neo4j_search` is configured
   - Performs hybrid GraphRAG search on Neo4j
   - Cost-efficient alternative to Azure Search
   - **Function**: `run(self, goal: str) -> str`

3. **AiSearchAgent** (`agents/agents.py`, lines 400-500):
   - Default fallback agent
   - Uses Azure Cognitive Search
   - Production-ready search solution
   - **Function**: `run(self, goal: str) -> str`

---

## Search Execution

### SQLAgent Flow

**File**: `agents/agents.py` → `SQLAgent.run()` (lines 660-895)

**Process**:

```python
async def run(self, query: str, database_id: Optional[str] = None) -> str:
    # Step 1: Try PostgreSQL SQL Query (if database_id provided)
    if database_id:
        # 1.1: Initialize TextToSQLTool
        from ..tools_sql.text_to_sql_tool import TextToSQLTool
        sql_tool = TextToSQLTool()

        # 1.2: Query database (retrieves schema, generates SQL, executes)
        sql_result = sql_tool.query_database(
            natural_language_query=query,
            database_id=database_id,
            auto_detect_database=True
        )

        # 1.3: If SQL query successful, return results
        if sql_result.get("success") and sql_result.get("results"):
            return formatted_sql_results

    # Step 2: Fallback to CSV query (if SQL fails)
    # Step 3: Fallback to Vector/GraphRAG search (if CSV fails)
```

**Schema Retrieval Process** (inside `TextToSQLTool.query_database()`):

**File**: `tools_sql/text_to_sql_tool.py` → `query_database()` (lines 61-307)

1. **Auto-Detect Best Database** (if `auto_detect_database=True`):

   ```python
   from .schema_retrieval import SchemaRetriever
   schema_retriever = SchemaRetriever()

   # Get initial schema from provided database_id
   initial_schema = schema_retriever.get_relevant_schema(
       query=natural_language_query,
       database_id=database_id,
       top_k=top_k,
       similarity_threshold=similarity_threshold
   )

   # Search for best database across all databases
   best_db_id = schema_retriever.find_best_database(
       query=natural_language_query,
       candidate_database_ids=None,  # Search all databases
       top_k=top_k * 2,
       similarity_threshold=max(0.3, similarity_threshold - 0.2)
   )
   ```

2. **Get Relevant Schema from Neo4j Backend**:

   **File**: `tools_sql/schema_retrieval.py` → `SchemaRetriever.get_relevant_schema()` (lines 52-150)

   ```python
   # API Call to Neo4j Backend
   url = f"{self.neo4j_api_url}/api/databases/{database_id}/schema/search"
   payload = {
       "query": query,
       "top_k": top_k,
       "similarity_threshold": similarity_threshold,
       "element_types": element_types,  # ["table", "column", "metric"] or None
       "use_keyword_search": True,
       "use_graph_expansion": True,
       "max_hops": 1
   }
   response = requests.post(url, json=payload, timeout=30.0)
   data = response.json()
   ```

   **Schema Search Process** (in Neo4j Backend):
   - **Semantic Search**: Embeds query → finds relevant tables/columns by similarity
   - **Keyword Search**: Matches query terms against table/column names
   - **Graph Expansion**: Expands via relationships (e.g., table-column relationships)
   - **Re-ranking**: Combines signals to rank schema elements

   **Returns**:

   ```python
   {
       "results": [
           {
               "element_type": "table" | "column" | "metric",
               "name": "table_name" | "column_name",
               "description": "...",
               "similarity": 0.85,
               "metadata": {...}
           }
       ],
       "schema_slice": {
           "tables": [
               {
                   "name": "table_name",
                   "columns": [
                       {"name": "col1", "type": "varchar", ...},
                       {"name": "col2", "type": "integer", ...}
                   ]
               }
           ]
       },
       "result_count": 10
   }
   ```

3. **Generate SQL from Schema**:

   **File**: `tools_sql/sql_generator.py` → `SQLGenerator.generate_sql()` (lines 28-150)

   ```python
   # Uses SchemaRetriever to get schema
   schema_result = self.schema_retriever.get_relevant_schema(
       query=query,
       database_id=database_id,
       top_k=top_k,
       similarity_threshold=similarity_threshold
   )

   # Extract schema slice (tables and columns)
   schema_slice = schema_result.get("schema_slice", {})

   # Call Foundry agent to generate SQL
   # (Foundry agent receives query + schema_slice)
   sql = generate_sql_with_foundry_agent(query, schema_slice)
   ```

4. **Execute SQL Query**:

   **File**: `tools_sql/text_to_sql_tool.py` → `query_database()` (lines 150-200)

   ```python
   # API Call to Neo4j Backend to execute SQL
   url = f"{self.backend_url}/api/databases/{database_id}/execute"
   payload = {
       "query": generated_sql
   }
   response = requests.post(url, json=payload, timeout=30.0)
   execution_result = response.json()

   # Returns:
   # {
   #     "success": True,
   #     "rows": [...],
   #     "columns": [...],
   #     "row_count": 10
   # }
   ```

**Complete SQLAgent Flow**:

1. **Triage Agent** detects SQL intent → returns `database_id`
2. **SQLAgent** receives `database_id` from Triage
3. **Schema Retrieval**:
   - `SchemaRetriever.get_relevant_schema()` → Calls Neo4j backend `/api/databases/{database_id}/schema/search`
   - Neo4j backend performs hybrid search (semantic + keyword) on stored schemas
   - Returns relevant tables, columns, and metadata
4. **SQL Generation**:
   - `SQLGenerator.generate_sql()` → Uses schema + query
   - Calls Foundry agent with schema context
   - Returns generated SQL query
5. **SQL Execution**:
   - `TextToSQLTool.query_database()` → Calls Neo4j backend `/api/databases/{database_id}/execute`
   - Neo4j backend executes SQL against actual database (PostgreSQL, etc.)
   - Returns query results (rows, columns, row_count)
6. **Fallback Chain** (if SQL fails):
   - Try CSV query (if CSV files found)
   - Fallback to Vector/GraphRAG search

### Neo4jGraphRAGAgent Flow

**File**: `agents/agents.py` → `Neo4jGraphRAGAgent.run()` (lines 600-700)

```python
async def run(self, goal: str) -> str:
    # Step 1: Call Neo4j GraphRAG search
    from ..tools.neo4j_graphrag import graphrag_search

    results = graphrag_search(
        query=goal,
        top_k=10,
        similarity_threshold=0.7
    )

    # Step 2: Format results as context
    context = format_search_results(results)

    # Step 3: Call Foundry agent with goal and context
    result = run_foundry_agent(
        agent_id=self.agent_id,
        goal=goal,
        thread_id=thread_id,
        context=context
    )

    return result
```

### AiSearchAgent Flow

**File**: `agents/agents.py` → `AiSearchAgent.run()` (lines 400-700)

**NEW: Query Classification & Section Routing** (lines 480-565)

The AiSearchAgent now includes intelligent query classification to route queries appropriately:

**Step 1: Query Classification** (lines 490-525)

```python
async def run(self, goal: str) -> str:
    # Step 1.1: Extract person names from query
    person_names = self._extract_person_names(goal)
    is_person_query = len(person_names) > 0

    # Step 1.2: Classify query intent
    query_intent = self._classify_query_intent(goal, person_names)

    # Query intent structure:
    # {
    #     'type': 'section_based_scoped' | 'section_based_cross_document' | 'semantic',
    #     'routing': 'hard' | 'soft',
    #     'section_query': Optional[str],  # e.g., "skills", "experience"
    #     'file_scope': bool
    # }
```

**Classification Logic** (`_classify_query_intent()`, lines 1100-1220):

- **Detects section-based queries** by looking for attribute keywords:
  - `skill`, `experience`, `education`, `qualification`, `role`, `position`, etc.
- **Routes queries**:
  - **HARD routing** (`section_based_scoped`): Person + attribute → "Kevin's skills" → section-scoped search
  - **HARD routing** (`section_based_cross_document`): Attribute only → "All Python skills" → cross-document section search
  - **SOFT routing** (`semantic`): General queries → "Tell me about AI" → semantic search with section boosting

**Step 1.3: Extract Section Query** (`_extract_attribute_phrase()`, lines 1157-1220):

```python
def _extract_attribute_phrase(self, query: str, attribute_keywords: List[str]) -> str:
    """
    Extract clean section query, EXCLUDING person names.

    Examples:
        "Kevin's industry experience" → "experience"
        "Alexis Skills section only" → "skills"
        "technical skills summary" → "technical skills"
    """
    # Exclude person names and stop words
    stop_words = {'what', 'are', 'is', 'the', 'tell', 'me', 'about',
                  'show', 'get', 'find', 'list', 'give', 'only',
                  'section', 'from', 'of', "'s", 's'}

    # Extract core attribute phrase (excluding person names)
    # Returns clean keyword like "skills" instead of "alexis skills section"
```

**Step 2: Execute GraphRAG Search with Routing** (lines 550-580)

```python
    # Step 2.1: Determine keywords and boost
    keywords = [name.lower() for name in person_names]
    if query_intent['type'] == 'section_based_scoped':
        # Add attribute keyword for keyword matching
        keywords.append(query_intent['section_query'])
        keyword_boost = 0.4
    else:
        keyword_boost = 0.3

    # Step 2.2: Call GraphRAG with section routing parameters
    results = await graphrag_search(
        query=goal,
        top_k=12,
        similarity_threshold=0.75,
        keywords=keywords,
        keyword_boost=keyword_boost,
        is_person_query=is_person_query,
        person_names=person_names,
        section_query=query_intent.get('section_query') if query_intent['routing'] == 'hard' else None,
        use_section_routing=query_intent['routing'] == 'hard'
    )
```

**Section Routing Flow**:

When `use_section_routing=True` and `section_query` is provided:

1. **Agent sends**:
   - `section_query: "skills"`
   - `use_section_routing: True`

2. **Neo4j Backend** (`services/graphrag_retrieval.py`, lines 130-200):
   - Receives parameters
   - Calls `section_scoped_search()` instead of regular hybrid search
   - **Generates section embedding** from `section_query`
   - **Finds matching Section nodes** (e.g., "Skills" section with 0.86+ similarity)
   - **Traverses IN_SECTION relationships** to get chunks from matching sections
   - **Ranks chunks** by combined score: `section_similarity * 0.6 + chunk_similarity * 0.4`

3. **Returns structured results**:
   ```python
   {
       "id": "chunk_id",
       "text": "chunk text...",
       "similarity": 0.36,
       "hybrid_score": 0.66,
       "metadata": {
           "vector_score": 0.36,
           "section_similarity": 0.86,  # Section match score
           "combined_score": 0.66,
           "section_name": "Skills",    # Matched section
           "section_path": "Level 1: ... > Level 2: Skills",
           "section_level": 2,
           "routing": "hard_section",
           "section_query": "skills"
       },
       "source": "section_routing"
   }
   ```

**Step 3: Filter Results** (lines 590-650)

```python
    # Step 3.1: Filter by person name (for person queries)
    filtered = self.filter_results_by_exact_match(
        results,
        goal,
        is_person_query=is_person_query,
        person_names=person_names
    )

    # Step 3.2: Apply intent-specific filtering
    if query_intent['routing'] == 'hard':
        # HARD routing: Trust section routing results
        # Minimal filtering - section graph already did the work
        filtered = [r for r in filtered if r.get('metadata', {}).get('section_name')]
    else:
        # SOFT routing: Standard hybrid filtering
        filtered = [r for r in filtered if r.get('hybrid_score', 0) >= 0.3]

    # Step 3.3: Format as context
    context = format_search_results(filtered)

    # Step 3.4: Call Foundry agent with context
    result = run_foundry_agent(
        agent_id=self.agent_id,
        goal=goal,
        thread_id=thread_id,
        context=context
    )

    return result
```

**Query Classification Examples**:

| Query                              | Classification                 | Routing | Section Query     | Execution                              |
| ---------------------------------- | ------------------------------ | ------- | ----------------- | -------------------------------------- |
| "Alexis Skills section only"       | `section_based_scoped`         | `hard`  | `"skills"`        | Section-scoped search on Alexis's file |
| "Kevin's industry experience"      | `section_based_scoped`         | `hard`  | `"experience"`    | Section-scoped search on Kevin's file  |
| "All Python skills"                | `section_based_cross_document` | `hard`  | `"python skills"` | Cross-document section search          |
| "Tell me about AI"                 | `semantic`                     | `soft`  | `None`            | Semantic search with section boosting  |
| "What projects did Sarah work on?" | `section_based_scoped`         | `hard`  | `"projects"`      | Section-scoped search on Sarah's file  |

**Benefits of Section Routing**:

- **Structural precision**: Uses document structure (Section nodes) instead of text matching
- **Efficient traversal**: Graph relationships (IN_SECTION) ensure accurate results
- **Semantic section matching**: Embeds section names for fuzzy matching ("Skills" ≈ "Technical Expertise")
- **Combined scoring**: Balances section relevance with chunk relevance
- **No false positives**: Structural facts prevent semantic drift

---

## Structural Index Implementation

### Overview

The structural index adds first-class **Section nodes** to the graph to enable document structure-aware retrieval. Instead of relying purely on semantic similarity, queries can now use document structure (section headers) to find relevant content.

**Schema Extension** (from `STRUCTURAL_INDEX_IMPLEMENTATION.md`):

### Node Types

**1. Section Node** (Document-Specific Instance)

```python
Section {
    id: str,              # "file123:section:Level_2_Technical_Expertise"
    file_id: str,         # Reference to parent File
    level: int,           # 1, 2, 3 (hierarchy depth)
    name: str,            # "Technical Expertise" (exact header_text)
    normalized_name: str, # "technical_expertise" (lowercase)
    path: str,            # "Level 1: Profile > Level 2: Technical Expertise"
    parent_path: str,     # "Level 1: Profile" or null for top-level
    chunk_count: int,     # Number of chunks in this section
    start_chunk_idx: int, # First chunk index in file
    end_chunk_idx: int,   # Last chunk index in file
    embedding: [float],   # 1536-dim vector of section name
    createdAt: datetime
}
```

**2. SectionType Node** (Cross-Document Pattern)

```python
SectionType {
    id: str,              # "section_type:skills"
    canonical_name: str,  # "Skills" (learned from clustering)
    normalized_name: str, # "skills"
    member_count: int,    # Number of Section instances
    centroid: [float],    # 1536-dim centroid of all member embeddings
    common_variants: [str], # ["Skills", "Technical Expertise", "Core Competencies"]
    createdAt: datetime,
    updatedAt: datetime
}
```

### Relationship Types

```python
# Document structure
(f:File)-[:HAS_SECTION]->(s:Section)

# Section hierarchy
(parent:Section)-[:PARENT_SECTION]->(child:Section)

# Section contains chunks
(s:Section)-[:IN_SECTION]->(c:Chunk)

# Section belongs to type (cross-document)
(s:Section)-[:INSTANCE_OF]->(st:SectionType)
```

**Real Example from Schema Discovery** (from logs):

```
HAS_SECTION: 34 edges
PARENT_SECTION: 29 edges
IN_SECTION: 36 edges
INSTANCE_OF: 35 edges
```

### Ingestion Pipeline

**When a document is uploaded**:

1. **Extract Sections from Chunks** (`extract_sections_from_chunks()`):
   - Analyzes chunk metadata (header_text, header_path, header_level)
   - Creates unique Section definitions for each header path
   - Determines section hierarchy from header levels

2. **Generate Section Embeddings** (`generate_section_embeddings()`):
   - Embeds section names using Azure OpenAI text-embedding-3-small
   - Creates 1536-dimensional vectors for semantic matching

3. **Create Section Nodes** (`create_section_nodes()`):
   - MERGE Section nodes into Neo4j
   - Store metadata: level, name, path, chunk_count, boundaries

4. **Build Section Hierarchy** (`create_section_hierarchy()`):
   - Create PARENT_SECTION relationships
   - Maps parent paths to parent Section nodes
   - Preserves document structure

5. **Link Section → Chunk** (`link_sections_to_chunks()`):
   - Create IN_SECTION relationships
   - Maps chunks to their containing sections via chunk indices

6. **Discover Section Types** (Optional, `SectionClusteringService`):
   - Cluster sections by embedding similarity (DBSCAN)
   - Create SectionType nodes from clusters
   - Learn that "Skills" ≈ "Technical Expertise" ≈ "Core Competencies"
   - Create INSTANCE_OF relationships

### Query Routing Based on Structure

**Classification Process**:

The Agent's query classification determines if a query should use structural routing:

```
Query: "Alexis Skills section only"
  ├─ Extract person names: ["Alexis"]
  ├─ Detect attribute keywords: "skills" found
  └─ Classification: section_based_scoped, routing=hard

Query Classification sends to backend:
  section_query: "skills"
  use_section_routing: True
  file_id: <Alexis's file>
```

**Section Matching Process** (in backend):

```
1. Embed section_query "skills" → vector
2. Find Section nodes with high cosine similarity:
   - "Skills" section: 0.8653 ✓ MATCH
   - "Technical Expertise": 0.7821 ✓ MATCH
   - "Education": 0.2145 ✗ NO MATCH (< 0.50 threshold)
3. Traverse IN_SECTION to get chunks from matching sections
4. Rank chunks by combined score:
   combined_score = section_similarity * 0.6 + chunk_similarity * 0.4
```

### Integration Points

**1. Agent Layer** (`agents/agents.py`):

- `_classify_query_intent()`: Detects section-based queries
- `_extract_attribute_phrase()`: Extracts clean section query
- Passes `section_query` and `use_section_routing` to GraphRAG

**2. GraphRAG Client** (`tools/neo4j_graphrag.py`):

- `hybrid_retrieve()`: Accepts section routing parameters
- Sends payload with `section_query` and `use_section_routing` flags

**3. Backend Router** (`routers/graphrag.py`):

- `GraphRAGQuery` model accepts `section_query` and `use_section_routing`
- Passes parameters to retrieval service

**4. Retrieval Service** (`services/graphrag_retrieval.py`):

- **HARD routing check** (line 130):
  ```python
  if use_section_routing and section_query:
      results = self.section_scoped_search(...)
  ```
- Falls back to hybrid search if `use_section_routing=False`

### Retrieval Strategies

**HARD Routing** (Structure-Driven):

- Query: "Alexis Skills section only"
- Process: Section embedding → find matching sections → return chunks from those sections
- Result: High precision, enforces document structure
- Metadata returned: `section_name`, `section_similarity`, `routing: hard_section`

**SOFT Routing** (Semantic with Boosting):

- Query: "Tell me about AI"
- Process: Full query embedding → find chunks semantically + boost if in relevant sections
- Result: Better recall, more flexible
- Metadata returned: `routing: soft`, section info optional

**CROSS-DOCUMENT**:

- Query: "All Python skills"
- Process: Search for "python skills" sections across all documents
- Result: Finds matching sections in multiple files
- Example: "Skills" sections from 5 different resumes with Python mentioned

### Query Examples and Execution

| Query                            | Intent                       | Routing | Execution                                                  |
| -------------------------------- | ---------------------------- | ------- | ---------------------------------------------------------- |
| "Alexis Skills section only"     | section_based_scoped         | HARD    | Find "Skills" section in Alexis's file, return its chunks  |
| "Kevin's industry experience"    | section_based_scoped         | HARD    | Find "Experience/Industry" section in Kevin's file         |
| "All Python skills"              | section_based_cross_document | HARD    | Find "Skills" sections across all files, filter for Python |
| "Tell me about machine learning" | semantic                     | SOFT    | Semantic search + section boosting if in Projects section  |
| "What's Sarah's background?"     | section_based_scoped         | HARD    | Find "Education/Background" section in Sarah's file        |

### Benefits

✅ **Structural Precision**: Uses document structure instead of text matching  
✅ **Efficient Traversal**: Graph relationships (IN_SECTION) ensure accuracy  
✅ **Semantic Fuzzy Matching**: Embeds section names for "Skills" ≈ "Technical Expertise"  
✅ **No False Positives**: Structural constraints prevent semantic drift  
✅ **Cross-Document Discovery**: Find equivalent sections across multiple documents  
✅ **Hierarchical Understanding**: Preserves section parent-child relationships

### Current Status

**✅ Implemented**:

- Section and SectionType node definitions in schema
- Section extraction and embedding during ingestion
- Section hierarchy creation (PARENT_SECTION)
- Section-to-chunk linking (IN_SECTION)
- Query classification in Agent
- Section routing in backend (`section_scoped_search()`)
- Debug logging and monitoring

**Verified in Graph**:

```
Node Types:  File, Chunk, Section, SectionType (19 total)
Relationships: HAS_SECTION (34), PARENT_SECTION (29), IN_SECTION (36), INSTANCE_OF (35)
Section Similarity: "skills" → "Skills" section = 0.8653 (above 0.50 threshold)
```

**⏳ Pending**:

- Neo4j backend service restart (to load new code into memory)
- Full integration test with section-based query
- Cross-document section clustering (SectionType discovery)

---

### Location: `neo4j_backend/services/graphrag_retrieval.py`

**Section-Scoped Search** (`section_scoped_search()`, lines 1911-2100)

When the Agent sends `use_section_routing=True` with a `section_query`, the backend executes structured retrieval:

**Step 1: Generate Section Embedding** (lines 1920-1930)

```python
def section_scoped_search(
    query_embedding: List[float],
    section_query: str,
    file_id: Optional[str] = None,
    top_k: int = 10,
    section_similarity_threshold: float = 0.50
):
    # Embed the section query (e.g., "skills")
    section_embedding = self.embedding_client.embed([section_query])[0]
```

**Step 2: Find Matching Section Nodes** (lines 1935-1970)

```cypher
// Neo4j Cypher Query
MATCH (s:Section)
WHERE s.embedding IS NOT NULL
  AND ($file_id IS NULL OR s.file_id = $file_id)

// Calculate section similarity
WITH s,
     gds.similarity.cosine(s.embedding, $section_embedding) AS section_similarity
WHERE section_similarity >= $section_threshold

// Example results:
// "Skills" section: similarity = 0.8653 ✓ (above 0.50 threshold)
// "Technical Expertise" section: similarity = 0.7821 ✓
// "Education" section: similarity = 0.2145 ✗ (below threshold)
```

**Step 3: Get Chunks via IN_SECTION Relationships** (lines 1975-2000)

```cypher
// Traverse to chunks in matching sections
MATCH (s)-[:IN_SECTION]->(c:Chunk)
WHERE c.embedding IS NOT NULL

// Rank chunks by query relevance
WITH c, s, section_similarity,
     gds.similarity.cosine(c.embedding, $query_embedding) AS chunk_similarity

// Combined scoring (section match + chunk relevance)
WITH c, s, section_similarity, chunk_similarity,
     (section_similarity * 0.6 + chunk_similarity * 0.4) AS combined_score

ORDER BY combined_score DESC
LIMIT $top_k

RETURN c.id AS chunk_id,
       c.text AS text,
       chunk_similarity,
       section_similarity,
       combined_score,
       s.name AS section_name,
       s.path AS section_path,
       s.level AS section_level
```

**Graph Structure Used**:

```
File
 └─[:HAS_SECTION]─> Section {name: "Skills", embedding: [...]}
                      │
                      └─[:IN_SECTION]─> Chunk {text: "Python, Java, ...", embedding: [...]}
                      └─[:IN_SECTION]─> Chunk {text: "10 years experience...", embedding: [...]}
```

**Section Similarity Scoring**:

| Section Query | Section Name              | Similarity | Match?              |
| ------------- | ------------------------- | ---------- | ------------------- |
| "skills"      | "Skills"                  | 0.8653     | ✓ Yes (0.86 > 0.50) |
| "skills"      | "Technical Expertise"     | 0.7821     | ✓ Yes (0.78 > 0.50) |
| "skills"      | "Core Competencies"       | 0.6912     | ✓ Yes (0.69 > 0.50) |
| "skills"      | "Education"               | 0.2145     | ✗ No (0.21 < 0.50)  |
| "experience"  | "Industry Experience"     | 0.8234     | ✓ Yes (0.82 > 0.50) |
| "experience"  | "Professional Background" | 0.7456     | ✓ Yes (0.75 > 0.50) |

**Comparison: Section Routing vs. Semantic Search**:

| Query: "Alexis Skills section only" | Section Routing (HARD)                                                            | Semantic Search (SOFT)                      |
| ----------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------- |
| **Method**                          | 1. Embed "skills" → Find Section nodes<br>2. Traverse IN_SECTION → Get chunks     | 1. Embed full query → Find chunks directly  |
| **Precision**                       | High - structural constraint                                                      | Medium - semantic similarity                |
| **Section Match**                   | 0.86 similarity to "Skills" section                                               | N/A - no section matching                   |
| **Graph Traversal**                 | Yes - IN_SECTION relationships                                                    | No - direct chunk search                    |
| **Result Metadata**                 | `section_name: "Skills"`<br>`section_similarity: 0.86`<br>`routing: hard_section` | `routing: soft`<br>No section info          |
| **False Positives**                 | Low - structure enforces correctness                                              | Higher - may match "skill" in wrong section |

**Key Implementation Details**:

1. **Section Threshold**: 0.50 (lenient to catch variations like "Skills" ≈ "Technical Skills")
2. **Combined Scoring**: `section_similarity * 0.6 + chunk_similarity * 0.4`
   - Section match weighted higher (60%) to prioritize structural relevance
   - Chunk content still matters (40%) for final ranking
3. **File Scope**: Optional `file_id` parameter scopes search to specific person's document
4. **Graph Schema**: Requires Section nodes with embeddings and IN_SECTION relationships

---

## Reporter Agent

### Location: `agents/agents.py` → `NewsReporterAgent` (lines 300-400)

**Class**: `NewsReporterAgent`

**Function**: `run(self, goal: str, context: str = "") -> str`

**Process**:

```python
async def run(self, goal: str, context: str = "") -> str:
    # Build prompt with goal and context
    prompt = f"Goal: {goal}\n\nContext:\n{context}"

    # Call Foundry agent
    result = run_foundry_agent(
        agent_id=self.agent_id,
        goal=prompt,
        thread_id=thread_id
    )

    return result
```

**Usage**:

- Called when `"news_script" in tri.intents`
- Receives search results as `context` parameter
- Generates script/article based on goal and context

---

## Review Agent

### Location: `agents/agents.py` → `ReviewAgent` (lines 500-600)

**Class**: `ReviewAgent`

**Function**: `run(self, goal: str, script: str) -> Dict[str, Any]`

**Process**:

```python
async def run(self, goal: str, script: str) -> Dict[str, Any]:
    # Build review prompt
    prompt = f"Review this script against the goal:\n\nGoal: {goal}\n\nScript:\n{script}"

    # Call Foundry agent
    result = run_foundry_agent_json(
        agent_id=self.agent_id,
        goal=prompt,
        thread_id=thread_id
    )

    # Parse JSON response
    parsed = json.loads(result)
    return {
        "decision": parsed.get("decision"),  # "accept" or "revise"
        "reason": parsed.get("reason", ""),
        "suggested_changes": parsed.get("suggested_changes", ""),
        "revised_script": parsed.get("revised_script", script)
    }
```

**Review Loop**:

- Maximum 3 review passes
- If `decision == "accept"`, return script
- If `decision == "revise"`, improve script and review again
- After 3 passes, return script with note

---

## Neo4j GraphRAG Search

### Location: `tools/neo4j_graphrag.py` → `graphrag_search()` (lines 200-240)

**Function**: `graphrag_search(query: str, top_k: int = 10, ...) -> List[Dict[str, Any]]`

**Process**:

1. **Initialize Retriever**:

   ```python
   retriever = Neo4jGraphRAGRetriever(neo4j_api_url=settings.neo4j_api_url)
   ```

2. **Call Hybrid Retrieve**:

   ```python
   results = retriever.hybrid_retrieve(
       query=query,
       top_k_vector=top_k,
       max_hops=2,
       similarity_threshold=similarity_threshold,
       use_keyword_search=use_keyword_search,
       keywords=keywords,
       keyword_match_type=keyword_match_type,
       keyword_boost=keyword_boost
   )
   ```

3. **Hybrid Retrieval Process** (`Neo4jGraphRAGRetriever.hybrid_retrieve()`, lines 57-140):

   The hybrid retrieval combines multiple search strategies:
   - **Vector Search**: Embed query → find top-k chunks by similarity
   - **Keyword Search**: Text matching on chunk keywords (see [Keyword Search Details](#keyword-search-details) below)
   - **Graph Expansion**: 1-2 hops via `SEMANTICALLY_SIMILAR` relationships
   - **Re-ranking**: Multi-signal scoring (similarity + keyword + graph signals)

### Keyword Search Details

Keyword search is a critical component of the hybrid retrieval system that complements semantic (vector) search:

1. **Keyword Extraction**:
   - **Automatic**: If `keywords=None`, keywords are automatically extracted from the query
   - **Manual**: Keywords can be explicitly provided (e.g., person names extracted from query)
   - **Person Name Extraction**: In `routers/chat_sessions.py`, person names are extracted from queries for targeted filtering:
     ```python
     person_names = extract_person_names(user_message_content)
     # Returns capitalized words (length > 2) excluding common words
     ```

2. **Keyword Matching**:
   - **Match Type**: Controlled by `keyword_match_type` parameter:
     - `"any"` (OR): Chunk matches if ANY keyword appears in chunk keywords
     - `"all"` (AND): Chunk matches if ALL keywords appear in chunk keywords
   - **Default**: `"any"` for broader matching
   - **Matching Location**: Keywords are matched against the `keywords` property stored on Chunk nodes in Neo4j

3. **Keyword Boost**:
   - **Parameter**: `keyword_boost` (default: 0.3, range: 0.0 to 1.0)
   - **Purpose**: Controls the weight of keyword matches in the final hybrid score
   - **Scoring**: `hybrid_score = similarity_score + (keyword_match_score * keyword_boost)`
   - **Effect**: Higher `keyword_boost` values give more weight to exact keyword matches

4. **Integration with Vector Search**:
   - Keywords are used alongside vector similarity for retrieval
   - Chunks that match keywords get a boost in their final hybrid score
   - This helps surface relevant chunks even if vector similarity is slightly lower

5. **Usage in Agent Flow**:

   ```python
   # In routers/chat_sessions.py (lines 188-198)
   person_names = extract_person_names(user_message_content)
   search_results = graphrag_search(
       query=user_message_content,
       top_k=12,
       similarity_threshold=0.75,
       keywords=person_names if person_names else None,  # Explicit keywords
       keyword_match_type="any",  # OR matching
       keyword_boost=0.4  # 40% weight for keyword matches
   )
   ```

6. **API Call** (lines 92-140):
   ```python
   url = f"{self.neo4j_api_url}/api/graphrag/query"
   payload = {
       "query": query,
       "top_k_vector": top_k_vector,
       "max_hops": max_hops,
       "similarity_threshold": similarity_threshold,
       "use_keyword_search": use_keyword_search,
       "keywords": keywords,
       "keyword_match_type": keyword_match_type,
       "keyword_boost": keyword_boost
   }
   response = requests.post(url, json=payload, timeout=30)
   results = response.json().get("results", [])
   ```

**Result Format**:

```python
[
    {
        "text": "chunk text content...",
        "file_name": "document.pdf",
        "file_path": "/path/to/document.pdf",
        "directory_name": "resumes",
        "similarity": 0.85,
        "hybrid_score": 0.92,
        "metadata": {
            "chunk_index": 0,
            "file_id": "...",
            ...
        }
    },
    ...
]
```

---

## Response Generation

### Location: `routers/chat_sessions.py` → `add_message()` (lines 474-545)

**Step 1: Run Agent Workflow** (lines 475-491)

```python
try:
    assistant_response = await run_sequential_goal(cfg, user_message_content)
except RuntimeError as e:
    error_msg = str(e)
    # Check if it's a Foundry access error
    if "Foundry" in error_msg or "foundry" in error_msg or "AZURE_AI_PROJECT" in error_msg:
        raise HTTPException(
            status_code=503,
            detail=f"Foundry access is required but not available. Error: {error_msg}"
        )
    raise HTTPException(status_code=500, detail=f"Agent workflow failed: {error_msg}")
except Exception as e:
    logging.exception("[add_message] Failed to process query: %s", e)
    raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")
```

**Step 2: Insert Assistant Message** (lines 493-502)

```python
assistant_message = {
    "sessionId": session_id,
    "userId": user_id,
    "role": "assistant",
    "content": assistant_response,
    "sources": sources if sources else None,
    "createdAt": datetime.utcnow(),
}
result = messages_collection.insert_one(assistant_message)
```

**Step 3: Update Session Timestamp** (lines 504-508)

```python
sessions_collection.update_one(
    {"_id": ObjectId(session_id)},
    {"$set": {"updatedAt": datetime.utcnow()}}
)
```

**Step 4: Serialize Response** (lines 510-545)

```python
raw_response = {
    "response": assistant_response,
    "sources": sources,
    "conversation_id": session_id,
}

# Recursively serialize MongoDB objects (ObjectId, datetime) to JSON-safe types
safe_response = recursive_serialize(raw_response)
return safe_response
```

**Serialization Function** (lines 548-563):

```python
def recursive_serialize(obj):
    """Recursively convert Pymongo/Datetime objects to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: recursive_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_serialize(v) for v in obj]
    elif isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat() + 'Z'
    elif hasattr(obj, "isoformat"):
        return obj.isoformat()
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        return str(obj)
```

---

## Complete Code Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. API Request: POST /api/chat/sessions/{session_id}/messages │
│    File: routers/chat_sessions.py:373                          │
│    Parameters: session_id, message (content)                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Authentication                                              │
│    File: routers/chat_sessions.py:377 →                        │
│         routers/auth.py:278                                    │
│    • Extract JWT token from Authorization header                │
│    • Decode and verify token                                   │
│    • Get user from MongoDB                                      │
│    • Return user object (dependency injection)                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Session Validation                                           │
│    File: routers/chat_sessions.py:387-394                     │
│    • Verify session exists in MongoDB                           │
│    • Verify session belongs to authenticated user               │
│    • Validate message content is not empty                      │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Store User Message                                           │
│    File: routers/chat_sessions.py:403-411                      │
│    • Insert user message into messages_collection               │
│    • Message structure: {sessionId, userId, role: "user",     │
│      content, createdAt}                                        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Get Sources from Neo4j (if enabled) - HYBRID SEARCH         │
│    File: routers/chat_sessions.py:419-472                       │
│                                                                 │
│    IF cfg.use_neo4j_search:                                     │
│      • Extract person names from query (keyword extraction)     │
│      • Call graphrag_search() → hybrid_retrieve()              │
│                                                                 │
│      HYBRID SEARCH PROCESS (semantic + keyword):               │
│      ┌─────────────────────────────────────────────────────┐  │
│      │ 5.1 VECTOR SEARCH (Semantic):                       │  │
│      │     • Embed query → generate embedding vector        │  │
│      │     • Find top-k chunks by cosine similarity         │  │
│      │     • Returns chunks with similarity scores          │  │
│      └─────────────────────────────────────────────────────┘  │
│      ┌─────────────────────────────────────────────────────┐  │
│      │ 5.2 KEYWORD SEARCH (Exact Match):                    │  │
│      │     • Extract keywords from query (person names)     │  │
│      │     • Match keywords against chunk.keywords property │  │
│      │     • Match type: "any" (OR) or "all" (AND)         │  │
│      │     • Returns chunks with keyword match scores       │  │
│      └─────────────────────────────────────────────────────┘  │
│      ┌─────────────────────────────────────────────────────┐  │
│      │ 5.3 GRAPH EXPANSION:                                 │  │
│      │     • Expand from vector/keyword results             │  │
│      │     • 1-2 hops via SEMANTICALLY_SIMILAR relationships│  │
│      │     • Find related chunks in graph                  │  │
│      └─────────────────────────────────────────────────────┘  │
│      ┌─────────────────────────────────────────────────────┐  │
│      │ 5.4 HYBRID RE-RANKING:                               │  │
│      │     • Combine signals: similarity + keyword + graph  │  │
│      │     • hybrid_score = similarity +                    │  │
│      │       (keyword_match * keyword_boost) +              │  │
│      │       graph_signals                                   │  │
│      │     • Sort by hybrid_score (descending)              │  │
│      └─────────────────────────────────────────────────────┘  │
│                                                                 │
│      • Filter results by exact name match (post-processing)    │
│      • Limit to top 8 results                                  │
│      • Format sources array                                     │
│    ELSE:                                                         │
│      sources = []                                               │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. Agent Workflow Execution                                     │
│    File: routers/chat_sessions.py:475 →                        │
│         workflows/workflow_factory.py:11                       │
│                                                                 │
│    6.1 TRIAGE AGENT:                                            │
│        • TriageAgent.run(goal)                                  │
│        • Returns IntentResult: {intents, preferred_agent,       │
│          database_id, database_type}                            │
│                                                                 │
│    6.2 SELECT SEARCH AGENT:                                     │
│        IF preferred_agent == "sql" AND agent_id_aisearch_sql:  │
│          → SQLAgent                                            │
│        ELIF use_neo4j_search AND agent_id_neo4j_search:         │
│          → Neo4jGraphRAGAgent (uses hybrid search)             │
│        ELSE:                                                     │
│          → AiSearchAgent                                        │
│                                                                 │
│    6.3 EXECUTE SEARCH (if "ai_search" in intents):             │
│        IF SQLAgent:                                             │
│          → search_agent.run(goal, database_id=tri.database_id)  │
│          ┌─────────────────────────────────────────────────┐  │
│          │ 6.3.1 SCHEMA RETRIEVAL FROM NEO4J:               │  │
│          │     • SchemaRetriever.get_relevant_schema()      │  │
│          │     • API: POST /api/databases/{db_id}/schema/    │  │
│          │       search                                       │  │
│          │     • Neo4j backend performs hybrid search:       │  │
│          │       - Semantic: Embed query → find tables/cols  │  │
│          │       - Keyword: Match query terms to names       │  │
│          │       - Graph: Expand via relationships           │  │
│          │     • Returns: relevant tables, columns, metadata │  │
│          └─────────────────────────────────────────────────┘  │
│          ┌─────────────────────────────────────────────────┐  │
│          │ 6.3.2 SQL GENERATION:                            │  │
│          │     • SQLGenerator.generate_sql()                 │  │
│          │     • Uses schema + query                        │  │
│          │     • Calls Foundry agent with schema context    │  │
│          │     • Returns: generated SQL query               │  │
│          └─────────────────────────────────────────────────┘  │
│          ┌─────────────────────────────────────────────────┐  │
│          │ 6.3.3 SQL EXECUTION:                             │  │
│          │     • TextToSQLTool.query_database()             │  │
│          │     • API: POST /api/databases/{db_id}/execute  │  │
│          │     • Neo4j backend executes SQL on database      │  │
│          │     • Returns: rows, columns, row_count          │  │
│          └─────────────────────────────────────────────────┘  │
│          • If SQL fails → Fallback to CSV query              │
│          • If CSV fails → Fallback to Vector/GraphRAG search │
│        ELIF Neo4jGraphRAGAgent:                                 │
│          → search_agent.run(goal)                               │
│          → Calls graphrag_search() → hybrid_retrieve()         │
│          → HYBRID SEARCH: vector + keyword + graph expansion   │
│        ELSE (AiSearchAgent):                                    │
│          → search_agent.run(goal)                               │
│          → Azure Cognitive Search                               │
│                                                                 │
│    6.4 REPORTER AGENT (if "news_script" in intents):           │
│        → reporter.run(goal, latest_search_results)              │
│                                                                 │
│    6.5 REVIEW AGENT (max 3 passes):                             │
│        FOR i in range(1, 4):                                    │
│          verdict = reviewer.run(goal, script)                  │
│          IF verdict.decision == "accept":                        │
│            RETURN script                                        │
│          ELSE:                                                  │
│            script = reporter.run(goal, improve_context)         │
│        RETURN script                                            │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. Store Assistant Message                                      │
│    File: routers/chat_sessions.py:493-502                      │
│    • Insert assistant message into messages_collection          │
│    • Message structure: {sessionId, userId, role: "assistant",│
│      content, sources, createdAt}                               │
│    • Update session.updatedAt timestamp                        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. Serialize and Return Response                               │
│    File: routers/chat_sessions.py:510-545                      │
│    • Build response: {response, sources, conversation_id}      │
│    • Recursively serialize MongoDB objects (ObjectId, datetime)│
│    • Return JSON response                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Code Files and Functions

| Component                    | File                                           | Function/Class                              | Lines     |
| ---------------------------- | ---------------------------------------------- | ------------------------------------------- | --------- |
| **API Endpoint**             | `routers/chat_sessions.py`                     | `add_message()`                             | 373-545   |
| **Authentication**           | `routers/auth.py`                              | `get_current_user()`                        | 278-295   |
| **Workflow Orchestration**   | `workflows/workflow_factory.py`                | `run_sequential_goal()`                     | 11-120    |
| **Triage Agent**             | `agents/agents.py`                             | `TriageAgent.run()`                         | 148-200   |
| **SQL Agent**                | `agents/agents.py`                             | `SQLAgent.run()`                            | 800-900   |
| **Neo4j GraphRAG Agent**     | `agents/agents.py`                             | `Neo4jGraphRAGAgent.run()`                  | 600-700   |
| **Azure Search Agent**       | `agents/agents.py`                             | `AiSearchAgent.run()`                       | 400-700   |
| **Query Classification**     | `agents/agents.py`                             | `AiSearchAgent._classify_query_intent()`    | 1100-1220 |
| **Section Query Extraction** | `agents/agents.py`                             | `AiSearchAgent._extract_attribute_phrase()` | 1157-1220 |
| **Reporter Agent**           | `agents/agents.py`                             | `NewsReporterAgent.run()`                   | 300-400   |
| **Review Agent**             | `agents/agents.py`                             | `ReviewAgent.run()`                         | 500-600   |
| **Neo4j Search**             | `tools/neo4j_graphrag.py`                      | `graphrag_search()`                         | 286-355   |
| **Neo4j Retriever**          | `tools/neo4j_graphrag.py`                      | `Neo4jGraphRAGRetriever.hybrid_retrieve()`  | 110-220   |
| **Section Scoped Search**    | `neo4j_backend/services/graphrag_retrieval.py` | `section_scoped_search()`                   | 1911-2100 |
| **Foundry Runner**           | `foundry_runner.py`                            | `run_foundry_agent()`                       | 100-200   |
| **Foundry JSON Runner**      | `foundry_runner.py`                            | `run_foundry_agent_json()`                  | 200-300   |

---

## Data Structures

### IntentResult Model

```python
class IntentResult(BaseModel):
    intents: List[str]  # e.g., ["ai_search", "news_script"]
    preferred_agent: Optional[str]  # e.g., "sql", "neo4j", "azure_search"
    database_id: Optional[str]  # Database ID if SQL query detected
    database_type: Optional[str]  # Database type if detected
```

### Chat Session (MongoDB)

```python
{
    "_id": ObjectId("..."),
    "userId": "user_id_string",
    "title": "New Chat",
    "createdAt": datetime.utcnow(),
    "updatedAt": datetime.utcnow()
}
```

### Chat Message (MongoDB)

```python
{
    "_id": ObjectId("..."),
    "sessionId": "session_id_string",
    "userId": "user_id_string",
    "role": "user" | "assistant",
    "content": "message text",
    "sources": [  # Optional, only for assistant messages
        {
            "file_name": "document.pdf",
            "file_path": "/path/to/document.pdf",
            "directory_name": "resumes",
            "text": "chunk text...",
            "similarity": 0.85,
            "hybrid_score": 0.92,
            "metadata": {...}
        }
    ],
    "createdAt": datetime.utcnow()
}
```

### Review Verdict

```python
{
    "decision": "accept" | "revise",
    "reason": "explanation text",
    "suggested_changes": "suggestions text",
    "revised_script": "revised script text"  # Optional
}
```

---

## Error Handling Points

1. **Authentication Failure** (auth.py:278): Returns 401 Unauthorized
2. **Session Not Found** (chat_sessions.py:389-394): Returns 404 Not Found
3. **Empty Message Content** (chat_sessions.py:397-399): Returns 400 Bad Request
4. **MongoDB Unavailable** (chat_sessions.py:382-383): Returns 503 Service Unavailable
5. **Foundry Access Error** (chat_sessions.py:479-486): Returns 503 Service Unavailable
6. **Agent Workflow Failure** (chat_sessions.py:487-491): Returns 500 Internal Server Error
7. **Neo4j Search Failure** (chat_sessions.py:470-472): Logs error, continues with empty sources

---

## Configuration

### Environment Variables

- `MONGO_AGENT_URL`: MongoDB connection string for chat sessions
- `NEO4J_API_URL`: Neo4j backend API URL (e.g., "http://localhost:8000")
- `AZURE_AI_PROJECT`: Azure AI Project name (for Foundry agents)
- `USE_NEO4J_SEARCH`: Enable Neo4j GraphRAG search (true/false)
- `CORS_ORIGINS`: Comma-separated list of allowed CORS origins

### Agent Configuration (Settings)

- `agent_id_triage`: Triage agent ID
- `agent_id_aisearch`: Azure Search agent ID
- `agent_id_aisearch_sql`: SQL agent ID (optional)
- `agent_id_neo4j_search`: Neo4j GraphRAG agent ID (optional)
- `agent_id_reviewer`: Review agent ID
- `reporter_ids`: List of reporter agent IDs
- `multi_route_always`: Always use multiple reporters (true/false)
- `use_neo4j_search`: Use Neo4j instead of Azure Search (true/false)

---

## Summary

The Agent flow follows this exact code path:

1. **Entry**: `routers/chat_sessions.py::add_message()` receives POST request
2. **Authentication**: `get_current_user` dependency verifies JWT token and gets user
3. **Session Validation**: Verifies session exists and belongs to user
4. **Store User Message**: Inserts user message into MongoDB
5. **Get Sources** (optional): If `use_neo4j_search`, performs GraphRAG search and filters results
6. **Agent Workflow**:
   - **Triage**: `TriageAgent` analyzes query and returns intents/preferences
   - **Search Selection**: Chooses SQL/Neo4j/Azure Search agent based on triage results
   - **Search Execution**: Runs selected search agent (if "ai_search" intent)
   - **Reporter**: Generates script/article (if "news_script" intent)
   - **Review**: Reviews and improves script (max 3 passes)
7. **Store Assistant Message**: Inserts assistant response with sources into MongoDB
8. **Response**: Serializes and returns JSON response with response and sources

All code references are based on the actual implementation in the codebase.

---

## Graph Workflow Implementation Details

### Architecture

The graph-based workflow system was implemented in Phases 1-9:

- **Phase 1**: Graph schema definition (`graph_schema.py`)
- **Phase 2**: WorkflowState model (`workflow_state.py`)
- **Phase 3**: AgentRunner compatibility layer (`agent_runner.py`)
- **Phase 4**: Node implementations (`nodes/` directory)
- **Phase 5**: Graph executor (`graph_executor.py`)
- **Phase 6**: Default workflow conversion (`default_workflow.json`)
- **Phase 7-9**: Advanced features (persistence, security, cost management, marketplace, etc.)

### Key Files

- **Graph Executor**: `workflows/graph_executor.py`
- **Graph Schema**: `workflows/graph_schema.py`
- **Graph Loader**: `workflows/graph_loader.py`
- **Workflow State**: `workflows/workflow_state.py`
- **Node Types**: `workflows/nodes/` (agent_node.py, conditional_node.py, fanout_node.py, loop_node.py, merge_node.py)
- **Default Workflow**: `workflows/default_workflow.json`
- **Workflow Factory**: `workflows/workflow_factory.py`

### Migration from Sequential to Graph

The graph workflow is functionally equivalent to the sequential workflow but provides:

- **Declarative Definition**: Workflow structure defined in JSON
- **Better Observability**: Execution trace, metrics, state tracking
- **Flexibility**: Easy to modify workflow without code changes
- **Advanced Features**: Cost tracking, debugging, governance, etc.

To enable graph workflow in `chat_sessions.py`, change:

```python
# Current (sequential):
assistant_response = await run_sequential_goal(cfg, user_message_content)

# To (graph-based):
assistant_response = await run_graph_workflow(cfg, user_message_content)
```

The graph executor will automatically fall back to sequential if graph execution fails.
