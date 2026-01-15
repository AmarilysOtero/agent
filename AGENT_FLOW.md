# Agent Flow - Code Implementation Trace

This document traces the actual code flow for chat sessions and agent interactions in the Agent service, based on the real implementation.

## Table of Contents

1. [Entry Point](#entry-point)
2. [Authentication](#authentication)
3. [Chat Session Management](#chat-session-management)
4. [Message Processing](#message-processing)
5. [Agent Workflow Execution](#agent-workflow-execution)
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
- `from ..workflows.workflow_factory import run_sequential_goal`
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

**Function**: `run_sequential_goal(cfg: Settings, goal: str)` (lines 11-120)

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

**File**: `agents/agents.py` → `AiSearchAgent.run()` (lines 400-500)

```python
async def run(self, goal: str) -> str:
    # Step 1: Call Azure Cognitive Search
    from ..tools.azure_search import hybrid_search

    results = hybrid_search(
        query=goal,
        top_k=10
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

| Component                  | File                            | Function/Class                             | Lines   |
| -------------------------- | ------------------------------- | ------------------------------------------ | ------- |
| **API Endpoint**           | `routers/chat_sessions.py`      | `add_message()`                            | 373-545 |
| **Authentication**         | `routers/auth.py`               | `get_current_user()`                       | 278-295 |
| **Workflow Orchestration** | `workflows/workflow_factory.py` | `run_sequential_goal()`                    | 11-120  |
| **Triage Agent**           | `agents/agents.py`              | `TriageAgent.run()`                        | 148-200 |
| **SQL Agent**              | `agents/agents.py`              | `SQLAgent.run()`                           | 800-900 |
| **Neo4j GraphRAG Agent**   | `agents/agents.py`              | `Neo4jGraphRAGAgent.run()`                 | 600-700 |
| **Azure Search Agent**     | `agents/agents.py`              | `AiSearchAgent.run()`                      | 400-500 |
| **Reporter Agent**         | `agents/agents.py`              | `NewsReporterAgent.run()`                  | 300-400 |
| **Review Agent**           | `agents/agents.py`              | `ReviewAgent.run()`                        | 500-600 |
| **Neo4j Search**           | `tools/neo4j_graphrag.py`       | `graphrag_search()`                        | 200-240 |
| **Neo4j Retriever**        | `tools/neo4j_graphrag.py`       | `Neo4jGraphRAGRetriever.hybrid_retrieve()` | 57-140  |
| **Foundry Runner**         | `foundry_runner.py`             | `run_foundry_agent()`                      | 100-200 |
| **Foundry JSON Runner**    | `foundry_runner.py`             | `run_foundry_agent_json()`                 | 200-300 |

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
