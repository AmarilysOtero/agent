# Sequential Workflow Orchestration Architecture

## Overview

This document describes the sequential workflow execution system used to coordinate multiple AI agents in a simple, linear flow. Unlike the graph-based orchestration system, sequential execution provides a straightforward agent pipeline where each agent executes in order, passing results to the next agent. This approach is ideal for simpler workflows that don't require complex conditional routing, parallel execution, or dynamic branching.

The sequential workflow is implemented as the `run_sequential_goal()` function in `workflow_factory.py` and serves as the fallback mechanism when graph-based execution fails or is unavailable.

## Table of Contents

1. [Workflow Factory](#workflow-factory)
2. [Sequential Workflow Function](#sequential-workflow-function)
3. [Agent Pipeline Construction](#agent-pipeline-construction)
4. [Agent Types and Execution](#agent-types-and-execution)
5. [State Management](#state-management)
6. [Execution Flow](#execution-flow)

---

## Workflow Factory

### Location

`src/news_reporter/workflows/workflow_factory.py`

### Purpose

The `workflow_factory.py` module provides the entry point for executing agent workflows. It offers two execution modes:

1. **Sequential execution** (primary): Traditional sequential agent execution through a defined pipeline
2. **Graph-based execution** (alternative): Uses `GraphExecutor` for complex workflows when sequential mode is insufficient

### Key Functions

#### `run_sequential_goal(cfg, goal)`

The primary entry point for sequential workflow execution.

**Flow:**

```
1. Execute Triage Agent
   └─> Analyze goal and determine routing
       ├─> Classify user intents
       ├─> Detect best database/schema (if applicable)
       └─> Set preferred search agent type

2. Select and Execute Search Agent
   └─> Based on triage results:
       ├─> SQLAgent - For SQL/PostgreSQL queries
       ├─> Neo4jGraphRAGAgent - For graph search
       └─> AiSearchAgent - Default vector search

3. Execute Assistant Agent(s)
   └─> Generate response using context:
       ├─> Single mode - One assistant
       └─> Multi mode - Multiple assistants in parallel

4. Review Loop (Max 1 iteration currently)
   └─> ReviewAgent validates response:
       ├─> Accept - Return response
       └─> Revise - Request improvements from assistant

5. Return final output
   └─> Return approved response to caller
```

**Parameters:**

- `cfg: Settings` - Application configuration with agent IDs
- `goal: str` - User query/goal

**Returns:**

- `str` - Final output from workflow execution

**Error Handling:**

- Catches agent execution errors
- Logs failures with detailed context
- Continues or stops based on error severity

#### Agent Selection Logic

The sequential workflow makes key decisions at specific pipeline stages:

```python
# Stage 1: Search Agent Selection (based on triage results)
if tri.preferred_agent == "sql" and hasattr(cfg, 'agent_id_aisearch_sql') and cfg.agent_id_aisearch_sql:
    search_agent = SQLAgent(cfg.agent_id_aisearch_sql)
    search_database_id = tri.database_id  # May be None for auto-detect
elif cfg.use_neo4j_search and cfg.agent_id_neo4j_search:
    search_agent = Neo4jGraphRAGAgent(cfg.agent_id_neo4j_search)
else:
    search_agent = AiSearchAgent(cfg.agent_id_aisearch)  # Default

# Stage 2: Assistant Selection
do_multi = ("multi" in tri.intents) or cfg.multi_route_always
# Note: reporter_ids is a legacy config name, it contains assistant agent IDs
targets = cfg.reporter_ids if do_multi else [cfg.reporter_ids[0]]

# Stage 3: Review Loop (Max 1 iteration currently)
for i in range(1, max_iters + 1):  # max_iters = 1
    verdict = await reviewer.run(goal, response)
    decision = (verdict.get("decision") or "revise").lower()
    if decision == "accept":
        return revised or response
    # Request improvement from assistant
    response = await assistant.run(goal, improve_context)
```

---

## Sequential Workflow Function

### Location

`src/news_reporter/workflows/workflow_factory.py`

### Function: `run_sequential_goal()`

The sequential workflow is implemented as a function rather than a class-based executor. It provides a simple, linear pipeline for agent coordination.

**Function Signature:**

```python
async def run_sequential_goal(cfg: Settings, goal: str) -> str
```

**Parameters:**

- `cfg: Settings` - Application configuration containing agent IDs
- `goal: str` - User query/goal

**Returns:**

- `str` - Final response from the workflow

### Purpose

The `run_sequential_goal()` function implements linear workflow execution where agents execute one after another. It manages:

- **Linear pipeline execution** - Agents run in predefined order
- **Agent selection logic** - Chooses appropriate search agent based on triage
- **Context passing** - Results flow from one agent to the next
- **Error handling** - Graceful failure management with informative messages
- **Multi-assistant support** - Parallel execution of multiple assistants if needed
- **Review validation** - Quality check with reviewer agent

### Execution Model: Linear Pipeline

The function processes agents sequentially through a fixed pipeline:

```
┌──────────────────────────────────────┐
│ 1. Triage Agent                      │
│ - Analyze goal                        │
│ - Classify intents                    │
│ - Detect database/schema              │
│ - Set preferred_agent                 │
└──────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ 2. Search Agent Selection             │
│ - If preferred_agent="sql" → SQLAgent │
│ - If use_neo4j_search → Neo4jAgent    │
│ - Default → AiSearchAgent             │
└──────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ 3. Search Agent Execution             │
│ - Retrieve context based on goal      │
│ - Return formatted results            │
└──────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ 4. Assistant Agent(s)                │
│ - Single: One assistant               │
│ - Multi: Parallel assistants          │
│ - Generate response with context      │
└──────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ 5. Review Loop (Max 1 iteration)     │
│ - ReviewAgent validates               │
│ - If approved → Return result         │
│ - If needs revision → Assistant fixes │
└──────────────────────────────────────┘
```

### Implementation Details

#### Nested Execution Function

The workflow uses a nested `run_one()` function to handle individual assistant execution:

```python
async def run_one(assistant_id: str) -> str:
    assistant = AssistantAgent(assistant_id)

    # Search step - retrieve context
    if isinstance(search_agent, SQLAgent):
        context = await search_agent.run(goal, database_id=search_database_id)
    else:
        context = await search_agent.run(goal)

    # Assistant step - generate response
    response = await assistant.run(goal, context or "")

    if not response:
        return "I apologize, but I couldn't generate a response. Please try rephrasing your question."

    # Review step (max 1 pass currently)
    max_iters = 1
    for i in range(1, max_iters + 1):
        verdict = await reviewer.run(goal, response)
        decision = (verdict.get("decision") or "revise").lower()

        if decision == "accept":
            return verdict.get("revised_script", response) or response

        # Ask assistant to improve
        improve_context = (
            f"Previous response needs improvement:\n{verdict.get('suggested_changes') or verdict.get('reason')}\n\n"
            f"Original response:\n{response}\n\n"
            f"Context available:\n{context}"
        )
        response = await assistant.run(goal, improve_context)

    return f"[After {max_iters} review passes]\n{response}"
```

#### Multi-Assistant Execution

For multi-assistant mode, the workflow executes multiple assistants in parallel:

```python
if len(targets) > 1:
    # Execute all assistants in parallel
    results = await asyncio.gather(*[run_one(rid) for rid in targets])

    # Stitch results together
    stitched = []
    for rid, out in zip(targets, results):
        stitched.append(f"### AssistantAgent={rid}\n{out}")
    return "\n\n---\n\n".join(stitched)

# Single assistant mode
return await run_one(targets[0])
```

---

## Agent Pipeline Construction

### Pipeline Definition

Sequential workflows follow a fixed pipeline structure with decision points:

```
START
  ↓
TRIAGE AGENT
  ├─→ Analyze goal
  ├─→ Classify intents
  ├─→ Detect database/schema (if AI search or unknown intent)
  └─→ Set preferred_agent ("sql", "csv", "vector", or None)
       ↓
  SEARCH AGENT SELECTION
       ├─→ preferred_agent="sql" AND agent configured? → SQLAgent
       ├─→ use_neo4j_search enabled? → Neo4jGraphRAGAgent
       └─→ Default → AiSearchAgent
            ↓
       SEARCH AGENT EXECUTION
            └─→ Retrieve context/information
                 ↓
            ASSISTANT SELECTION
                 ├─→ "multi" intent OR multi_route_always? → Multiple assistants
                 └─→ Default → Single assistant
                      ↓
                 ASSISTANT EXECUTION
                      ├─→ Single: Direct generation with context
                      └─→ Multi: Parallel execution + stitch results
                           ↓
                      REVIEW LOOP (max 1 iteration)
                           ├─→ ReviewAgent validates
                           ├─→ Decision = accept? → END
                           └─→ Else → Assistant revises → loop back
                                ↓
                           FINAL OUTPUT
END
```

### Agent Execution Sequence

#### Stage 1: Triage (Required)

**Purpose:** Understand user goal and determine routing

**Agent:** `TriageAgent`

**Inputs:**

```python
goal: str  # User query
```

**Outputs:**

```python
IntentResult(
    intents: List[str],              # Detected intents (e.g., ["ai_search", "multi"])
    confidence: float,               # Confidence in classification
    rationale: str,                  # Explanation
    targets: List[str],              # Target entities (optional)
    database_type: Optional[str],    # "postgresql", "csv", "other"
    database_id: Optional[str],      # Best matching database ID
    preferred_agent: Optional[str]   # "sql", "csv", "vector"
)
```

    "confidence": float            # Confidence in routing decision

}

````

#### Stage 2: Search Agent Execution (Required)

**Execution:** Based on `triage.preferred_agent`

**Options:**

1. **SQLAgent** - If preferred_agent == "sql"

   ```python
   inputs = {
       "goal": state.goal,
       "database_id": state.triage.database_id
   }
   output = str  # SQL results
````

2. **Neo4jGraphRAGAgent** - If use_neo4j_search enabled

   ```python
   inputs = {
       "goal": state.goal,
       "neo4j_query": Optional[str]
   }
   output = str  # Graph RAG results
   ```

3. **AiSearchAgent** - Default
   ```python
   inputs = {
       "goal": state.goal
   }
   output = str  # Search results
   ```

**Special Feature: Schema Detection**

When the triage agent detects "ai_search" or "unknown" intents, it automatically performs schema detection:

```python
# 1. List all available databases
all_databases = schema_retriever.list_databases()

# 2. Find best matching database for the query
best_db_id = schema_retriever.find_best_database(query=goal)

# 3. Determine database type and set preferred agent
if "postgresql" in db_type:
    preferred_agent = "sql"
    database_type = "postgresql"
elif "csv" in db_type or "csv" in db_name:
    preferred_agent = "csv"
    database_type = "csv"
else:
    preferred_agent = "vector"
    database_type = "other"
```

#### Stage 2: Search Agent Execution (Required)

**Execution:** Based on `triage.preferred_agent`

**Options:**

1. **SQLAgent** - If preferred_agent == "sql" and agent configured

   ```python
   # Agent creation
   search_agent = SQLAgent(cfg.agent_id_aisearch_sql)

   # Execution
   context = await search_agent.run(goal, database_id=search_database_id)
   # Returns: SQL query results as formatted string
   ```

2. **Neo4jGraphRAGAgent** - If use_neo4j_search enabled

   ```python
   # Agent creation
   search_agent = Neo4jGraphRAGAgent(cfg.agent_id_neo4j_search)

   # Execution
   context = await search_agent.run(goal)
   # Returns: Graph query results as formatted string
   ```

3. **AiSearchAgent** - Default vector search

   ```python
   # Agent creation
   search_agent = AiSearchAgent(cfg.agent_id_aisearch)

   # Execution
   context = await search_agent.run(goal)
   # Returns: Vector search results as formatted string
   ```

#### Stage 3: Assistant Execution (Required)

**Agent:** `AssistantAgent` (formerly called NewsReporterAgent)

**Single Assistant Mode:**

```python
assistant = AssistantAgent(assistant_id)

# Execute with goal and retrieved context
response = await assistant.run(goal, context or "")

# Fallback if no response
if not response:
    response = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
```

**Multi-Assistant Mode:**

```python
# Determine number of assistants
# Note: reporter_ids is a legacy config name, it contains assistant agent IDs
targets = cfg.reporter_ids if do_multi else [cfg.reporter_ids[0]]

# Execute all assistants in parallel
results = await asyncio.gather(*[run_one(rid) for rid in targets])

# Stitch results together
stitched = []
for rid, out in zip(targets, results):
    stitched.append(f"### AssistantAgent={rid}\n{out}")
final_output = "\n\n---\n\n".join(stitched)
```

#### Stage 4: Review Loop (Currently Max 1 Iteration)

**Agent:** `ReviewAgent`

**Iteration Inputs:**

```python
verdict = await reviewer.run(goal, response)
```

**Iteration Output:**

```python
{
    "decision": "accept" | "revise",
    "reason": str,                    # Explanation of decision
    "suggested_changes": str,         # What to improve (empty if accept)
    "revised_script": str             # Improved version (empty if accept)
}
```

**Loop Logic:**

```python
max_iters = 1  # Currently limited to 1 iteration
for i in range(1, max_iters + 1):
    verdict = await reviewer.run(goal, response)
    decision = (verdict.get("decision") or "revise").lower()

    if decision == "accept":
        # Return either the revised script from reviewer or original response
        return verdict.get("revised_script", response) or response

    # Build improvement context for assistant
    improve_context = (
        f"Previous response needs improvement:\n{verdict.get('suggested_changes') or verdict.get('reason')}\n\n"
        f"Original response:\n{response}\n\n"
        f"Context available:\n{context}"
    )

    # Ask assistant to revise
    response = await assistant.run(goal, improve_context)

# Max iterations reached
return f"[After {max_iters} review passes]\n{response}"
```

---

## Agent Types and Execution

### Agent Architecture

All agents in the sequential workflow use Palantir Foundry agents via `run_foundry_agent()`. They follow a consistent pattern:

```python
class SomeAgent:
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, *args) -> ReturnType:
        # Construct prompt
        prompt = build_prompt(*args)

        # Call Foundry agent
        result = run_foundry_agent(self._id, prompt)

        # Parse and return result
        return parse_result(result)
```

### Included Agents

#### 1. TriageAgent

**Location:** `src/news_reporter/agents/triage_agent.py`

**Purpose:** Intent classification and database routing

**Location:** `src/news_reporter/agents/triage_agent.py`

**Purpose:** Intent classification and database routing

**Execution:**

```python
triage = TriageAgent(cfg.agent_id_triage)
tri = await triage.run(goal)

# Returns IntentResult with:
# - intents: List of detected intents
# - preferred_agent: "sql", "csv", "vector", or None
# - database_id: Best matching database ID (if detected)
# - database_type: "postgresql", "csv", or "other"
# - confidence: Confidence score
# - rationale: Explanation
```

**Key Features:**

- Uses Foundry agent for LLM-based intent classification
- Automatic schema detection for ai_search/unknown intents
- Lists all available databases for transparency
- Finds best matching database using SchemaRetriever
- Sets preferred_agent based on database type

#### 2. Search Agents

**AiSearchAgent:**

```python
# Location: src/news_reporter/agents/ai_search_agent.py
# Purpose: Vector search using Azure AI Search or Neo4j GraphRAG

search_agent = AiSearchAgent(cfg.agent_id_aisearch)
context = await search_agent.run(goal)
# Returns: Formatted search results as string
```

**SQLAgent:**

```python
# Location: src/news_reporter/agents/sql_agent.py
# Purpose: SQL queries with fallback to CSV and Vector search

search_agent = SQLAgent(cfg.agent_id_aisearch_sql)
context = await search_agent.run(goal, database_id=database_id)
# Returns: SQL query results formatted as string
# Supports: PostgreSQL, CSV files, vector search fallback
```

**Neo4jGraphRAGAgent:**

```python
# Location: src/news_reporter/agents/neo4j_graphrag_agent.py
# Purpose: Graph-based RAG using Neo4j

search_agent = Neo4jGraphRAGAgent(cfg.agent_id_neo4j_search)
context = await search_agent.run(goal)
# Returns: Graph query results as string
```

#### 3. AssistantAgent

**Location:** `src/news_reporter/agents/assistant_agent.py`

**Purpose:** Generate natural language responses using RAG context

**Execution:**

```python
assistant = AssistantAgent(foundry_agent_id)
response = await assistant.run(query, context)

# Prompt structure:
# - User Question: {query}
# - Retrieved Context: {context}
# - Instructions: Answer using context, be conversational
# - Fallback: Provide general guidance if no specific docs found
```

**Key Features:**

- Uses Foundry agent for response generation
- Handles empty context gracefully
- Allows general guidance when no docs found
- Cites sources from context when available

**Alias:** `NewsReporterAgent = AssistantAgent` (backward compatibility)

#### 4. ReviewAgent

**Location:** `src/news_reporter/agents/review_agent.py`

**Purpose:** Validate assistant responses for accuracy and completeness

**Execution:**

```python
reviewer = ReviewAgent(cfg.agent_id_reviewer)
verdict = await reviewer.run(goal, candidate_response)

# Returns dict with:
# - decision: "accept" or "revise"
# - reason: Brief explanation
# - suggested_changes: What to improve (empty if accept)
# - revised_script: Improved version (empty if accept)
```

**Review Criteria:**

1. **Accuracy** - Does it correctly answer the question?
2. **Completeness** - Is the answer sufficient?
3. **Clarity** - Is it easy to understand?

**Error Handling:**

- Fallsafe: Accepts response on parse error to avoid infinite loops
- Always returns valid dict structure
- Validates decision field ("accept" or "revise")

---

## State Management

### State Flow in Sequential Workflow

Unlike a class-based executor with a `WorkflowState` object, the sequential workflow passes data through local variables and function parameters:

```python
async def run_sequential_goal(cfg: Settings, goal: str) -> str:
    # Stage 1: Triage
    tri = await triage.run(goal)  # IntentResult object

    # Stage 2: Search Agent Selection
    if tri.preferred_agent == "sql" and cfg.agent_id_aisearch_sql:
        search_agent = SQLAgent(cfg.agent_id_aisearch_sql)
        search_database_id = tri.database_id
    elif cfg.use_neo4j_search:
        search_agent = Neo4jGraphRAGAgent(cfg.agent_id_neo4j_search)
    else:
        search_agent = AiSearchAgent(cfg.agent_id_aisearch)

    # Stage 3: Determine assistant mode
    do_multi = ("multi" in tri.intents) or cfg.multi_route_always
    targets = cfg.reporter_ids if do_multi else [cfg.reporter_ids[0]]

    # Stage 4: Execute assistants (nested function)
    async def run_one(assistant_id: str) -> str:
        # Search
        context = await search_agent.run(goal, ...)

        # Assistant
        response = await assistant.run(goal, context)

        # Review loop
        for i in range(1, max_iters + 1):
            verdict = await reviewer.run(goal, response)
            if verdict["decision"] == "accept":
                return response
            response = await assistant.run(goal, improve_context)

        return response

    # Execute and return
    if len(targets) > 1:
        results = await asyncio.gather(*[run_one(rid) for rid in targets])
        return stitch_results(results)
    return await run_one(targets[0])
```

### Data Flow Through Pipeline

```
┌────────────────────┐
│ Input              │
│ goal: str          │
│ cfg: Settings      │
└────────────────────┘
         ↓
    TriageAgent
         ↓
┌────────────────────┐
│ tri: IntentResult  │
│ - intents          │
│ - preferred_agent  │
│ - database_id      │
└────────────────────┘
         ↓
  Select SearchAgent
         ↓
┌────────────────────┐
│ search_agent       │
│ search_database_id │
└────────────────────┘
         ↓
   Execute Search
         ↓
┌────────────────────┐
│ context: str       │
└────────────────────┘
         ↓
  Execute Assistant
         ↓
┌────────────────────┐
│ response: str      │
└────────────────────┘
         ↓
   Review Loop
         ↓
┌────────────────────┐
│ final: str         │
└────────────────────┘
         ↓
   Return result
```

### Key Variables

**Top-level variables:**

- `goal: str` - User query (immutable throughout)
- `tri: IntentResult` - Triage results
- `search_agent` - Selected search agent instance
- `search_database_id: Optional[str]` - Database ID for SQL queries
- `targets: List[str]` - Assistant agent IDs to use
- `do_multi: bool` - Whether to use multiple assistants

**run_one() function scope:**

- `context: str` - Retrieved context from search
- `response: str` - Current response being refined
- `verdict: dict` - Review feedback
- `improve_context: str` - Feedback for assistant revision

---

## Execution Flow

### Complete Execution Lifecycle

```
┌─────────────────────────────────────────────────────────┐
│ 1. Entry Point                                           │
│    run_sequential_goal(cfg, goal)                        │
│    - Called from workflow_factory                        │
│    - Fallback when graph workflow fails                  │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Triage Execution                                      │
│    triage = TriageAgent(cfg.agent_id_triage)             │
│    tri = await triage.run(goal)                          │
│    - Analyze goal using Foundry agent                    │
│    - Classify intents                                    │
│    - Auto-detect database/schema if needed               │
│    - Set preferred_agent and database_id                 │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Search Agent Selection                                │
│    - Check tri.preferred_agent                           │
│    - Check config for agent availability                 │
│    - Instantiate appropriate search agent                │
│    - Set search_database_id for SQL queries              │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Assistant Mode Selection                              │
│    do_multi = ("multi" in tri.intents) or                │
│               cfg.multi_route_always                     │
│    targets = cfg.reporter_ids if do_multi                │
│              else [cfg.reporter_ids[0]]                  │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 5. Define run_one() Nested Function                      │
│    async def run_one(assistant_id: str) -> str:          │
│    - Creates closure over search_agent, goal, etc.       │
│    - Handles: search → assistant → review loop           │
└─────────────────────────────────────────────────────────┘
                        │
                        ├─────────────────────────────────┐
                        │                                 │
                ┌───────▼────────┐         ┌──────▼──────┐
                │ Single Mode    │         │ Multi Mode  │
                │ 1 Assistant    │         │ N Assistants│
                └─────┬──────────┘         └──────┬──────┘
                      │                           │
                ┌─────▼──────────┐         ┌──────▼──────┐
                │ run_one(id)    │         │ gather()    │
                │ Sequential     │         │ Parallel    │
                └─────┬──────────┘         └──────┬──────┘
                      │                           │
                      │                    ┌──────▼──────┐
                      │                    │ Stitch      │
                      │                    │ Results     │
                      │                    └──────┬──────┘
                      │                           │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────┐
│ 6. Inside run_one(): Search Execution                    │
│    if isinstance(search_agent, SQLAgent):                │
│        context = await search_agent.run(                 │
│            goal, database_id=search_database_id)         │
│    else:                                                 │
│        context = await search_agent.run(goal)            │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 7. Inside run_one(): Assistant Execution                 │
│    assistant = AssistantAgent(assistant_id)              │
│    response = await assistant.run(goal, context or "")   │
│    - Generate response using retrieved context           │
│    - Fallback message if no response generated           │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 8. Inside run_one(): Review Loop                         │
│    for i in range(1, max_iters + 1):  # max_iters=1     │
│    - Execute reviewer.run(goal, response)                │
│    - Check decision (accept/revise)                       │
│    - If accept: return response                          │
│    - If revise: build improve_context and regenerate     │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 9. Return Final Output                                   │
│    - Single mode: return run_one(targets[0])             │
│    - Multi mode: return stitched results                 │
│    - Log execution context                               │
└─────────────────────────────────────────────────────────┘
```

### Error Handling Strategy

#### Agent-Level Error Handling

Each agent has built-in error handling that raises `RuntimeError` with informative messages:

```python
# TriageAgent
try:
    raw = run_foundry_agent(self._id, content).strip()
except RuntimeError as e:
    logger.error("TriageAgent Foundry error: %s", e)
    raise RuntimeError(
        f"Triage agent failed: {str(e)}. "
        "Please check your Foundry access and agent configuration."
    ) from e

# AssistantAgent
try:
    return run_foundry_agent(self._id, prompt)
except RuntimeError as e:
    logger.error("AssistantAgent Foundry error: %s", e)
    raise RuntimeError(
        f"Assistant agent failed: {str(e)}. "
        "Please check your Foundry access and agent configuration."
    ) from e

# ReviewAgent
try:
    data = run_foundry_agent_json(self._id, prompt, system_hint="...")
except RuntimeError as e:
    logger.error("ReviewAgent Foundry error: %s", e)
    raise RuntimeError(
        f"Review agent failed: {str(e)}. "
        "Please check your Foundry access and agent configuration."
    ) from e
```

#### Parsing Error Handling

Agents handle JSON parsing errors gracefully:

```python
# TriageAgent - returns default result on parse error
try:
    data = json.loads(raw)
    result = IntentResult(**data)
    return result
except (json.JSONDecodeError, ValidationError) as e:
    logger.error("Triage parse error: %s", e)
    return IntentResult(intents=["unknown"], confidence=0.0, rationale="parse_error")

# ReviewAgent - accepts response on parse error to avoid infinite loops
try:
    if not isinstance(data, dict) or "decision" not in data:
        raise ValueError("Invalid JSON shape from reviewer")
    decision = (data.get("decision") or "revise").lower()
    return {
        "decision": decision if decision in {"accept", "revise"} else "revise",
        "reason": data.get("reason", ""),
        "suggested_changes": data.get("suggested_changes", ""),
        "revised_script": data.get("revised_script", candidate_response),
    }
except Exception as e:
    logger.error("Review parse error: %s", e)
    # Fail-safe: accept last response to avoid infinite loops
    return {
        "decision": "accept",
        "reason": "parse_error",
        "suggested_changes": "",
        "revised_script": candidate_response,
    }
```

#### Workflow-Level Error Handling

The workflow handles missing responses:

```python
# In run_one() function
response = await assistant.run(goal, context or "")

if not response:
    logger.warning(f"🔍 Workflow: No response generated - returning default message")
    return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
```

#### Fallback to Sequential Workflow

The graph workflow falls back to sequential on errors:

```python
# In run_graph_workflow()
try:
    # ... execute graph workflow ...
    return result
except (ValueError, FileNotFoundError) as e:
    # Configuration errors - fall back
    logger.error(f"Graph workflow configuration error: {e}", exc_info=True)
    logger.warning("Falling back to sequential workflow due to configuration error")
    return await run_sequential_goal(cfg, goal)
except Exception as e:
    # Execution errors - fall back
    logger.error(f"Graph workflow execution failed: {e}", exc_info=True)
    logger.warning("Falling back to sequential workflow")
    return await run_sequential_goal(cfg, goal)
```

---

## Comparison: Sequential vs Graph Execution

### Sequential Workflow

**Implementation:**

- Function-based: `run_sequential_goal(cfg, goal)` in `workflow_factory.py`
- ~90 lines of code
- Linear execution with nested `run_one()` function

**Characteristics:**

- Linear pipeline of agents
- Fixed execution order
- Simple variable passing (no state object)
- Straightforward error handling with fallback messages
- Nested function closure for context sharing
- Good for: Basic workflows, simple agent chains, predictable paths

**Example Flow:**

```
Triage → [Search Agent Selection] → [Search] → [Assistant] → [Review] → Output
```

**Advantages:**

- Simple to understand and debug
- Low overhead
- Easy to modify
- Clear data flow
- Fast execution

**Limitations:**

- No conditional branching
- No parallel branches (except multi-assistant mode)
- No loops (except review loop with max 1 iteration)
- Fixed pipeline structure

### Graph Workflow

**Implementation:**

- Class-based: `GraphExecutor` in `graph_executor.py`
- ~2000+ lines of code
- Queue-based execution model

**Characteristics:**

- Complex node types (agent, fanout, loop, conditional, merge)
- Dynamic routing based on conditions
- Parallel branches with fanout/merge
- Loop iterations with configurable limits
- Good for: Complex branching, multi-path workflows, conditional execution

**Example Flow:**

```
Triage → [Conditional] → [SQL/Neo4j/AiSearch] → Fanout → [Assistant 1, 2, 3] → Merge → Loop → Review
```

**Advantages:**

- Complex workflow patterns
- Conditional routing
- Parallel execution
- Loop constructs
- Configurable via JSON

**Limitations:**

- Higher complexity
- More overhead
- Harder to debug
- Requires workflow definition

### When to Use Sequential

1. **Simple agent chains** - Few agents, fixed order
2. **Rapid development** - Quick iteration without JSON configs
3. **Easy debugging** - Linear execution path
4. **Resource constrained** - Lower memory overhead
5. **Fallback mode** - When graph workflow fails
6. **Learning purposes** - Understanding agent coordination

### When to Use Graph

1. **Complex workflows** - Multiple branching paths
2. **Conditional routing** - Different agents based on analysis
3. **Parallel execution** - Multiple agents simultaneously
4. **Dynamic loops** - Iterative refinement beyond simple review
5. **Multi-path scenarios** - Different handling for different intent types
6. **Configurable workflows** - Non-developer workflow modifications

---

## Summary

The sequential workflow system provides:

1. **Simple Function-Based Pipeline** - No classes, just a function with nested execution
2. **Flexible Agent Selection** - Different search agents based on triage analysis
3. **Variable-Based State** - Simple data flow through function parameters
4. **Error Handling** - Graceful failure with informative messages
5. **Multi-Assistant Support** - Parallel execution via asyncio.gather()
6. **Review Loop** - Single-iteration validation (currently max_iters=1)
7. **Easy to Understand** - Linear code flow matching business logic
8. **Foundry Integration** - All agents use run_foundry_agent()
9. **Schema Detection** - Automatic database detection for routing

The sequential model is ideal for workflows where the execution path is clear and straightforward, while the graph model is better suited for complex scenarios with dynamic routing and parallel execution patterns.

**Key Files:**

- `src/news_reporter/workflows/workflow_factory.py` - Main sequential workflow function
- `src/news_reporter/agents/triage_agent.py` - Intent classification
- `src/news_reporter/agents/assistant_agent.py` - Response generation
- `src/news_reporter/agents/review_agent.py` - Response validation
- `src/news_reporter/agents/sql_agent.py` - SQL search
- `src/news_reporter/agents/ai_search_agent.py` - Vector search
- `src/news_reporter/agents/neo4j_graphrag_agent.py` - Graph search
