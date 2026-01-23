# Sequential Workflow Orchestration Architecture

## Overview

This document describes the sequential workflow execution system used to coordinate multiple AI agents in a simple, linear flow. Unlike the graph-based orchestration system, sequential execution provides a straightforward agent pipeline where each agent executes in order, passing results to the next agent. This approach is ideal for simpler workflows that don't require complex conditional routing, parallel execution, or dynamic branching.

## Table of Contents

1. [Workflow Factory](#workflow-factory)
2. [Sequential Executor](#sequential-executor)
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
1. Initialize workflow state
   └─> Create WorkflowState with goal
       └─> Prepares shared state for agents

2. Execute agent pipeline
   └─> Execute agents in order:
       ├─> TriageAgent - Analyze goal
       ├─> SearchAgent - Gather information
       ├─> ReporterAgent - Generate content
       └─> ReviewerAgent - Validate and improve (≤3 iterations)

3. Handle multi-reporter fanout
   └─> If multiple reporters needed:
       ├─> Execute reporters in parallel
       └─> Stitch results together

4. Return final output
   └─> Collect and format final result
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

The sequential executor makes key decisions at specific pipeline stages:

```python
# Stage 1: Search Agent Selection
if "sql" in triage.intents and cfg.agent_id_aisearch_sql:
    search_agent = SQLAgent(cfg)
elif cfg.use_neo4j_search:
    search_agent = Neo4jGraphRAGAgent(cfg)
else:
    search_agent = AiSearchAgent(cfg)  # Default

# Stage 2: Reporter Selection
if "multi" in triage.intents or cfg.multi_route_always:
    reporters = [ReporterAgent(cfg) for _ in range(num_reporters)]
else:
    reporters = [ReporterAgent(cfg)]

# Stage 3: Review Loop
for iteration in range(max_review_iterations):
    feedback = reviewer.execute(script)
    if feedback.decision == "accept":
        break
    script = reporter.revise(feedback)
```

---

## Sequential Executor

### Location

`src/news_reporter/workflows/sequential_executor.py`

### Purpose

The `SequentialExecutor` implements linear workflow execution where agents execute one after another. It manages:

- **Linear pipeline execution** - Agents run in predefined order
- **State passing** - Results flow from one agent to the next
- **Error handling** - Graceful failure management
- **Agent orchestration** - Creation and lifecycle management
- **Result aggregation** - Combining multi-agent outputs
- **Iteration control** - Loop management for review cycles

### Architecture

#### Core Components

```python
class SequentialExecutor:
    - config: Settings                # Application configuration
    - runner: AgentRunner             # Agent execution layer
    - triage_agent: TriageAgent       # Initial analyzer
    - search_agent: SearchAgent       # Information gathering
    - reporter_agents: List           # Content generators
    - reviewer_agent: ReviewerAgent   # Quality validator
    - state: WorkflowState            # Shared execution state
    - error_handler: ErrorHandler     # Failure management
    - metrics_collector: Metrics      # Performance tracking
```

#### Execution Model: Linear Pipeline

The executor processes agents sequentially through a fixed pipeline:

```
┌──────────────────────────────────────┐
│ 1. Triage Agent                      │
│ - Analyze goal                        │
│ - Determine intents                   │
│ - Route selection                     │
└──────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ 2. Search Agent Selection             │
│ - If SQL intent → SQLAgent            │
│ - If Graph search → Neo4jAgent        │
│ - Default → AiSearchAgent             │
└──────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ 3. Reporter Agent(s)                 │
│ - Single: Sequential execution        │
│ - Multi: Parallel execution           │
│ - Aggregate results                   │
└──────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ 4. Review Loop (Max 3 iterations)    │
│ - ReviewerAgent validates            │
│ - If approved → Return result         │
│ - If needs revision → Loop back       │
└──────────────────────────────────────┘
```

### Execution Flow

#### 1. Initialization (`execute()`)

```python
async def execute(self, goal: str) -> str:
    # 1. Create WorkflowState
    state = WorkflowState(goal=goal)

    # 2. Initialize error handler
    error_handler = ErrorHandler()

    # 3. Try-catch wrapper
    try:
        result = await self._execute_pipeline(state)
        return result
    except Exception as e:
        # Log error
        error_handler.record_error(e)
        # Return fallback
        return error_handler.get_fallback_response()
```

#### 2. Pipeline Execution (`_execute_pipeline()`)

```python
async def _execute_pipeline(self, state: WorkflowState) -> str:
    # Stage 1: Triage
    triage_result = await self._execute_triage(state)
    state.set("triage", triage_result)

    # Stage 2: Search Agent Selection
    search_agent = self._select_search_agent(triage_result)
    search_result = await self._execute_search(search_agent, state)
    state.set("latest", search_result)

    # Stage 3: Reporter(s)
    if self._is_multi_report_mode(triage_result):
        reporter_results = await self._execute_multi_reporters(state)
        state.set("drafts", reporter_results)
    else:
        reporter_result = await self._execute_single_reporter(state)
        state.set("drafts", {"default": reporter_result})

    # Stage 4: Review Loop
    final_result = await self._execute_review_loop(state)
    state.set("final", final_result)

    return final_result
```

#### 3. Agent Execution (`_execute_triage()`)

```python
async def _execute_triage(self, state: WorkflowState) -> Dict:
    # Create agent instance
    agent = self.triage_agent

    # Prepare inputs
    inputs = {
        "goal": state.goal
    }

    # Execute with error handling
    try:
        result = await agent.execute(inputs)

        # Validate output
        if not self._validate_triage_output(result):
            return self._default_triage_result()

        # Return result
        return result
    except Exception as e:
        # Log error
        self.logger.error(f"Triage execution failed: {e}")
        # Return default
        return self._default_triage_result()
```

#### 4. Search Agent Selection

```python
def _select_search_agent(self, triage_result: Dict) -> SearchAgent:
    """Select search agent based on triage output"""

    preferred_agent = triage_result.get("preferred_agent", "default")

    # Priority 1: SQL Agent
    if preferred_agent == "sql" and self.config.agent_id_aisearch_sql:
        return SQLAgent(self.config)

    # Priority 2: Neo4j GraphRAG
    if self.config.use_neo4j_search:
        return Neo4jGraphRAGAgent(self.config)

    # Default: AiSearch
    return AiSearchAgent(self.config)
```

#### 5. Multi-Reporter Execution

```python
async def _execute_multi_reporters(self, state: WorkflowState) -> Dict:
    """Execute multiple reporters in parallel"""

    # Create reporter instances
    num_reporters = state.triage.get("num_reporters", 2)
    reporters = [
        ReporterAgent(self.config)
        for _ in range(num_reporters)
    ]

    # Execute in parallel
    tasks = [
        reporter.execute({
            "goal": state.goal,
            "latest_news": state.latest
        })
        for reporter in reporters
    ]
    results = await asyncio.gather(*tasks)

    # Aggregate results with keys
    return {
        f"reporter_{i+1}": result
        for i, result in enumerate(results)
    }
```

#### 6. Review Loop Execution

```python
async def _execute_review_loop(self, state: WorkflowState) -> str:
    """Execute review loop with max 3 iterations"""

    max_iterations = 3
    current_script = state.get("drafts.default")

    for iteration in range(max_iterations):
        # Execute reviewer
        feedback = await self.reviewer_agent.execute({
            "topic": state.goal,
            "candidate_script": current_script
        })

        # Check decision
        if feedback.get("decision") == "accept":
            return current_script

        # Request revision from reporter
        current_script = await self.reporter_agent.execute({
            "goal": state.goal,
            "feedback": feedback,
            "current_script": current_script
        })

    # Max iterations reached, return current script
    return current_script
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
  ├─→ Determine intents (sql, multi, neo4j, etc.)
  └─→ Set routing preferences
       ↓
  SEARCH AGENT SELECTION
       ├─→ SQL intent? → SQLAgent
       ├─→ Graph search? → Neo4jGraphRAGAgent
       └─→ Default → AiSearchAgent
            ↓
       SEARCH AGENT EXECUTION
            └─→ Retrieve information
                 ↓
            REPORTER SELECTION
                 ├─→ Multi intent? → Multiple reporters
                 └─→ Default → Single reporter
                      ↓
                 REPORTER EXECUTION
                      ├─→ Single: Direct generation
                      └─→ Multi: Parallel execution + stitch
                           ↓
                      REVIEW LOOP (max 3 iterations)
                           ├─→ ReviewerAgent validates
                           ├─→ Decision = accept? → END
                           └─→ Else → ReporterAgent revises → loop back
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
{
    "goal": str                    # User query
}
```

**Outputs:**

```python
{
    "intents": List[str],          # Detected intents (e.g., ["sql", "multi"])
    "preferred_agent": str,        # "sql", "neo4j", or "default"
    "num_reporters": int,          # Number of reporters if multi
    "database_id": Optional[str],  # Target database if SQL
    "confidence": float            # Confidence in routing decision
}
```

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
   ```

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

#### Stage 3: Reporter Execution (Required)

**Single Reporter Mode:**

```python
inputs = {
    "goal": state.goal,
    "latest_news": state.latest
}
output = str  # Generated script
```

**Multi-Reporter Mode:**

```python
# Each reporter receives same inputs
# Executed in parallel
# Results aggregated as:
{
    "reporter_1": str,
    "reporter_2": str,
    ...
}

# Stitched together if needed
final_output = stitch(reporter_1, reporter_2, ...)
```

#### Stage 4: Review Loop (Optional but Default)

**Iteration Control:** Max 3 iterations

**Iteration Inputs:**

```python
{
    "topic": state.goal,
    "candidate_script": current_script
}
```

**Iteration Output:**

```python
{
    "decision": "accept" | "revise",
    "feedback": str,
    "score": float
}
```

**Loop Logic:**

- If decision == "accept" → Exit loop, return script
- If decision == "revise" → Revise script, continue loop
- If max iterations reached → Exit loop, return current script

---

## Agent Types and Execution

### Agent Base Class

All agents inherit from `BaseAgent`:

```python
class BaseAgent(ABC):
    def __init__(self, config: Settings):
        self.config = config
        self.llm = None
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def execute(self, inputs: Dict[str, Any]) -> Any:
        """Execute agent logic with given inputs"""
        pass

    async def execute_with_retry(self, inputs, max_retries=2):
        """Execute with retry on failure"""
        for attempt in range(max_retries):
            try:
                return await self.execute(inputs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

### Included Agents

#### 1. TriageAgent

**Purpose:** Analyze user goal and determine routing

**Execution:**

```python
# 1. Extract intents from goal
intents = nlp.extract_intents(goal)

# 2. Determine preferred search agent
if "sql" in intents and has_sql_agent:
    preferred = "sql"
elif "graph" in intents:
    preferred = "neo4j"
else:
    preferred = "default"

# 3. Determine multi-reporter mode
multi_mode = "multi" in intents or config.multi_route_always

# 4. Return routing decision
return {
    "intents": intents,
    "preferred_agent": preferred,
    "num_reporters": 2 if multi_mode else 1,
    ...
}
```

#### 2. SearchAgent (Variants)

**AiSearchAgent:**

```python
# 1. Vectorize goal
query_vector = embedder.embed(goal)

# 2. Search knowledge base
results = vector_db.search(query_vector, top_k=5)

# 3. Format results
formatted = format_search_results(results)

# 4. Return formatted results
return formatted
```

**SQLAgent:**

```python
# 1. Generate SQL query
sql_query = llm.generate_sql(goal, database_schema)

# 2. Execute query
results = database.execute(sql_query, database_id)

# 3. Format results
formatted = format_sql_results(results)

# 4. Return formatted results
return formatted
```

**Neo4jGraphRAGAgent:**

```python
# 1. Generate Cypher query
cypher_query = llm.generate_cypher(goal)

# 2. Execute graph query
results = neo4j.execute(cypher_query)

# 3. Format results
formatted = format_graph_results(results)

# 4. Return formatted results
return formatted
```

#### 3. ReporterAgent

**Purpose:** Generate content based on research

**Execution:**

```python
# 1. Format context
context = format_context(goal, latest_news)

# 2. Generate script
prompt = create_reporter_prompt(context, style)
script = llm.generate(prompt)

# 3. Validate script
if not validate_script_quality(script):
    script = regenerate_with_constraints(prompt)

# 4. Return script
return script
```

#### 4. ReviewerAgent

**Purpose:** Validate and provide feedback

**Execution:**

```python
# 1. Analyze script
analysis = llm.analyze_script(
    candidate_script,
    topic,
    criteria=[accuracy, clarity, completeness]
)

# 2. Determine decision
decision = "accept" if analysis.score > threshold else "revise"

# 3. Generate feedback
feedback = llm.generate_feedback(analysis)

# 4. Return verdict
return {
    "decision": decision,
    "feedback": feedback,
    "score": analysis.score,
    "issues": analysis.issues
}
```

---

## State Management

### WorkflowState

The `WorkflowState` is a simple dictionary-like object passed through the pipeline:

```python
class WorkflowState:
    goal: str                           # Original user goal
    triage: Optional[Dict]              # Triage agent output
    latest: Optional[str]               # Search results
    drafts: Optional[Dict[str, str]]    # Reporter outputs
    final: Optional[str]                # Final approved output
    logs: List[Dict]                    # Execution logs
```

### State Access Patterns

#### Simple Attribute Access

```python
# Get/Set at top level
state.goal = "new goal"
current_goal = state.goal

# Get nested value
triage_result = state.triage
preferred_agent = state.triage.get("preferred_agent")

# Set nested value
state.triage = {"intents": ["sql"], "preferred_agent": "sql"}
```

#### Dot Notation (Optional)

```python
# Get using dot notation
value = state.get("triage.preferred_agent")

# Set using dot notation
state.set("drafts.reporter_1", "script text")
```

### State Flow Through Pipeline

```
┌────────────────────┐
│ Initial State      │
│ goal: "user query" │
└────────────────────┘
         ↓
    TriageAgent
         ↓
┌────────────────────┐
│ After Triage       │
│ + triage: {...}    │
└────────────────────┘
         ↓
    SearchAgent
         ↓
┌────────────────────┐
│ After Search       │
│ + latest: "..."    │
└────────────────────┘
         ↓
    ReporterAgent(s)
         ↓
┌────────────────────┐
│ After Reporters    │
│ + drafts: {...}    │
└────────────────────┘
         ↓
    ReviewerAgent
         ↓
┌────────────────────┐
│ Final State        │
│ + final: "..."     │
└────────────────────┘
         ↓
   Return final
```

---

## Execution Flow

### Complete Execution Lifecycle

```
┌─────────────────────────────────────────────────────────┐
│ 1. Entry Point                                           │
│    run_sequential_goal(cfg, goal)                        │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 2. State Initialization                                  │
│    state = WorkflowState(goal=goal)                      │
│    - Create empty state object                           │
│    - Initialize logging                                  │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Triage Execution                                      │
│    triage_agent.execute({"goal": goal})                  │
│    - Analyze goal                                        │
│    - Determine routing                                   │
│    - Store result in state.triage                        │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Search Agent Selection                                │
│    _select_search_agent(state.triage)                    │
│    - Check preferred_agent                               │
│    - Instantiate appropriate search agent                │
│    - Return selected agent                               │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 5. Search Execution                                      │
│    search_agent.execute({"goal": goal, ...})             │
│    - Execute search with appropriate agent               │
│    - Retrieve and format results                         │
│    - Store result in state.latest                        │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 6. Reporter Selection                                    │
│    _select_reporter_mode(state.triage)                   │
│    - Check for multi-reporter intent                     │
│    - Instantiate reporter agent(s)                       │
│    - Return reporter configuration                       │
└─────────────────────────────────────────────────────────┘
                        │
                        ├─────────────────────────────────┐
                        │                                 │
                ┌───────▼────────┐         ┌──────▼──────┐
                │ Single Mode    │         │ Multi Mode  │
                │ Reporter       │         │ Reporters   │
                └─────┬──────────┘         └──────┬──────┘
                      │                           │
                ┌─────▼──────────┐         ┌──────▼──────┐
                │ Sequential     │         │ Parallel    │
                │ Execution      │         │ Execution   │
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
│ 7. Reporter Execution                                    │
│    reporter_agent(s).execute({...})                      │
│    - Generate content based on search results            │
│    - Aggregate multi-reporter results                    │
│    - Store result in state.drafts                        │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 8. Review Loop Execution                                 │
│    for iteration in range(max_iterations=3):             │
│    - Execute reviewer agent                              │
│    - Check decision (accept/revise)                       │
│    - If accept: break loop                               │
│    - If revise: regenerate and loop                       │
│    - Store final result in state.final                    │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 9. Return Final Output                                   │
│    return state.final                                    │
│    - Return approved script to caller                    │
│    - Log execution metrics                               │
└─────────────────────────────────────────────────────────┘
```

### Error Handling Strategy

#### By Stage

```python
# Stage 1: Triage
try:
    triage_result = await triage_agent.execute(...)
except Exception as e:
    logger.error(f"Triage failed: {e}")
    # Use default triage result
    triage_result = get_default_triage_result()

# Stage 2: Search
try:
    search_result = await search_agent.execute(...)
except Exception as e:
    logger.error(f"Search failed: {e}")
    # Use empty search result
    search_result = ""

# Stage 3: Reporter
try:
    reporter_result = await reporter_agent.execute(...)
except Exception as e:
    logger.error(f"Reporter failed: {e}")
    # Return error message
    return f"Failed to generate content: {e}"

# Stage 4: Review Loop
try:
    # Retry logic built in
    final_result = await review_loop(...)
except Exception as e:
    logger.error(f"Review loop failed: {e}")
    # Return last generated script
    return last_generated_script
```

### Retry and Recovery

```python
# Retry on transient failures
async def execute_with_retry(agent, inputs, max_retries=2):
    for attempt in range(max_retries):
        try:
            return await agent.execute(inputs)
        except TransientError as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise

# Fallback on persistent failures
try:
    result = await execute_with_retry(agent, inputs)
except Exception as e:
    logger.error(f"Agent failed after retries: {e}")
    result = get_fallback_result()
```

---

## Comparison: Sequential vs Graph Execution

### Sequential Workflow

**Characteristics:**

- Linear pipeline of agents
- Fixed execution order
- Simple state passing
- Straightforward error handling
- Good for: Basic workflows, simple agent chains, predictable paths

**Example:**

```
Triage → [Search] → [Reporter] → [Review] → Output
```

**Code Simplicity:**

- ~500 lines for core executor
- Clear, linear logic flow
- Easy to debug and understand

### Graph Workflow

**Characteristics:**

- Complex node types (agent, fanout, loop, conditional, merge)
- Dynamic routing based on conditions
- Parallel branches
- Loop iterations
- Good for: Complex branching, multi-path workflows, conditional execution

**Example:**

```
Triage → [SQL/Neo4j/AiSearch] → Fanout → [Reporter 1, 2, 3] → Merge → Loop → Review
```

**Code Complexity:**

- ~2000+ lines for full executor
- Queue-based execution model
- Supports complex patterns

### When to Use Sequential

1. **Simple agent chains** - Few agents, fixed order
2. **Rapid development** - Quick iteration
3. **Easy debugging** - Linear execution path
4. **Resource constrained** - Lower memory overhead
5. **Learning purposes** - Understanding agent coordination

### When to Use Graph

1. **Complex workflows** - Multiple branching paths
2. **Conditional routing** - Different agents based on analysis
3. **Parallel execution** - Multiple agents simultaneously
4. **Dynamic loops** - Iterative refinement
5. **Multi-path scenarios** - Different user intent handling

---

## Summary

The sequential workflow system provides:

1. **Simple Agent Pipeline** - Linear execution of agents in order
2. **Flexible Agent Selection** - Different search/reporter agents based on analysis
3. **State Management** - Shared state passed through pipeline
4. **Error Handling** - Graceful failure recovery at each stage
5. **Multi-Agent Support** - Parallel reporters with result stitching
6. **Review Loop** - Iterative validation and refinement
7. **Easy to Understand** - Linear code flow matching business logic

The sequential model is ideal for workflows where the execution path is clear and straightforward, while the graph model is better suited for complex scenarios with dynamic routing and parallel execution patterns.
