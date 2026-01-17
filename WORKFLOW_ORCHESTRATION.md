# Workflow Factory and Orchestration Architecture

## Overview

This document describes the workflow factory pattern and graph-based orchestration system used to coordinate multiple AI agents in a flexible, declarative workflow. The system transforms sequential agent execution into a graph-based execution model that supports conditional routing, parallel execution, loops, and complex state management.

## Table of Contents

1. [Workflow Factory](#workflow-factory)
2. [Graph Executor (Orchestration Engine)](#graph-executor-orchestration-engine)
3. [Agent Orchestration Construction](#agent-orchestration-construction)
4. [Node Types and Execution](#node-types-and-execution)
5. [State Management](#state-management)
6. [Execution Flow](#execution-flow)

---

## Workflow Factory

### Location

`src/news_reporter/workflows/workflow_factory.py`

### Purpose

The `workflow_factory.py` module provides the entry point for executing agent workflows. It offers two execution modes:

1. **Graph-based execution** (primary): Uses `GraphExecutor` to run declarative JSON-defined workflows
2. **Sequential execution** (fallback): Traditional sequential agent execution when graph execution fails

### Key Functions

#### `run_graph_workflow(cfg, goal, graph_path=None)`

The primary entry point for graph-based workflow execution.

**Flow:**

```
1. Load graph definition from JSON
   └─> load_graph_definition(graph_path, config)
       ├─> Reads JSON file (default: default_workflow.json)
       ├─> Substitutes ${VAR} placeholders with actual agent IDs
       └─> Returns GraphDefinition object

2. Create GraphExecutor
   └─> GraphExecutor(graph_def, cfg)
       ├─> Validates graph structure
       ├─> Builds execution graph (adjacency lists)
       └─> Initializes metrics, caching, retry handlers

3. Execute workflow
   └─> executor.execute(goal)
       ├─> Creates WorkflowState with goal
       ├─> Finds entry nodes
       └─> Runs queue-based execution

4. Collect metrics
   └─> Analytics engine integration

5. Return final output
```

**Parameters:**

- `cfg: Settings` - Application configuration with agent IDs
- `goal: str` - User query/goal
- `graph_path: str | None` - Optional path to workflow JSON (defaults to `default_workflow.json`)

**Returns:**

- `str` - Final output from workflow execution

**Error Handling:**

- Falls back to `run_sequential_goal()` if graph execution fails
- Logs errors and continues with sequential execution

#### `run_sequential_goal(cfg, goal)`

Fallback sequential execution mode that maintains backward compatibility.

**Flow:**

```
TRIAGE → AISEARCH → REPORTER → REVIEWER (≤3 passes)
```

**Agent Selection Logic:**

1. **Triage Agent** - Analyzes goal and determines routing
2. **Search Agent Selection:**
   - If `preferred_agent == "sql"` and SQL agent configured → Use `SQLAgent`
   - If `use_neo4j_search` → Use `Neo4jGraphRAGAgent`
   - Otherwise → Use `AiSearchAgent` (default)
3. **Reporter Agent** - Generates news script
4. **Review Agent** - Reviews and provides feedback (max 3 iterations)

**Multi-reporter Support:**

- If `"multi" in tri.intents` or `multi_route_always` config enabled
- Executes multiple reporter agents in parallel
- Stitches results together

---

## Graph Executor (Orchestration Engine)

### Location

`src/news_reporter/workflows/graph_executor.py`

### Purpose

The `GraphExecutor` is the core orchestration engine that executes graph-based workflows. It implements a queue-based execution model that supports:

- **Dynamic routing** based on state conditions
- **Parallel execution** of independent nodes
- **Cycles and loops** for iterative processing
- **Fanout/merge** patterns for parallel branches
- **State checkpointing** for recovery
- **Performance metrics** and monitoring
- **Retry mechanisms** for fault tolerance
- **Caching** for optimization

### Architecture

#### Core Components

```python
class GraphExecutor:
    - graph_def: GraphDefinition      # Graph structure (nodes, edges)
    - config: Settings                # Application configuration
    - runner: AgentRunner             # Agent execution layer
    - nodes: Dict[str, NodeConfig]    # Node lookup table
    - outgoing_edges: Dict           # Adjacency list (forward)
    - incoming_edges: Dict            # Adjacency list (backward)
    - limits: GraphLimits             # Execution limits
    - checkpoint_manager              # State checkpointing
    - metrics_collector               # Performance tracking
    - cache_manager                   # Result caching
    - retry_handler                  # Retry logic
    - monitor                        # Real-time monitoring
```

#### Execution Model: Queue-Based System

The executor uses a **queue-based execution model** rather than recursive traversal:

```
┌─────────────────────────────────────────┐
│  Execution Queue (deque)                │
│  - ExecutionToken objects               │
│  - Contains: node_id, context, parent   │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  Execution State Tracking                 │
│  - executed: Set[(node_id, iteration)]   │
│  - executing: Set[node_id]               │
│  - node_results: Dict[node_id, Result]   │
│  - branch_contexts: Dict[branch_id, Ctx]│
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  Node Execution                          │
│  1. Check cache                         │
│  2. Create node instance                 │
│  3. Execute with retry                  │
│  4. Update state                         │
│  5. Determine next nodes                 │
└─────────────────────────────────────────┘
```

### Execution Flow

#### 1. Initialization (`execute()`)

```python
async def execute(self, goal: str) -> str:
    # 1. Create WorkflowState
    state = WorkflowState(goal=goal)

    # 2. Find entry nodes
    entry_nodes = self.graph_def.get_entry_nodes()

    # 3. Create root execution context
    root_context = ExecutionContext(node_id=entry_nodes[0])

    # 4. Try to restore from checkpoint
    if checkpoint_manager:
        restored_state = checkpoint_manager.restore_state(run_id)

    # 5. Execute with timeout
    await self._execute_queue_based(state, entry_nodes, root_context)

    # 6. Get final output
    return self._get_final_output(state)
```

#### 2. Queue-Based Execution (`_execute_queue_based()`)

```python
async def _execute_queue_based(...):
    # Initialize
    queue = deque()
    executed = set()
    executing = set()
    tracker = ExecutionTracker()

    # Add entry nodes to queue
    for node_id in entry_nodes:
        context = root_context.create_child_branch(node_id)
        queue.append(ExecutionToken(node_id, context))

    # Process queue
    while queue and step_count < max_steps:
        token = queue.popleft()

        # Execute node
        result = await self._execute_node(node_id, state, context)

        # Apply state updates
        self._apply_state_updates(state, result.state_updates)

        # Handle special nodes (fanout, loop, merge)
        next_nodes = await self._handle_special_nodes(...)

        # Determine next nodes
        if next_nodes is None:
            next_nodes = self._determine_next_nodes(node_id, result, state)

        # Add next nodes to queue
        for next_node_id in next_nodes:
            queue.append(ExecutionToken(next_node_id, ...))
```

#### 3. Node Execution (`_execute_node()`)

```python
async def _execute_node(self, node_id, state, context, parent_result):
    # 1. Check cache
    cached_result = self.cache_manager.get(node_id, node_inputs)
    if cached_result:
        return cached_result

    # 2. Create node instance
    node = create_node(node_type, config, state, runner, settings)

    # 3. Execute with retry
    result, retry_count = await self.retry_handler.execute_with_retry(
        node_id, execute_fn=node.execute
    )

    # 4. Cache successful results
    if result.status == SUCCESS:
        self.cache_manager.set(node_id, node_inputs, result)

    # 5. Record metrics
    self.metrics_collector.record_node_execution(...)

    return result
```

#### 4. Next Node Determination (`_determine_next_nodes()`)

```python
def _determine_next_nodes(self, node_id, result, state):
    # 1. Check if NodeResult specifies next_nodes
    if result.next_nodes:
        return result.next_nodes

    # 2. Use graph edges
    outgoing = self.outgoing_edges.get(node_id, [])
    next_nodes = []

    for edge in outgoing:
        # Check edge condition
        if edge.condition:
            condition_met = ConditionEvaluator.evaluate(edge.condition, state)
            if not condition_met:
                continue  # Skip this edge

        next_nodes.append(edge.to_node)

    return next_nodes
```

### Special Node Handling

#### Fanout Nodes

Fanout nodes create parallel execution branches:

```python
async def _handle_fanout_node(...):
    # 1. Get fanout items and branch node IDs
    fanout_items = result.artifacts.get("fanout_items", [])
    branch_node_ids = result.artifacts.get("branches", [])

    # 2. Find merge node
    merge_node_id = find_merge_node(node_id)

    # 3. Register fanout in tracker
    fanout_tracker = tracker.register_fanout(...)

    # 4. Create branches for each item
    for item in fanout_items:
        for branch_node_id in branch_node_ids:
            branch_context = context.create_child_branch(branch_node_id)
            queue.append(ExecutionToken(branch_node_id, branch_context))

    # 5. Don't continue to merge yet - wait for all branches
    return []
```

#### Loop Nodes

Loop nodes enable iterative execution:

```python
async def _handle_loop_node(...):
    should_continue = artifacts.get("should_continue", False)
    body_node_id = artifacts.get("body_node_id")

    if not should_continue:
        return self._determine_next_nodes(node_id, result, state)

    # Loop should continue
    if body_node_id:
        # Increment iteration
        new_iter = tracker.increment_loop_iteration(node_id)

        # Loop back to body
        body_context = context.create_iteration(body_node_id)
        queue.append(ExecutionToken(body_node_id, body_context))

    return []
```

#### Merge Nodes

Merge nodes wait for all branches to complete (join barrier):

```python
async def _handle_merge_node(...):
    expected_keys = node_config.params.get("expected_keys")

    if not expected_keys:
        return self._determine_next_nodes(node_id, result, state)

    # Check if all expected keys are present
    items = state.get(merge_key, {})
    missing_keys = set(expected_keys) - set(items.keys())

    if missing_keys:
        # Join barrier not met - wait or re-queue
        if fanout_tracker:
            # Wait for fanout branches
            while not fanout_tracker.all_branches_complete():
                await asyncio.sleep(0.1)
        else:
            # Re-queue merge node
            queue.append(ExecutionToken(node_id, context))
            return []

    # All keys present, proceed
    return self._determine_next_nodes(node_id, result, state)
```

---

## Agent Orchestration Construction

### Graph Definition Structure

Agent orchestrations are defined declaratively in JSON format. The graph structure consists of:

1. **Nodes** - Execution units (agents, conditionals, loops, etc.)
2. **Edges** - Connections between nodes with optional conditions
3. **Metadata** - Name, description, version, limits

### Graph Definition Schema

```json
{
	"name": "Workflow Name",
	"description": "Workflow description",
	"version": "1.0.0",
	"entry_node_id": "triage",
	"nodes": [
		{
			"id": "node_id",
			"type": "agent|fanout|loop|conditional|merge",
			"agent_id": "${AGENT_ID_TRIAGE}",
			"inputs": {
				"input_key": "state.path"
			},
			"outputs": {
				"output_key": "state.path"
			},
			"params": {}
		}
	],
	"edges": [
		{
			"from_node": "node1",
			"to_node": "node2",
			"condition": "optional_condition_expression"
		}
	],
	"limits": {
		"max_steps": 1000,
		"timeout_ms": 300000,
		"max_parallel": 5
	}
}
```

### Node Types

#### 1. Agent Nodes

Execute AI agents (TriageAgent, AiSearchAgent, ReporterAgent, etc.)

```json
{
	"id": "triage",
	"type": "agent",
	"agent_id": "${AGENT_ID_TRIAGE}",
	"inputs": {
		"goal": "goal"
	},
	"outputs": {
		"result": "triage"
	}
}
```

**Execution:**

- Reads inputs from `WorkflowState` using dot notation paths
- Executes agent via `AgentRunner`
- Writes outputs to `WorkflowState` using dot notation paths

#### 2. Conditional Nodes

Route execution based on state conditions

```json
{
	"id": "select_search",
	"type": "conditional",
	"condition": "triage.preferred_agent == \"sql\" and agent_id_aisearch_sql is not None"
}
```

**Execution:**

- Evaluates condition using `ConditionEvaluator`
- Routes to different edges based on condition result
- No state updates (pure routing)

#### 3. Fanout Nodes

Create parallel execution branches

```json
{
	"id": "report_fanout",
	"type": "fanout",
	"branches": ["report_branch"],
	"params": {
		"fanout_items": ["reporter_1", "reporter_2"]
	}
}
```

**Execution:**

- Creates multiple branches for each item
- Each branch executes independently
- Results collected at merge node

#### 4. Loop Nodes

Iterative execution with condition checking

```json
{
	"id": "review_loop",
	"type": "loop",
	"max_iters": 3,
	"loop_condition": "verdicts.current_fanout_item[-1].decision != \"accept\""
}
```

**Execution:**

- Checks loop condition before each iteration
- Executes body nodes until condition is false or max_iters reached
- Maintains iteration counter in `ExecutionContext`

#### 5. Merge Nodes

Join barrier for parallel branches

```json
{
	"id": "merge_reports",
	"type": "merge",
	"params": {
		"expected_keys": ["reporter_1", "reporter_2"],
		"merge_key": "final"
	}
}
```

**Execution:**

- Waits for all expected keys to be present in state
- Proceeds only when join barrier is met
- Supports timeout for incomplete branches

### Edge Conditions

Edges can have conditional routing:

```json
{
	"from_node": "triage",
	"to_node": "search_sql",
	"condition": "triage.preferred_agent == \"sql\""
}
```

**Condition Syntax:**

- Operators: `==`, `!=`, `in`, `not in`, `is None`, `is not None`, `and`, `or`
- State paths: `triage.preferred_agent`, `state.selected_search`
- Literals: `"string"`, `123`, `true`, `false`

### Agent ID Substitution

Agent IDs can use environment variable placeholders:

```json
{
	"agent_id": "${AGENT_ID_TRIAGE}"
}
```

**Substitution Process:**

1. `load_graph_definition()` calls `_substitute_agent_ids()`
2. Creates mapping from config:
   ```python
   substitutions = {
       "AGENT_ID_TRIAGE": config.agent_id_triage,
       "AGENT_ID_AISEARCH": config.agent_id_aisearch,
       "AGENT_ID_AISEARCHSQL": config.agent_id_aisearch_sql,
       ...
   }
   ```
3. Replaces `${VAR}` with actual values
4. Falls back to environment variables if not in config

### Example: Complete Workflow Construction

```json
{
	"name": "Default News Reporter Workflow",
	"entry_node_id": "triage",
	"nodes": [
		{
			"id": "triage",
			"type": "agent",
			"agent_id": "${AGENT_ID_TRIAGE}",
			"inputs": { "goal": "goal" },
			"outputs": { "result": "triage" }
		},
		{
			"id": "select_search",
			"type": "conditional",
			"condition": "triage.preferred_agent == \"sql\""
		},
		{
			"id": "search_sql",
			"type": "agent",
			"agent_id": "${AGENT_ID_AISEARCHSQL}",
			"inputs": { "goal": "goal", "database_id": "triage.database_id" },
			"outputs": { "result": "latest" }
		},
		{
			"id": "search_aisearch",
			"type": "agent",
			"agent_id": "${AGENT_ID_AISEARCH}",
			"inputs": { "goal": "goal" },
			"outputs": { "result": "latest" }
		},
		{
			"id": "report_fanout",
			"type": "fanout",
			"branches": ["report_branch"]
		},
		{
			"id": "report_branch",
			"type": "agent",
			"agent_id": "${AGENT_ID_REPORTER}",
			"inputs": { "goal": "goal", "latest_news": "latest" },
			"outputs": { "result": "drafts.current_fanout_item" }
		},
		{
			"id": "review_loop",
			"type": "loop",
			"max_iters": 3
		},
		{
			"id": "review",
			"type": "agent",
			"agent_id": "${AGENT_ID_REVIEWER}",
			"inputs": {
				"topic": "goal",
				"candidate_script": "drafts.current_fanout_item"
			},
			"outputs": { "result": "verdicts.current_fanout_item" }
		}
	],
	"edges": [
		{ "from_node": "triage", "to_node": "select_search" },
		{
			"from_node": "select_search",
			"to_node": "search_sql",
			"condition": "triage.preferred_agent == \"sql\""
		},
		{
			"from_node": "select_search",
			"to_node": "search_aisearch",
			"condition": "triage.preferred_agent != \"sql\""
		},
		{ "from_node": "search_sql", "to_node": "report_fanout" },
		{ "from_node": "search_aisearch", "to_node": "report_fanout" },
		{ "from_node": "report_fanout", "to_node": "report_branch" },
		{ "from_node": "report_branch", "to_node": "review_loop" },
		{ "from_node": "review_loop", "to_node": "review" },
		{
			"from_node": "review",
			"to_node": "review_loop",
			"condition": "verdicts.current_fanout_item[-1].decision != \"accept\""
		}
	]
}
```

**Execution Flow:**

```
triage → select_search → [search_sql | search_aisearch]
  → report_fanout → report_branch (parallel)
  → review_loop → review → [loop back | exit]
```

---

## Node Types and Execution

### Node Factory Pattern

Nodes are created using a factory function:

```python
# src/news_reporter/workflows/nodes/__init__.py

NODE_TYPES = {
    "agent": AgentNode,
    "fanout": FanoutNode,
    "loop": LoopNode,
    "conditional": ConditionalNode,
    "merge": MergeNode,
}

def create_node(node_type, config, state, runner, settings):
    node_class = NODE_TYPES.get(node_type)
    return node_class(config, state, runner, settings)
```

### Base Node Interface

All nodes inherit from `BaseNode`:

```python
class BaseNode(ABC):
    def __init__(self, config, state, runner, settings):
        self.config = config
        self.state = state
        self.runner = runner
        self.settings = settings

    @abstractmethod
    async def execute(self) -> NodeResult:
        """Execute node logic and return NodeResult"""
        pass

    def get_input(self, input_key, default=None):
        """Get input from state using input mapping"""
        path = self.config.inputs[input_key]
        return self.state.get(path, default)

    def set_output(self, output_key, value):
        """Set output to state using output mapping"""
        state_path = self.config.outputs[output_key]
        self.state.set(state_path, value)
```

### NodeResult Structure

All nodes return a `NodeResult`:

```python
class NodeResult:
    status: NodeStatus  # SUCCESS, FAILED, SKIPPED
    state_updates: Dict[str, Any]  # State changes
    artifacts: Dict[str, Any]  # Additional data
    next_nodes: Optional[List[str]]  # Override next nodes
    error: Optional[str]  # Error message if failed
```

---

## State Management

### WorkflowState

The `WorkflowState` is a shared state object that all nodes read from and write to:

```python
class WorkflowState(BaseModel):
    goal: str
    triage: Optional[Dict[str, Any]]
    selected_search: Optional[str]
    database_id: Optional[str]
    targets: List[str]
    latest: str
    drafts: Dict[str, str]
    final: Dict[str, str]
    verdicts: Dict[str, List[Dict[str, Any]]]
    logs: List[Dict[str, Any]]
    execution_trace: List[Dict[str, Any]]
```

### State Access Methods

#### Get State Value

```python
# Dot notation path
value = state.get("triage.preferred_agent")
value = state.get("drafts.reporter_1", default="")
```

#### Set State Value

```python
# Direct attribute
state.set("goal", "new goal")

# Nested path
state.set("triage.preferred_agent", "sql")
state.set("drafts.reporter_1", "draft text")
```

### State Propagation

State updates from `NodeResult.state_updates` are automatically applied:

```python
def _apply_state_updates(self, state, updates):
    for path, value in updates.items():
        state.set(path, value)
```

---

## Execution Flow

### Complete Execution Lifecycle

```
┌─────────────────────────────────────────────────────────┐
│ 1. Workflow Factory Entry                                │
│    run_graph_workflow(cfg, goal, graph_path)             │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Graph Loading                                         │
│    load_graph_definition()                                │
│    - Read JSON file                                       │
│    - Substitute agent IDs                                 │
│    - Parse nodes and edges                                │
│    - Create GraphDefinition                               │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Executor Initialization                               │
│    GraphExecutor(graph_def, config)                      │
│    - Validate graph                                       │
│    - Build execution graph                                │
│    - Initialize components                                │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Execution Start                                       │
│    executor.execute(goal)                                │
│    - Create WorkflowState                                │
│    - Find entry nodes                                     │
│    - Create root context                                  │
│    - Check for checkpoint                                 │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 5. Queue-Based Execution                                 │
│    _execute_queue_based()                                 │
│    - Initialize queue with entry nodes                    │
│    - Process queue:                                       │
│      ├─> Execute node                                     │
│      ├─> Apply state updates                              │
│      ├─> Handle special nodes                             │
│      ├─> Determine next nodes                             │
│      └─> Add next nodes to queue                          │
│    - Continue until queue empty or max_steps              │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 6. Node Execution                                        │
│    _execute_node()                                        │
│    - Check cache                                          │
│    - Create node instance                                 │
│    - Execute with retry                                   │
│    - Cache result                                         │
│    - Record metrics                                       │
│    - Return NodeResult                                    │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 7. Next Node Determination                               │
│    _determine_next_nodes()                                │
│    - Check NodeResult.next_nodes                          │
│    - Evaluate edge conditions                             │
│    - Return list of next node IDs                         │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 8. Final Output Extraction                               │
│    _get_final_output()                                    │
│    - Check terminal nodes                                 │
│    - Fallback to state.latest                             │
│    - Return final string                                  │
└─────────────────────────────────────────────────────────┘
```

### Execution Context and Branching

The executor maintains execution contexts for branch tracking:

```python
class ExecutionContext:
    run_id: str              # Unique run identifier
    branch_id: str           # Branch identifier
    node_id: str             # Current node
    iteration: int           # Loop iteration
    parent_branch_id: str    # Parent branch

    def create_child_branch(self, node_id) -> ExecutionContext:
        """Create new branch context"""

    def create_iteration(self, node_id) -> ExecutionContext:
        """Create iteration context for loops"""
```

### Error Handling and Recovery

1. **Node Failures:**

   - Retry with exponential backoff
   - Continue or stop based on `error_strategy`
   - Log errors to state

2. **Timeout Handling:**

   - Global timeout from `limits.timeout_ms`
   - Per-node timeouts in node config
   - Graceful shutdown on timeout

3. **Checkpointing:**
   - Periodic checkpoints every N steps
   - Restore from checkpoint on restart
   - Save state and metadata

---

## Summary

The workflow factory and orchestration system provides:

1. **Declarative Workflow Definition** - JSON-based graph definitions
2. **Flexible Execution** - Queue-based model supporting complex patterns
3. **State Management** - Shared state with dot notation access
4. **Agent Coordination** - Seamless integration of multiple AI agents
5. **Fault Tolerance** - Retry, caching, checkpointing
6. **Observability** - Metrics, logging, tracing
7. **Extensibility** - Easy to add new node types and agents

The system transforms sequential agent execution into a powerful graph-based orchestration engine that can handle complex workflows with conditional routing, parallel execution, loops, and state management.
