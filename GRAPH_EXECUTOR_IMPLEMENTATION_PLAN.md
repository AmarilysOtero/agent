# Graph Executor Implementation Plan

This document outlines the implementation plan for the graph-based workflow executor system.

## Implementation Status

### Phase 1: Define Graph Contract (JSON Schema) ✅ COMPLETE
**Status**: Implemented

1.1 Create Graph Definition Schema
- **File**: `src/news_reporter/workflows/graph_schema.py`
- **Components**: `NodeConfig`, `EdgeConfig`, `GraphDefinition`
- **Features**:
  - Node types: agent, fanout, loop, merge, conditional
  - Entry node support (`entry_node_id`)
  - Toolsets, policy profiles, limits

1.2 Node Types
- **Supported**: agent, fanout, loop, merge, conditional

1.3 Condition Expression Language
- **File**: `src/news_reporter/workflows/condition_evaluator.py`
- **Features**: Safe expression evaluator (parser-based, no eval())
- **Operators**: ==, !=, in, not in, is None, is not None, and, or

1.4 Entry Node
- **Feature**: Explicit `entry_node_id` field in GraphDefinition

---

### Phase 2: Create WorkflowState ✅ COMPLETE
**Status**: Implemented

2.1 Define WorkflowState Model
- **File**: `src/news_reporter/workflows/workflow_state.py`
- **Components**: `WorkflowState` (Pydantic BaseModel)
- **Fields**: goal, triage, selected_search, database_id, latest, drafts, final, verdicts, logs, execution_trace

2.2 State Access Methods
- **Methods**: `get(path)`, `set(path, value)`, `append_log()`, `add_trace()`

---

### Phase 3: Create AgentRunner Compatibility Layer ✅ COMPLETE
**Status**: Implemented

3.1 AgentRunner Class
- **File**: `src/news_reporter/workflows/agent_runner.py`
- **Features**: Wraps `run_foundry_agent()` for compatibility

3.2 Agent Type Detection
- **File**: `src/news_reporter/workflows/agent_adapter.py`
- **Features**: Agent adapter registry for input/output mapping

---

### Phase 4: Refactor Sequential Workflow into Nodes ✅ COMPLETE
**Status**: Implemented

4.1 Node Result Structure
- **File**: `src/news_reporter/workflows/node_result.py`
- **Components**: `NodeResult` with state_updates, artifacts, next_nodes, status, metrics

4.2 Node Handler Base Class
- **File**: `src/news_reporter/workflows/nodes/base.py`
- **Components**: `BaseNode` abstract class

4.3 Implement Node Types
- **Files**:
  - `src/news_reporter/workflows/nodes/agent_node.py` - AgentNode
  - `src/news_reporter/workflows/nodes/fanout_node.py` - FanoutNode (parallel execution)
  - `src/news_reporter/workflows/nodes/loop_node.py` - LoopNode (iterative execution)
  - `src/news_reporter/workflows/nodes/conditional_node.py` - ConditionalNode (routing)
  - `src/news_reporter/workflows/nodes/merge_node.py` - MergeNode (combines inputs)

4.4 Execution Context
- **File**: `src/news_reporter/workflows/execution_context.py`
- **Features**: Branch identity system for fanout/loop isolation

4.5 Node Registry
- **File**: `src/news_reporter/workflows/nodes/__init__.py`
- **Features**: Factory function to create nodes by type

---

### Phase 5: Workflow Visualization and Management ✅ COMPLETE
**Status**: Implemented

5.1 Workflow Visualizer
- **File**: `src/news_reporter/workflows/workflow_visualizer.py`
- **Features**: Generate visualizations (Mermaid, DOT, JSON)

5.2 Workflow Versioning
- **File**: `src/news_reporter/workflows/workflow_versioning.py`
- **Features**: Version management for workflows

5.3 API Endpoints
- **File**: `src/news_reporter/routers/workflows.py`
- **Endpoints**: Visualization, versioning, management

---

### Phase 6: Workflow Optimization, Scheduling, Analytics, Testing, Composition ✅ COMPLETE
**Status**: Implemented

6.1 Workflow Optimizer
- **File**: `src/news_reporter/workflows/workflow_optimizer.py`
- **Features**: Performance optimization recommendations

6.2 Workflow Scheduler
- **File**: `src/news_reporter/workflows/workflow_scheduler.py`
- **Features**: Schedule workflow executions

6.3 Workflow Analytics
- **File**: `src/news_reporter/workflows/workflow_analytics.py`
- **Features**: Analytics and reporting

6.4 Workflow Tester
- **File**: `src/news_reporter/workflows/workflow_tester.py`
- **Features**: Testing framework for workflows

6.5 Workflow Composer
- **File**: `src/news_reporter/workflows/workflow_composer.py`
- **Features**: Compose workflows from templates

6.6 Performance Metrics
- **File**: `src/news_reporter/workflows/performance_metrics.py`
- **Features**: Collect and report execution metrics

6.7 Cache Manager
- **File**: `src/news_reporter/workflows/cache_manager.py`
- **Features**: Caching for workflow execution

6.8 Retry Handler
- **File**: `src/news_reporter/workflows/retry_handler.py`
- **Features**: Retry logic for failed nodes

---

### Phase 7: Persistence, Security, Collaboration, Notifications, Integrations, Deployment ✅ COMPLETE
**Status**: Implemented

7.1 Workflow Persistence
- **File**: `src/news_reporter/workflows/workflow_persistence.py`
- **Features**: Save/load workflows and execution records

7.2 Workflow Security
- **File**: `src/news_reporter/workflows/workflow_security.py`
- **Features**: Permissions, roles, access control

7.3 Workflow Collaboration
- **File**: `src/news_reporter/workflows/workflow_collaboration.py`
- **Features**: Sharing, collaboration features

7.4 Workflow Notifications
- **File**: `src/news_reporter/workflows/workflow_notifications.py`
- **Features**: Notification system

7.5 Workflow Integrations
- **File**: `src/news_reporter/workflows/workflow_integrations.py`
- **Features**: Webhooks, event subscriptions

7.6 Workflow Deployment
- **File**: `src/news_reporter/workflows/workflow_deployment.py`
- **Features**: Deployment management

7.7 Workflow Templates
- **File**: `src/news_reporter/workflows/workflow_templates.py`
- **Features**: Template registry

---

### Phase 8: Cost Management, Backup, Debugger, Governance, AI, Documentation ✅ COMPLETE
**Status**: Implemented

8.1 Workflow Cost Management
- **File**: `src/news_reporter/workflows/workflow_cost.py`
- **Features**: Cost tracking, budgets, reports

8.2 Workflow Backup
- **File**: `src/news_reporter/workflows/workflow_backup.py`
- **Features**: Backup and restore workflows

8.3 Workflow Debugger
- **File**: `src/news_reporter/workflows/workflow_debugger.py`
- **Features**: Breakpoints, tracing, watch expressions

8.4 Workflow Governance
- **File**: `src/news_reporter/workflows/workflow_governance.py`
- **Features**: Policy management, compliance, validation

8.5 Workflow AI
- **File**: `src/news_reporter/workflows/workflow_ai.py`
- **Features**: AI predictions, recommendations, anomaly detection

8.6 Workflow Documentation (Management System)
- **File**: `src/news_reporter/workflows/workflow_documentation.py`
- **Features**: Documentation management, knowledge base, auto-generation

---

### Phase 9: Marketplace, Patterns, Migration, Alerting, Multi-Tenant, Gateway ✅ COMPLETE
**Status**: Implemented

9.1 Workflow Marketplace
- **File**: `src/news_reporter/workflows/workflow_marketplace.py`
- **Features**: Workflow sharing, discovery, reviews, ratings

9.2 Workflow Patterns
- **File**: `src/news_reporter/workflows/workflow_patterns.py`
- **Features**: State machines, event-driven patterns

9.3 Workflow Migration
- **File**: `src/news_reporter/workflows/workflow_migration.py`
- **Features**: Workflow migration and transformation tools

9.4 Workflow Alerting
- **File**: `src/news_reporter/workflows/workflow_alerting.py`
- **Features**: Advanced monitoring and alerting system

9.5 Workflow Multi-Tenant
- **File**: `src/news_reporter/workflows/workflow_multitenant.py`
- **Features**: Multi-tenant support and isolation

9.6 Workflow Gateway
- **File**: `src/news_reporter/workflows/workflow_gateway.py`
- **Features**: API gateway and rate limiting

---

### Phase 10: Documentation (User Guides) ⏳ PENDING
**Status**: Not Yet Implemented

**Note**: The original plan specified Phase 9 as Documentation (user guides), but implementation proceeded with advanced features. This phase should be implemented to provide comprehensive user documentation.

10.1 Graph Definition Format Documentation
- **Goal**: Document JSON schema with examples
- **Content**:
  - Complete schema reference
  - Example workflows
  - Best practices
  - Common patterns

10.2 Node Types Reference
- **Goal**: Document each node type, parameters, inputs/outputs
- **Content**:
  - Agent node documentation
  - Fanout node documentation
  - Loop node documentation
  - Conditional node documentation
  - Merge node documentation
  - Configuration options
  - Examples for each type

10.3 Migration Guide
- **Goal**: Document how to convert sequential workflows to graphs
- **Content**:
  - Step-by-step migration process
  - Before/after examples
  - Common migration patterns
  - Troubleshooting guide

10.4 API Documentation
- **Goal**: Complete API reference
- **Content**:
  - All endpoints documented
  - Request/response examples
  - Authentication
  - Error handling

10.5 Tutorials and Examples
- **Goal**: Practical tutorials for common use cases
- **Content**:
  - Getting started guide
  - Building your first workflow
  - Advanced patterns
  - Integration examples

---

## Implementation Order (As Implemented)

1. **Graph Schema (Phase 1)** - Define contracts early ✅
2. **Condition Evaluator (Phase 1.3)** - Safe parser (no eval) ✅
3. **WorkflowState (Phase 2)** - Foundation for everything ✅
4. **ExecutionContext (Phase 4.4)** - Branch identity system ✅
5. **NodeResult (Phase 4.1)** - Structured output format ✅
6. **Agent Adapter Registry (Phase 3.2)** - Explicit agent mappings ✅
7. **AgentRunner (Phase 3)** - Compatibility layer ✅
8. **BaseNode + AgentNode (Phase 4.2)** - Core execution ✅
9. **Graph Executor (Phase 5)** - Queue-based orchestration ✅
10. **Fanout + Merge (Phase 4.3)** - Parallel execution with join barriers ✅
11. **Loop + Conditional (Phase 4.3)** - Control flow ✅
12. **Default Graph (Phase 6)** - Convert current workflow ✅
13. **Workflow Factory Update (Phase 7)** - Integration ✅
14. **Advanced Features (Phases 5-9)** - Extended capabilities ✅
15. **Documentation (Phase 10)** - User guides ⏳

---

## Key Design Decisions

### Queue-Based Executor
- **Decision**: Use queue-based executor instead of topological sort
- **Reason**: Supports cycles, conditionals, and dynamic fanout naturally
- **Implementation**: `src/news_reporter/workflows/graph_executor.py`

### Explicit Entry Node
- **Decision**: `entry_node_id` field instead of inferring from "no incoming edges"
- **Reason**: Removes ambiguity and helps UI/future features

### Execution Context
- **Decision**: Branch identity system for fanout/loop isolation
- **Reason**: Prevents output collisions in parallel execution

### NodeResult Structure
- **Decision**: Structured output with state patches instead of raw dicts
- **Reason**: Clear separation of concerns, easier to reason about

### Agent Adapter Registry
- **Decision**: Explicit mappings instead of runtime detection
- **Reason**: Better type safety and maintainability

### Safe Condition Evaluator
- **Decision**: Parser-based (no eval()) for security
- **Reason**: Prevents code injection vulnerabilities

---

## Future Extensibility

The design allows future additions without refactoring:

- **Tools**: Add to AgentRunner.run() - nodes unchanged
- **Policies**: Add to GraphExecutor - enforce in AgentRunner
- **Budgets**: Add to GraphDefinition.limits - enforce in executor
- **UI**: Generate/edit JSON graphs - executor unchanged

---

## Testing Status

- **Unit Tests**: ✅ Comprehensive test coverage for all phases
- **Integration Tests**: ✅ Full workflow execution tests
- **Test Files**:
  - `tests/unit/workflows/test_phase8.py` - Phase 8 tests
  - `tests/unit/workflows/test_phase9.py` - Phase 9 tests
  - Additional tests in `tests/unit/workflows/`

---

## Notes

- **Phase 8 Original Plan**: Was "Testing & Validation" but implemented as "Cost Management, Backup, Debugger, Governance, AI, Documentation"
- **Phase 9 Original Plan**: Was "Documentation" but implemented as "Marketplace, Patterns, Migration, Alerting, Multi-Tenant, Gateway"
- **Phase 10**: Should be "Documentation (User Guides)" to complete the original plan
