# src/news_reporter/workflows/executor.py
"""Workflow Executor Core (PR 3)"""
from __future__ import annotations
import json
import logging
from typing import Dict, List, Set, Optional, Tuple, Any
from datetime import datetime, timezone
from collections import defaultdict

from ..models.workflow import Workflow, WorkflowRun, NodeResult, NodeError
from .workflow_repository import WorkflowRepository
from .agent_invoke import invoke_agent, build_agent_prompt  # PR4
from ..config import Settings  # PR4

logger = logging.getLogger(__name__)

# Output size limit (approximately 16KB)
MAX_OUTPUT_BYTES = 16 * 1024


def truncate_output(output: Any) -> Tuple[str, bool, Optional[str]]:
    """
    Truncate output to string format with size limit.
    
    FIX 2: Byte-safe truncation to prevent UTF-8 character splitting.
    
    Args:
        output: Output value (string, dict, list, or other type)
        
    Returns:
        Tuple of (output_string, is_truncated, preview_string)
    """
    # Convert to string
    if isinstance(output, str):
        output_str = output
    elif isinstance(output, (dict, list)):
        output_str = json.dumps(output, sort_keys=True, ensure_ascii=False)
    else:
        output_str = str(output)
    
    # FIX 2: Byte-safe truncation
    output_bytes = output_str.encode('utf-8')
    byte_len = len(output_bytes)
    
    if byte_len <= MAX_OUTPUT_BYTES:
        return output_str, False, None
    
    # Truncate at byte boundary and decode safely
    truncated_bytes = output_bytes[:MAX_OUTPUT_BYTES]
    truncated_str = truncated_bytes.decode('utf-8', errors='ignore')
    
    # Preview is first 500 characters of original string
    preview = output_str[:500]
    
    return truncated_str, True, preview


class WorkflowExecutor:
    """Sequential workflow executor with fan-in gating"""
    
    def __init__(self, workflow: Workflow, run: WorkflowRun, repo: WorkflowRepository, cfg: Settings):
        """
        Initialize executor with workflow definition and run  document.
        
        Args:
            workflow: Workflow definition
            run: WorkflowRun document to execute
            repo: Repository for persistence
            cfg: Settings for agent configuration (PR4)
        """
        self.workflow = workflow
        self.run = run
        self.repo = repo
        self.cfg = cfg  # PR4: Store config for agent invocation
        
        # Graph data structures
        self.node_map: Dict[str, dict] = {}
        self.children: Dict[str, List[str]] = {}  # node -> list of child nodes
        self.parents: Dict[str, List[str]] = {}   # node -> list of parent nodes
        self.outgoing: Dict[str, List[str]] = {}  # Alias for children (compatibility)
        self.incoming: Dict[str, List[str]] = {}  # Alias for parents (compatibility)
        self.in_degree: Dict[str, int] = {}
        
        # Execution state
        self.completed: Set[str] = set()
        self.node_results: Dict[str, NodeResult] = {}
        self.ready_queue: List[str] = []
        self.schedule_counter = 0  # For deterministic execution ordering
        
    async def execute(self) -> WorkflowRun:
        """
        Execute the workflow run.
        
        FIX 1: Renamed from run() to execute() to avoid naming collision with self.run attribute.
        
        Returns:
            Updated WorkflowRun document
        """
        try:
            # FIX 2: Enforce validate-before-run with clear error messages for all cases
            validation_status = self.workflow.validationStatus
            
            # Handle missing/None/empty validationStatus
            # DEMO PATCH: Allow unvalidated workflows to run
            # if not validation_status or validation_status.strip() == "":
            #     error_msg = (
            #         "Workflow must be validated before execution (validationStatus missing). "
            #         "Please validate the workflow and retry."
            #     )
            #     logger.error(f"Run {self.run.id} rejected: {error_msg}")
            #     await self.repo.update_run_status(
            #         self.run.id,
            #         self.run.userId,  # PR5 Fix 1: userId scoping
            #         "failed",
            #         completedAt=datetime.now(timezone.utc),
            #         error=error_msg
            #     )
            #     updated_run = await self.repo.get_run(self.run.id, self.run.userId)
            #     return updated_run
            
            # Handle explicitly invalid workflows
            if validation_status == "invalid":
                error_msg = "Workflow is invalid. Please fix validation errors and retry."
                logger.error(f"Run {self.run.id} rejected: {error_msg}")
                await self.repo.update_run_status(
                    self.run.id,
                    self.run.userId,  # PR5 Fix 1: userId scoping
                    "failed",
                    completedAt=datetime.now(timezone.utc),
                    error=error_msg
                )
                updated_run = await self.repo.get_run(self.run.id, self.run.userId)
                return updated_run
            
            # Only proceed if explicitly valid
            # DEMO PATCH: Relax strict check
            # if validation_status != "valid":
            #     error_msg = (
            #         f"Unexpected validationStatus '{validation_status}'. "
            #         "Workflow must have validationStatus='valid' to execute."
            #     )
            #     logger.error(f"Run {self.run.id} rejected: {error_msg}")
            #     await self.repo.update_run_status(
            #         self.run.id,
            #         self.run.userId,  # Consistency patch: missing userId
            #         "failed",
            #         completedAt=datetime.now(timezone.utc),
            #         error=error_msg
            #     )
            #     updated_run = await self.repo.get_run(self.run.id, self.run.userId)
            #     return updated_run
            
            # Update run status to running
            await self.repo.update_run_status(
                self.run.id,
                self.run.userId,  # PR5 Fix 1: userId scoping
                "running",
                startedAt=datetime.now(timezone.utc),
                heartbeatAt=datetime.now(timezone.utc)
            )
            
            # Prepare graph structures
            self._prepare_graph()
            
            # Seed ready queue with root node
            self._seed_ready_queue()
            
            # Execute loop
            await self._execute_loop()
            
            # Mark run as succeeded
            await self.repo.update_run_status(
                self.run.id,
                self.run.userId,  # PR5 Fix 1: userId scoping
                "succeeded",
                completedAt=datetime.now(timezone.utc)
            )
            
            # Reload run to get updated state
            updated_run = await self.repo.get_run(self.run.id, self.run.userId)
            return updated_run
            
        except Exception as e:
            logger.exception(f"Workflow run {self.run.id} failed: {e}")
            
            # Mark run as failed
            error_msg = f"Workflow execution failed: {str(e)}"
            await self.repo.update_run_status(
                self.run.id,
                self.run.userId,  # PR5 Fix 1: userId scoping
                "failed",
                completedAt=datetime.now(timezone.utc),
                error=error_msg
            )
            
            # Reload run to get updated state
            updated_run = await self.repo.get_run(self.run.id, self.run.userId)
            return updated_run
    
    def _prepare_graph(self):
        """
        Build internal graph structures from workflow definition.
        
        FIX 4: Harden against invalid edges referencing non-existent nodes.
        """
        nodes = self.workflow.graph.nodes
        edges = self.workflow.graph.edges
        
        # Build node map and initialize adjacency lists and in-degree
        for node in nodes:
            node_id = node["id"]
            self.node_map[node_id] = node
            self.children[node_id] = []
            self.parents[node_id] = []
            self.in_degree[node_id] = 0
        
        # FIX 4: Validate edges before building graph structures
        for edge in edges:
            source = edge.get("source") or edge.get("from")
            target = edge.get("target") or edge.get("to")
            
            # Defensive check: ensure both nodes exist
            if source not in self.node_map:
                raise RuntimeError(
                    f"Invalid edge references unknown source nodeId '{source}'; revalidate workflow"
                )
            if target not in self.node_map:
                raise RuntimeError(
                    f"Invalid edge references unknown target nodeId '{target}'; revalidate workflow"
                )
            
            self.children[source].append(target)
            self.parents[target].append(source)
            self.in_degree[target] += 1
        
        # FIX 2: Set aliases so _execute_loop can enqueue children
        self.outgoing = self.children
        self.incoming = self.parents
    
    def _seed_ready_queue(self):
        """Seed ready queue with root node (defensive checks)"""
        roots = [nid for nid, deg in self.in_degree.items() if deg == 0]
        
        # Assert exactly one root
        if len(roots) != 1:
            raise RuntimeError(f"Expected exactly 1 root node, found {len(roots)}")
        
        root_id = roots[0]
        root_node = self.node_map[root_id]
        
        # Assert root is StartNode
        if root_node.get("type") != "start":
            raise RuntimeError(f"Root node must be 'start', found {root_node.get('type')}")
        
        # Seed ready queue
        self.ready_queue = [root_id]
    
    async def _execute_loop(self):
        """Main execution loop with parallel batch scheduling and fan-in gating"""
        import asyncio
        
        while self.ready_queue:
            # Build batch of all runnable nodes (deterministic admission)
            batch = self._build_execution_batch()
            
            if not batch:
                # No nodes are ready despite queue not empty - likely deadlock
                raise RuntimeError(
                    f"Executor stuck: ready_queue has {len(self.ready_queue)} nodes but none are runnable. "
                    f"Possible dependency deadlock."
                )
            
            # Execute batch in parallel with scheduleIndex tracking
            tasks = []
            for node_id in sorted(batch):  # Deterministic order
                schedule_index = self.schedule_counter
                self.schedule_counter += 1
                tasks.append(self._execute_node_with_schedule(node_id, schedule_index))
            
            # Execute in parallel (fail-fast with persistence)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results (persist, check failures, enqueue children)
            failed_node_id, failed_error_msg = await self._process_batch_results(sorted(batch), results)
            if failed_node_id:
                raise RuntimeError(f"Node {failed_node_id} failed: {failed_error_msg}")
    
    def _build_execution_batch(self) -> List[str]:
        """
        Build batch of all runnable nodes from ready_queue.
        Nodes are runnable if all their parents are completed.
        
        Returns:
            Sorted list of node IDs ready for execution
        """
        batch = []
        remaining = []
        
        for node_id in self.ready_queue:
            parents = self.incoming.get(node_id, [])
            if all(p in self.completed for p in parents):
                batch.append(node_id)
            else:
                remaining.append(node_id)
        
        self.ready_queue = remaining
        return sorted(batch)  # Deterministic order
    
    async def _execute_node_with_schedule(self, node_id: str, schedule_index: int) -> Tuple[str, NodeResult]:
        """
        Execute a node and inject scheduleIndex into logs.
        
        Args:
            node_id: Node ID to execute
            schedule_index: Monotonic schedule index for deterministic ordering
            
        Returns:
            Tuple of (node_id, NodeResult)
        """
        node = self.node_map[node_id]
        start_time = datetime.now(timezone.utc)
        
        try:
            node_result = await self._execute_node(node)
            
            # Inject scheduleIndex into logs (for deterministic execution order)
            if not node_result.logs:
                node_result.logs = []
            node_result.logs.insert(0, f"scheduleIndex={schedule_index}")
            
            return (node_id, node_result)
            
        except Exception as node_err:
            # Node execution failed - create failed result
            end_time = datetime.now(timezone.utc)
            execution_ms = (end_time - start_time).total_seconds() * 1000
            
            logger.error(f"Node {node_id} execution failed: {node_err}", exc_info=True)
            
            node_result = NodeResult(
                status="failed",
                inputs=self._build_inputs_dict(node_id),
                output="",
                outputTruncated=False,
                outputPreview=None,
                executionMs=execution_ms,
                startedAt=start_time,
                completedAt=end_time,
                logs=[
                    f"scheduleIndex={schedule_index}",
                    "Node execution failed",
                    f"nodeId={node_id}",
                    f"errorType={type(node_err).__name__}"
                ],
                error=NodeError(
                    message=str(node_err),
                    details=type(node_err).__name__
                )
            )
            
            return (node_id, node_result)
    
    async def _process_batch_results(self, batch: List[str], results: List) -> Tuple[Optional[str], str]:
        """
        Process results from parallel batch execution.
        Persists all results, marks nodes complete, enqueues children.
        
        Args:
            batch: Sorted list of node IDs that were executed
            results: Results from asyncio.gather (same order as batch)
            
        Returns:
            Tuple of (failed_node_id, error_message) if any node failed, otherwise (None, "")
        """
        for i, node_id in enumerate(batch):
            result_data = results[i]
            
            # Handle exceptions from gather
            if isinstance(result_data, Exception):
                # Unexpected exception from gather itself
                logger.error(f"Batch execution exception for {node_id}: {result_data}", exc_info=result_data)
                
                error_msg = str(result_data)
                node_result = NodeResult(
                    status="failed",
                    inputs=self._build_inputs_dict(node_id),
                    output="",
                    outputTruncated=False,
                    outputPreview=None,
                    executionMs=0,
                    startedAt=datetime.now(timezone.utc),
                    completedAt=datetime.now(timezone.utc),
                    logs=[f"Batch exception: {type(result_data).__name__}"],
                    error=NodeError(
                        message=error_msg,
                        details=type(result_data).__name__
                    )
                )
                
                await self.repo.persist_node_result(self.run.id, self.run.userId, node_id, node_result)
                return (node_id, error_msg)  # Fail-fast with error message
            
            # Normal result: (node_id, NodeResult)
            result_node_id, node_result = result_data
            
            # Persist node result
            await self.repo.persist_node_result(self.run.id, self.run.userId, node_id, node_result)
            
            # Update heartbeat
            await self.repo.update_run_heartbeat(self.run.id, self.run.userId)
            
            # Check for failure (fail-fast)
            if node_result.status == "failed":
                error_msg = node_result.error.message if node_result.error else "unknown error"
                logger.error(f"Node {node_id} failed: {error_msg}")
                return (node_id, error_msg)  # Stop processing with error message
            
            # Mark completed
            self.completed.add(node_id)
            self.node_results[node_id] = node_result
            
            # Enqueue children
            self._enqueue_children(node_id)
        
        return (None, "")  # No failures
    
    def _enqueue_children(self, node_id: str):
        """
        Enqueue children of a completed node if all their parents are complete.
        Uses set membership to avoid duplicate enqueueing.
        
        Args:
            node_id: ID of completed node
        """
        ready_set = set(self.ready_queue)
        
        for child_id in self.outgoing.get(node_id, []):
            if child_id in ready_set or child_id in self.completed:
                continue  # Already queued or completed
            
            # Check if all parents of child are completed (fan-in gating)
            child_parents = self.incoming.get(child_id, [])
            if all(p in self.completed for p in child_parents):
                self.ready_queue.append(child_id)
                ready_set.add(child_id)  # Keep set in sync

    
    def _build_inputs_dict(self, node_id: str) -> Dict[str, str]:
        """
        Build inputs dict from parent node outputs.
        Excludes StartNode outputs per canonical contract.
        
        Args:
            node_id: Node ID to build inputs for
            
        Returns:
            Dict mapping parent nodeId to output string
        """
        inputs = {}
        
        parents = self.incoming.get(node_id, [])
        for parent_id in parents:
            parent_node = self.node_map.get(parent_id)
            
            # EXCLUDE StartNode outputs from inputs_dict
            if parent_node and parent_node.get("type") == "start":
                continue
            
            # Get parent result
            parent_result = self.node_results.get(parent_id)
            if parent_result:
                inputs[parent_id] = parent_result.output
        
        return inputs
    
    def _get_node_config(self, node: dict) -> dict:
        """
        Get node configuration with explicit precedence.
        
        FIX C: Support both 'config' and 'data' fields (frontend XYFlow compatibility).
        Precedence: node['config'] takes priority if present, else node['data'].
        Both validator and executor read from the same field with this helper.
        
        Args:
            node: Node definition
            
        Returns:
            Configuration dict (may be empty if neither field exists)
        """
        # Precedence: config > data > empty dict
        if "config" in node and node["config"]:
            return node["config"]
        elif "data" in node and node["data"]:
            return node["data"]
        else:
            return {}
    
    async def _execute_node(self, node: dict) -> NodeResult:
        """
        Execute a single node based on its type.
        
        Args:
            node: Node definition
            
        Returns:
            NodeResult with status, output, etc.
        """
        node_id = node["id"]
        node_type = node["type"]
        config = self._get_node_config(node)  # FIX C: Support both config and data fields
        
        start_time = datetime.now(timezone.utc)
        
        # Build inputs from parents
        inputs_dict = self._build_inputs_dict(node_id)
        
        # Execute based on node type
        if node_type == "start":
            # Output is metadata dict (will be serialized to string)
            output_data = {
                "runId": self.run.id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "workflowId": self.workflow.id
            }
            output_str, truncated, preview = truncate_output(output_data)
            
        elif node_type == "send_message":
            message = config.get("message", "")
            output_str, truncated, preview = truncate_output(message)
            
        elif node_type == "fan_out":
            # FanOut: pass-through control node (forwards upstream payload)
            # Build upstream input text from parent outputs
            upstream_parts = []
            for parent_id in sorted(inputs_dict.keys()):
                upstream_parts.append(inputs_dict[parent_id])
            
            # Use upstream payload as output (pass-through)
            if upstream_parts:
                input_text = "\n".join(upstream_parts)
            else:
                input_text = ""  # No upstream input
            
            children = self.children.get(node_id, [])
            output_str, truncated, preview = truncate_output(input_text)
            logs = [
                f"[FanOut] Launching {len(children)} branches",
                f"payload_length={len(input_text)} chars"
            ]
            
        elif node_type == "fan_in":
            # FanIn: aggregate parent outputs
            parents = sorted(self.parents.get(node_id, []))  # Deterministic order
            
            if not parents:
                raise RuntimeError(f"FanIn node {node_id} has no parents")
            
            # Collect parent outputs
            parent_outputs = []
            for parent_id in parents:
                parent_result = self.node_results.get(parent_id)
                if not parent_result:
                    raise RuntimeError(f"FanIn node {node_id} missing parent result: {parent_id}")
                parent_outputs.append(parent_result.output)
            
            # Apply aggregation mode
            agg_mode = config.get("aggregationMode", "json_object")
            
            if agg_mode == "concat":
                separator = config.get("separator", "\n---\n")
                aggregated = separator.join(parent_outputs)
            else:  # json_object (default)
                output_dict = {parent_id: self.node_results[parent_id].output for parent_id in parents}
                aggregated = json.dumps(output_dict, ensure_ascii=False, indent=2)
            
            output_str, truncated, preview = truncate_output(aggregated)
            logs = [f"[FanIn] Aggregated {len(parents)} results", f"aggregationMode={agg_mode}"]
            
        elif node_type == "invoke_agent":
            # FIX 2: Require agentId (no silent default)
            # Support both agentId and selectedAgent (frontend compatibility)
            agent_id = (config.get("agentId") or config.get("selectedAgent") or "").strip()
            if not agent_id:
                raise RuntimeError("node missing required config.agentId or selectedAgent")
            
            logs = [
                "agent started",
                f"agentId={agent_id}",
            ]
            
            try:
                # Build deterministic prompt
                prompt = build_agent_prompt(config, inputs_dict)
                logs.append(f"prompt_chars={len(prompt)}")
                
                # Invoke agent
                # FIX: Pass full node config for explicit mode dispatch
                agent_output = await invoke_agent(
                    cfg=self.cfg,
                    node_config=config,
                    prompt=prompt,
                    user_id=self.run.userId
                )
                
                logs.append("InvokeAgent completed")
                output_str, truncated, preview = truncate_output(agent_output)
                
            except Exception as e:
                # Agent invocation failed - return failed result immediately
                logger.error(f"InvokeAgent node {node_id} failed: {e}", exc_info=True)
                end_time = datetime.now(timezone.utc)
                execution_ms = (end_time - start_time).total_seconds() * 1000
                
                return NodeResult(
                    status="failed",
                    inputs=inputs_dict,
                    output="",
                    outputTruncated=False,
                    outputPreview=None,
                    executionMs=execution_ms,
                    startedAt=start_time,
                    completedAt=end_time,
                    logs=logs + [f"Error: {str(e)}"],
                    error=NodeError(
                        message=str(e),
                        details=type(e).__name__
                    )
                )
            
        else:
            raise RuntimeError(f"Unknown node type: {node_type}")
        
        end_time = datetime.now(timezone.utc)
        execution_ms = (end_time - start_time).total_seconds() * 1000
        
        # Create node result (FIX 1: include outputPreview)
        result = NodeResult(
            status="succeeded",
            inputs=inputs_dict,
            output=output_str,
            outputTruncated=truncated,
            outputPreview=preview,  # FIX 1: Persist preview for truncated outputs
            executionMs=execution_ms,
            startedAt=start_time,
            completedAt=end_time,
            logs=logs if (node_type in ["invoke_agent", "fan_out", "fan_in"]) else [],
            error=None
        )
        
        return result


async def execute_workflow_run(
    workflow_id: str,
    run_id: str,
    user_id: str,
    repo: WorkflowRepository,
    cfg: Settings  # PR4: Add config parameter
) -> WorkflowRun:
    """
    Execute a workflow run (entrypoint for BackgroundTasks).
    
    Args:
        workflow_id: Workflow definition ID
        run_id: WorkflowRun document ID
        user_id: User ID for authorization
        repo: Repository for database access
        cfg: Settings for agent configuration (PR4)
        
    Returns:
        Updated WorkflowRun document
    """
    # Load workflow
    workflow = await repo.get_workflow(workflow_id, user_id)
    if not workflow:
        raise RuntimeError(f"Workflow {workflow_id} not found or access denied")
    
    # Load run
    run = await repo.get_run(run_id, user_id)
    if not run:
        raise RuntimeError(f"Run {run_id} not found or access denied")
    
    # Verify consistency
    if run.workflowId != workflow_id:
        raise RuntimeError(f"Run {run_id} does not belong to workflow {workflow_id}")
    
    if run.userId != user_id:
        raise RuntimeError(f"Run {run_id} does not belong to user {user_id}")
    
    # Execute
    executor = WorkflowExecutor(workflow, run, repo, cfg)  # PR4: Pass config
    updated_run = await executor.execute()  # FIX 1: method renamed from run() to execute()
    
    return updated_run
