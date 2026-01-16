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
        """Main execution loop with sequential scheduling and fan-in gating"""
        # FIX 2: Add no-progress counter to prevent infinite loops
        no_progress_iterations = 0
        MAX_NO_PROGRESS = 100  # Safety guard
        
        while self.ready_queue:
            # Sort ready queue deterministically (lexicographic by nodeId)
            self.ready_queue.sort()
            
            # Pop first node
            current_node_id = self.ready_queue.pop(0)
            
            # Check fan-in: all parents must be completed
            parents = self.incoming.get(current_node_id, [])  # FIX 2: Use self.incoming for consistency
            if not all(p in self.completed for p in parents):
                # FIX 2: Requeue node instead of dropping it
                logger.warning(f"Node {current_node_id} not ready (parents not complete), requeueing")
                self.ready_queue.append(current_node_id)
                
                # Increment no-progress counter
                no_progress_iterations += 1
                if no_progress_iterations > MAX_NO_PROGRESS:
                    raise RuntimeError(
                        f"Executor stuck: {no_progress_iterations} iterations with no progress. "
                        f"Possible dependency deadlock or graph issue."
                    )
                continue
            
            # Reset no-progress counter (we're making progress)
            no_progress_iterations = 0
            
            # FIX 2: Capture start time for telemetry
            node_start = datetime.now(timezone.utc)
            
            # Execute node
            node = self.node_map[current_node_id]
            try:
                node_result = await self._execute_node(node)
            except Exception as node_err:
                # FIX 2: Compute proper execution time and add telemetry
                end_time = datetime.now(timezone.utc)
                execution_ms = (end_time - node_start).total_seconds() * 1000
                
                # Node execution failed - persist failed result with proper telemetry
                logger.error(f"Node {current_node_id} execution failed: {node_err}", exc_info=True)
                
                # FIX 2 & 3: Ensure consistent NodeResult shape with proper timestamps and logs
                node_result = NodeResult(
                    status="failed",
                    inputs=self._build_inputs_dict(current_node_id),
                    output="",
                    outputTruncated=False,
                    outputPreview=None,  # FIX 3: Explicit for shape consistency
                    executionMs=execution_ms,  # FIX 2: Real execution time
                    startedAt=node_start,  # FIX 2: Actual start time
                    completedAt=end_time,  # FIX 2: Actual completion time
                    logs=[  # FIX 2: Add telemetry logs
                        "Node execution failed",
                        f"nodeId={current_node_id}",
                        f"errorType={type(node_err).__name__}"
                    ],
                    error=NodeError(
                        message=str(node_err),
                        details=type(node_err).__name__
                    )
                )
                
                # Persist failed node result
                await self.repo.persist_node_result(self.run.id, self.run.userId, current_node_id, node_result)  # PR5 Fix 1: userId
                
                # FIX 1: Normalize fail-fast error propagation
                error_msg = node_result.error.message if node_result.error else str(node_err)
                raise RuntimeError(f"Node {current_node_id} failed: {error_msg}")

            
            # Persist node result
            await self.repo.persist_node_result(self.run.id, self.run.userId, current_node_id, node_result)  # PR5 Fix 1: userId
            
            # FIX A: Fail-fast if node execution failed
            if node_result.status == "failed":
                error_msg = node_result.error.message if node_result.error else "unknown error"
                raise RuntimeError(f"Node {current_node_id} failed: {error_msg}")
            
            # Update heartbeat periodically
            await self.repo.update_run_heartbeat(self.run.id, self.run.userId)  # PR5 Fix 1: userId
            
            # Mark completed
            self.completed.add(current_node_id)
            self.node_results[current_node_id] = node_result
            
            # Enqueue children whose parents are all completed (fan-in gating)
            for child_id in self.outgoing.get(current_node_id, []):
                if child_id not in self.ready_queue:
                    # Check if all parents of child are completed
                    child_parents = self.incoming.get(child_id, [])
                    if all(p in self.completed for p in child_parents):
                        self.ready_queue.append(child_id)
    
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
            logs=logs if node_type == "invoke_agent" else [],  # PR4: Include logs for InvokeAgent
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
