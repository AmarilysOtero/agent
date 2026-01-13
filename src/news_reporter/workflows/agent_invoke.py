# src/news_reporter/workflows/agent_invoke.py
"""Agent invocation helper for workflow executor (PR4)"""
import logging
from typing import Optional
from ..config import Settings
from .workflow_factory import run_sequential_goal

logger = logging.getLogger(__name__)


async def invoke_agent(agent_id: str, prompt: str, user_id: str, cfg: Optional[Settings] = None) -> str:
    """
    Invoke an agent with the given prompt and return plain text output.
    
    FIX 1: agentId now explicitly dispatches to execution pathway (not magic prefix).
    Currently, only run_sequential_goal is available, so we use it as fallback
    and log the limitation.
    
    Args:
        agent_id: Agent identifier (e.g., "TRIAGE", "SEARCH", "REPORTER", "SEQUENTIAL_GOAL")
        prompt: Input prompt constructed from workflow node inputs
        user_id: User ID for authorization/scoping
        cfg: Optional settings (will load from env if not provided)
        
    Returns:
        Plain text output from agent execution
        
    Raises:
        Exception: If agent execution fails for any reason
    """
    try:
        # Load config if not provided
        if cfg is None:
            cfg = Settings.load()
        
        logger.info(f"Invoking agent: {agent_id}, user: {user_id}, prompt_length: {len(prompt)}")
        
        # FIX 1: Explicit dispatch based on agentId
        # NOTE: Currently only run_sequential_goal is available as entrypoint.
        # Individual agent classes (TriageAgent, etc.) exist but are used internally
        # by run_sequential_goal. Future: expose per-agent execution if needed.
        
        if agent_id in ["TRIAGE", "SEARCH", "REPORTER", "REVIEWER", "SEQUENTIAL_GOAL"]:
            # Use sequential goal orchestration (current only pathway)
            logger.info(f"AgentId '{agent_id}' routed to run_sequential_goal (orchestration layer)")
            result = await run_sequential_goal(cfg, prompt)
        else:
            # Unknown agentId - use fallback and warn
            logger.warning(f"Unknown agentId '{agent_id}', falling back to run_sequential_goal")
            result = await run_sequential_goal(cfg, prompt)
        
        logger.info(f"Agent {agent_id} completed successfully, output_length: {len(result)}")
        
        return result
        
    except Exception as e:
        logger.error(f"Agent {agent_id} execution failed: {e}", exc_info=True)
        raise


def build_agent_prompt(
    node_config: dict,
    inputs_dict: dict[str, str]
) -> str:
    """
    Build deterministic prompt for InvokeAgent from node config and parent outputs.
    
    PR4 Requirements:
    - Sort parent node IDs lexicographically
    - Build parts in order: [prefix] + [sorted parent outputs]
    - Join with exact delimiter: \\n\\n---\\n\\n
    
    Args:
        node_config: Node configuration dict (may contain 'input' prefix)
        inputs_dict: Map of parent nodeId → output string
        
    Returns:
        Constructed prompt string
    """
    parts = []
    
    # Optional  prefix from config.input
    prefix = node_config.get("input", "").strip()
    if prefix:
        parts.append(prefix)
    
    # Parent outputs in lexicographic order
    sorted_parents = sorted(inputs_dict.keys())
    for parent_id in sorted_parents:
        output = inputs_dict[parent_id]
        if output:  # Skip empty outputs
            parts.append(output)
    
    # Join with exact delimiter
    delimiter = "\n\n---\n\n"
    prompt = delimiter.join(parts)
    
    return prompt
