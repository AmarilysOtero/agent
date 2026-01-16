# src/news_reporter/workflows/agent_invoke.py
"""Agent invocation helper for workflow executor (PR4)"""
import logging
import asyncio
from typing import Optional
from ..config import Settings
from .workflow_factory import run_sequential_goal

logger = logging.getLogger(__name__)


from ..foundry_runner import run_foundry_agent

async def invoke_agent(cfg: Settings, node_config: dict, prompt: str, user_id: str) -> str:
    """
    Invoke an agent based on node configuration mode.
    
    Modes:
    - "foundry_agent": Direct execution of a specific Foundry agent (default for demo).
    - "sequential_goal": Legacy orchestration pipeline (Start -> Triage -> Search -> Reporter).
    
    Args:
        cfg: App settings
        node_config: Node configuration dict (must contain 'mode' and 'agentId' if mode='foundry_agent')
        prompt: Input prompt
        user_id: User ID for scoping
        
    Returns:
        Agent output string
    """
    mode = node_config.get("mode")
    agent_id = (node_config.get("agentId") or node_config.get("selectedAgent") or "").strip()
    
    # Strict validation per requirements
    if not mode:
        # Fallback for legacy/demo workflows created before this change
        if agent_id:
             mode = "foundry_agent"
             logger.warning(f"Node missing 'mode', defaulting to 'foundry_agent' because agentId='{agent_id}' is present")
        else:
             raise RuntimeError("InvokeAgent node missing required 'mode' configuration (e.g., 'foundry_agent')")

    logger.info(f"\n\nInvoking agent: {agent_id}, mode: {mode}, user: {user_id}, prompt_len: {len(prompt)}")
    
    if mode == "foundry_agent":
        if not agent_id:
            raise RuntimeError("InvokeAgent mode='foundry_agent' requires 'agentId'")
            
        # Execute direct Foundry path
        logger.info(f"Invoking Foundry agent {agent_id} (node_config mode='foundry_agent')...")
        try:
            # Note: run_foundry_agent is blocking, but that matches existing pattern in agents.py
            output = await asyncio.to_thread(run_foundry_agent, agent_id, prompt)
            logger.info(f"Foundry agent {agent_id} completed, output_len={len(output)}")
            return output
        except Exception as e:
            logger.error(f"Foundry agent {agent_id} failed: {e}")
            raise RuntimeError(f"Foundry agent execution failed: {e}") from e

    elif mode == "sequential_goal":
        # Legacy/Chat-Session orchestration
        logger.info(f"Routing to run_sequential_goal (orchestration layer)")
        return await run_sequential_goal(cfg, prompt)
        
    else:
        raise RuntimeError(f"Invalid InvokeAgent mode: '{mode}'. Must be 'foundry_agent' or 'sequential_goal'")


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
