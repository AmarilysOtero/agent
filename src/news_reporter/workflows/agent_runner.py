"""AgentRunner - Compatibility layer for agent execution"""

from __future__ import annotations
from typing import Any, Dict, Optional
import logging

from .workflow_state import WorkflowState
from ..foundry_runner import run_foundry_agent
from ..config import Settings

logger = logging.getLogger(__name__)


class AgentRunner:
    """
    Compatibility layer for executing agents.
    
    Currently wraps run_foundry_agent() for simple LLM calls.
    Future: Will support tools, policies, and budgets without changing node code.
    """
    
    def __init__(self, config: Settings):
        self.config = config
    
    @staticmethod
    async def run(
        agent_id: str,
        input_data: str | Dict[str, Any],
        context: WorkflowState,
        **params: Any
    ) -> Any:
        """
        Execute an agent with given input and context.
        
        Args:
            agent_id: Foundry agent ID
            input_data: Input string or dict for the agent
            context: WorkflowState for context access
            **params: Additional parameters (e.g., database_id for SQLAgent)
        
        Returns:
            Agent output (type depends on agent)
        """
        # Convert input_data to string if it's a dict
        if isinstance(input_data, dict):
            # For agents that expect structured input, format appropriately
            if "goal" in input_data and "latest_news" in input_data:
                # NewsReporterAgent format
                user_content = f"Goal: {input_data['goal']}\n\nLatest News: {input_data['latest_news']}"
            elif "topic" in input_data and "candidate_script" in input_data:
                # ReviewAgent format
                user_content = f"Topic: {input_data['topic']}\n\nCandidate Script:\n{input_data['candidate_script']}"
            elif "goal" in input_data:
                # Simple goal-based input
                user_content = str(input_data.get("goal", ""))
            else:
                # Generic dict to string
                user_content = str(input_data)
        else:
            user_content = str(input_data)
        
        # Add system hint if needed (for SQL generation, etc.)
        system_hint = params.get("system_hint")
        
        # Execute agent
        logger.info(f"AgentRunner: Executing agent {agent_id}")
        result = run_foundry_agent(
            agent_id=agent_id,
            user_content=user_content,
            system_hint=system_hint
        )
        
        # Future: Handle tool calls here
        # Future: Apply policies and budgets here
        
        return result
    
    @staticmethod
    def detect_agent_type(agent_id: str, config: Settings) -> str:
        """
        Detect agent type based on agent_id and config.
        
        Returns:
            Agent type: "triage", "sql", "aisearch", "neo4j", "reporter", "reviewer"
        """
        if agent_id == config.agent_id_triage:
            return "triage"
        elif agent_id == getattr(config, 'agent_id_aisearch_sql', None):
            return "sql"
        elif agent_id == config.agent_id_aisearch:
            return "aisearch"
        elif agent_id == getattr(config, 'agent_id_neo4j_search', None):
            return "neo4j"
        elif agent_id in config.reporter_ids:
            return "reporter"
        elif agent_id == config.agent_id_reviewer:
            return "reviewer"
        else:
            return "unknown"
