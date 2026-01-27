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
        config: Optional[Settings] = None,
        **params: Any
    ) -> Any:
        """
        Execute an agent with given input and context.
        
        Args:
            agent_id: Foundry agent ID
            input_data: Input string or dict for the agent
            context: WorkflowState for context access
            config: Settings config (required for agent class instantiation)
            **params: Additional parameters (e.g., database_id for SQLAgent)
        
        Returns:
            Agent output (type depends on agent)
        """
        # Detect agent type and use actual agent class if available
        if config:
            agent_type = AgentRunner.detect_agent_type(agent_id, config)
            
            # For search agents, use the actual Python class methods
            if agent_type == "aisearch":
                from ..agents.agents import AiSearchAgent
                goal = input_data if isinstance(input_data, str) else input_data.get("goal", str(input_data))
                logger.info(f"AgentRunner: Using AiSearchAgent.run() for agent {agent_id}")
                agent = AiSearchAgent(agent_id)
                return await agent.run(goal)
            
            elif agent_type == "neo4j":
                from ..agents.agents import Neo4jGraphRAGAgent
                goal = input_data if isinstance(input_data, str) else input_data.get("goal", str(input_data))
                logger.info(f"AgentRunner: Using Neo4jGraphRAGAgent.run() for agent {agent_id}")
                agent = Neo4jGraphRAGAgent(agent_id)
                return await agent.run(goal)
            
            elif agent_type == "sql":
                from ..agents.agents import SQLAgent
                goal = input_data if isinstance(input_data, str) else input_data.get("goal", str(input_data))
                database_id = params.get("database_id") or context.get("triage.database_id")
                logger.info(f"AgentRunner: Using SQLAgent.run() for agent {agent_id}, database_id={database_id}")
                agent = SQLAgent(agent_id)
                return await agent.run(goal, database_id=database_id)
        
        # For other agents (triage, reporter, reviewer), use Foundry agent directly
        # Convert input_data to string if it's a dict
        if isinstance(input_data, dict):
            # For agents that expect structured input, format appropriately
            if "input" in input_data and len(input_data) == 1:
                # Chained node - use parent output directly (no goal)
                user_content = str(input_data["input"])
            elif "goal" in input_data and "input" in input_data:
                # Structured chaining format (generic agents in linear workflows)
                user_content = f"Goal: {input_data['goal']}\n\nInput: {input_data['input']}"
            elif "goal" in input_data and "latest_news" in input_data:
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
        
        # Log executed Foundry Agent for traceability
        user_content_preview = user_content[:200] + "..." if len(user_content) > 200 else user_content
        logger.info(f"AgentRunner: agent_id={agent_id} user_content_preview=\"{user_content_preview}\"")
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
