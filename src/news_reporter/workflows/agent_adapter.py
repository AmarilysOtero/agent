"""Agent Adapter Registry - Explicit adapters for different agent types"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

from .workflow_state import WorkflowState

logger = logging.getLogger(__name__)


class AgentAdapter(ABC):
    """
    Adapter for a specific agent type.
    
    Provides:
    - build_input: Convert state to agent input format
    - parse_output: Convert agent output to state updates
    - get_state_paths: Which state paths this agent reads/writes
    """
    
    @abstractmethod
    def build_input(self, state: WorkflowState, params: Dict[str, Any]) -> str | Dict[str, Any]:
        """
        Build input for the agent from workflow state.
        
        Args:
            state: Current workflow state
            params: Node-specific parameters
        
        Returns:
            Input string or dict for the agent
        """
        pass
    
    @abstractmethod
    def parse_output(self, agent_output: Any, state: WorkflowState) -> Dict[str, Any]:
        """
        Parse agent output and return state updates.
        
        Args:
            agent_output: Raw output from agent
            state: Current workflow state (for context)
        
        Returns:
            Dictionary of state path -> value updates
        """
        pass
    
    @abstractmethod
    def get_state_paths(self) -> Dict[str, List[str]]:
        """
        Get state paths this agent reads/writes.
        
        Returns:
            Dict with keys:
            - "reads": List of state paths read
            - "writes": List of state paths written
        """
        pass


class TriageAdapter(AgentAdapter):
    """Adapter for TriageAgent"""
    
    def build_input(self, state: WorkflowState, params: Dict[str, Any]) -> str:
        return state.goal
    
    def parse_output(self, agent_output: Any, state: WorkflowState) -> Dict[str, Any]:
        """Parse triage output (JSON string) to state updates"""
        import json
        try:
            if isinstance(agent_output, str):
                triage_data = json.loads(agent_output)
            else:
                triage_data = agent_output
            
            updates = {
                "triage": triage_data
            }
            
            # Extract specific fields if present
            if isinstance(triage_data, dict):
                if "preferred_agent" in triage_data:
                    updates["triage.preferred_agent"] = triage_data["preferred_agent"]
                if "database_id" in triage_data:
                    updates["triage.database_id"] = triage_data["database_id"]
                if "intents" in triage_data:
                    updates["triage.intents"] = triage_data["intents"]
            
            return updates
        except Exception as e:
            logger.error(f"Failed to parse triage output: {e}")
            return {"triage": {"error": str(e), "raw_output": str(agent_output)}}
    
    def get_state_paths(self) -> Dict[str, List[str]]:
        return {
            "reads": ["goal"],
            "writes": ["triage", "triage.preferred_agent", "triage.database_id", "triage.intents"]
        }


class SQLAdapter(AgentAdapter):
    """Adapter for SQLAgent"""
    
    def build_input(self, state: WorkflowState, params: Dict[str, Any]) -> str:
        goal = state.goal
        database_id = params.get("database_id") or state.get("triage.database_id")
        if database_id:
            return f"{goal}\n\nDatabase ID: {database_id}"
        return goal
    
    def parse_output(self, agent_output: Any, state: WorkflowState) -> Dict[str, Any]:
        """Parse SQL agent output (SQL query string)"""
        return {
            "latest": str(agent_output),
            "selected_search": "sql"
        }
    
    def get_state_paths(self) -> Dict[str, List[str]]:
        return {
            "reads": ["goal", "triage.database_id"],
            "writes": ["latest", "selected_search"]
        }


class AiSearchAdapter(AgentAdapter):
    """Adapter for AiSearchAgent"""
    
    def build_input(self, state: WorkflowState, params: Dict[str, Any]) -> str:
        return state.goal
    
    def parse_output(self, agent_output: Any, state: WorkflowState) -> Dict[str, Any]:
        """Parse AiSearch output (search results string)"""
        output_str = str(agent_output)
        logger.info(f"📊 AiSearchAdapter: Storing search results in 'latest' (length: {len(output_str)})")
        logger.debug(f"📊 AiSearchAdapter: First 500 chars of output: {output_str[:500]}")
        return {
            "latest": output_str,
            "selected_search": "aisearch"
        }
    
    def get_state_paths(self) -> Dict[str, List[str]]:
        return {
            "reads": ["goal"],
            "writes": ["latest", "selected_search"]
        }


class Neo4jAdapter(AgentAdapter):
    """Adapter for Neo4jGraphRAGAgent"""
    
    def build_input(self, state: WorkflowState, params: Dict[str, Any]) -> str:
        return state.goal
    
    def parse_output(self, agent_output: Any, state: WorkflowState) -> Dict[str, Any]:
        """Parse Neo4j output (graph search results string)"""
        return {
            "latest": str(agent_output),
            "selected_search": "neo4j"
        }
    
    def get_state_paths(self) -> Dict[str, List[str]]:
        return {
            "reads": ["goal"],
            "writes": ["latest", "selected_search"]
        }


class NewsReporterAdapter(AgentAdapter):
    """Adapter for NewsReporterAgent"""
    
    def build_input(self, state: WorkflowState, params: Dict[str, Any]) -> Dict[str, Any]:
        reporter_id = params.get("reporter_id") or state.get("current_fanout_item")
        latest_news = state.latest or ""
        logger.info(f"📊 NewsReporterAdapter: Building input for reporter - goal length: {len(state.goal)}, latest_news length: {len(latest_news)}")
        logger.debug(f"📊 NewsReporterAdapter: latest_news first 500 chars: {latest_news[:500] if latest_news else 'None'}")
        return {
            "goal": state.goal,
            "latest_news": latest_news,
            "reporter_id": reporter_id
        }
    
    def parse_output(self, agent_output: Any, state: WorkflowState) -> Dict[str, Any]:
        """Parse reporter output (draft script string)"""
        reporter_id = state.get("current_fanout_item", "unknown")
        return {
            f"drafts.{reporter_id}": str(agent_output)
        }
    
    def get_state_paths(self) -> Dict[str, List[str]]:
        return {
            "reads": ["goal", "latest", "current_fanout_item"],
            "writes": ["drafts"]
        }


class ReviewAdapter(AgentAdapter):
    """Adapter for ReviewAgent"""
    
    def build_input(self, state: WorkflowState, params: Dict[str, Any]) -> Dict[str, Any]:
        reporter_id = params.get("reporter_id") or state.get("current_fanout_item")
        draft = state.get(f"drafts.{reporter_id}", "")
        return {
            "topic": state.goal,
            "candidate_script": draft,
            "reporter_id": reporter_id
        }
    
    def parse_output(self, agent_output: Any, state: WorkflowState) -> Dict[str, Any]:
        """Parse review output (verdict JSON or text)"""
        import json
        reporter_id = state.get("current_fanout_item", "unknown")
        
        try:
            if isinstance(agent_output, str):
                verdict_data = json.loads(agent_output)
            else:
                verdict_data = agent_output
            
            # Store verdict
            verdicts = state.get("verdicts", {})
            if reporter_id not in verdicts:
                verdicts[reporter_id] = []
            verdicts[reporter_id].append(verdict_data)
            
            return {
                f"verdicts.{reporter_id}": verdicts[reporter_id]
            }
        except Exception:
            # If not JSON, store as text
            verdicts = state.get("verdicts", {})
            if reporter_id not in verdicts:
                verdicts[reporter_id] = []
            verdicts[reporter_id].append({"text": str(agent_output)})
            return {
                f"verdicts.{reporter_id}": verdicts[reporter_id]
            }
    
    def get_state_paths(self) -> Dict[str, List[str]]:
        return {
            "reads": ["goal", "drafts", "current_fanout_item"],
            "writes": ["verdicts"]
        }


class AgentAdapterRegistry:
    """Registry for agent adapters"""
    
    _adapters: Dict[str, AgentAdapter] = {}
    _agent_id_to_type: Dict[str, str] = {}
    
    @classmethod
    def register(cls, agent_type: str, adapter: AgentAdapter) -> None:
        """Register an adapter for an agent type"""
        cls._adapters[agent_type] = adapter
        logger.info(f"Registered adapter for agent type: {agent_type}")
    
    @classmethod
    def register_agent_id(cls, agent_id: str, agent_type: str) -> None:
        """Register an agent ID to type mapping"""
        cls._agent_id_to_type[agent_id] = agent_type
        logger.debug(f"Registered agent ID {agent_id} -> type {agent_type}")
    
    @classmethod
    def get_adapter(cls, agent_type: str) -> Optional[AgentAdapter]:
        """Get adapter for agent type"""
        return cls._adapters.get(agent_type)
    
    @classmethod
    def get_adapter_by_agent_id(cls, agent_id: str) -> Optional[AgentAdapter]:
        """Get adapter by agent ID"""
        agent_type = cls._agent_id_to_type.get(agent_id)
        if agent_type:
            return cls.get_adapter(agent_type)
        return None
    
    @classmethod
    def initialize_defaults(cls, config: Any) -> None:
        """Initialize default adapters and mappings from config"""
        # Register default adapters
        cls.register("triage", TriageAdapter())
        cls.register("sql", SQLAdapter())
        cls.register("aisearch", AiSearchAdapter())
        cls.register("neo4j", Neo4jAdapter())
        cls.register("reporter", NewsReporterAdapter())
        cls.register("reviewer", ReviewAdapter())
        
        # Register agent ID mappings
        if hasattr(config, 'agent_id_triage'):
            cls.register_agent_id(config.agent_id_triage, "triage")
        if hasattr(config, 'agent_id_aisearch_sql'):
            cls.register_agent_id(config.agent_id_aisearch_sql, "sql")
        if hasattr(config, 'agent_id_aisearch'):
            cls.register_agent_id(config.agent_id_aisearch, "aisearch")
        if hasattr(config, 'agent_id_neo4j_search'):
            cls.register_agent_id(config.agent_id_neo4j_search, "neo4j")
        if hasattr(config, 'reporter_ids'):
            for reporter_id in config.reporter_ids:
                cls.register_agent_id(reporter_id, "reporter")
        if hasattr(config, 'agent_id_reviewer'):
            cls.register_agent_id(config.agent_id_reviewer, "reviewer")
        
        logger.info("Initialized default agent adapters")


# Initialize on import
# Note: Will be re-initialized with actual config when GraphExecutor is created
