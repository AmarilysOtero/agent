"""Main agents module - imports all agent classes from separate files.

This module provides the public API for all agent classes:
- TriageAgent: Intent classification and database routing
- AiSearchAgent: Search using Neo4j GraphRAG
- SQLAgent: SQL queries with fallback to CSV and Vector search
- Neo4jGraphRAGAgent: Neo4j GraphRAG search
- AssistantAgent: Natural language response generation
- ReviewAgent: Response validation and refinement

Usage:
    from agents import AiSearchAgent, SQLAgent, TriageAgent
    
    ai_search = AiSearchAgent(foundry_agent_id="...")
    triage = TriageAgent(foundry_agent_id="...")
"""

from __future__ import annotations
import logging

# Import all agent classes from their respective modules
from .ai_search_agent import AiSearchAgent
from .sql_agent import SQLAgent
from .neo4j_graphrag_agent import Neo4jGraphRAGAgent
from .assistant_agent import AssistantAgent, NewsReporterAgent
from .review_agent import ReviewAgent
from .triage_agent import TriageAgent, IntentResult
from .utils import (
    infer_header_from_chunk,
    extract_person_names_and_mode,
    extract_person_names,
    filter_results_by_exact_match,
)

logger = logging.getLogger(__name__)

# Public API
__all__ = [
    "TriageAgent",
    "AiSearchAgent",
    "SQLAgent",
    "Neo4jGraphRAGAgent",
    "AssistantAgent",
    "NewsReporterAgent",
    "ReviewAgent",
    "IntentResult",
    "infer_header_from_chunk",
    "extract_person_names_and_mode",
    "extract_person_names",
    "filter_results_by_exact_match",
]
