from __future__ import annotations
import asyncio
import logging
import json
from typing import Any, Dict, List

from ..config import Settings
from ..agents.agents import TriageAgent, AiSearchAgent, Neo4jGraphRAGAgent, NewsReporterAgent, ReviewAgent, SQLAgent
from .graph_executor import GraphExecutor
from .graph_loader import load_graph_definition

# Optional analytics import
try:
    from .workflow_analytics import get_analytics_engine
except ImportError:
    def get_analytics_engine():
        return None

logger = logging.getLogger(__name__)

async def run_graph_workflow(
    cfg: Settings, 
    goal: str, 
    graph_path: str | None = None,
    workflow_definition: Dict[str, Any] | None = None
) -> str:
    """
    Execute workflow using graph executor.
    
    Args:
        cfg: Settings configuration
        goal: User goal/query
        graph_path: Optional path to graph JSON (default: default_workflow.json)
        workflow_definition: Optional workflow definition dict (from agent builder)
    
    Returns:
        Final output string
    """
    try:
        # Load graph definition from workflow_definition if provided, otherwise from graph_path
        if workflow_definition:
            logger.debug(f"Loading workflow from definition dict (entry_node_id: {workflow_definition.get('entry_node_id', 'triage')})")
            # Convert workflow definition dict to GraphDefinition
            from .graph_schema import GraphDefinition, NodeConfig, EdgeConfig, GraphLimits
            
            try:
                # Substitute agent IDs if needed
                if cfg:
                    workflow_definition = _substitute_agent_ids_in_dict(workflow_definition, cfg)
                
                # Parse nodes
                nodes = []
                for node_data in workflow_definition.get("nodes", []):
                    try:
                        nodes.append(NodeConfig(**node_data))
                    except Exception as node_error:
                        logger.error(f"Failed to parse node {node_data.get('id', 'unknown')}: {node_error}", exc_info=True)
                        raise ValueError(f"Invalid node configuration: {node_error}")
                
                # Parse edges
                edges = []
                for edge_data in workflow_definition.get("edges", []):
                    try:
                        edges.append(EdgeConfig(**edge_data))
                    except Exception as edge_error:
                        logger.error(f"Failed to parse edge {edge_data.get('from_node', '?')} -> {edge_data.get('to_node', '?')}: {edge_error}", exc_info=True)
                        raise ValueError(f"Invalid edge configuration: {edge_error}")
                
                # Create graph definition
                limits = None
                if workflow_definition.get("limits"):
                    try:
                        limits = GraphLimits(**workflow_definition["limits"])
                    except Exception as limits_error:
                        logger.warning(f"Failed to parse limits, using defaults: {limits_error}")
                
                graph_def = GraphDefinition(
                    nodes=nodes,
                    edges=edges,
                    entry_node_id=workflow_definition.get("entry_node_id", "triage"),
                    toolsets=workflow_definition.get("toolsets", []),
                    policy_profile=workflow_definition.get("policy_profile", "read_only"),
                    limits=limits,
                    name=workflow_definition.get("name"),
                    description=workflow_definition.get("description"),
                    version=workflow_definition.get("version")
                )
                
                # Validate workflow before execution
                validation_errors = graph_def.validate()
                if validation_errors:
                    logger.warning(f"Workflow validation warnings ({len(validation_errors)}): {validation_errors}")
                    # Continue execution but log warnings
                    # If there are critical errors, we might want to fail early
                    critical_errors = [e for e in validation_errors if "missing agent_id" in e or "not found" in e]
                    if critical_errors:
                        logger.error(f"Critical validation errors detected: {critical_errors}")
                        raise ValueError(f"Workflow validation failed: {critical_errors}")
            except (ValueError, KeyError, TypeError) as parse_error:
                logger.error(f"Failed to parse workflow definition: {parse_error}", exc_info=True)
                raise
        else:
            # Load from graph_path
            logger.debug(f"Loading workflow from graph_path: {graph_path}")
            try:
                graph_def = load_graph_definition(graph_path=graph_path, config=cfg)
            except FileNotFoundError as file_error:
                logger.error(f"Graph definition file not found: {file_error}")
                raise
            except Exception as load_error:
                logger.error(f"Failed to load graph definition: {load_error}", exc_info=True)
                raise
        
        # Create executor
        logger.debug(f"Creating GraphExecutor with {len(graph_def.nodes)} nodes and {len(graph_def.edges)} edges")
        executor = GraphExecutor(graph_def, cfg)
        
        # Execute
        logger.info("=" * 100)
        logger.info("=" * 100)
        logger.info(f"🚀 NEW WORKFLOW RUN STARTING - Goal: {goal[:100]}...")
        logger.info("=" * 100)
        logger.info("=" * 100)
        result = await executor.execute(goal)
        logger.info("=" * 100)
        logger.info("=" * 100)
        logger.info("✅ Graph workflow execution completed successfully")
        logger.info("=" * 100)
        logger.info("=" * 100)
        
        # Phase 6: Collect metrics for analytics (simplified - would get run_id properly)
        try:
            analytics_engine = get_analytics_engine()
            if analytics_engine:
                # Get the most recent metrics
                all_metrics = executor.metrics_collector.get_all_metrics()
                if all_metrics:
                    analytics_engine.add_metrics(all_metrics[-1])
        except Exception as e:
            logger.warning(f"Failed to add metrics to analytics: {e}")
        
        return result
    except (ValueError, FileNotFoundError) as e:
        # These are parsing/configuration errors - should fall back
        logger.error(f"Graph workflow configuration error: {e}", exc_info=True)
        logger.warning("Falling back to sequential workflow due to configuration error")
        return await run_sequential_goal(cfg, goal)
    except Exception as e:
        logger.error(f"Graph workflow execution failed: {e}", exc_info=True)
        logger.warning("Falling back to sequential workflow")
        return await run_sequential_goal(cfg, goal)


def _substitute_agent_ids_in_dict(graph_data: dict, config: Settings) -> dict:
    """Substitute ${VAR} placeholders in agent_id fields with actual values"""
    import os
    
    # Create mapping of env vars to config values
    substitutions = {
        "AGENT_ID_TRIAGE": config.agent_id_triage,
        "AGENT_ID_AISEARCH": config.agent_id_aisearch,
        "AGENT_ID_AISEARCHSQL": getattr(config, 'agent_id_aisearch_sql', None),
        "AGENT_ID_NEO4J_SEARCH": getattr(config, 'agent_id_neo4j_search', None),
        "AGENT_ID_REVIEWER": config.agent_id_reviewer,
        "AGENT_ID_REPORTER": config.reporter_ids[0] if config.reporter_ids else None,
    }
    
    # Create a copy to avoid modifying the original
    graph_data = json.loads(json.dumps(graph_data)) if isinstance(graph_data, dict) else graph_data.copy()
    
    # Substitute in nodes
    for node in graph_data.get("nodes", []):
        if "agent_id" in node and isinstance(node["agent_id"], str):
            agent_id = node["agent_id"]
            if agent_id.startswith("${") and agent_id.endswith("}"):
                var_name = agent_id[2:-1]
                if var_name in substitutions and substitutions[var_name]:
                    node["agent_id"] = substitutions[var_name]
                else:
                    # Try environment variable
                    node["agent_id"] = os.getenv(var_name, agent_id)
    
    return graph_data


async def run_sequential_goal(cfg: Settings, goal: str) -> str:
    """
    Foundry-defined agents + local orchestration (no external AF dependency).

    Flow:
      TRIAGE -> AISEARCH -> REPORTER -> REVIEWER (≤3 passes)
    """
    # ---- 1) TRIAGE ----
    triage = TriageAgent(cfg.agent_id_triage)
    tri = await triage.run(goal)
    print("Triage:", tri.model_dump())

    # Decide whether to fan out across multiple reporter agents
    do_multi = ("multi" in tri.intents) or cfg.multi_route_always
    targets: List[str] = cfg.reporter_ids if do_multi else [cfg.reporter_ids[0]]

    # Choose search agent based on TriageAgent routing or config
    # Check if TriageAgent detected a preferred agent type
    print(f"🔍 Workflow: Checking TriageAgent results:")
    print(f"   - intents: {tri.intents}")
    print(f"   - preferred_agent: {tri.preferred_agent}")
    print(f"   - database_id: {tri.database_id}")
    print(f"   - database_type: {getattr(tri, 'database_type', 'N/A')}")
    print(f"   - has agent_id_aisearch_sql: {hasattr(cfg, 'agent_id_aisearch_sql')}")
    logger.info(f"🔍 Workflow: TriageAgent results - intents={tri.intents}, preferred_agent={tri.preferred_agent}, database_id={tri.database_id}, database_type={getattr(tri, 'database_type', 'N/A')}")
    if hasattr(cfg, 'agent_id_aisearch_sql'):
        print(f"   - agent_id_aisearch_sql value: {cfg.agent_id_aisearch_sql}")
        logger.info(f"   - agent_id_aisearch_sql value: {cfg.agent_id_aisearch_sql}")
    
    search_database_id = None
    if tri.preferred_agent == "sql" and hasattr(cfg, 'agent_id_aisearch_sql') and cfg.agent_id_aisearch_sql:
        print(f"✅ Using SQL Agent (PostgreSQL → CSV → Vector) for database_id: {tri.database_id}")
        search_agent = SQLAgent(cfg.agent_id_aisearch_sql)
        search_database_id = tri.database_id  # Pass database_id to SQL agent (may be None for auto-detect)
    elif cfg.use_neo4j_search and cfg.agent_id_neo4j_search:
        print("Using Neo4j GraphRAG Agent (cost-efficient)")
        search_agent = Neo4jGraphRAGAgent(cfg.agent_id_neo4j_search)
    else:
        print("Using Azure Search Agent (production)")
        if tri.preferred_agent == "sql":
            print(f"⚠️  WARNING: TriageAgent set preferred_agent='sql' but SQL agent not configured or not available")
        search_agent = AiSearchAgent(cfg.agent_id_aisearch)
    
    reviewer = ReviewAgent(cfg.agent_id_reviewer)

    # ---- 2) Actual execution logic ----
    async def run_one(reporter_id: str) -> str:
        reporter = NewsReporterAgent(reporter_id)

        # AI Search step (works with either agent)
        # Also handle "unknown" intents if schema detection found a database (might be misclassified search query)
        should_search = "ai_search" in tri.intents or (
            "unknown" in tri.intents and tri.preferred_agent and tri.database_id
        )
        if should_search:
            # Pass database_id to SQLAgent if it's a SQLAgent (database_id may be None for auto-detect)
            if isinstance(search_agent, SQLAgent):
                latest = await search_agent.run(goal, database_id=search_database_id)
            else:
                latest = await search_agent.run(goal)
        else:
            latest = ""

        # Reporter step
        logger.info(f"🔍 Workflow: Reporter step - news_script in intents: {'news_script' in tri.intents}, latest length: {len(latest) if latest else 0}")
        print(f"🔍 Workflow: Reporter step - news_script in intents: {'news_script' in tri.intents}, latest length: {len(latest) if latest else 0}")
        script = (
            await reporter.run(goal, latest or "No ai-search content")
            if ("news_script" in tri.intents)
            else latest
        )
        logger.info(f"🔍 Workflow: Final script length: {len(script) if script else 0}")
        print(f"🔍 Workflow: Final script length: {len(script) if script else 0}")
        if not script:
            logger.warning(f"🔍 Workflow: No script generated - returning 'No action taken.'")
            print(f"⚠️ Workflow: No script generated - returning 'No action taken.'")
            return "No action taken."

        # Review step (max 3 passes)
        max_iters = 3
        for i in range(1, max_iters + 1):
            print(f"Review pass {i}/{max_iters}...")
            verdict = await reviewer.run(goal, script)
            decision = (verdict.get("decision") or "revise").lower()
            reason = verdict.get("reason", "")
            suggested = verdict.get("suggested_changes", "")
            revised = verdict.get("revised_script", script)

            print(f"Decision: {decision} | Reason: {reason}")

            if decision == "accept":
                return revised or script

            # Ask reporter to improve using reviewer notes
            improve_context = (
                f"Apply these review notes strictly:\n{suggested or reason}\n\n"
                f"Original draft:\n{script}"
            )
            script = await reporter.run(goal, improve_context)

        return f"[After {max_iters} review passes]\n{script}"

    if len(targets) > 1:
        results = await asyncio.gather(*[run_one(rid) for rid in targets])
        stitched = []
        for rid, out in zip(targets, results):
            stitched.append(f"### ReporterAgent={rid}\n{out}")
        return "\n\n---\n\n".join(stitched)

    return await run_one(targets[0])
