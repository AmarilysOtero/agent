from __future__ import annotations
import asyncio
import logging
import json
import os
from typing import Any, Dict, List

from ..config import Settings
from ..agents.agents import TriageAgent, AiSearchAgent, Neo4jGraphRAGAgent, AssistantAgent, ReviewAgent, SQLAgent
from .graph_executor import GraphExecutor
from .graph_loader import load_graph_definition
from ..retrieval.file_expansion import expand_to_full_files, filter_chunks_by_relevance, log_expanded_chunks
from ..tools.neo4j_graphrag import expand_files_via_api
from ..retrieval.chunk_logger import log_chunks_to_markdown, ensure_rlm_enable_log_files
from ..retrieval.recursive_summarizer import recursive_summarize_files, log_file_summaries_to_markdown
from ..retrieval.phase_5_answer_generator import generate_final_answer, log_final_answer_to_markdown

# Optional analytics import
try:
    from .workflow_analytics import get_analytics_engine
except ImportError:
    def get_analytics_engine():
        return None

logger = logging.getLogger(__name__)


def _search_results_to_expanded_files(
    search_results: List[Dict[str, Any]],
    query: str | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Build expanded_files dict from search agent results for RLM pipeline.

    Groups chunks by file_id (or file_name) so recursive_summarize_files can run.
    Each chunk provides chunk_id, text, and metadata. Keywords are taken only from
    the chunk (metadata.keywords or top-level keywords); no hardcoded extraction.
    """
    expanded: Dict[str, Dict[str, Any]] = {}
    for r in search_results:
        chunk_id = r.get("chunk_id") or r.get("id") or ""
        text = r.get("text", "")
        file_id = (
            r.get("file_id")
            or (r.get("metadata") or {}).get("file_id")
            or r.get("file_name")
            or "unknown"
        )
        file_name = r.get("file_name", str(file_id))
        meta = dict(r.get("metadata") or {})
        # Keywords only from chunk: metadata.keywords or top-level keywords (no hardcoded fallback)
        if "keywords" not in meta and r.get("keywords") is not None:
            meta["keywords"] = r["keywords"] if isinstance(r["keywords"], list) else []
        if file_id not in expanded:
            expanded[file_id] = {
                "chunks": [],
                "file_name": file_name,
                "entry_chunk_count": 0,
            }
        expanded[file_id]["chunks"].append({
            "chunk_id": chunk_id,
            "text": text,
            "metadata": meta,
        })
    for fd in expanded.values():
        fd["entry_chunk_count"] = len(fd["chunks"])
    return expanded


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
            from ..models.graph_schema import GraphDefinition, NodeConfig, EdgeConfig, GraphLimits
            
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
    General RAG assistance workflow (no external AF dependency).

    Flow:
      TRIAGE -> SEARCH (retrieve context) -> ASSISTANT (generate response) -> REVIEWER (≤3 passes)
    
    Note: When RLM_ENABLED=true, execution routes through RLM-specific branch after Search.
    """
    # ---- 1) TRIAGE ----
    triage = TriageAgent(cfg.agent_id_triage)
    tri = await triage.run(goal)
    print("Triage:", tri.model_dump())

    # RLM Branch Selection
    if cfg.rlm_enabled:
        logger.info("RLM branch selected")
        print("\n🔄 RLM MODE ACTIVATED (currently routes through default sequential for Phase 1)")
        # Ensure logs/chunk_analysis/enable exists and initial .md files are created for RLM runs
        try:
            ensure_rlm_enable_log_files(query=goal)
        except Exception as e:
            logger.warning("Could not init RLM chunk log (enable): %s", e)
        # Phase 1: Both paths still call the existing flow (no behavioral change)
    else:
        logger.debug("Default sequential branch selected (RLM not enabled)")

    # Decide whether to fan out across multiple assistant agents
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
    async def run_one(assistant_id: str) -> str:
        # Ensure all variables are defined before use
        high_recall_mode = bool(cfg.rlm_enabled)
        if high_recall_mode:
            logger.info("🔍 Workflow: High-recall retrieval enabled (RLM mode)")
        raw_results = None
        context = None
        expanded_context = None
        expanded_files = {}
        file_summaries = []
        final_answer = None
        response = None

        # --- Retrieve context from the selected search agent ---
        raw_results: List[Dict[str, Any]] = []
        try:
            logger.info(f"🔍 Workflow: Calling search agent {search_agent.__class__.__name__} (ID: {getattr(search_agent, 'agent_id', 'N/A')})")
            print(f"🔍 Workflow: Calling search agent {search_agent.__class__.__name__} (ID: {getattr(search_agent, 'agent_id', 'N/A')})")
            if isinstance(search_agent, SQLAgent):
                context = await search_agent.run(goal, database_id=search_database_id)
            elif cfg.rlm_enabled and (
                isinstance(search_agent, AiSearchAgent) or isinstance(search_agent, Neo4jGraphRAGAgent)
            ):
                out = await search_agent.run(goal, high_recall_mode=True, return_results=True)
                if isinstance(out, tuple):
                    context, raw_results = out[0], (out[1] if len(out) > 1 else [])
                else:
                    context = out
            else:
                context = await search_agent.run(goal)
            logger.info(f"🔍 Workflow: Search agent returned context of length {len(context) if context else 0}")
            print(f"🔍 Workflow: Search agent returned context of length {len(context) if context else 0}")
        except Exception as e:
            logger.error(f"Search agent failed: {e}", exc_info=True)
            context = None

        # --- RLM pipeline: recursive summarization + Phase 5 when RLM enabled and we have structured results ---
        if cfg.rlm_enabled and raw_results:
            try:
                expanded_files = _search_results_to_expanded_files(raw_results, goal)
                # When RLM is enabled, expand to all chunks per file (Phase 3) so the chunk log includes every chunk (e.g. 106 for a PDF).
                if expanded_files:
                    file_ids = list(expanded_files.keys())
                    expanded_full = await asyncio.to_thread(expand_files_via_api, file_ids)
                    if expanded_full:
                        total_before = sum(len(f["chunks"]) for f in expanded_files.values())
                        total_after = sum(len(f["chunks"]) for f in expanded_full.values())
                        if total_after >= total_before:
                            expanded_files = expanded_full
                            logger.info(f"🔍 Workflow: Expanded to full files: {total_before} -> {total_after} chunks")
                            print(f"🔍 Workflow: Expanded to full files: {total_before} -> {total_after} chunks")
                if expanded_files:
                    logger.info(f"🔍 Workflow: Running RLM pipeline for {len(expanded_files)} files")
                    print("🔍 Workflow: Running RLM pipeline (recursive summarization + final answer)")
                    file_summaries = await recursive_summarize_files(
                        expanded_files, goal, rlm_enabled=True
                    )
                    if file_summaries:
                        answer_result = await generate_final_answer(
                            file_summaries,
                            goal,
                            citation_policy=getattr(cfg, "rlm_citation_policy", "best_effort"),
                            max_files=getattr(cfg, "rlm_max_files", 10),
                            max_chunks=getattr(cfg, "rlm_max_chunks", 50),
                        )
                        final_answer = answer_result.answer_text if hasattr(answer_result, "answer_text") else str(answer_result)
                        expanded_context = final_answer
                        logger.info("🔍 Workflow: RLM pipeline produced final answer")
                    else:
                        logger.warning("🔍 Workflow: RLM pipeline returned no summaries; using search context")
                else:
                    logger.warning("🔍 Workflow: No expanded files from search results; using search context")
            except Exception as e:
                logger.warning("🔍 Workflow: RLM pipeline failed, falling back to search context: %s", e, exc_info=True)

        # If expanded_context is still None, use context or empty string
        if expanded_context is None:
            expanded_context = context if context is not None else ""

        # Assistant step - use Azure OpenAI for LLM response (even if RLM is disabled)
        if not final_answer:
            logger.info(f"🔍 Workflow: Assistant step (Azure OpenAI) - context length: {len(expanded_context) if expanded_context else 0}")
            print(f"🔍 Workflow: Assistant step (Azure OpenAI) - context length: {len(expanded_context) if expanded_context else 0}")
            logger.info(f"🔍 Workflow: Assistant step - context string:\n{expanded_context}")
            print(f"🔍 Workflow: Assistant step - context string:\n{expanded_context}")
            try:
                from openai import AsyncAzureOpenAI
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
                api_key = os.getenv("AZURE_OPENAI_API_KEY")
                api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
                model_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "o3-mini")
                if not (azure_endpoint and api_key):
                    logger.warning("⚠️  Assistant step: Azure OpenAI credentials not configured; cannot generate response.")
                    return "I apologize, but the LLM is not configured. Please contact your administrator."
                llm_client = AsyncAzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=azure_endpoint
                )
                # Compose the prompt as before
                prompt = (
                    f"User Question: {goal}\n\n"
                    f"Retrieved Context:\n{expanded_context if expanded_context and expanded_context.strip() else '(No specific documentation found in knowledge base)'}\n\n"
                    "Instructions:\n"
                    f"- Answer the user's question using {'the context above' if expanded_context and expanded_context.strip() else 'general knowledge'}\n"
                    f"- Be conversational, concise, and accurate\n"
                    "- Cite specific details from the context when available\n"
                    "- If citing context, mention the source"
                )
                # Call Azure OpenAI
                response_obj = await llm_client.chat.completions.create(
                    model=model_deployment,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=1024
                )
                response = response_obj.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Assistant step (Azure OpenAI) failed: {e}", exc_info=True)
                return f"I apologize, but I couldn't generate a response due to an internal error: {e}"

            # Review step (max 3 passes) - only for Assistant-generated responses
            max_iters = 2
            for i in range(1, max_iters + 1):
                print(f"Review pass {i}/{max_iters}...")
                verdict = await reviewer.run(goal, response)
                decision = (verdict.get("decision") or "revise").lower()
                reason = verdict.get("reason", "")
                suggested = verdict.get("suggested_changes", "")
                revised = verdict.get("revised_script", response)

                print(f"Decision: {decision} | Reason: {reason}")

                if decision == "accept":
                    return revised or response

                # Ask assistant to improve using reviewer feedback
                improve_context = (
                    f"Previous response needs improvement:\n{suggested or reason}\n\n"
                    f"Original response:\n{response}\n\n"
                    f"Context available:\n{expanded_context}"
                )
                # Re-call Azure OpenAI for improvement
                try:
                    improve_prompt = (
                        f"User Question: {goal}\n\n"
                        f"Reviewer Feedback: {suggested or reason}\n\n"
                        f"Original Response: {response}\n\n"
                        f"Retrieved Context: {expanded_context}\n\n"
                        "Instructions:\n"
                        "- Revise the response based on the reviewer feedback above.\n"
                        "- Be conversational, concise, and accurate.\n"
                        "- Cite specific details from the context when available.\n"
                        "- If citing context, mention the source."
                    )
                    improve_response_obj = await llm_client.chat.completions.create(
                        model=model_deployment,
                        messages=[{"role": "user", "content": improve_prompt}],
                        max_completion_tokens=1024
                    )
                    response = improve_response_obj.choices[0].message.content.strip()
                except Exception as e:
                    logger.error(f"Assistant improvement step (Azure OpenAI) failed: {e}", exc_info=True)
                    return f"[After {i} review passes] {response}\n\n(Note: Could not further improve due to error: {e})"

            return f"[After {max_iters} review passes]\n{response}"
        else:
            # RLM/Phase 5 generated answer - return directly without review
            logger.info("🔍 Workflow: RLM/Phase 5 answer (skipping review step)")
            print("🔍 Workflow: RLM/Phase 5 answer (skipping review step)")
            return final_answer or response

    if len(targets) > 1:
        results = await asyncio.gather(*[run_one(rid) for rid in targets])
        stitched = []
        for rid, out in zip(targets, results):
            stitched.append(f"### ReporterAgent={rid}\n{out}")
        return "\n\n---\n\n".join(stitched)

    return await run_one(targets[0])
