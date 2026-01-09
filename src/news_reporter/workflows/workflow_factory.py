from __future__ import annotations
import asyncio
import logging
from typing import Any, Dict, List

from ..config import Settings
from ..agents.agents import TriageAgent, AiSearchAgent, Neo4jGraphRAGAgent, NewsReporterAgent, ReviewAgent, SQLAgent

logger = logging.getLogger(__name__)

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
