from __future__ import annotations
import asyncio
import logging
from typing import Any, Dict, List

from ..config import Settings
from ..agents.agents import TriageAgent, AiSearchAgent, Neo4jGraphRAGAgent, NewsReporterAgent, ReviewAgent

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

    # Choose search agent based on config
    if cfg.use_neo4j_search and cfg.agent_id_neo4j_search:
        print("Using Neo4j GraphRAG Agent (cost-efficient)")
        search_agent = Neo4jGraphRAGAgent(cfg.agent_id_neo4j_search)
    else:
        print("Using Azure Search Agent (production)")
        search_agent = AiSearchAgent(cfg.agent_id_aisearch)
    
    reviewer = ReviewAgent(cfg.agent_id_reviewer)

    # ---- 2) Actual execution logic ----
    async def run_one(reporter_id: str) -> str:
        reporter = NewsReporterAgent(reporter_id)

        # AI Search step (works with either agent)
        latest = await search_agent.run(goal) if ("ai_search" in tri.intents) else ""

        # Reporter step
        script = (
            await reporter.run(goal, latest or "No ai-search content")
            if ("news_script" in tri.intents)
            else latest
        )
        if not script:
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
