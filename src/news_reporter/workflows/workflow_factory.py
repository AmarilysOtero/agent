from __future__ import annotations
import logging
import asyncio
from typing import List
from agent_framework.azure import AzureOpenAIChatClient
from ..config import Settings
from ..agents.agents import TriageAgent, WebSearchAgent, NewsReporterAgent, ReviewAgent  # ⟵ added ReviewAgent

logger = logging.getLogger(__name__)

def build_chat_client(cfg: Settings) -> AzureOpenAIChatClient:
    """Creates an AzureOpenAIChatClient using endpoint and API key."""
    client = AzureOpenAIChatClient(
        endpoint=cfg.endpoint,
        api_key=cfg.api_key,
        api_version=cfg.api_version,
        deployment_name=cfg.default_deployment,
    )
    return client

async def run_sequential_goal(cfg: Settings, goal: str) -> str:
    """
    Executes: (optional) WebSearchAgent -> NewsReporterAgent -> ReviewAgent (up to 3 passes)
    We keep the simple sequential orchestration you’re using now (no WorkflowBuilder wrapper),
    and add a reviewer loop that can send the script back for revision up to 3 times.
    """
    client = build_chat_client(cfg)

    # 1) Classify intent with the triage agent
    triage = TriageAgent(client, cfg.default_deployment)
    tri = await triage.run(goal)
    print("Triage:", tri.model_dump())  # keep print

    # Decide whether to fan out
    do_multi = ("multi" in tri.intents) or cfg.multi_route_always
    targets: List[str] = tri.targets or cfg.routing_deployments

    async def run_one(deployment: str) -> str:
        web = WebSearchAgent(client, deployment)
        news = NewsReporterAgent(client, deployment)
        review = ReviewAgent(client, deployment)  # ⟵ new reviewer

        # Step A: optional web step
        latest = await web.run(goal) if ("web_search" in tri.intents) else ""
        # Step B: produce initial draft
        script = await news.run(goal, latest or "No web content") if ("news_script" in tri.intents) else latest
        if not script:
            return "No action taken."

        # Step C: up to 3 review passes
        max_iters = 3
        for i in range(1, max_iters + 1):
            print(f"Review pass {i}/{max_iters}...")  # keep print
            verdict = await review.run(goal, script)
            decision = (verdict.get("decision") or "revise").lower()
            reason = verdict.get("reason", "")
            suggested = verdict.get("suggested_changes", "")
            revised = verdict.get("revised_script", script)

            print(f"Decision: {decision} | Reason: {reason}")  # keep print

            if decision == "accept":
                return revised or script

            # If revise requested, feed reviewer guidance back to the reporter
            improve_context = (
                f"Apply these review notes strictly:\n{suggested or reason}\n\n"
                f"Original draft:\n{script}"
            )
            script = await news.run(goal, improve_context)

        # If all review passes used, return the last script with a note
        return f"[After {max_iters} review passes]\n{script}"

    if do_multi:
        results = await asyncio.gather(*[run_one(dep) for dep in targets])
        stitched = []
        for dep, out in zip(targets, results):
            stitched.append(f"### Model={dep}\n{out}")
        return "\n\n---\n\n".join(stitched)

    return await run_one(cfg.default_deployment)
