from __future__ import annotations
import logging
import asyncio
from typing import List
from agent_framework.azure import AzureOpenAIChatClient
from ..config import Settings
from ..agents.agents import TriageAgent, WebSearchAgent, NewsReporterAgent

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
    Executes: (optional) WebSearchAgent -> NewsReporterAgent
    NOTE: We intentionally avoid WorkflowBuilder here because it requires
    an Executor/AgentProtocol instance, not a bare method. Since we only
    need a simple 2-step chain, direct async calls are clearer and avoid
    wrapping complexity.
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

        latest = await web.run(goal) if ("web_search" in tri.intents) else ""
        script = await news.run(goal, latest or "No web content") if ("news_script" in tri.intents) else latest
        return script or latest or "No action taken."

    if do_multi:
        results = await asyncio.gather(*[run_one(dep) for dep in targets])
        stitched = []
        for dep, out in zip(targets, results):
            stitched.append(f"### Model={dep}\n{out}")
        return "\n\n---\n\n".join(stitched)

    return await run_one(cfg.default_deployment)
