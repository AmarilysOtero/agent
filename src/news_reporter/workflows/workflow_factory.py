from __future__ import annotations
import logging
import asyncio
from typing import Any, Dict, List, Callable, Awaitable
from agent_framework import WorkflowBuilder  # local orchestration/structure
from ..config import Settings
from ..agents.agents import TriageAgent, WebSearchAgent, NewsReporterAgent, ReviewAgent

logger = logging.getLogger(__name__)


class FoundryExecutor:
    """
    Minimal adapter that looks like an Executor/AgentProtocol to WorkflowBuilder.
    It exposes __call__ / run / execute so it passes duck-typing checks across
    agent_framework builds. We still run the chain manually for full control.
    """
    def __init__(self, name: str, coro: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]):
        self._name = name
        self._coro = coro

    @property
    def name(self) -> str:
        return self._name

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await self._coro(state)

    # Some AF builds look for .run or .execute — provide both.
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await self._coro(state)

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await self._coro(state)


async def run_sequential_goal(cfg: Settings, goal: str) -> str:
    """
    Foundry-defined agents + local orchestration:
      TRIAGE (Foundry) -> decide flow / multi-route
      WEBSEARCH (Foundry) -> latest
      REPORTER (Foundry)  -> draft
      REVIEWER (Foundry)  -> up to 3 passes (revise/accept)

    We register Executor-shaped wrappers with WorkflowBuilder so the graph is valid,
    but we still perform the actual execution manually (clearer for the review loop).
    """
    # ---- 1) TRIAGE ----
    triage = TriageAgent(cfg.agent_id_triage)
    tri = await triage.run(goal)
    print("Triage:", tri.model_dump())  # keep print

    # Decide whether to fan out across multiple reporter agents
    do_multi = ("multi" in tri.intents) or cfg.multi_route_always
    targets: List[str] = cfg.reporter_ids if do_multi else [cfg.reporter_ids[0]]

    # Shared agents
    web = WebSearchAgent(cfg.agent_id_websearch)
    reviewer = ReviewAgent(cfg.agent_id_reviewer)

    # ---- 2) Build a structural workflow graph (for topology/observability) ----
    # We create lightweight wrappers that "could" be executed by the framework.
    async def _web_step(state: Dict[str, Any]) -> Dict[str, Any]:
        if "web_search" in tri.intents:
            print("WebSearch step: querying goal...")  # keep print
            latest = await web.run(state["goal"])
        else:
            latest = ""
        state["latest"] = latest or "No web content"
        return state

    async def _reporter_step(state: Dict[str, Any]) -> Dict[str, Any]:
        # This wrapper is used only for graph shape; real execution below per-reporter.
        # Returning passthrough keeps the builder happy.
        return state

    async def _review_step(state: Dict[str, Any]) -> Dict[str, Any]:
        # Graph-only placeholder. Real review loop executed below.
        return state

    web_exec = FoundryExecutor("WebSearchAgent", _web_step)
    reporter_exec = FoundryExecutor("NewsReporterAgent", _reporter_step)
    review_exec = FoundryExecutor("ReviewAgent", _review_step)

    builder = WorkflowBuilder("NewsWorkflow (Foundry-backed)")
    builder.set_start_executor(web_exec)              # ✅ instance, not class
    builder.add_edge(web_exec, reporter_exec)         # ✅ instance -> instance
    builder.add_edge(reporter_exec, review_exec)      # ✅ instance -> instance
    _workflow = builder.build()  # We don't call it; we run our loop below.

    # ---- 3) Actual execution logic (preserves your behavior) ----
    async def run_one(reporter_id: str) -> str:
        reporter = NewsReporterAgent(reporter_id)

        # Web step (real)
        latest = await web.run(goal) if ("web_search" in tri.intents) else ""
        # Reporter step (real)
        script = (
            await reporter.run(goal, latest or "No web content")
            if ("news_script" in tri.intents)
            else latest
        )
        if not script:
            return "No action taken."

        # Review step (real) — up to 3 passes
        max_iters = 3
        for i in range(1, max_iters + 1):
            print(f"Review pass {i}/{max_iters}...")  # keep print
            verdict = await reviewer.run(goal, script)
            decision = (verdict.get("decision") or "revise").lower()
            reason = verdict.get("reason", "")
            suggested = verdict.get("suggested_changes", "")
            revised = verdict.get("revised_script", script)

            print(f"Decision: {decision} | Reason: {reason}")  # keep print

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
