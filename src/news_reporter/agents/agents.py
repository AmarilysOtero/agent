from __future__ import annotations
import json
import logging
from pydantic import BaseModel, Field, ValidationError
from ..foundry_runner import run_foundry_agent

logger = logging.getLogger(__name__)

# ---------- TRIAGE (Foundry) ----------

class IntentResult(BaseModel):
    intents: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""
    targets: list[str] = Field(default_factory=list)

class TriageAgent:
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, goal: str) -> IntentResult:
        content = f"Classify and return JSON only. User goal: {goal}"
        print("TriageAgent: using Foundry agent:", self._id)  # keep print
        raw = run_foundry_agent(self._id, content).strip()
        print("Triage raw:", raw)  # keep print
        try:
            data = json.loads(raw)
            return IntentResult(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error("Triage parse error: %s", e)
            return IntentResult(intents=["unknown"], confidence=0.0, rationale="parse_error")

# ---------- WEB SEARCH (Foundry) ----------

class WebSearchAgent:
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, query: str) -> str:
        print("WebSearchAgent: using Foundry agent:", self._id)  # keep print
        # Your Foundry WebSearch agent should already be wired with Bing Grounding/tools.
        return run_foundry_agent(self._id, query)

# ---------- REPORTER (Foundry) ----------

class NewsReporterAgent:
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, topic: str, latest_news: str) -> str:
        content = (
            f"Topic: {topic}\n"
            f"Latest info:\n{latest_news}\n"
            "Write a 60-90s news broadcast script."
        )
        print("NewsReporterAgent: using Foundry agent:", self._id)  # keep print
        return run_foundry_agent(self._id, content)

# ---------- REVIEWER (Foundry) ----------

class ReviewAgent:
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, topic: str, candidate_script: str) -> dict:
        """
        Reviewer must return STRICT JSON:
        {"decision":"accept"|"revise","reason":string,
         "suggested_changes":string,"revised_script":string}
        """
        prompt = (
            f"Topic: {topic}\n\n"
            f"Candidate script:\n{candidate_script}\n\n"
            "Return STRICT JSON with keys decision (accept|revise), reason, "
            "suggested_changes, revised_script. Be strict about factuality, "
            "source clarity, neutral tone, explicit dates, 60-90s length."
        )
        print("ReviewAgent: using Foundry agent:", self._id)  # keep print
        raw = run_foundry_agent(self._id, prompt).strip()
        print("Review raw:", raw)  # keep print
        try:
            data = json.loads(raw)
            if not isinstance(data, dict) or "decision" not in data:
                raise ValueError("Invalid review JSON")
            return {
                "decision": (data.get("decision") or "revise"),
                "reason": data.get("reason", ""),
                "suggested_changes": data.get("suggested_changes", ""),
                "revised_script": data.get("revised_script", candidate_script),
            }
        except Exception as e:
            logger.error("Review parse error: %s", e)
            # Fail-safe: accept last script to avoid infinite loops
            return {
                "decision": "accept",
                "reason": "parse_error",
                "suggested_changes": "",
                "revised_script": candidate_script,
            }
