from __future__ import annotations
import json
import logging
from pydantic import BaseModel, Field, ValidationError
from ..foundry_runner import run_foundry_agent, run_foundry_agent_json
from ..tools.azure_search import hybrid_search

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

# ---------- AI SEARCH (Foundry) ----------

class AiSearchAgent:
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, query: str) -> str:
        print("AiSearchAgent: using Foundry agent:", self._id)  # keep print
        results = hybrid_search(
            search_text=query,
            top_k=8,
            select=["file_name", "content", "url", "last_modified"],
            semantic=False
        )

        if not results:
            return "No results found in Azure AI Search."

        findings = []
        for res in results:
            content = (res.get("content") or "").replace("\n", " ")
            findings.append(f"- {res.get('file_name')}: {content[:300]}...")

        # print("AiSearchAgent list of sources/content\n\n" + "\n".join(findings))
        return "\n".join(findings)


# ---------- REPORTER (Foundry) ----------

class NewsReporterAgent:
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, topic: str, latest_news: str) -> str:
        content = (
            f"Topic: {topic}\n"
            f"Latest info:\n{latest_news}\n"
            # "Write a 60-90s news broadcast script."
            "Write a description about the information in the tone of a news reporter." 
        )
        print("NewsReporterAgent: using Foundry agent:", self._id)  # keep print
        return run_foundry_agent(self._id, content)

# ---------- REVIEWER (Foundry, strict JSON) ----------

class ReviewAgent:
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, topic: str, candidate_script: str) -> dict:
        """
        Foundry system prompt already defines the JSON schema. We still remind at user layer.
        Returns a dict with keys: decision, reason, suggested_changes, revised_script.
        Return ONLY STRICT JSON (no markdown, no prose) as per your schema.
        """
        prompt = (
            f"Topic: {topic}\n\n"
            f"Candidate script:\n{candidate_script}\n\n"
            # "Evaluate factual accuracy, clarity, neutral tone, explicit dates, and 60-90s length. "
            "Evaluate factual accuracy, relevance, and tone of a news reporter. " 
            "Return ONLY STRICT JSON (no markdown, no prose) as per your schema."
        )
        print("ReviewAgent: using Foundry agent:", self._id)  # keep print
        try:
            data = run_foundry_agent_json(
                self._id,
                prompt,
                system_hint="You are a reviewer that returns STRICT JSON only."
            )
            if not isinstance(data, dict) or "decision" not in data:
                raise ValueError("Invalid JSON shape from reviewer")
            decision = (data.get("decision") or "revise").lower()
            return {
                "decision": decision if decision in {"accept", "revise"} else "revise",
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
