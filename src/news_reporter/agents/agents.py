from __future__ import annotations
import json
import logging
from pydantic import BaseModel, Field, ValidationError
from typing import List
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework import ChatMessage

logger = logging.getLogger(__name__)

class IntentResult(BaseModel):
    intents: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""
    targets: List[str] = Field(default_factory=list)

TRIAGE_SYS = (
    "You classify the user's goal for a news agent system. "
    "Return STRICT JSON: {\"intents\":[\"web_search\"|\"news_script\"|\"plan\"|\"multi\"|\"unknown\"...], "
    "\"confidence\":0..1, \"rationale\":string, \"targets\":[string...]}. No prose."
)

class TriageAgent:
    def __init__(self, client: AzureOpenAIChatClient, deployment: str):
        self._client = client
        self._deployment = deployment

    async def run(self, goal: str) -> IntentResult:
        messages = [
            ChatMessage(role="system", text=TRIAGE_SYS),
            ChatMessage(role="user", text=f"Classify and return JSON only. User goal: {goal}")
        ]
        print("TriageAgent: sending messages...")  # keep print
        resp = await self._client.get_response(messages=messages, deployment_name=self._deployment)
        text = resp.messages[0].text.strip()
        print("Triage raw:", text)  # keep print
        try:
            data = json.loads(text)
            return IntentResult(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error("Triage parse error: %s", e)
            return IntentResult(intents=["unknown"], confidence=0.0, rationale="parse_error")

NEWS_SYS = (
    "You are a news anchor named John at MSinghTV. "
    "Given a topic and latest info, write a short, factual news script with attribution hints."
)

class WebSearchAgent:
    def __init__(self, client: AzureOpenAIChatClient, deployment: str):
        self._client = client
        self._deployment = deployment

    async def run(self, query: str) -> str:
        messages = [
            ChatMessage(role="system", text="You retrieve fresh, verifiable facts and suggest 3 credible sources with dates."),
            ChatMessage(role="user", text=f"Find the latest information (with dates) about: {query}. Return a concise bullet list.")
        ]
        print("WebSearchAgent: querying:", query)  # keep print
        resp = await self._client.get_response(messages=messages, deployment_name=self._deployment)
        return resp.messages[0].text

class NewsReporterAgent:
    def __init__(self, client: AzureOpenAIChatClient, deployment: str):
        self._client = client
        self._deployment = deployment

    async def run(self, topic: str, latest_news: str) -> str:
        messages = [
            ChatMessage(role="system", text=NEWS_SYS),
            ChatMessage(role="user", text=f"Topic: {topic}\nLatest info:\n{latest_news}\nWrite a 60-90s script for broadcast.")
        ]
        print("NewsReporterAgent: generating script...")  # keep print
        resp = await self._client.get_response(messages=messages, deployment_name=self._deployment)
        return resp.messages[0].text

# ---------------- NEW: Reviewer agent ----------------

REVIEW_SYS = (
    "You are a critical news editor reviewing a broadcast script. "
    "You must be strict about: factuality, source clarity, tone (objective), "
    "time references (use explicit dates), and length (60-90s). "
    "Return STRICT JSON with keys: "
    "{\"decision\":\"accept\"|\"revise\",\"reason\":string,"
    "\"suggested_changes\":string,\"revised_script\":string}. "
    "No prose outside JSON."
)

class ReviewAgent:
    def __init__(self, client: AzureOpenAIChatClient, deployment: str):
        self._client = client
        self._deployment = deployment

    async def run(self, topic: str, candidate_script: str) -> dict:
        """Return a dict with decision, reason, suggested_changes, revised_script."""
        prompt = (
            f"Topic: {topic}\n\n"
            f"Candidate script:\n{candidate_script}\n\n"
            "If the script is not satisfactory, set decision to 'revise' and provide a revised_script "
            "that applies your suggested_changes. If the script is acceptable, set decision to 'accept' "
            "and you may keep revised_script identical or with minimal edits for clarity."
        )
        messages = [
            ChatMessage(role="system", text=REVIEW_SYS),
            ChatMessage(role="user", text=prompt),
        ]
        print("ReviewAgent: reviewing script...")  # keep print
        resp = await self._client.get_response(messages=messages, deployment_name=self._deployment)
        raw = resp.messages[0].text.strip()
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
            # If reviewer fails, fall back to accept to avoid infinite loops
            return {
                "decision": "accept",
                "reason": "parse_error",
                "suggested_changes": "",
                "revised_script": candidate_script,
            }
