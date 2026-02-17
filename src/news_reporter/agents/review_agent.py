"""Review Agent for response validation."""

import logging
from typing import Dict, Any

from ..foundry_runner import run_foundry_agent_json

logger = logging.getLogger(__name__)


class ReviewAgent:
    """Review assistant responses for accuracy and completeness"""
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, query: str, candidate_response: str) -> dict:
        """
        Review assistant response and decide if it needs improvement.
        Returns a dict with keys: decision, reason, suggested_changes, revised_script.
        """
        prompt = (
            f"User Query: {query}\n\n"
            f"Assistant Response:\n{candidate_response}\n\n"
            "Review the response for:\n"
            "1. Accuracy - Does it correctly answer the question?\n"
            "2. Completeness - Is the answer sufficient?\n"
            "3. Clarity - Is it easy to understand?\n\n"
            "Return ONLY STRICT JSON (no markdown, no prose) with keys:\n"
            '"decision": "accept" or "revise"\n'
            '"reason": brief explanation\n'
            '"suggested_changes": what to improve (empty if accept)\n'
            '"revised_script": improved version (empty if accept)'
        )
        logger.info(f"🤖 [AGENT INVOKED] ReviewAgent (ID: {self._id})")
        print(f"🤖 [AGENT INVOKED] ReviewAgent (ID: {self._id})")
        print("ReviewAgent: using Foundry agent:", self._id)  # keep print
        try:
            data = run_foundry_agent_json(
                self._id,
                prompt,
                system_hint="You are a reviewer that returns STRICT JSON only."
            )
        except RuntimeError as e:
            logger.error("ReviewAgent Foundry error: %s", e)
            raise RuntimeError(
                f"Review agent failed: {str(e)}. "
                "Please check your Foundry access and agent configuration."
            ) from e
        
        try:
            if not isinstance(data, dict) or "decision" not in data:
                raise ValueError("Invalid JSON shape from reviewer")
            decision = (data.get("decision") or "revise").lower()
            decision = decision if decision in {"accept", "revise"} else "revise"
            # On accept, leave revised_script empty so callers use the original response (agent may echo instructions).
            revised_script = "" if decision == "accept" else data.get("revised_script", candidate_response)
            return {
                "decision": decision,
                "reason": data.get("reason", ""),
                "suggested_changes": data.get("suggested_changes", ""),
                "revised_script": revised_script or candidate_response,
            }
        except Exception as e:
            logger.error("Review parse error: %s", e)
            # Fail-safe: accept last response to avoid infinite loops
            return {
                "decision": "accept",
                "reason": "parse_error",
                "suggested_changes": "",
                "revised_script": candidate_response,
            }
