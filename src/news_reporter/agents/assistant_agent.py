"""Assistant Agent for natural language response generation."""

import logging

from ..foundry_runner import run_foundry_agent

logger = logging.getLogger(__name__)


class AssistantAgent:
    """Generate natural language responses using RAG context"""
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, query: str, context: str) -> str:
        logger.info(f"🤖 [AGENT INVOKED] AssistantAgent (ID: {self._id})")
        print(f"🤖 [AGENT INVOKED] AssistantAgent (ID: {self._id})")
        
        # If no context found, allow LLM to provide helpful general guidance
        context_instruction = "the context above" if context and context.strip() else "general knowledge"
        fallback_permission = "" if context and context.strip() else "\n- If no specific documentation is available, you may provide general best-practice guidance."
        
        prompt = (
            f"User Question: {query}\n\n"
            f"Retrieved Context:\n{context if context and context.strip() else '(No specific documentation found in knowledge base)'}\n\n"
            "Instructions:\n"
            f"- Answer the user's question using {context_instruction}\n"
            f"- Be conversational, concise, and accurate{fallback_permission}\n"
            "- Cite specific details from the context when available\n"
            "- If citing context, mention the source"
        )
        print("AssistantAgent: using Foundry agent:", self._id)  # keep print
        try:
            return run_foundry_agent(self._id, prompt)
        except RuntimeError as e:
            logger.error("AssistantAgent Foundry error: %s", e)
            raise RuntimeError(
                f"Assistant agent failed: {str(e)}. "
                "Please check your Foundry access and agent configuration."
            ) from e


# Backward compatibility alias
NewsReporterAgent = AssistantAgent
