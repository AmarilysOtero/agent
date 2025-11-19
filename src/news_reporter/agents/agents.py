from __future__ import annotations
import json
import logging
from pydantic import BaseModel, Field, ValidationError
from ..foundry_runner import run_foundry_agent, run_foundry_agent_json

logger = logging.getLogger(__name__)

# Lazy import for Azure Search to avoid import errors when using Neo4j only
def _get_hybrid_search():
    try:
        from ..tools.azure_search import hybrid_search
        return hybrid_search
    except ImportError as e:
        logger.warning(f"Azure Search not available: {e}")
        return None

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
        try:
            raw = run_foundry_agent(self._id, content).strip()
        except RuntimeError as e:
            logger.error("TriageAgent Foundry error: %s", e)
            raise RuntimeError(
                f"Triage agent failed: {str(e)}. "
                "Please check your Foundry access and agent configuration."
            ) from e
        print("Triage raw:", raw)  # keep print
        try:
            data = json.loads(raw)
            return IntentResult(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error("Triage parse error: %s", e)
            return IntentResult(intents=["unknown"], confidence=0.0, rationale="parse_error")

# ---------- AI SEARCH (Foundry) ----------

class AiSearchAgent:
    """Search agent using Azure AI Search"""
    
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, query: str) -> str:
        print("AiSearchAgent: using Foundry agent:", self._id)  # keep print
        hybrid_search = _get_hybrid_search()
        if hybrid_search is None:
            return "Azure Search is not available. Please configure Azure Search or use Neo4j search instead."
        
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


# ---------- NEO4J GRAPHRAG SEARCH (Foundry) ----------

class Neo4jGraphRAGAgent:
    """Search agent using Neo4j GraphRAG retrieval"""
    
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, query: str) -> str:
        print("Neo4jGraphRAGAgent: using Foundry agent:", self._id)  # keep print
        from ..tools.neo4j_graphrag import graphrag_search
        
        results = graphrag_search(
            query=query,
            top_k=8,
            similarity_threshold=0.7
        )

        if not results:
            return "No results found in Neo4j GraphRAG."

        findings = []
        for res in results:
            text = res.get("text", "").replace("\n", " ")
            file_name = res.get("file_name", "Unknown")
            directory = res.get("directory_name", "")
            file_path = res.get("file_path", "")
            
            # Include comprehensive metadata for agent context
            metadata_parts = []
            if "hybrid_score" in res and res["hybrid_score"] is not None:
                metadata_parts.append(f"score:{res['hybrid_score']:.2f}")
            if "similarity" in res and res["similarity"] is not None:
                metadata_parts.append(f"similarity:{res['similarity']:.2f}")
            if "metadata" in res and res["metadata"]:
                meta = res["metadata"]
                if meta.get("hop_count", 0) > 0:
                    metadata_parts.append(f"hops:{meta['hop_count']}")
                if meta.get("vector_score") is not None:
                    metadata_parts.append(f"vector:{meta['vector_score']:.3f}")
                if meta.get("keyword_score") is not None:
                    metadata_parts.append(f"keyword:{meta['keyword_score']:.3f}")
                if meta.get("path_score") is not None:
                    metadata_parts.append(f"path:{meta['path_score']:.3f}")
                if meta.get("chunk_index") is not None:
                    metadata_parts.append(f"chunk_idx:{meta['chunk_index']}")
                if meta.get("chunk_size") is not None:
                    metadata_parts.append(f"size:{meta['chunk_size']}")
                if meta.get("file_id"):
                    metadata_parts.append(f"file_id:{meta['file_id']}")
            
            metadata_str = f" [{', '.join(metadata_parts)}]" if metadata_parts else ""
            
            # Format source info with file path (preferred) or directory/name
            if file_path:
                source_info = file_path
            elif directory:
                source_info = f"{directory}/{file_name}"
            else:
                source_info = file_name
            
            findings.append(f"- {source_info}: {text[:300]}...{metadata_str}")

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
        try:
            return run_foundry_agent(self._id, content)
        except RuntimeError as e:
            logger.error("NewsReporterAgent Foundry error: %s", e)
            raise RuntimeError(
                f"Reporter agent failed: {str(e)}. "
                "Please check your Foundry access and agent configuration."
            ) from e

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
