# src/news_reporter/tools/embeddings.py
from __future__ import annotations
from typing import List

# --- Robust imports: prefer package-relative, fallback to absolute ---
try:
    from ..config import Settings
    from .util import normalize_ai_project_endpoint
except ImportError:
    import sys, pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root.parent))
    from src.news_reporter.config import Settings
    from src.news_reporter.tools.util import normalize_ai_project_endpoint

settings = Settings.load()


class EmbeddingsProvider:
    """
    Embeddings via Azure AI Project (Foundry) using your split env vars:

      AZURE_AI_PROJECT_ENDPOINT  (may include /api/projects/<ProjectName>)
      AZURE_AI_SUBSCRIPTION_ID
      AZURE_AI_RESOURCE_GROUP
      AZURE_AI_ACCOUNT_NAME
      AZURE_AI_PROJECT_NAME
      AZURE_AI_EMBEDDING_DEPLOYMENT_NAME  (e.g., text-embedding-3-large)
    """

    def __init__(self) -> None:
        from azure.identity import DefaultAzureCredential
        from azure.ai.projects import AIProjectClient
        import os
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            "[EmbeddingsProvider] Env check: OPENAI_API_VERSION=%s, AZURE_OPENAI_API_VERSION=%s",
            os.getenv("OPENAI_API_VERSION"),
            os.getenv("AZURE_OPENAI_API_VERSION"),
        )

        # ✅ explicit runtime + type guard to satisfy Pylance
        if settings.ai_project_endpoint is None:
            raise ValueError("AZURE_AI_PROJECT_ENDPOINT is required")

        base_endpoint, project_name = normalize_ai_project_endpoint(
            settings.ai_project_endpoint,
            fallback_project=settings.ai_project_name,
        )

        # Build AIProjectClient
        client = AIProjectClient(
            endpoint=base_endpoint,
            subscription_id=settings.ai_subscription_id,
            resource_group_name=settings.ai_resource_group,
            account_name=settings.ai_account_name,
            credential=DefaultAzureCredential(),
        )

        # Use get_openai_client() instead of .openai
        self._openai = client.get_openai_client()
        self._emb_model = settings.ai_embedding_deployment

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for a list of texts."""
        if not texts:
            return []
        resp = self._openai.embeddings.create(model=self._emb_model, input=texts)
        return [d.embedding for d in resp.data]
