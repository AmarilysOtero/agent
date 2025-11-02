# src/news_reporter/tools/util.py
"""
Utility helpers for hashing, connection string parsing, and normalizing
Azure AI Project endpoints.

Used by: embeddings.py, azure_search.py, and ingestion pipelines.
"""

from __future__ import annotations
import hashlib
from urllib.parse import urlparse


def hash_text(text: str) -> str:
    """
    Return a SHA-256 hash of the given text.
    Used for deduplication and stable chunk IDs.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_connection_string(conn: str) -> dict[str, str]:
    """
    Parse a semicolon-separated connection string into a dictionary.

    Example:
        "endpoint=https://...;project=MyProj;subscription_id=1234"
        → {"endpoint": "...", "project": "MyProj", "subscription_id": "1234"}
    """
    if not conn:
        raise ValueError("Connection string cannot be empty")

    parts = dict(p.split("=", 1) for p in conn.split(";") if "=" in p)
    # Normalize trailing slash for endpoint, if present
    if "endpoint" in parts:
        parts["endpoint"] = parts["endpoint"].rstrip("/")
    return parts


def normalize_ai_project_endpoint(
    endpoint: str,
    fallback_project: str | None = None
) -> tuple[str, str]:
    """
    Normalize a Foundry / Azure AI Project endpoint string.

    Accepts either:
      https://...services.ai.azure.com
      https://...services.ai.azure.com/api/projects/<ProjectName>

    Returns:
        (base_endpoint, project_name)

    Raises:
        ValueError if neither a project in the path nor a fallback is provided.
    """
    if not endpoint:
        raise ValueError("AZURE_AI_PROJECT_ENDPOINT is required")

    e = endpoint.rstrip("/")
    if "/api/projects/" in e:
        base = e.split("/api/projects/")[0]
        project = e.split("/api/projects/")[1]
        return base, project

    if not fallback_project:
        raise ValueError(
            "AZURE_AI_PROJECT_ENDPOINT has no '/api/projects/<name>'; "
            "set AZURE_AI_PROJECT_NAME as fallback."
        )
    return e, fallback_project
