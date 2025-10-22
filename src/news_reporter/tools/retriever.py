# src/news_reporter/tools/retriever.py
from __future__ import annotations
from typing import Dict, Any, List, Optional

try:
    from ..config import Settings
except ImportError:
    import sys, pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root.parent))
    from src.news_reporter.config import Settings

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorQuery  # optional if you use vector mode


def _client(settings: Settings) -> SearchClient:
    if settings.search_endpoint is None or settings.search_api_key is None or settings.search_index is None:
        raise ValueError("AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, AZURE_SEARCH_INDEX are required in .env")
    return SearchClient(
        endpoint=settings.search_endpoint,
        index_name=settings.search_index,
        credential=AzureKeyCredential(settings.search_api_key),
    )


def search(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    k: int = 8,
    mode: str = "hybrid",              # "hybrid" (text BM25) or "vector"
    query_vector: Optional[List[float]] = None,   # if mode=="vector"
) -> List[Dict[str, Any]]:
    """
    Query the Azure AI Search index populated by the indexer.
    Fields expected in the index: id, fileName, blobUri, chunk, vector
    """
    settings = Settings.load()
    client = _client(settings)

    # example OData filter (extend as you like)
    filter_expr = None
    if filters and "fileName" in filters:
        val = str(filters["fileName"]).replace("'", "''")
        filter_expr = f"fileName eq '{val}'"

    if mode == "vector":
        if not query_vector:
            raise ValueError("vector mode requires query_vector")
        vq = VectorQuery(vector=query_vector, k=k, fields="vector")
        results = client.search(
            search_text=None,
            vector_queries=[vq],
            filter=filter_expr,
            select=["id", "chunk", "blobUri", "fileName"],
        )
    else:
        # hybrid (BM25 text over 'chunk')
        results = client.search(
            search_text=query,
            filter=filter_expr,
            top=k,
            select=["id", "chunk", "blobUri", "fileName"],
        )

    out: List[Dict[str, Any]] = []
    for r in results:
        out.append({
            "id": getattr(r, "id", None),
            "fileName": getattr(r, "fileName", None),
            "blobUri": getattr(r, "blobUri", None),
            "text": getattr(r, "chunk", None),
        })
    return out
