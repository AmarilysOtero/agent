# src/news_reporter/tools/azure_search.py
from __future__ import annotations
from typing import List, Dict, Any, Optional

# --- Package-safe settings import ---
try:
    from ..config import Settings
except ImportError:
    import sys, pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root.parent))
    from src.news_reporter.config import Settings

# Try to import Azure Search - if it fails, the module can still be loaded
# but functions will raise ImportError when called
# Catch both ImportError and TypeError (TypeError can occur with version incompatibilities)
try:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        SearchIndex,
        SearchField,
        SimpleField,
        SearchableField,
        SearchFieldDataType,
        VectorSearch,
        VectorSearchProfile,
        HnswAlgorithmConfiguration,
        HnswParameters,
        VectorSearchAlgorithmKind,
        SemanticConfiguration,
        SemanticPrioritizedFields,
        SemanticField,
    )
    _AZURE_SEARCH_AVAILABLE = True
except (ImportError, TypeError) as e:
    _AZURE_SEARCH_AVAILABLE = False
    _AZURE_SEARCH_ERROR = str(e)
    # Create dummy classes to prevent NameError
    AzureKeyCredential = None
    SearchClient = None
    SearchIndexClient = None

# Load settings once
settings = Settings.load()

# --- Local validators to satisfy both runtime and Pylance types ---
def _require_str(val: Optional[str], name: str) -> str:
    if not val:
        raise RuntimeError(f"Missing required setting: {name}")
    return val

# Only initialize Azure Search constants if available
if _AZURE_SEARCH_AVAILABLE:
    ENDPOINT: str = _require_str(settings.azure_search_endpoint, "AZURE_SEARCH_ENDPOINT")
    API_KEY: str = _require_str(settings.azure_search_api_key, "AZURE_SEARCH_API_KEY")
    INDEX_NAME: str = _require_str(settings.azure_search_index, "AZURE_SEARCH_INDEX")
    VECTOR_DIM: int = settings.embedding_vector_dim  # e.g., 3072 for text-embedding-3-large
else:
    ENDPOINT = ""
    API_KEY = ""
    INDEX_NAME = ""
    VECTOR_DIM = 0


def _index_client() -> SearchIndexClient:
    if not _AZURE_SEARCH_AVAILABLE:
        raise ImportError(f"Azure Search is not available: {_AZURE_SEARCH_ERROR}")
    return SearchIndexClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(API_KEY),
    )


def _search_client() -> SearchClient:
    if not _AZURE_SEARCH_AVAILABLE:
        raise ImportError(f"Azure Search is not available: {_AZURE_SEARCH_ERROR}")
    return SearchClient(
        endpoint=ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(API_KEY),
    )


def ensure_index() -> None:
    """
    Create a vector-capable index if it doesn't already exist.

    Fields:
      - id (key)
      - text (searchable preview)
      - doc_id (filterable)
      - page (filterable)
      - blob_uri (filterable)
      - timestamp (filterable)
      - vector (Collection(Single), with HNSW profile)

    Notes:
      * This definition omits 'vectorizers' for broad SDK compatibility.
      * We pass vectors at query time.
    """
    sic = _index_client()
    try:
        sic.get_index(INDEX_NAME)
        return  # already exists
    except Exception:
        pass  # create below

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(
            name="text",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=False,
            sortable=False,
            facetable=False,
        ),
        SimpleField(name="doc_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True, facetable=True),
        SimpleField(name="blob_uri", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="timestamp", type=SearchFieldDataType.String, filterable=True),
        SearchField(
            name="vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            filterable=False,
            facetable=False,
            sortable=False,
            vector_search_dimensions=VECTOR_DIM,
            vector_search_profile_name="vprof",
        ),
    ]

    # Minimal, version-friendly vector search setup (no vectorizers)
    vector_search = VectorSearch(
        profiles=[
            VectorSearchProfile(
                name="vprof",
                algorithm_configuration_name="hnsw",
            )
        ],
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw",
                kind=VectorSearchAlgorithmKind.HNSW,
                parameters=HnswParameters(),  # defaults; tune later if needed
            )
        ],
    )

    # Optional semantic config (kept for hybrid/semantic ranking)
    semantic = [
        SemanticConfiguration(
            name="semconf",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="doc_id"),
                content_fields=[SemanticField(field_name="text")],
            ),
        )
    ]

    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_configurations=semantic,
    )
    sic.create_index(index)


def upsert_documents(docs: List[Dict[str, Any]]) -> None:
    """
    Upsert documents into the index.

    Each document should include:
      - id: str
      - text: str                # short preview / chunk text
      - vector: List[float]      # embedding with dim == VECTOR_DIM
      - doc_id: str
      - page: int
      - blob_uri: str
      - timestamp: str (ISO)
    """
    if not docs:
        return
    sc = _search_client()
    sc.upload_documents(docs)  # mergeOrUpload semantics on recent SDKs


def vector_search(
    query_vector: List[float],
    top_k: int = 8,
    filter_expr: Optional[str] = None,
    select: Optional[List[str]] = None,
):
    """
    Pure vector search with optional OData filter.

    Examples:
      filter_expr="doc_id eq 'abcd-1234'"
      filter_expr="doc_id in ('a','b') and page ge 2"
    """
    sc = _search_client()
    if select is None:
        select = ["id", "doc_id", "page", "text", "blob_uri", "timestamp"]

    results = sc.search(
        search_text="",  # vector-only
        vector={"value": query_vector, "fields": "vector", "k": top_k},
        filter=filter_expr,
        select=select,
    )
    return results


def hybrid_search(
    search_text: str,
    query_vector: Optional[List[float]] = None,
    top_k: int = 8,
    filter_expr: Optional[str] = None,
    select: Optional[List[str]] = None,
    semantic: bool = True,
):
    """
    Hybrid keyword + optional vector search.
    Set semantic=False if your service doesn't enable semantic search.
    """
    if not _AZURE_SEARCH_AVAILABLE:
        raise ImportError(f"Azure Search is not available: {_AZURE_SEARCH_ERROR}")
    sc = _search_client()
    if select is None:
        select = ["id", "doc_id", "page", "text", "blob_uri", "timestamp"]

    kwargs: Dict[str, Any] = dict(
        search_text=search_text or "*",
        filter=filter_expr,
        select=select,
        top=top_k,
    )
    if query_vector is not None:
        kwargs["vector"] = {"value": query_vector, "fields": "vector", "k": top_k}
    if semantic:
        kwargs["query_type"] = "semantic"
        kwargs["semantic_configuration_name"] = "semconf"

    return sc.search(**kwargs)


def delete_index() -> None:
    """Dev utility: drop the index (for local resets)."""
    sic = _index_client()
    try:
        sic.delete_index(INDEX_NAME)
    except Exception:
        pass
