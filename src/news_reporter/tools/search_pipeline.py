from __future__ import annotations
import json
import logging
import os
import re
from typing import Dict, Optional
import requests

# Match your pdf_ingestor import pattern
try:
    from ..config import Settings
except ImportError:
    import sys, pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root.parent))
    from src.news_reporter.config import Settings  # type: ignore

__all__ = ["ensure_pipeline", "run_indexer_now", "get_indexer_status"]

API_VERSION = "2024-07-01"

# ----------------------- helpers -----------------------

def _require(v: Optional[str], name: str) -> str:
    if not v or not str(v).strip():
        raise RuntimeError(f"Missing required setting: {name}")
    return str(v).strip()

def _opt(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    v = str(v).strip()
    return v or None

def _safe_name(s: str, prefix: str = "", max_len: int = 128) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\-]", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    if prefix:
        s = f"{prefix}-{s}" if s else prefix
    return s[:max_len].strip("-")

def _fail_verbose(what: str, url: str, payload: Dict, response: requests.Response):
    try:
        body = response.json()
    except Exception:
        body = response.text
    logging.error(
        "[search_pipeline] %s upsert failed\nURL: %s\nStatus: %s\nPayload:\n%s\nResponse:\n%s",
        what, url, f"{response.status_code}",
        json.dumps(payload, indent=2),
        json.dumps(body, indent=2) if isinstance(body, dict) else body,
    )
    raise RuntimeError(f"{what} upsert failed: {response.status_code} {response.text}")

def _build_base_endpoint(cfg) -> str:
    endpoint = _opt(getattr(cfg, "azure_search_endpoint", None)) or _opt(os.getenv("AZURE_SEARCH_ENDPOINT"))
    if endpoint:
        ep = endpoint.rstrip("/")
        if not ep.startswith("https://"):
            ep = "https://" + ep
        return ep
    service = _opt(getattr(cfg, "azure_search_service", None)) or _opt(os.getenv("AZURE_SEARCH_SERVICE"))
    if service:
        return f"https://{service}.search.windows.net"
    raise RuntimeError("Missing AZURE_SEARCH_ENDPOINT or AZURE_SEARCH_SERVICE in configuration.")

def _admin_key(cfg) -> str:
    return _require(
        _opt(getattr(cfg, "azure_search_admin_key", None))
        or _opt(os.getenv("AZURE_SEARCH_ADMIN_KEY"))
        or _opt(getattr(cfg, "azure_search_api_key", None))
        or _opt(os.getenv("AZURE_SEARCH_API_KEY")),
        "AZURE_SEARCH_ADMIN_KEY (or AZURE_SEARCH_API_KEY)"
    )

def _headers(admin_key: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    base = {"Content-Type": "application/json", "api-key": admin_key}
    if extra:
        base.update(extra)
    return base

def _build_blob_connection(cfg) -> Dict[str, str]:
    conn = _opt(getattr(cfg, "azure_blob_conn_str", None)) or _opt(os.getenv("AZURE_BLOB_CONN_STR"))
    if conn:
        return {"connectionString": conn}
    acct = _opt(getattr(cfg, "azure_storage_account", None)) or _opt(os.getenv("AZURE_STORAGE_ACCOUNT"))
    key  = _opt(getattr(cfg, "azure_storage_key", None)) or _opt(os.getenv("AZURE_STORAGE_KEY"))
    if acct and key:
        return {"connectionString": f"DefaultEndpointsProtocol=https;AccountName={acct};AccountKey={key};EndpointSuffix=core.windows.net"}
    raise RuntimeError("Missing AZURE_BLOB_CONN_STR or (AZURE_STORAGE_ACCOUNT + AZURE_STORAGE_KEY).")

def _raw_container(cfg) -> str:
    return _require(_opt(getattr(cfg, "blob_container_raw", None)) or _opt(os.getenv("BLOB_CONTAINER_RAW")), "BLOB_CONTAINER_RAW")

def _index_name(cfg) -> str:
    idx = _opt(getattr(cfg, "azure_search_index", None)) or _opt(os.getenv("AZURE_SEARCH_INDEX"))
    return _safe_name(idx, "") if idx else _safe_name("raw", "idx")

def _index_url(base: str, idx_name: str) -> str:
    return f"{base}/indexes/{idx_name}?api-version={API_VERSION}"

def _ds_url(base: str, ds_name: str) -> str:
    return f"{base}/datasources/{ds_name}?api-version={API_VERSION}"

def _ixr_url(base: str, ixr_name: str) -> str:
    return f"{base}/indexers/{ixr_name}?api-version={API_VERSION}"

# ----------------------- main -----------------------

def ensure_pipeline() -> None:
    """
    Create/Update:
      - Data Source (Blob)
      - Index (key = 'id')
      - Indexer mapping metadata_storage_path -> id (base64Encode)
    """
    cfg = Settings.load()
    base      = _build_base_endpoint(cfg)
    admin_key = _admin_key(cfg)
    container = _raw_container(cfg)

    label    = _safe_name("raw")
    ds_name  = _safe_name(label, "ds")
    idx_name = _index_name(cfg)
    ixr_name = _safe_name(label, "ixr")

    # Data source (upsert OK)
    ds_url = _ds_url(base, ds_name)
    ds_payload = {
        "name": ds_name,
        "type": "azureblob",
        "credentials": _build_blob_connection(cfg),
        "container": {"name": container},
        "description": "Raw blob container for ingestion",
    }
    r = requests.put(ds_url, headers=_headers(admin_key), data=json.dumps(ds_payload))
    if r.status_code >= 300:
        _fail_verbose("Data source", ds_url, ds_payload, r)

    # Index (create-only; 'id' is the key)
    idx_url = _index_url(base, idx_name)
    idx_payload = {
        "name": idx_name,
        "fields": [
            {"name": "id",            "type": "Edm.String",        "key": True,  "searchable": False, "filterable": False, "sortable": False, "facetable": False},
            {"name": "content",       "type": "Edm.String",        "searchable": True,  "filterable": False, "sortable": False, "facetable": False},
            {"name": "file_name",     "type": "Edm.String",        "searchable": False, "filterable": True,  "sortable": True,  "facetable": False},
            {"name": "content_type",  "type": "Edm.String",        "searchable": False, "filterable": True,  "sortable": True,  "facetable": True},
            {"name": "last_modified", "type": "Edm.DateTimeOffset","searchable": False, "filterable": True,  "sortable": True,  "facetable": False},
            {"name": "url",           "type": "Edm.String",        "searchable": False, "filterable": False, "sortable": False, "facetable": False},
        ],
    }
    r = requests.put(idx_url, headers=_headers(admin_key, {"If-None-Match": "*"}), data=json.dumps(idx_payload))
    if r.status_code >= 300 and r.status_code not in (409, 412):
        _fail_verbose("Index", idx_url, idx_payload, r)
    elif r.status_code in (409, 412):
        logging.info("[search_pipeline] Index '%s' already exists; leaving as-is.", idx_name)

    # Indexer: map path -> id (base64), plus basic metadata and content
    ixr_url = _ixr_url(base, ixr_name)
    ixr_payload = {
        "name": ixr_name,
        "dataSourceName": ds_name,
        "targetIndexName": idx_name,
        "parameters": {
            "configuration": {
                "parsingMode": "default",
                "dataToExtract": "contentAndMetadata",
            }
        },
        # IMPORTANT:
        # - fieldMappings: raw blob metadata fields (NO '/document' prefix)
        # - outputFieldMappings: cracked fields (WITH '/document/...'), e.g. content
        "fieldMappings": [
            {
                "sourceFieldName": "metadata_storage_path",   # FIXED: no /document
                "targetFieldName": "id",
                "mappingFunction": {"name": "base64Encode"}
            },
            {"sourceFieldName": "metadata_storage_name",          "targetFieldName": "file_name"},
            {"sourceFieldName": "metadata_storage_content_type",  "targetFieldName": "content_type"},
            {"sourceFieldName": "metadata_storage_last_modified", "targetFieldName": "last_modified"},
            {"sourceFieldName": "metadata_storage_path",          "targetFieldName": "url"},
        ],
        "outputFieldMappings": [
            {"sourceFieldName": "/document/content", "targetFieldName": "content"},
        ],
        "schedule": {
            "interval": "PT5M"   # Run every 5 minutes; change to PT10M or PT15M if you prefer
    }
    }
    r = requests.put(ixr_url, headers=_headers(admin_key), data=json.dumps(ixr_payload))
    if r.status_code >= 300:
        _fail_verbose("Indexer", ixr_url, ixr_payload, r)

    logging.info("[search_pipeline] ✅ Pipeline ready: datasource=%s, index=%s, indexer=%s", ds_name, idx_name, ixr_name)

# ----------------------- ops -----------------------

def run_indexer_now(indexer_name: Optional[str] = None) -> None:
    cfg = Settings.load()
    base = _build_base_endpoint(cfg)
    admin_key = _admin_key(cfg)

    ixr = _safe_name(indexer_name) if indexer_name else _safe_name("raw", "ixr")
    url = f"{base}/indexers/{ixr}/run?api-version={API_VERSION}"
    r = requests.post(url, headers=_headers(admin_key))
    if r.status_code >= 300:
        _fail_verbose("Indexer Run", url, {}, r)

def get_indexer_status(indexer_name: Optional[str] = None) -> Dict:
    cfg = Settings.load()
    base = _build_base_endpoint(cfg)
    admin_key = _admin_key(cfg)

    ixr = _safe_name(indexer_name) if indexer_name else _safe_name("raw", "ixr")
    url = f"{base}/indexers/{ixr}/status?api-version={API_VERSION}"
    r = requests.get(url, headers=_headers(admin_key))
    if r.status_code >= 300:
        _fail_verbose("Indexer Status", url, {}, r)
    return r.json()
