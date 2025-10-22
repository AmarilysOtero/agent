from __future__ import annotations
import json
import logging
import os
import re
from typing import Dict, Optional

import requests

API_VERSION = "2024-07-01"
log = logging.getLogger("search_pipeline")


# ---------- helpers ----------
def _opt(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    v = str(v).strip()
    return v or None


def _require(v: Optional[str], name: str) -> str:
    v = _opt(v)
    if not v:
        raise RuntimeError(f"Missing required setting: {name}")
    return v


def _headers(key: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    h = {"Content-Type": "application/json", "api-key": key}
    if extra:
        h.update(extra)
    return h


def _safe_name(s: str, prefix: str = "", max_len: int = 128) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\-]", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    if prefix:
        s = f"{prefix}-{s}" if s else prefix
    return s[:max_len].strip("-")


def _search_base(cfg) -> str:
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
        or _opt(getattr(cfg, "azure_search_api_key", None))
        or _opt(os.getenv("AZURE_SEARCH_ADMIN_KEY"))
        or _opt(os.getenv("AZURE_SEARCH_API_KEY")),
        "AZURE_SEARCH_ADMIN_KEY / AZURE_SEARCH_API_KEY",
    )


def _blob_connection(cfg) -> Dict[str, str]:
    conn = _opt(getattr(cfg, "azure_blob_conn_str", None)) or _opt(os.getenv("AZURE_BLOB_CONN_STR"))
    if conn:
        return {"connectionString": conn}
    acct = _opt(getattr(cfg, "azure_storage_account", None)) or _opt(os.getenv("AZURE_STORAGE_ACCOUNT"))
    key = _opt(getattr(cfg, "azure_storage_key", None)) or _opt(os.getenv("AZURE_STORAGE_KEY"))
    if acct and key:
        return {
            "connectionString": f"DefaultEndpointsProtocol=https;AccountName={acct};AccountKey={key};EndpointSuffix=core.windows.net"
        }
    raise RuntimeError("Missing AZURE_BLOB_CONN_STR or (AZURE_STORAGE_ACCOUNT + AZURE_STORAGE_KEY).")


def _raw_container(cfg) -> str:
    return _require(
        _opt(getattr(cfg, "blob_container_raw", None)) or _opt(os.getenv("BLOB_CONTAINER_RAW")),
        "BLOB_CONTAINER_RAW",
    )


def _index_name(cfg) -> str:
    return _require(
        _opt(getattr(cfg, "azure_search_index", None)) or _opt(os.getenv("AZURE_SEARCH_INDEX")),
        "AZURE_SEARCH_INDEX",
    )


def _fail_verbose(what: str, url: str, payload: Dict, response: requests.Response):
    try:
        body = response.json()
    except Exception:
        body = response.text
    log.error(
        "[search_pipeline] %s upsert failed\nURL: %s\nStatus: %s\nPayload:\n%s\nResponse:\n%s",
        what,
        url,
        response.status_code,
        json.dumps(payload, indent=2),
        json.dumps(body, indent=2) if isinstance(body, dict) else body,
    )
    raise RuntimeError(f"{what} upsert failed: {response.status_code} {response.text}")


# ---------- main ----------
def ensure_pipeline() -> None:
    # late import to avoid cycles
    try:
        from ..config import Settings
    except ImportError:
        import sys, pathlib
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        sys.path.append(str(repo_root.parent))
        from src.news_reporter.config import Settings  # type: ignore

    cfg = Settings.load()
    base = _search_base(cfg)
    key = _admin_key(cfg)
    cont = _raw_container(cfg)
    idx = _index_name(cfg)

    ds_name = _safe_name("raw", "ds")
    ixr_name = _safe_name("raw", "ixr")

    # Data source
    ds_url = f"{base}/datasources/{ds_name}?api-version={API_VERSION}"
    ds_payload = {
        "name": ds_name,
        "type": "azureblob",
        "credentials": _blob_connection(cfg),
        "container": {"name": cont},
        "description": "Raw blob container for ingestion",
    }
    r = requests.put(ds_url, headers=_headers(key), data=json.dumps(ds_payload))
    if r.status_code >= 300:
        _fail_verbose("Data source", ds_url, ds_payload, r)

    # Index (create-only)
    idx_url = f"{base}/indexes/{idx}?api-version={API_VERSION}"
    idx_payload = {
        "name": idx,
        "fields": [
            {"name": "id", "type": "Edm.String", "key": True, "searchable": False, "filterable": False, "sortable": False, "facetable": False},
            {"name": "content", "type": "Edm.String", "searchable": True, "filterable": False, "sortable": False, "facetable": False},
            {"name": "file_name", "type": "Edm.String", "searchable": False, "filterable": True, "sortable": True, "facetable": False},
            {"name": "content_type", "type": "Edm.String", "searchable": False, "filterable": True, "sortable": True, "facetable": True},
            {"name": "last_modified", "type": "Edm.DateTimeOffset", "searchable": False, "filterable": True, "sortable": True, "facetable": False},
            {"name": "url", "type": "Edm.String", "searchable": False, "filterable": False, "sortable": False, "facetable": False},
        ],
    }
    r = requests.put(idx_url, headers=_headers(key, {"If-None-Match": "*"}), data=json.dumps(idx_payload))
    if r.status_code >= 300 and r.status_code not in (409, 412):
        _fail_verbose("Index", idx_url, idx_payload, r)
    elif r.status_code in (409, 412):
        log.info("[search_pipeline] Index '%s' already exists; leaving as-is.", idx)

    # Indexer (scheduled)
    ixr_url = f"{base}/indexers/{ixr_name}?api-version={API_VERSION}"
    ixr_payload = {
        "name": ixr_name,
        "dataSourceName": ds_name,
        "targetIndexName": idx,
        "parameters": {
            "configuration": {
                "parsingMode": "default",
                "dataToExtract": "contentAndMetadata",
            }
        },
        # RAW blob fields here (no /document)
        "fieldMappings": [
            {"sourceFieldName": "metadata_storage_path", "targetFieldName": "id", "mappingFunction": {"name": "base64Encode"}},
            {"sourceFieldName": "metadata_storage_name", "targetFieldName": "file_name"},
            {"sourceFieldName": "metadata_storage_content_type", "targetFieldName": "content_type"},
            {"sourceFieldName": "metadata_storage_last_modified", "targetFieldName": "last_modified"},
            {"sourceFieldName": "metadata_storage_path", "targetFieldName": "url"},
        ],
        # Cracked fields here (WITH /document)
        "outputFieldMappings": [
            {"sourceFieldName": "/document/content", "targetFieldName": "content"},
        ],
        # schedule: every 5 minutes
        "schedule": {"interval": "PT5M"},
    }
    r = requests.put(ixr_url, headers=_headers(key), data=json.dumps(ixr_payload))
    if r.status_code >= 300:
        _fail_verbose("Indexer", ixr_url, ixr_payload, r)

    log.info("[search_pipeline] ✅ Pipeline ready: datasource=%s, index=%s, indexer=%s", ds_name, idx, ixr_name)


def run_indexer_now(indexer_name: Optional[str] = None) -> None:
    try:
        from ..config import Settings
    except ImportError:
        import sys, pathlib
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        sys.path.append(str(repo_root.parent))
        from src.news_reporter.config import Settings  # type: ignore

    cfg = Settings.load()
    base = _search_base(cfg)
    key = _admin_key(cfg)
    ixr = _safe_name(indexer_name) if indexer_name else _safe_name("raw", "ixr")

    url = f"{base}/indexers/{ixr}/run?api-version={API_VERSION}"
    r = requests.post(url, headers=_headers(key))
    if r.status_code not in (200, 202):
        _fail_verbose("Indexer Run", url, {}, r)


def get_indexer_status(indexer_name: Optional[str] = None) -> Dict:
    try:
        from ..config import Settings
    except ImportError:
        import sys, pathlib
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        sys.path.append(str(repo_root.parent))
        from src.news_reporter.config import Settings  # type: ignore

    cfg = Settings.load()
    base = _search_base(cfg)
    key = _admin_key(cfg)
    ixr = _safe_name(indexer_name) if indexer_name else _safe_name("raw", "ixr")

    url = f"{base}/indexers/{ixr}/status?api-version={API_VERSION}"
    r = requests.get(url, headers=_headers(key))
    if r.status_code >= 300:
        _fail_verbose("Indexer Status", url, {}, r)
    return r.json()
