# src/news_reporter/tools/pdf_ingestor.py

from __future__ import annotations
import io
import datetime
import base64
import re
from typing import Dict, Any

from azure.storage.blob import BlobServiceClient

try:
    from ..config import Settings
except ImportError:
    import sys, pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root.parent))
    from src.news_reporter.config import Settings

def _safe_name(s: str, prefix: str = "", max_len: int = 128) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\-]", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    if prefix:
        s = f"{prefix}-{s}" if s else prefix
    return s[:max_len].strip("-")

def ingest_pdf(raw_bytes: bytes, filename: str | None, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uploads the PDF to Blob Storage with a deterministic blob path, so re-uploads
    replace the existing blob and update the same index document.
    """
    settings = Settings.load()

    if settings.azure_blob_conn_str is None:
        raise ValueError("AZURE_BLOB_CONN_STR is missing in .env")
    if settings.blob_container_raw is None:
        raise ValueError("BLOB_CONTAINER_RAW is missing in .env")

    blob_service = BlobServiceClient.from_connection_string(settings.azure_blob_conn_str)
    container = blob_service.get_container_client(settings.blob_container_raw)
    try:
        container.create_container()
    except Exception:
        pass  # already exists

    # ----------------------
    # ✅ DETERMINISTIC NAME
    # ----------------------
    safe_name = _safe_name(filename or "uploaded.pdf")
    blob_name = f"uploads/{safe_name}"  # consistent path
    doc_id = base64.b64encode(blob_name.encode("utf-8")).decode("utf-8")  # index key

    # Optional: delete old blob first (not strictly necessary if overwriting)
    try:
        container.delete_blob(blob_name)
    except Exception:
        pass  # blob may not exist

    # Upload the file (overwrite=True is important)
    container.upload_blob(name=blob_name, data=io.BytesIO(raw_bytes), overwrite=True)
    blob_uri = f"{container.url}/{blob_name}"
    ts = datetime.datetime.utcnow().isoformat()

    return {
        "doc_id": doc_id,
        "blob_uri": blob_uri,
        "container": settings.blob_container_raw,
        "uploaded_utc": ts,
        "meta": meta or {},
        "note": "Azure Search indexer will process this file (chunk + embed).",
    }
