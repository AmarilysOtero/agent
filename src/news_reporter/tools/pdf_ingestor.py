# src/news_reporter/tools/pdf_ingestor.py
from __future__ import annotations
import io
import uuid
import datetime
from typing import Dict, Any

from azure.storage.blob import BlobServiceClient

try:
    from ..config import Settings
except ImportError:
    import sys, pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root.parent))
    from src.news_reporter.config import Settings


def ingest_pdf(raw_bytes: bytes, filename: str | None, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uploads the PDF to Blob Storage. An Azure Search Indexer (provisioned separately)
    will crack, chunk, embed, and index it automatically.
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
        pass  # exists

    doc_id = str(uuid.uuid4())
    ts = datetime.datetime.utcnow().isoformat()
    safe_name = filename or "uploaded.pdf"
    blob_name = f"{doc_id}/{safe_name}"

    container.upload_blob(name=blob_name, data=io.BytesIO(raw_bytes), overwrite=True)
    blob_uri = f"{container.url}/{blob_name}"

    return {
        "doc_id": doc_id,
        "blob_uri": blob_uri,
        "container": settings.blob_container_raw,
        "uploaded_utc": ts,
        "meta": meta or {},
        "note": "Azure Search indexer will process this file (chunk + embed).",
    }
