from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from .config import Settings
from src.news_reporter.tools_upload_index.search_pipeline import ensure_pipeline, run_indexer_now
from src.news_reporter.tools_upload_index.pdf_ingestor import ingest_pdf


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logging.info("[lifespan] Loading configuration...")
        _ = Settings.load()

        logging.info("[lifespan] Ensuring Azure Search pipeline...")
        ensure_pipeline()
        logging.info("[lifespan] Pipeline ensured successfully.")

        # Optionally trigger indexer at startup (safe even if nothing new)
        try:
            run_indexer_now()
            logging.info("[lifespan] Indexer triggered on startup.")
        except Exception as ie:
            logging.warning("[lifespan] Indexer trigger skipped: %s", ie)

    except Exception as e:
        logging.exception("[lifespan] Startup provisioning failed: %s", e)

    yield
    logging.info("[lifespan] API shutting down.")

app = FastAPI(
    title="News Reporter Uploader",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/", response_class=HTMLResponse)
def upload_form():
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8"/>
    <title>Upload PDF</title>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <style>
      body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 40px; }
      .card { max-width: 560px; padding: 24px; border: 1px solid #e5e7eb; border-radius: 16px; box-shadow: 0 1px 2px rgba(0,0,0,.06); }
      h1 { margin-top: 0; }
      label { display:block; margin: 12px 0 6px; font-weight: 600; }
      input[type=file], input[type=text] { padding: 10px; border: 1px solid #e5e7eb; border-radius: 8px; width: 100%; }
      button { margin-top: 16px; padding: 10px 16px; border: 0; border-radius: 10px; background: #111827; color: white; cursor: pointer; }
      .hint { color: #6b7280; font-size: 14px; }
      .footer { margin-top: 24px; color: #6b7280; font-size: 13px;}
    </style>
  </head>
  <body>
    <div class="card">
      <h1>Upload a PDF</h1>
      <p class="hint">Choose a PDF. The app uploads it to Blob; the Azure Search indexer ingests it automatically.</p>
      <form method="post" action="/upload" enctype="multipart/form-data">
        <label for="file">PDF file</label>
        <input id="file" type="file" name="file" accept="application/pdf" required />
        <label for="tags">Tags (optional)</label>
        <input id="tags" name="tags" type="text" placeholder="e.g. finance, weekly-report" />
        <button type="submit">Upload & Index</button>
      </form>
      <div class="footer">
        <p>Docs at <a href="/docs">/docs</a> • Health at <a href="/healthz">/healthz</a></p>
      </div>
    </div>
  </body>
</html>
    """

@app.post("/upload")
async def upload(file: UploadFile = File(...), tags: Optional[str] = Form(None)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    try:
        raw_bytes = await file.read()
        meta = {"source": "web-upload"}
        if tags:
            meta["tags"] = [t.strip() for t in tags.split(",") if t.strip()]

        result = ingest_pdf(raw_bytes, file.filename, meta)

        kicked = False
        try:
            run_indexer_now()
            kicked = True
        except Exception as ie:
            logging.warning("[upload] Could not trigger indexer: %s", ie)

        return JSONResponse(
            {
                "status": "uploaded",
                "file_name": file.filename,
                "blob_uri": result.get("blob_uri"),
                "container": result.get("container"),
                "doc_id": result.get("doc_id"),
                "uploaded_utc": result.get("uploaded_utc"),
                "indexer_triggered": kicked,
                "note": "The Search indexer will ingest this document shortly.",
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("[upload] Failed to ingest: %s", e)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# ---------- Allow `python -m src.news_reporter.api` to start the server ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.news_reporter.api:app", host="127.0.0.1", port=8787, reload=True)
