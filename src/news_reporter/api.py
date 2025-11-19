from __future__ import annotations
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Optional
from pydantic import BaseModel

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings
from .workflows.workflow_factory import run_sequential_goal

# Optional imports for upload functionality
try:
    from src.news_reporter.tools_upload_index.search_pipeline import ensure_pipeline, run_indexer_now
    from src.news_reporter.tools_upload_index.pdf_ingestor import ingest_pdf
    _UPLOAD_AVAILABLE = True
except ImportError:
    _UPLOAD_AVAILABLE = False
    ensure_pipeline = None
    run_indexer_now = None
    ingest_pdf = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logging.info("[lifespan] Loading configuration...")
        _ = Settings.load()

        if _UPLOAD_AVAILABLE:
            logging.info("[lifespan] Ensuring Azure Search pipeline...")
            try:
                ensure_pipeline()
                logging.info("[lifespan] Pipeline ensured successfully.")

                # Optionally trigger indexer at startup (safe even if nothing new)
                try:
                    run_indexer_now()
                    logging.info("[lifespan] Indexer triggered on startup.")
                except Exception as ie:
                    logging.warning("[lifespan] Indexer trigger skipped: %s", ie)
            except Exception as e:
                logging.warning("[lifespan] Pipeline setup skipped: %s", e)
        else:
            logging.info("[lifespan] Upload functionality not available (Azure dependencies missing)")

    except Exception as e:
        logging.exception("[lifespan] Startup provisioning failed: %s", e)

    yield
    logging.info("[lifespan] API shutting down.")

app = FastAPI(
    title="News Reporter Uploader",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None

class Source(BaseModel):
    file_name: Optional[str] = None
    file_path: Optional[str] = None
    directory_name: Optional[str] = None
    text: Optional[str] = None
    similarity: Optional[float] = None
    hybrid_score: Optional[float] = None
    metadata: Optional[dict] = None

class ChatResponse(BaseModel):
    response: str
    sources: list[Source] = []
    conversation_id: str

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
    if not _UPLOAD_AVAILABLE:
        raise HTTPException(status_code=503, detail="Upload functionality is not available. Azure dependencies are missing.")
    
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

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint that processes user queries through the Agent workflow.
    Returns response with sources from Neo4j GraphRAG.
    """
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Generate or use provided conversation ID
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Load settings
        cfg = Settings.load()
        
        # Get sources from Neo4j if using Neo4j search
        sources = []
        if cfg.use_neo4j_search:
            try:
                # Try relative import first, then absolute import
                try:
                    from ..tools.neo4j_graphrag import graphrag_search
                except ImportError:
                    from src.news_reporter.tools.neo4j_graphrag import graphrag_search
                
                search_results = graphrag_search(
                    query=request.query,
                    top_k=8,
                    similarity_threshold=0.7
                )
                
                if search_results:
                    sources = [
                        Source(
                            file_name=res.get("file_name"),
                            file_path=res.get("file_path"),
                            directory_name=res.get("directory_name"),
                            text=res.get("text", "")[:500] if res.get("text") else None,  # Truncate for response
                            similarity=res.get("similarity"),
                            hybrid_score=res.get("hybrid_score"),
                            metadata=res.get("metadata")
                        )
                        for res in search_results
                    ]
                    logging.info(f"Retrieved {len(sources)} sources from Neo4j GraphRAG")
                else:
                    logging.warning("No search results returned from Neo4j GraphRAG")
            except Exception as e:
                logging.error(f"Failed to get Neo4j sources: {e}", exc_info=True)
                sources = []
        
        # Run the agent workflow
        try:
            response_text = await run_sequential_goal(cfg, request.query)
        except RuntimeError as e:
            error_msg = str(e)
            # Check if it's a Foundry access error
            if "Foundry" in error_msg or "foundry" in error_msg or "AZURE_AI_PROJECT" in error_msg:
                logging.error("[chat] Foundry access error: %s", e)
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Foundry access is required but not available. "
                        "Please configure Foundry access or contact your administrator. "
                        f"Error: {error_msg}"
                    )
                )
            # Re-raise other RuntimeErrors
            raise HTTPException(status_code=500, detail=f"Agent workflow failed: {error_msg}")
        
        return ChatResponse(
            response=response_text,
            sources=sources,
            conversation_id=conversation_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("[chat] Failed to process query: %s", e)
        error_msg = str(e)
        # Provide user-friendly error message
        if "Foundry" in error_msg or "foundry" in error_msg:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Foundry access error. "
                    "Please check your Foundry configuration and access permissions. "
                    f"Error: {error_msg}"
                )
            )
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {error_msg}")

# ---------- Allow `python -m src.news_reporter.api` to start the server ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.news_reporter.api:app", host="127.0.0.1", port=8787, reload=True)
