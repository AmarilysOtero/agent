from __future__ import annotations
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Optional, List, Dict
from datetime import datetime
from pydantic import BaseModel

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
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

# SQL generation imports
try:
    from src.news_reporter.tools_sql.sql_generator import SQLGenerator
    from src.news_reporter.tools_sql.schema_retrieval import SchemaRetriever
    _SQL_AVAILABLE = True
except ImportError:
    _SQL_AVAILABLE = False
    SQLGenerator = None
    SchemaRetriever = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def extract_person_names(query: str) -> List[str]:
    """Extract potential person names from query (capitalized words that might be names)
    
    Args:
        query: User query text
        
    Returns:
        List of potential person names (capitalized words)
    """
    # Split query into words
    words = query.split()
    # Extract capitalized words that are likely names (length > 2, starts with capital)
    names = [w.strip('.,!?;:') for w in words if w and w[0].isupper() and len(w.strip('.,!?;:')) > 2]
    # Remove common words that start with capital but aren't names
    common_words = {'The', 'This', 'That', 'These', 'Those', 'What', 'When', 'Where', 'Who', 'Why', 'How', 'Tell', 'Show', 'Give', 'Find', 'Search', 'Get'}
    names = [n for n in names if n not in common_words]
    return names


def filter_results_by_exact_match(results: List[dict], query: str, min_similarity: float = 0.9) -> List[dict]:
    """Filter search results to require query name appears in chunk text or very high similarity
    
    Args:
        results: List of search result dictionaries
        query: Original query text
        min_similarity: Minimum similarity to keep result without exact match
        
    Returns:
        Filtered list of results
    """
    if not results:
        return results
    
    # Extract potential name words from query using the same function that filters common words
    names = extract_person_names(query)
    query_words = [n.lower() for n in names]
    
    # If no capitalized words, apply minimum similarity threshold only
    if not query_words:
        # Still filter out very low similarity results
        return [res for res in results if res.get("similarity", 0.0) >= 0.3]
    
    # Get first name (first name word) - critical for distinguishing names
    first_name = query_words[0] if query_words else None
    last_name = query_words[-1] if len(query_words) > 1 else None
    
    logging.info(f"Filtering {len(results)} results for query '{query}' (first_name='{first_name}', last_name='{last_name}')")
    
    filtered = []
    for res in results:
        text = res.get("text", "").lower()
        similarity = res.get("similarity", 0.0)
        
        # Apply absolute minimum similarity threshold (reject very low scores)
        if similarity < 0.3:
            logging.debug(f"Filtered out result: similarity={similarity:.3f} < 0.3 (absolute minimum)")
            continue
        
        # Check if first name appears in text (required for name queries)
        # This prevents "Axel Torres" from matching "Alexis Torres" queries
        first_name_found = first_name in text if first_name else True
        
        # If we have both first and last name, require both to match
        if first_name and last_name:
            last_name_found = last_name in text
            name_match = first_name_found and last_name_found
        else:
            # Only first name available, require it to match
            name_match = first_name_found
        
        # Keep if: (name matches AND similarity >= 0.3) OR similarity is very high (>= min_similarity)
        # Lower threshold for name matches to allow more results through
        if (name_match and similarity >= 0.3) or similarity >= min_similarity:
            filtered.append(res)
            logging.debug(f"Kept result: similarity={similarity:.3f}, first_name_match={first_name_found}, name_match={name_match}")
        else:
            logging.info(f"Filtered out result: similarity={similarity:.3f}, first_name_match={first_name_found}, name_match={name_match}, text_preview={text[:100]}")
    
    logging.info(f"Filtered {len(results)} results down to {len(filtered)} results")
    return filtered

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

class SessionCreate(BaseModel):
    user_id: str
    title: Optional[str] = "New Chat"

class SessionUpdate(BaseModel):
    title: Optional[str] = None

class Session(BaseModel):
    id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str

# In-memory session store (in production, use a database)
_sessions: Dict[str, Session] = {}

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
                
                # Extract person names from query for keyword filtering
                person_names = extract_person_names(request.query)
                
                # Use higher similarity threshold and reduced hops to prevent false matches
                search_results = graphrag_search(
                    query=request.query,
                    top_k=12,  # Get more results initially for filtering
                    similarity_threshold=0.75,  # Increased from 0.7 to reduce false matches
                    keywords=person_names if person_names else None,
                    keyword_match_type="any",
                    keyword_boost=0.4  # Increase keyword weight for name matching
                )
                
                # Filter results to require exact name match or very high similarity
                filtered_results = filter_results_by_exact_match(
                    search_results, 
                    request.query, 
                    min_similarity=0.7  # Very high threshold for results without exact match
                )
                
                # Limit to top 8 after filtering
                filtered_results = filtered_results[:8]
                
                if filtered_results:
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
                        for res in filtered_results
                    ]
                    logging.info(f"Retrieved {len(sources)} sources from Neo4j GraphRAG (filtered from {len(search_results)} initial results)")
                else:
                    logging.warning(f"No search results after filtering (had {len(search_results)} initial results)")
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

# SQL Generation Endpoints

class SQLGenerationRequest(BaseModel):
    """Request model for SQL generation"""
    query: str
    database_id: str
    top_k: int = 10
    similarity_threshold: float = 0.7


class SchemaSearchRequest(BaseModel):
    """Request model for schema search"""
    query: str
    database_id: str
    top_k: int = 10
    similarity_threshold: float = 0.7
    element_types: Optional[list] = None


@app.post("/api/sql/generate")
async def generate_sql(request: SQLGenerationRequest):
    """Generate SQL from natural language query"""
    if not _SQL_AVAILABLE:
        raise HTTPException(status_code=503, detail="SQL generation functionality is not available. Required dependencies are missing.")
    try:
        generator = SQLGenerator()
        result = generator.generate_sql(
            query=request.query,
            database_id=request.database_id,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        return JSONResponse(result)
    except Exception as e:
        logging.exception("[sql/generate] Failed: %s", e)
        raise HTTPException(status_code=500, detail=f"SQL generation failed: {str(e)}")


@app.post("/api/schema/search")
async def search_schema(request: SchemaSearchRequest):
    """Search database schema for relevant elements"""
    if not _SQL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Schema search functionality is not available. Required dependencies are missing.")
    try:
        retriever = SchemaRetriever()
        result = retriever.get_relevant_schema(
            query=request.query,
            database_id=request.database_id,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            element_types=request.element_types
        )
        return JSONResponse(result)
    except Exception as e:
        logging.exception("[schema/search] Failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Schema search failed: {str(e)}")


# Session management endpoints
@app.get("/api/sessions")
async def get_sessions(user_id: str = Query(..., description="User ID")):
    """Get all sessions for a user"""
    user_sessions = [session.dict() for session in _sessions.values() if session.user_id == user_id]
    return JSONResponse(user_sessions)

@app.post("/api/sessions", response_model=Session)
async def create_session(request: SessionCreate):
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    
    session = Session(
        id=session_id,
        user_id=request.user_id,
        title=request.title or "New Chat",
        created_at=now,
        updated_at=now
    )
    
    _sessions[session_id] = session
    logging.info(f"Created session {session_id} for user {request.user_id}")
    return JSONResponse(session.dict())

@app.get("/api/sessions/{session_id}", response_model=Session)
async def get_session(session_id: str, user_id: str = Query(..., description="User ID")):
    """Get a specific session"""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = _sessions[session_id]
    if session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    return JSONResponse(session.dict())

@app.patch("/api/sessions/{session_id}", response_model=Session)
async def update_session(session_id: str, request: SessionUpdate, user_id: str = Query(..., description="User ID")):
    """Update a session"""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = _sessions[session_id]
    if session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    if request.title is not None:
        session.title = request.title
        session.updated_at = datetime.utcnow().isoformat()
    
    _sessions[session_id] = session
    logging.info(f"Updated session {session_id}")
    return JSONResponse(session.dict())

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, user_id: str = Query(..., description="User ID")):
    """Delete a session"""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = _sessions[session_id]
    if session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    del _sessions[session_id]
    logging.info(f"Deleted session {session_id}")
    return JSONResponse({"status": "deleted"})

# ---------- Allow `python -m src.news_reporter.api` to start the server ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.news_reporter.api:app", host="127.0.0.1", port=8787, reload=True)
