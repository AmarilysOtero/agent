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




@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logging.info("[lifespan] Loading configuration...")
        _ = Settings.load()

        # Initialize workflow persistence (MongoDB connection) on startup
        try:
            logging.info("[lifespan] Initializing workflow persistence...")
            from .workflows.workflow_persistence import get_workflow_persistence
            persistence = get_workflow_persistence()
            if persistence.storage_backend:
                logging.info("[lifespan] Workflow persistence initialized with MongoDB backend")
            else:
                logging.info("[lifespan] Workflow persistence initialized with in-memory storage only")
        except Exception as e:
            logging.warning("[lifespan] Workflow persistence initialization skipped: %s", e)

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

# Include auth router
try:
    logging.info("Attempting to import auth router...")
    from .routers.auth import router as auth_router
    logging.info(f"Auth router imported successfully. Prefix: {auth_router.prefix}, Routes: {len(auth_router.routes)}")
    app.include_router(auth_router)
    logging.info("Auth router mounted successfully")
except ImportError as e:
    logging.warning(f"Auth router not available (ImportError): {e}", exc_info=True)
except Exception as e:
    logging.error(f"Failed to mount auth router: {e}", exc_info=True)

# Include chat sessions router
try:
    logging.info("Attempting to import chat sessions router...")
    from .routers.chat_sessions import router as chat_router
    logging.info(f"Chat sessions router imported successfully. Prefix: {chat_router.prefix}, Routes: {len(chat_router.routes)}")
    app.include_router(chat_router)
    logging.info("Chat sessions router mounted successfully")
except ImportError as e:
    logging.warning(f"Chat sessions router not available (ImportError): {e}", exc_info=True)
except Exception as e:
    logging.error(f"Failed to mount chat sessions router: {e}", exc_info=True)

# Include workflows router (Phase 4)
try:
    logging.info("Attempting to import workflows router...")
    from .routers.workflows import router as workflows_router
    logging.info(f"Workflows router imported successfully. Prefix: {workflows_router.prefix}, Routes: {len(workflows_router.routes)}")
    app.include_router(workflows_router)
    logging.info("Workflows router mounted successfully")
except ImportError as e:
    logging.warning(f"Workflows router not available (ImportError): {e}", exc_info=True)
except Exception as e:
    logging.error(f"Failed to mount workflows router: {e}", exc_info=True)

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


# Agent Management Endpoints
# IMPORTANT: These routes must be registered BEFORE the workflows router
# to avoid any potential route conflicts

class CreateAgentRequest(BaseModel):
    """Request model for creating an agent"""
    name: str
    model: str
    instructions: str
    description: Optional[str] = None
    tools: Optional[List[str]] = None


class UpdateAgentRequest(BaseModel):
    """Request model for updating an agent"""
    name: Optional[str] = None
    model: Optional[str] = None
    instructions: Optional[str] = None
    description: Optional[str] = None
    tools: Optional[List[str]] = None


# Register POST route FIRST to ensure it's available before GET
@app.post("/api/agents", name="create_agent")
async def create_agent(request: CreateAgentRequest):
    """Create a new agent in Foundry
    
    Required fields:
    - name: Agent name (string)
    - model: Model deployment name (e.g., 'gpt-4o-mini', 'gpt-4o')
    - instructions: System instructions for the agent (string)
    
    Optional fields:
    - description: Agent description (defaults to name if not provided)
    - tools: List of tool names to enable for the agent
    """
    try:
        logging.info(f"[POST /api/agents] Creating agent: name={request.name}, model={request.model}")
        from .services.agent_service import create_foundry_agent
        agent = create_foundry_agent(
            name=request.name,
            model=request.model,
            instructions=request.instructions,
            description=request.description,
            tools=request.tools
        )
        logging.info(f"[POST /api/agents] Successfully created agent: {agent.get('id', 'unknown')}")
        return JSONResponse(agent, status_code=201)
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("[POST /api/agents] Failed to create agent")
        raise HTTPException(status_code=500, detail=str(e))


# Agents endpoint for workflow builder (GET must come after POST to avoid route conflicts)
@app.get("/api/agents")
async def get_agents():
    """Get list of available agents for workflow configuration (from config)"""
    try:
        config = Settings.load()
        agents = []
        
        # Add required agents
        if config.agent_id_triage:
            agents.append({
                "id": config.agent_id_triage,
                "name": "Triage Agent",
                "description": "Routes and categorizes incoming requests"
            })
        
        if config.agent_id_aisearch:
            agents.append({
                "id": config.agent_id_aisearch,
                "name": "AI Search Agent",
                "description": "Searches Azure AI Search for relevant content"
            })
        
        if config.agent_id_reviewer:
            agents.append({
                "id": config.agent_id_reviewer,
                "name": "Review Agent",
                "description": "Reviews and validates generated content"
            })
        
        # Add reporter agents
        for i, reporter_id in enumerate(config.reporter_ids, 1):
            agents.append({
                "id": reporter_id,
                "name": f"Reporter Agent {i}" if len(config.reporter_ids) > 1 else "Reporter Agent",
                "description": "Generates news reports and content"
            })
        
        # Add optional agents
        if config.agent_id_neo4j_search:
            agents.append({
                "id": config.agent_id_neo4j_search,
                "name": "Neo4j GraphRAG Agent",
                "description": "Searches Neo4j graph database for relevant content"
            })
        
        if config.agent_id_aisearch_sql:
            agents.append({
                "id": config.agent_id_aisearch_sql,
                "name": "SQL Search Agent",
                "description": "Queries PostgreSQL databases and converts to vector search"
            })
        
        return agents
    except Exception as e:
        logging.exception("Failed to get agents list")
        # Return empty list on error so frontend doesn't break
        return []


@app.get("/api/agents/all")
async def list_agents():
    """List all agents from Foundry (not just config)"""
    try:
        from .services.agent_service import list_all_foundry_agents
        agents = list_all_foundry_agents()
        return agents
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Failed to list agents")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agents/{agent_id}")
async def get_agent_details(agent_id: str):
    """Get details of a specific agent"""
    try:
        from .services.agent_service import get_foundry_agent
        agent = get_foundry_agent(agent_id)
        return agent
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Failed to get agent")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/agents/{agent_id}")
async def update_agent(agent_id: str, request: UpdateAgentRequest):
    """Update an existing agent"""
    try:
        from .services.agent_service import update_foundry_agent
        agent = update_foundry_agent(
            agent_id=agent_id,
            name=request.name,
            model=request.model,
            instructions=request.instructions,
            description=request.description,
            tools=request.tools
        )
        return agent
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Failed to update agent")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/agents/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete an agent from Foundry"""
    try:
        from .services.agent_service import delete_foundry_agent
        success = delete_foundry_agent(agent_id)
        return {"success": success, "message": f"Agent {agent_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Failed to delete agent")
        raise HTTPException(status_code=500, detail=str(e))


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
        
# ---------- Allow `python -m src.news_reporter.api` to start the server ----------
if __name__ == "__main__":
    import uvicorn
    import os
    # In Docker, bind to 0.0.0.0; locally, use 127.0.0.1
    host = "0.0.0.0" if os.getenv("DOCKER_ENV") else "127.0.0.1"
    reload = os.getenv("DOCKER_ENV") is None
    uvicorn.run("src.news_reporter.api:app", host=host, port=8787, reload=reload)
