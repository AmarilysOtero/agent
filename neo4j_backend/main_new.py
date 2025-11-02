"""Main application entry point - Refactored version"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .database import get_neo4j_connection, Neo4jOperations
from .routers import health_router, machine_router, graph_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Include routers
app.include_router(health_router)
app.include_router(machine_router)
app.include_router(graph_router)


@app.on_event("startup")
async def startup_event():
    """Initialize Neo4j connection and create constraints"""
    try:
        neo4j_conn = get_neo4j_connection()
        neo4j_conn.connect()
        with neo4j_conn.get_session() as session:
            ops = Neo4jOperations(session)
            ops.create_constraints()
        logger.info("Neo4j backend initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Neo4j: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Close Neo4j connection"""
    neo4j_conn = get_neo4j_connection()
    neo4j_conn.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_new:app", host="0.0.0.0", port=8000, reload=True)


