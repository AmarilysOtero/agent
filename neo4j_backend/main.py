"""Main FastAPI application for Neo4j GraphRAG backend"""

import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .routers import graph

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Neo4j GraphRAG Backend",
    description="Backend API for Neo4j GraphRAG operations including entity extraction and relationship management",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(graph.router)


@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "status": "online",
        "service": "Neo4j GraphRAG Backend",
        "version": "0.1.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from .database.operations import Neo4jOperations
    
    try:
        # Test Neo4j connection
        db = Neo4jOperations()
        with db.driver.session() as session:
            session.run("RETURN 1")
        db.close()
        neo4j_status = "connected"
    except Exception as e:
        logger.error(f"Neo4j health check failed: {e}")
        neo4j_status = f"disconnected: {str(e)}"
    
    return {
        "status": "healthy" if neo4j_status == "connected" else "unhealthy",
        "neo4j": neo4j_status,
        "openai": "configured" if os.getenv("AZURE_OPENAI_ENDPOINT") else "not configured"
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting Neo4j GraphRAG backend on {host}:{port}")
    
    uvicorn.run(
        "neo4j_backend.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
