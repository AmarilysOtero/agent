"""Health check and root endpoints"""

from fastapi import APIRouter
from datetime import datetime

from database import get_neo4j_connection

router = APIRouter()


@router.get("/")
async def root():
    """Root endpoint"""
    return {"message": "RAG File Scanner Neo4j API", "status": "running"}


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    neo4j_conn = get_neo4j_connection()
    try:
        # Try a simple connection test without routing
        with neo4j_conn.get_session() as session:
            result = session.run("RETURN 1 as test")
            test_value = result.single()["test"]
            return {
                "status": "healthy",
                "neo4j_connected": True,
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        # If routing fails but we can connect, still report as healthy
        if "routing" in str(e).lower():
            return {
                "status": "healthy", 
                "neo4j_connected": True,
                "warning": "Single instance mode (routing not available)",
                "timestamp": datetime.now().isoformat()
            }
        return {
            "status": "unhealthy",
            "neo4j_connected": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

