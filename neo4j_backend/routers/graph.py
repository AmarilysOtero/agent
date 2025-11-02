"""Graph operations endpoints"""

import logging
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException

from database import get_neo4j_connection, Neo4jOperations
from models import FileStructureRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/graph/store")
async def store_file_structure(request: FileStructureRequest):
    """Store file structure in Neo4j graph with RAG hierarchy"""
    neo4j_conn = get_neo4j_connection()
    try:
        with neo4j_conn.get_session() as session:
            ops = Neo4jOperations(session)
            ops.store_file_structure(
                request.data, 
                rag_data=request.rag_data,
                machine_id=request.machine_id
            )
            
        # Calculate root key using machineId:fullPath format
        root_key = f"{request.machine_id}:{request.data.fullPath}" if request.machine_id else request.data.id
            
        return {
            "message": "File structure stored successfully with RAG hierarchy",
            "root_id": root_key,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error storing file structure: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/graph/nodes")
async def get_all_nodes():
    """Get all nodes from the graph"""
    neo4j_conn = get_neo4j_connection()
    try:
        with neo4j_conn.get_session() as session:
            result = session.run("MATCH (n) RETURN n")
            nodes = [dict(record["n"]) for record in result]
            return {"nodes": nodes}
    except Exception as e:
        logger.error(f"Error getting nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/graph/relationships")
async def get_all_relationships():
    """Get all relationships from the graph"""
    neo4j_conn = get_neo4j_connection()
    try:
        with neo4j_conn.get_session() as session:
            result = session.run("MATCH (n)-[r]->(m) RETURN n, r, m")
            relationships = []
            for record in result:
                relationships.append({
                    "source": dict(record["n"]),
                    "relationship": dict(record["r"]),
                    "target": dict(record["m"])
                })
            return {"relationships": relationships}
    except Exception as e:
        logger.error(f"Error getting relationships: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/graph/visualization")
async def get_graph_visualization():
    """Get graph data formatted for visualization with RAG hierarchy"""
    neo4j_conn = get_neo4j_connection()
    try:
        with neo4j_conn.get_session() as session:
            # Get all nodes with their labels
            nodes_result = session.run("MATCH (n) RETURN n, labels(n) as labels")
            nodes = []
            for record in nodes_result:
                node_data = dict(record["n"])
                labels = record["labels"]
                
                # Determine node type and size based on labels
                node_type = "unknown"
                node_size = 15  # Default size
                
                if "File" in labels:
                    node_type = "file"
                    node_size = 2  # Very small for files
                elif "Directory" in labels:
                    node_type = "directory"
                    node_size = 50  # Larger for directories
                elif "RAGCheckbox" in labels:
                    node_type = "rag_checkbox"
                    node_size = 20
                elif "RAGStatus" in labels:
                    node_type = "rag_status"
                    node_size = 20
                
                nodes.append({
                    "id": node_data.get("id", str(node_data.get("name", ""))),
                    "name": node_data.get("name", "Unknown"),
                    "type": node_type,
                    "size": node_size,
                    "source": node_data.get("source", "unknown"),
                    "value": node_data.get("value", ""),
                    "labels": labels
                })
            
            # Get all relationships
            links_result = session.run("MATCH (n)-[r]->(m) RETURN n.id as source, m.id as target, type(r) as relationship_type")
            links = []
            for record in links_result:
                links.append({
                    "source": record["source"],
                    "target": record["target"],
                    "type": record["relationship_type"]
                })
            
            return {
                "nodes": nodes,
                "links": links
            }
    except Exception as e:
        logger.error(f"Error getting visualization data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/graph/stats")
async def get_graph_stats():
    """Get graph statistics"""
    neo4j_conn = get_neo4j_connection()
    try:
        with neo4j_conn.get_session() as session:
            ops = Neo4jOperations(session)
            stats = ops.get_graph_stats()
            return stats
    except Exception as e:
        # If routing fails, return empty stats instead of crashing
        if "routing" in str(e).lower():
            logger.warning(f"Routing error in stats, returning empty stats: {e}")
            return {
                "total_nodes": 0,
                "total_files": 0,
                "total_directories": 0,
                "sources": [],
                "warning": "Single instance mode (routing not available)"
            }
        logger.error(f"Error getting graph stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/graph/clear")
async def clear_graph():
    """Clear all data from the graph"""
    neo4j_conn = get_neo4j_connection()
    try:
        with neo4j_conn.get_session() as session:
            ops = Neo4jOperations(session)
            ops.clear_database()
            
        return {
            "message": "Graph cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/graph/search")
async def search_files(name: str, source: Optional[str] = None):
    """Search for files by name"""
    neo4j_conn = get_neo4j_connection()
    try:
        with neo4j_conn.get_session() as session:
            ops = Neo4jOperations(session)
            results = ops.search_files(name, source)
            
        return {
            "results": results,
            "count": len(results),
            "query": {"name": name, "source": source}
        }
    except Exception as e:
        logger.error(f"Error searching files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/graph/tree/{directory_id}")
async def get_directory_tree(directory_id: str, max_depth: int = 5):
    """Get directory tree structure"""
    neo4j_conn = get_neo4j_connection()
    try:
        with neo4j_conn.get_session() as session:
            ops = Neo4jOperations(session)
            tree = ops.get_directory_tree(directory_id, max_depth)
            
        return {
            "directory_id": directory_id,
            "max_depth": max_depth,
            "tree": tree
        }
    except Exception as e:
        logger.error(f"Error getting directory tree: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/graph/directory")
async def get_directory_structure(machine_id: str, full_path: str):
    """Retrieve directory structure from Neo4j by machineId and fullPath
    
    This endpoint retrieves a previously stored directory structure.
    Returns null if the directory hasn't been stored yet.
    """
    neo4j_conn = get_neo4j_connection()
    try:
        # Construct the root key using machineId:fullPath format
        root_key = f"{machine_id}:{full_path}" if machine_id and full_path else None
        
        if not root_key:
            raise HTTPException(status_code=400, detail="machine_id and full_path are required")
        
        with neo4j_conn.get_session() as session:
            ops = Neo4jOperations(session)
            structure = ops.retrieve_directory_structure(root_key)
            
        if structure is None:
            return {
                "found": False,
                "structure": None,
                "message": "Directory not found in Neo4j"
            }
        
        return {
            "found": True,
            "structure": structure,
            "root_key": root_key
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving directory structure: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/graph/upload-directory")
async def upload_directory_documents(machine_id: str, full_path: str, chunk_size: int = 1200, chunk_overlap: int = 200):
    """Upload directory documents: chunk, embed (Azure), and store in Neo4j.
    Requires Azure config set in environment. Returns processing summary.
    """
    neo4j_conn = get_neo4j_connection()
    try:
        with neo4j_conn.get_session() as session:
            ops = Neo4jOperations(session)
            summary = ops.store_chunks_for_directory(machine_id, full_path, chunk_size, chunk_overlap)
        return {
            "message": "Upload completed",
            "summary": summary,
            "directory_id": f"{machine_id}:{full_path}",
        }
    except Exception as e:
        logger.error(f"Error uploading directory documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/graph/nodes/{node_id}")
async def delete_node(node_id: str):
    """Delete a specific node and its relationships"""
    neo4j_conn = get_neo4j_connection()
    try:
        with neo4j_conn.get_session() as session:
            # First, delete all relationships connected to this node
            session.run("MATCH (n {id: $node_id})-[r]-() DELETE r", node_id=node_id)
            
            # Then delete the node itself
            result = session.run("MATCH (n {id: $node_id}) DELETE n RETURN count(n) as deleted_count", node_id=node_id)
            deleted_count = result.single()["deleted_count"]
            
            if deleted_count == 0:
                raise HTTPException(status_code=404, detail="Node not found")
            
        return {
            "message": f"Node {node_id} deleted successfully",
            "deleted_count": deleted_count,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting node: {e}")
        raise HTTPException(status_code=500, detail=str(e))

