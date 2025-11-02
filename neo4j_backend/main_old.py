from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uvicorn
from neo4j import GraphDatabase
import logging
from pathlib import Path
import os
from dotenv import load_dotenv
import hashlib
import uuid

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG File Scanner Neo4j API",
    description="API for storing file structures in Neo4j graph database",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Neo4j connection
class Neo4jConnection:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "@neo4j://127.0.0.1:7687")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None
        
    def connect(self):
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password),
                encrypted=False
            )
            logger.info("Connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        if self.driver:
            self.driver.close()
    
    def get_session(self):
        if not self.driver:
            self.connect()
        return self.driver.session()

# Initialize Neo4j connection
neo4j_conn = Neo4jConnection()

# Machine fingerprinting utility
def generate_fingerprint(request: Request) -> str:
    """Generate a fingerprint hash from IP address and user-agent"""
    # Get client IP (handling proxies)
    client_ip = request.client.host if request.client else "unknown"
    
    # Get user-agent
    user_agent = request.headers.get("user-agent", "unknown")
    
    # Combine and hash
    fingerprint_data = f"{client_ip}:{user_agent}"
    fingerprint_hash = hashlib.sha256(fingerprint_data.encode()).hexdigest()
    
    return fingerprint_hash

# Pydantic models
class FileNode(BaseModel):
    id: str
    type: str = Field(..., description="Either 'file' or 'directory'")
    name: str
    fullPath: str
    relativePath: str
    size: Optional[int] = None
    extension: Optional[str] = None
    modifiedTime: str
    createdAt: str
    source: str
    children: List['FileNode'] = []

class FileStructureRequest(BaseModel):
    data: FileNode
    metadata: Optional[Dict[str, Any]] = {}
    rag_data: Optional[Dict[str, Dict[str, Any]]] = {}
    machine_id: Optional[str] = None  # Machine ID to identify the client

class MachineRegistrationRequest(BaseModel):
    """Request model for machine registration (no body needed - uses fingerprint)"""
    pass

class GraphStats(BaseModel):
    total_nodes: int
    total_files: int
    total_directories: int
    sources: List[str]

# Neo4j operations
class Neo4jOperations:
    def __init__(self, session):
        self.session = session
    
    def create_constraints(self):
        """Create constraints and indexes for better performance"""
        constraints = [
            "CREATE CONSTRAINT file_id_unique IF NOT EXISTS FOR (f:File) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT directory_id_unique IF NOT EXISTS FOR (d:Directory) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT machine_id_unique IF NOT EXISTS FOR (m:Machine) REQUIRE m.machineId IS UNIQUE",
            "CREATE INDEX file_name_index IF NOT EXISTS FOR (f:File) ON (f.name)",
            "CREATE INDEX directory_name_index IF NOT EXISTS FOR (d:Directory) ON (d.name)",
            "CREATE INDEX file_source_index IF NOT EXISTS FOR (f:File) ON (f.source)",
            "CREATE INDEX directory_source_index IF NOT EXISTS FOR (d:Directory) ON (d.source)",
            "CREATE INDEX machine_fingerprint_index IF NOT EXISTS FOR (m:Machine) ON m.fingerprint"
        ]
        
        for constraint in constraints:
            try:
                self.session.run(constraint)
                logger.info(f"Created constraint/index: {constraint}")
            except Exception as e:
                logger.warning(f"Constraint/index may already exist: {e}")
    
    def clear_database(self):
        """Clear all nodes and relationships"""
        self.session.run("MATCH (n) DETACH DELETE n")
        logger.info("Cleared Neo4j database")
    
    def store_file_structure(self, file_node: FileNode, parent_id: Optional[str] = None, rag_data: Optional[dict] = None, machine_id: Optional[str] = None):
        """Recursively store file structure in Neo4j with RAG hierarchy
        
        Args:
            file_node: The file or directory node to store
            parent_id: The ID of the parent node (if any)
            rag_data: Dictionary of RAG checkbox and status data
            machine_id: The machine ID to use for creating unique keys (machineId:fullPath)
                        Using fullPath ensures each directory is stored independently
        """
        
        if file_node.type == "directory":
            # Generate unique key using machineId:fullPath if machine_id provided
            # Use fullPath instead of relativePath because:
            # - Root directories often have relativePath as "." or empty, causing collisions
            # - fullPath is unique per directory on the machine
            # - This ensures each directory is stored independently
            directory_key = f"{machine_id}:{file_node.fullPath}" if machine_id else file_node.id
            
            # Create directory node using the key for MERGE
            query = """
            MERGE (d:Directory {id: $id})
            SET d.name = $name,
                d.fullPath = $fullPath,
                d.relativePath = $relativePath,
                d.modifiedTime = $modifiedTime,
                d.createdAt = $createdAt,
                d.source = $source,
                d.machineKey = $machine_key,
                d.lastUpdated = datetime()
            RETURN d
            """
            
            result = self.session.run(query, 
                id=directory_key,  # Use machineKey as the actual ID for MERGE
                name=file_node.name,
                fullPath=file_node.fullPath,
                relativePath=file_node.relativePath,
                modifiedTime=file_node.modifiedTime,
                createdAt=file_node.createdAt,
                source=file_node.source,
                machine_key=directory_key
            )
            
            directory_node = result.single()["d"]
            
            # Create parent-child relationship if parent exists
            if parent_id:
                parent_query = """
                MATCH (parent:Directory {id: $parent_id})
                MATCH (child:Directory {id: $child_id})
                MERGE (parent)-[:CONTAINS]->(child)
                """
                self.session.run(parent_query, parent_id=parent_id, child_id=directory_key)
            
            # Process children (use directory_key as parent_id)
            for child in file_node.children:
                self.store_file_structure(child, directory_key, rag_data, machine_id)
                
        else:
            # Generate unique key using machineId:fullPath if machine_id provided
            # Use fullPath to ensure file identity survives localStorage clears
            # This also ensures files are unique per directory when same filename exists in different paths
            file_key = f"{machine_id}:{file_node.fullPath}" if machine_id else file_node.id
            
            # Create file node using the key for MERGE
            query = """
            MERGE (f:File {id: $id})
            SET f.name = $name,
                f.fullPath = $fullPath,
                f.relativePath = $relativePath,
                f.size = $size,
                f.extension = $extension,
                f.modifiedTime = $modifiedTime,
                f.createdAt = $createdAt,
                f.source = $source,
                f.machineKey = $machine_key,
                f.lastUpdated = datetime()
            RETURN f
            """
            
            result = self.session.run(query,
                id=file_key,  # Use machineKey as the actual ID for MERGE
                name=file_node.name,
                fullPath=file_node.fullPath,
                relativePath=file_node.relativePath,
                size=file_node.size,
                extension=file_node.extension,
                modifiedTime=file_node.modifiedTime,
                createdAt=file_node.createdAt,
                source=file_node.source,
                machine_key=file_key
            )
            
            file_node_result = result.single()["f"]
            
            # Create parent-child relationship if parent exists
            if parent_id:
                parent_query = """
                MATCH (parent:Directory {id: $parent_id})
                MATCH (child:File {id: $child_id})
                MERGE (parent)-[:CONTAINS]->(child)
                """
                self.session.run(parent_query, parent_id=parent_id, child_id=file_key)
            
            # Always create RAG nodes for every file with default values
            # Use file_key (machineId:relPath) instead of file_node.id for RAG data lookup
            rag_info = rag_data.get(file_key, {}) if rag_data else {}
            
            # Create RAG checkbox node (always create with default value)
            # Use file_key for RAG node IDs to maintain consistency
            rag_id = f"{file_key}_rag_checkbox"
            rag_checkbox_query = """
            MERGE (rag:RAGCheckbox {id: $rag_id})
            SET rag.name = 'RAG',
                rag.value = $rag_value,
                rag.type = 'checkbox',
                rag.source = $source,
                rag.lastUpdated = datetime()
            RETURN rag
            """
            
            rag_selected = rag_info.get('selected', False)
            rag_value = "1" if rag_selected else "0"  # Default to "0" if not selected
            
            self.session.run(rag_checkbox_query,
                rag_id=rag_id,
                rag_value=rag_value,
                source=file_node.source
            )
            
            # Create relationship from file to RAG checkbox
            rag_rel_query = """
            MATCH (f:File {id: $file_id})
            MATCH (rag:RAGCheckbox {id: $rag_id})
            MERGE (f)-[:HAS_RAG]->(rag)
            """
            self.session.run(rag_rel_query, file_id=file_key, rag_id=rag_id)
            
            # Create RAG Status node (always create with default value)
            # Use file_key for RAG status node IDs to maintain consistency
            rag_status_id = f"{file_key}_rag_status"
            rag_status_query = """
            MERGE (rag_status:RAGStatus {id: $rag_status_id})
            SET rag_status.name = 'RAG Status',
                rag_status.value = $rag_status_value,
                rag_status.type = 'status',
                rag_status.source = $source,
                rag_status.lastUpdated = datetime()
            RETURN rag_status
            """
            
            rag_status_value = rag_info.get('status', 'unselected')  # Default to "unselected" if empty
            
            self.session.run(rag_status_query,
                rag_status_id=rag_status_id,
                rag_status_value=rag_status_value,
                source=file_node.source
            )
            
            # Create relationship from file to RAG Status
            rag_status_rel_query = """
            MATCH (f:File {id: $file_id})
            MATCH (rag_status:RAGStatus {id: $rag_status_id})
            MERGE (f)-[:HAS_RAG_STATUS]->(rag_status)
            """
            self.session.run(rag_status_rel_query, file_id=file_key, rag_status_id=rag_status_id)
    
    def get_graph_stats(self) -> GraphStats:
        """Get statistics about the graph"""
        query = """
        MATCH (n)
        WITH 
            count(n) as total_nodes,
            count(CASE WHEN n:File THEN 1 END) as total_files,
            count(CASE WHEN n:Directory THEN 1 END) as total_directories,
            collect(DISTINCT n.source) as sources
        RETURN total_nodes, total_files, total_directories, sources
        """
        
        result = self.session.run(query)
        record = result.single()
        
        return GraphStats(
            total_nodes=record["total_nodes"],
            total_files=record["total_files"],
            total_directories=record["total_directories"],
            sources=record["sources"]
        )
    
    def search_files(self, name_pattern: str, source: Optional[str] = None):
        """Search for files by name pattern"""
        query = """
        MATCH (f:File)
        WHERE f.name CONTAINS $pattern
        """
        params = {"pattern": name_pattern}
        
        if source:
            query += " AND f.source = $source"
            params["source"] = source
            
        query += " RETURN f ORDER BY f.name LIMIT 100"
        
        result = self.session.run(query, **params)
        return [record["f"] for record in result]
    
    def get_directory_tree(self, directory_id: str, max_depth: int = 5):
        """Get directory tree structure"""
        query = """
        MATCH path = (d:Directory {id: $id})-[r:CONTAINS*0..$max_depth]->(n)
        RETURN path
        ORDER BY length(path)
        """
        
        result = self.session.run(query, id=directory_id, max_depth=max_depth)
        return [record["path"] for record in result]
    
    def retrieve_directory_structure(self, root_key: str):
        """Retrieve complete directory structure from Neo4j by root key (machineId:relativePath)
        
        Returns a tree structure matching the original FileNode format, or None if not found.
        """
        # First, check if root directory exists
        check_query = """
        MATCH (d:Directory {id: $root_key})
        RETURN d
        LIMIT 1
        """
        result = self.session.run(check_query, root_key=root_key)
        root_record = result.single()
        
        if not root_record:
            # Directory not found in Neo4j
            return None
        
        root_data = dict(root_record["d"])
        
        # Recursively build the tree structure
        def build_tree(node_key: str):
            """Recursively build tree structure from a node key"""
            # Get the node (could be Directory or File)
            node_query = """
            MATCH (n)
            WHERE n.id = $node_key
            AND (n:Directory OR n:File)
            RETURN n, labels(n) as labels
            LIMIT 1
            """
            node_result = self.session.run(node_query, node_key=node_key)
            node_record = node_result.single()
            
            if not node_record:
                return None
            
            node_data = dict(node_record["n"])
            labels = node_record["labels"]
            
            # Determine node type
            node_type = "directory" if "Directory" in labels else "file"
            
            # Build base node structure
            node = {
                "id": node_data.get("id", ""),
                "type": node_type,
                "name": node_data.get("name", ""),
                "fullPath": node_data.get("fullPath", ""),
                "relativePath": node_data.get("relativePath", ""),
                "modifiedTime": node_data.get("modifiedTime", ""),
                "createdAt": node_data.get("createdAt", ""),
                "source": node_data.get("source", ""),
                "children": []
            }
            
            # Add file-specific properties
            if node_type == "file":
                node["size"] = node_data.get("size")
                node["extension"] = node_data.get("extension")
                
                # Get RAG data for this file
                rag_query = """
                MATCH (f:File {id: $file_id})
                OPTIONAL MATCH (f)-[:HAS_RAG]->(rag:RAGCheckbox)
                OPTIONAL MATCH (f)-[:HAS_RAG_STATUS]->(rag_status:RAGStatus)
                RETURN rag.value as rag_selected, rag_status.value as rag_status
                LIMIT 1
                """
                rag_result = self.session.run(rag_query, file_id=node_key)
                rag_record = rag_result.single()
                
                # Initialize with defaults
                node["ragSelected"] = False
                node["ragStatus"] = "unselected"
                
                if rag_record:
                    # Handle RAG checkbox value
                    rag_selected_value = rag_record.get("rag_selected")
                    if rag_selected_value is not None:
                        node["ragSelected"] = rag_selected_value == "1" or rag_selected_value == "True"
                    
                    # Handle RAG status value
                    rag_status_value = rag_record.get("rag_status")
                    if rag_status_value is not None and rag_status_value != "unselected":
                        node["ragStatus"] = rag_status_value
            
            # If it's a directory, get children recursively
            if node_type == "directory":
                children_query = """
                MATCH (parent:Directory {id: $parent_id})-[:CONTAINS]->(child)
                WHERE child:Directory OR child:File
                RETURN child.id as child_id
                ORDER BY child.name
                """
                children_result = self.session.run(children_query, parent_id=node_key)
                
                for child_record in children_result:
                    child_key = child_record["child_id"]
                    child_node = build_tree(child_key)
                    if child_node:
                        node["children"].append(child_node)
            
            return node
        
        # Build the tree starting from root
        tree = build_tree(root_key)
        return tree

# API Routes
@app.on_event("startup")
async def startup_event():
    """Initialize Neo4j connection and create constraints"""
    try:
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
    neo4j_conn.close()

@app.get("/")
async def root():
    return {"message": "RAG File Scanner Neo4j API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
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

@app.post("/api/register-machine")
async def register_machine(http_request: Request):
    """Register a machine and return a persistent machineId
    
    Uses fingerprint (IP + user-agent hash) to detect if this is a returning client.
    If fingerprint matches an existing machine, returns that machineId.
    Otherwise, creates a new machineId and stores it in Neo4j.
    """
    try:
        # Generate fingerprint from client IP and user-agent
        fingerprint = generate_fingerprint(http_request)
        
        with neo4j_conn.get_session() as session:
            # Check if machine with this fingerprint already exists
            check_query = """
            MATCH (m:Machine {fingerprint: $fingerprint})
            RETURN m.machineId as machine_id
            LIMIT 1
            """
            result = session.run(check_query, fingerprint=fingerprint)
            record = result.single()
            
            if record:
                # Existing machine found - return the same machineId
                existing_machine_id = record["machine_id"]
                logger.info(f"Returning existing machineId {existing_machine_id} for fingerprint {fingerprint[:16]}...")
                return {
                    "machineId": existing_machine_id,
                    "isNew": False,
                    "timestamp": datetime.now().isoformat()
                }
            
            # New machine - generate UUID and store in Neo4j
            new_machine_id = str(uuid.uuid4())
            
            create_query = """
            CREATE (m:Machine {
                machineId: $machine_id,
                fingerprint: $fingerprint,
                createdAt: datetime(),
                lastSeen: datetime()
            })
            RETURN m.machineId as machine_id
            """
            session.run(create_query, machine_id=new_machine_id, fingerprint=fingerprint)
            
            logger.info(f"Created new machineId {new_machine_id} for fingerprint {fingerprint[:16]}...")
            return {
                "machineId": new_machine_id,
                "isNew": True,
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error registering machine: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/graph/store")
async def store_file_structure(request: FileStructureRequest):
    """Store file structure in Neo4j graph with RAG hierarchy"""
    try:
        with neo4j_conn.get_session() as session:
            ops = Neo4jOperations(session)
            # Pass machine_id to store_file_structure to generate machineId:relPath keys
            ops.store_file_structure(
                request.data, 
                rag_data=request.rag_data,
                machine_id=request.machine_id
            )
            
        # Calculate root key using machineId:fullPath format
        # Using fullPath ensures each directory is stored independently
        root_key = f"{request.machine_id}:{request.data.fullPath}" if request.machine_id else request.data.id
            
        return {
            "message": "File structure stored successfully with RAG hierarchy",
            "root_id": root_key,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error storing file structure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graph/nodes")
async def get_all_nodes():
    """Get all nodes from the graph"""
    try:
        with neo4j_conn.get_session() as session:
            result = session.run("MATCH (n) RETURN n")
            nodes = [record["n"] for record in result]
            return {"nodes": nodes}
    except Exception as e:
        logger.error(f"Error getting nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graph/relationships")
async def get_all_relationships():
    """Get all relationships from the graph"""
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

@app.get("/api/graph/visualization")
async def get_graph_visualization():
    """Get graph data formatted for visualization with RAG hierarchy"""
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
                    node_size = 25
                elif "Directory" in labels:
                    node_type = "directory"
                    node_size = 30
                elif "RAGCheckbox" in labels:
                    node_type = "rag_checkbox"
                    node_size = 15  # Smaller size for RAG checkbox
                elif "RAGStatus" in labels:
                    node_type = "rag_status"
                    node_size = 15  # Smaller size for RAG status
                
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
@app.get("/api/graph/stats")
async def get_graph_stats():
    """Get graph statistics"""
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

@app.post("/api/graph/clear")
async def clear_graph():
    """Clear all data from the graph"""
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

@app.get("/api/graph/search")
async def search_files(name: str, source: Optional[str] = None):
    """Search for files by name"""
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

@app.get("/api/graph/tree/{directory_id}")
async def get_directory_tree(directory_id: str, max_depth: int = 5):
    """Get directory tree structure"""
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

@app.get("/api/graph/directory")
async def get_directory_structure(machine_id: str, full_path: str):
    """Retrieve directory structure from Neo4j by machineId and fullPath
    
    This endpoint retrieves a previously stored directory structure.
    Returns null if the directory hasn't been stored yet.
    """
    try:
        # Construct the root key using machineId:fullPath format
        # Using fullPath ensures each directory is stored independently
        root_key = f"{machine_id}:{full_path}" if machine_id and full_path else None
        
        if not root_key:
            raise HTTPException(status_code=400, detail="machine_id and full_path are required")
        
        with neo4j_conn.get_session() as session:
            ops = Neo4jOperations(session)
            structure = ops.retrieve_directory_structure(root_key)
            
        if structure is None:
            # Directory not found in Neo4j
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

@app.delete("/api/graph/nodes/{node_id}")
async def delete_node(node_id: str):
    """Delete a specific node and its relationships"""
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
        logger.error(f"Error deleting node {node_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
