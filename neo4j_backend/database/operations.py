"""Neo4j database operations and queries"""

import logging
from typing import Optional, Dict, Any, List, Tuple

from models import FileNode, GraphStats
from utils.chunking import extract_and_chunk, is_supported_file

logger = logging.getLogger(__name__)


class Neo4jOperations:
    """Operations for interacting with Neo4j database"""
    
    def __init__(self, session):
        """Initialize with Neo4j session
        
        Args:
            session: Neo4j driver session
        """
        self.session = session
    
    def create_constraints(self):
        """Create constraints and indexes for better performance"""
        constraints = [
            "CREATE CONSTRAINT file_id_unique IF NOT EXISTS FOR (f:File) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT directory_id_unique IF NOT EXISTS FOR (d:Directory) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT machine_id_unique IF NOT EXISTS FOR (m:Machine) REQUIRE m.machineId IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
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

    # --------------------- Upload Documents (Chunks) ---------------------
    def _list_files_under_directory(self, root_directory_id: str) -> List[Dict[str, Any]]:
        """Return file nodes under a directory (recursive)."""
        query = """
        MATCH (d:Directory {id: $root_id})-[:CONTAINS*0..]->(f:File)
        RETURN f
        """
        result = self.session.run(query, root_id=root_directory_id)
        files: List[Dict[str, Any]] = []
        for record in result:
            f = dict(record["f"])
            files.append(f)
        return files

    def store_chunks_for_directory(self, machine_id: str, full_path: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> Dict[str, Any]:
        """Extract, chunk, embed, and store chunks for all supported files under the given directory.
        Returns summary counts.
        """
        root_id = f"{machine_id}:{full_path}"
        # Validate directory exists
        check = self.session.run("MATCH (d:Directory {id: $id}) RETURN d", id=root_id).single()
        if not check:
            return {"processed_files": 0, "created_chunks": 0, "skipped_files": 0, "message": "Directory not found"}

        files = self._list_files_under_directory(root_id)
        processed_files = 0
        created_chunks = 0
        skipped_files = 0

        # Lazy import to avoid raising if not configured until needed
        try:
            from utils.embeddings import EmbeddingClient
            embedder = EmbeddingClient()
        except Exception as e:
            raise RuntimeError(f"Embedding client not configured: {e}")

        for f in files:
            file_full_path = f.get("fullPath") or ""
            if not file_full_path or not is_supported_file(file_full_path):
                skipped_files += 1
                continue

            try:
                chunks = extract_and_chunk(file_full_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            except Exception:
                skipped_files += 1
                continue

            if not chunks:
                skipped_files += 1
                continue

            # Embed in batches (simple all-at-once for now)
            vectors = embedder.embed(chunks)

            file_id = f.get("id") or f"{machine_id}:{file_full_path}"

            for idx, (text, vector) in enumerate(zip(chunks, vectors)):
                chunk_id = f"{file_id}:chunk:{idx}"
                self.session.run(
                    """
                    MERGE (c:Chunk {id: $id})
                    SET c.index = $index,
                        c.text = $text,
                        c.dim = size($embedding),
                        c.embedding = $embedding,
                        c.createdAt = datetime()
                    WITH c
                    MATCH (f:File {id: $file_id})
                    MERGE (f)-[:HAS_CHUNK]->(c)
                    WITH c
                    MATCH (d:Directory {id: $dir_id})
                    MERGE (d)-[:CONTAINS_CHUNK]->(c)
                    """,
                    id=chunk_id,
                    index=idx,
                    text=text,
                    embedding=vector,
                    file_id=file_id,
                    dir_id=root_id,
                )
                created_chunks += 1
            processed_files += 1

        return {"processed_files": processed_files, "created_chunks": created_chunks, "skipped_files": skipped_files}
    
    def store_file_structure(
        self, 
        file_node: FileNode, 
        parent_id: Optional[str] = None, 
        rag_data: Optional[dict] = None, 
        machine_id: Optional[str] = None
    ):
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
                id=directory_key,
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
                id=file_key,
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
            rag_info = rag_data.get(file_key, {}) if rag_data else {}
            
            # Create RAG checkbox node (always create with default value)
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
            rag_value = "1" if rag_selected else "0"
            
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
            
            rag_status_value = rag_info.get('status', 'unselected')
            
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
        """Retrieve complete directory structure from Neo4j by root key
        
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
            return None
        
        root_data = dict(root_record["d"])
        
        # Recursively build the tree structure
        def build_tree(node_key: str):
            """Recursively build tree structure from a node key"""
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
    
    def store_document_chunks(
        self,
        file_id: str,
        chunks: list,
        embeddings: list = None
    ):
        """Store document chunks with embeddings in Neo4j
        
        Args:
            file_id: ID of the file node in Neo4j
            chunks: List of chunk dictionaries with 'text', 'index', and metadata
            embeddings: Optional list of embedding vectors (one per chunk)
        """
        if not chunks:
            logger.warning(f"No chunks provided for file {file_id}")
            return
        
        # Match the file node
        file_match_query = """
        MATCH (f:File {id: $file_id})
        RETURN f
        LIMIT 1
        """
        file_result = self.session.run(file_match_query, file_id=file_id)
        if not file_result.single():
            logger.error(f"File node {file_id} not found in Neo4j")
            return
        
        # Delete existing chunks for this file (if re-uploading)
        delete_query = """
        MATCH (f:File {id: $file_id})-[:HAS_CHUNK]->(c:Chunk)
        DETACH DELETE c
        """
        self.session.run(delete_query, file_id=file_id)
        
        # Store each chunk
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{file_id}:chunk:{chunk.get('index', idx)}"
            chunk_text = chunk.get('text', '')
            chunk_metadata = {k: v for k, v in chunk.items() if k not in ['text', 'index']}
            
            # Create chunk node
            chunk_query = """
            CREATE (c:Chunk {
                id: $chunk_id,
                text: $text,
                index: $index,
                file_id: $file_id,
                chunk_size: $chunk_size,
                createdAt: datetime()
            })
            SET c += $metadata
            RETURN c
            """
            
            self.session.run(
                chunk_query,
                chunk_id=chunk_id,
                text=chunk_text,
                index=chunk.get('index', idx),
                file_id=file_id,
                chunk_size=len(chunk_text),
                metadata=chunk_metadata or {}
            )
            
            # Link chunk to file
            link_query = """
            MATCH (f:File {id: $file_id})
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (f)-[:HAS_CHUNK]->(c)
            """
            self.session.run(link_query, file_id=file_id, chunk_id=chunk_id)
            
            # Store embedding if provided
            if embeddings and idx < len(embeddings) and embeddings[idx] is not None:
                embedding_vector = embeddings[idx]
                embedding_query = """
                MATCH (c:Chunk {id: $chunk_id})
                SET c.embedding = $embedding,
                    c.embedding_dimension = $dimension
                """
                self.session.run(
                    embedding_query,
                    chunk_id=chunk_id,
                    embedding=embedding_vector,
                    dimension=len(embedding_vector)
                )
        
        logger.info(f"Stored {len(chunks)} chunks for file {file_id}")
    
    def store_directory_chunks(
        self,
        directory_id: str,
        files_data: list
    ):
        """Store chunks for multiple files in a directory
        
        Args:
            directory_id: ID of staring directory node
            files_data: List of dicts with 'file_id', 'chunks', and optionally 'embeddings'
        """
        for file_data in files_data:
            file_id = file_data.get('file_id')
            chunks = file_data.get('chunks', [])
            embeddings = file_data.get('embeddings')
            
            if file_id and chunks:
                self.store_document_chunks(file_id, chunks, embeddings)
    
    def get_file_chunks(self, file_id: str):
        """Retrieve all chunks for a file
        
        Args:
            file_id: ID of the file node
            
        Returns:
            List of chunk dictionaries
        """
        query = """
        MATCH (f:File {id: $file_id})-[:HAS_CHUNK]->(c:Chunk)
        RETURN c
        ORDER BY c.index
        """
        
        result = self.session.run(query, file_id=file_id)
        chunks = []
        for record in result:
            chunk_node = record["c"]
            chunk_dict = dict(chunk_node)
            chunks.append(chunk_dict)
        
        return chunks

