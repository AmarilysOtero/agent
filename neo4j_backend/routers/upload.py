"""Document upload and processing endpoints"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from datetime import datetime
import os

from database import get_neo4j_connection, Neo4jOperations
from utils import DocumentChunker, read_file_content, get_embeddings_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/upload/directory")
async def upload_directory(directory_path: str, machine_id: str):
    """Upload and process all files in a directory
    
    Args:
        directory_path: Full path to the directory
        machine_id: Machine ID for identifying the directory
        
    Returns:
        Upload results with chunk counts
    """
    neo4j_conn = get_neo4j_connection()
    
    try:
        # Construct directory key
        directory_key = f"{machine_id}:{directory_path}"
        
        # Check if directory exists in Neo4j
        with neo4j_conn.get_session() as session:
            ops = Neo4jOperations(session)
            
            # Verify directory exists
            dir_check_query = """
            MATCH (d:Directory {id: $dir_id})
            RETURN d
            LIMIT 1
            """
            dir_result = session.run(dir_check_query, dir_id=directory_key)
            if not dir_result.single():
                raise HTTPException(
                    status_code=404,
                    detail=f"Directory {directory_path} not found in Neo4j. Store it first."
                )
            
            # Get all files in this directory (recursively)
            files_query = """
            MATCH (d:Directory {id: $dir_id})
            MATCH path = (d)-[:CONTAINS*]->(f:File)
            WHERE ALL(rel in relationships(path) WHERE type(rel) = 'CONTAINS')
            RETURN DISTINCT f.id as file_id, f.fullPath as full_path, f.name as name
            """
            files_result = session.run(files_query, dir_id=directory_key)
            files = [{"file_id": record["file_id"], "full_path": record["full_path"], "name": record["name"]} 
                    for record in files_result]
            
            if not files:
                return {
                    "message": "No files found in directory",
                    "directory_path": directory_path,
                    "files_processed": 0,
                    "chunks_created": 0,
                    "errors": []
                }
            
            # Process each file
            chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
            embeddings_service = get_embeddings_service()
            
            results = {
                "directory_path": directory_path,
                "files_processed": 0,
                "total_chunks": 0,
                "total_files": len(files),
                "errors": []
            }
            
            for file_info in files:
                file_path = file_info["full_path"]
                file_id = file_info["file_id"]
                
                try:
                    # Check if file exists on disk
                    if not os.path.exists(file_path):
                        results["errors"].append({
                            "file": file_path,
                            "error": "File not found on disk"
                        })
                        continue
                    
                    # Read file content
                    content = read_file_content(file_path)
                    if not content or not content.strip():
                        results["errors"].append({
                            "file": file_path,
                            "error": "File is empty or could not be read as text"
                        })
                        continue
                    
                    # Chunk the content
                    chunks = chunker.chunk_file(
                        file_path=file_path,
                        content=content,
                        file_metadata={
                            "machine_id": machine_id,
                            "uploaded_at": datetime.now().isoformat()
                        }
                    )
                    
                    if not chunks:
                        results["errors"].append({
                            "file": file_path,
                            "error": "No chunks created from file"
                        })
                        continue
                    
                    # Generate embeddings for all chunks
                    chunk_texts = [chunk["text"] for chunk in chunks]
                    embeddings = embeddings_service.generate_embeddings_batch(chunk_texts)
                    
                    # Store chunks and embeddings in Neo4j
                    ops.store_document_chunks(file_id, chunks, embeddings)
                    
                    results["files_processed"] += 1
                    results["total_chunks"] += len(chunks)
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    results["errors"].append({
                        "file": file_path,
                        "error": str(e)
                    })
            
            return {
                "message": f"Processed {results['files_processed']} files with {results['total_chunks']} chunks",
                **results,
                "timestamp": datetime.now().isoformat()
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading directory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/upload/file")
async def upload_file(file_path: str, machine_id: str):
    """Upload and process a single file
    
    Args:
        file_path: Full path to the file
        machine_id: Machine ID
        
    Returns:
        Upload results
    """
    neo4j_conn = get_neo4j_connection()
    
    try:
        # Construct file key
        file_key = f"{machine_id}:{file_path}"
        
        # Check if file exists in Neo4j
        with neo4j_conn.get_session() as session:
            ops = Neo4jOperations(session)
            
            file_check_query = """
            MATCH (f:File {id: $file_id})
            RETURN f
            LIMIT 1
            """
            file_result = session.run(file_check_query, file_id=file_key)
            if not file_result.single():
                raise HTTPException(
                    status_code=404,
                    detail=f"File {file_path} not found in Neo4j. Store it first."
                )
            
            # Check if file exists on disk
            if not os.path.exists(file_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"File {file_path} not found on disk"
                )
            
            # Read file content
            content = read_file_content(file_path)
            if not content or not content.strip():
                raise HTTPException(
                    status_code=400,
                    detail="File is empty or could not be read as text"
                )
            
            # Chunk the content
            chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
            chunks = chunker.chunk_file(
                file_path=file_path,
                content=content,
                file_metadata={
                    "machine_id": machine_id,
                    "uploaded_at": datetime.now().isoformat()
                }
            )
            
            if not chunks:
                raise HTTPException(
                    status_code=400,
                    detail="No chunks created from file"
                )
            
            # Generate embeddings
            embeddings_service = get_embeddings_service()
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = embeddings_service.generate_embeddings_batch(chunk_texts)
            
            # Store chunks and embeddings
            ops.store_document_chunks(file_key, chunks, embeddings)
            
            return {
                "message": f"Successfully processed file with {len(chunks)} chunks",
                "file_path": file_path,
                "chunks_created": len(chunks),
                "embeddings_created": sum(1 for e in embeddings if e is not None),
                "timestamp": datetime.now().isoformat()
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

