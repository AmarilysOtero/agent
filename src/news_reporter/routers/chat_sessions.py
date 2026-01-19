from fastapi import APIRouter, HTTPException, Depends

from typing import List, Optional
from datetime import datetime
import os
import logging

# Optional MongoDB imports
try:
    from pymongo import MongoClient
    from bson import ObjectId
    _MONGO_AVAILABLE = True
except ImportError:
    MongoClient = None
    ObjectId = None
    _MONGO_AVAILABLE = False

from .auth import get_current_user
from ..config import Settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

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

# MongoDB connection
print("[CHAT_SESSIONS] Initializing chat sessions router...")
agent_db = None
if _MONGO_AVAILABLE:
    try:
        settings = Settings.load()
        # Use the same auth database URI for chat sessions
        MONGO_AGENT_URL = os.getenv("MONGO_AGENT_URL")
        
        if not MONGO_AGENT_URL:
            # Fall back to auth URI if agent URI not set
            MONGO_AGENT_URL = settings.auth_api_url
            print(f"[CHAT_SESSIONS] Using auth_api_url as fallback")
        
        if not MONGO_AGENT_URL:
            print("[CHAT_SESSIONS] WARNING: No MongoDB URI available for chat sessions")
            agent_db = None
        else:
            print(f"[CHAT_SESSIONS] Connecting to MongoDB with URL: {MONGO_AGENT_URL.split('@')[0]}@***")  # Hide password in logs
            # Parse connection string to extract components
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(MONGO_AGENT_URL)
            db_name = parsed.path.lstrip('/').split('?')[0] if parsed.path else 'agent_db'
            query_params = parse_qs(parsed.query)
            auth_source = query_params.get('authSource', [db_name])[0]
            
            # Use explicit parameters to avoid URL parsing issues with special characters
            password = unquote(parsed.password) if parsed.password else ""
            print(f"[CHAT_SESSIONS] Using explicit parameters (password length: {len(password)})")
            
            # Use the parsed hostname from connection string (mongo in Docker, 127.0.0.1 locally)
            mongo_host = parsed.hostname or "127.0.0.1"
            mongo_port = parsed.port or 27017
            
            print(f"[CHAT_SESSIONS] Connecting to {mongo_host}:{mongo_port}...")
            client = MongoClient(
                host=mongo_host,
                port=mongo_port,
                username=parsed.username,
                password=password,
                authSource=auth_source,
                authMechanism="SCRAM-SHA-256",
                serverSelectionTimeoutMS=5000
            )
            agent_db = client[db_name]
            agent_db.command('ping')
            # Connection already tested above
            print(f"[CHAT_SESSIONS] Successfully connected to MongoDB database: {agent_db.name}")
    except Exception as e:
        print(f"[CHAT_SESSIONS] Failed to connect to MongoDB: {e}")
        print(f"[CHAT_SESSIONS] Connection string: {MONGO_AGENT_URL.split('@')[0] if MONGO_AGENT_URL else 'NOT SET'}@***")
        print(f"[CHAT_SESSIONS] Troubleshooting:")
        print(f"  1. Verify MongoDB is running (run from any directory):")
        print(f"     docker ps | findstr rag-mongo")
        print(f"  2. Check password matches MONGO_APP_PASS (run from RAG_Infra directory):")
        print(f"     cd c:\\Alexis\\Projects\\RAG_Infra")
        print(f"     # Check .env file for MONGO_APP_PASS value")
        print(f"  3. Verify user exists (run from RAG_Infra directory):")
        print(f"     cd c:\\Alexis\\Projects\\RAG_Infra")
        print(f"     docker exec rag-mongo mongosh -u root -p rootpassword --authenticationDatabase admin")
        print(f"  4. Test connection (run from RAG_Infra directory):")
        print(f"     cd c:\\Alexis\\Projects\\RAG_Infra")
        if MONGO_AGENT_URL:
            parsed_url = MONGO_AGENT_URL.split('/')
            db_name = parsed_url[-1].split('?')[0] if parsed_url else 'agent_db'
            query_part = MONGO_AGENT_URL.split('?')[1] if '?' in MONGO_AGENT_URL else 'authSource=agent_db'
            auth_source = query_part.split('=')[1] if 'authSource=' in query_part else 'agent_db'
            print(f"     docker exec rag-mongo mongosh \"{MONGO_AGENT_URL.split('@')[0]}@127.0.0.1:27017/{db_name}?authSource={auth_source}\"")
        else:
            print(f"     docker exec rag-mongo mongosh \"mongodb://user_rw:BestRAG.2026@127.0.0.1:27017/agent_db?authSource=agent_db\"")
        agent_db = None
else:
    print("[CHAT_SESSIONS] pymongo not available - MongoDB features disabled")

# Collections
if agent_db is not None:
    sessions_collection = agent_db["chat_sessions"]
    messages_collection = agent_db["chat_messages"]
    print(f"[CHAT_SESSIONS] Collections ready: chat_sessions, chat_messages")
else:
    sessions_collection = None
    messages_collection = None
    print("[CHAT_SESSIONS] WARNING: Collections unavailable!")


@router.get("/test")
async def test_chat_router():
    """Simple test endpoint to verify router is mounted."""
    return {
        "status": "ok",
        "message": "Chat router is working",
        "db_connected": agent_db is not None,
        "sessions_collection": sessions_collection is not None
    }


@router.get("/sessions")
async def get_sessions(user: dict = Depends(get_current_user)):
    """Get all chat sessions for the current user."""
    if sessions_collection is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    user_id = str(user["_id"])
    sessions = list(sessions_collection.find({"userId": user_id}).sort("updatedAt", -1))
    
    # Convert ObjectId to string and format response
    result = []
    for session in sessions:
        result.append({
            "id": str(session["_id"]),
            "userId": session["userId"],
            "title": session.get("title", "New Chat"),
            "createdAt": session["createdAt"].isoformat() + 'Z',
            "updatedAt": session["updatedAt"].isoformat() + 'Z',
        })
    
    return result


@router.post("/sessions")
async def create_session(user: dict = Depends(get_current_user)):
    """Create a new chat session."""
    if sessions_collection is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    print(f"\n\nUser: {user}")
    
    try:
        user_id = str(user["_id"])
        print(f"User ID: {user_id}")
        
        now = datetime.utcnow()
        
        new_session = {
            "userId": user_id,
            "title": "New Chat",
            "createdAt": now,
            "updatedAt": now,
        }
        
        print(f"Inserting session: {new_session}")
        result = sessions_collection.insert_one(new_session)
        print(f"Session created with ID: {result.inserted_id}")
        
        return {
            "id": str(result.inserted_id),
            "userId": user_id,
            "title": "New Chat",
            "createdAt": now.isoformat(),
            "updatedAt": now.isoformat(),
            "messages": [],
        }
    except Exception as e:
        print(f"Error creating session: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, user: dict = Depends(get_current_user)):
    """Get a specific session with its messages."""
    if sessions_collection is None or messages_collection is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    user_id = str(user["_id"])
    
    # Verify session exists and belongs to user
    try:
        session = sessions_collection.find_one({"_id": ObjectId(session_id), "userId": user_id})
    except Exception:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get messages for this session
    messages = list(messages_collection.find({"sessionId": session_id}).sort("createdAt", 1))
    
    # Format messages
    formatted_messages = []
    for msg in messages:
        formatted_messages.append({
            "id": str(msg["_id"]),
            "sessionId": msg["sessionId"],
            "role": msg["role"],
            "content": msg["content"],
            "createdAt": msg["createdAt"].isoformat() + 'Z',
        })
    
    return {
        "id": str(session["_id"]),
        "userId": session["userId"],
        "title": session.get("title", "New Chat"),
        "createdAt": session["createdAt"].isoformat() + 'Z',
        "updatedAt": session["updatedAt"].isoformat() + 'Z',
        "messages": formatted_messages,
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user: dict = Depends(get_current_user)):
    """Delete a chat session and all its messages."""
    if sessions_collection is None or messages_collection is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    user_id = str(user["_id"])
    
    # Verify session exists and belongs to user
    try:
        session = sessions_collection.find_one({"_id": ObjectId(session_id), "userId": user_id})
    except Exception:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Delete all messages for this session
    messages_collection.delete_many({"sessionId": session_id})
    
    # Delete the session
    sessions_collection.delete_one({"_id": ObjectId(session_id)})
    
    return {"message": "Session deleted successfully"}


@router.patch("/sessions/{session_id}")
async def update_session(
    session_id: str,
    body: dict,
    user: dict = Depends(get_current_user)
):
    """Update a chat session (e.g., title)."""
    if sessions_collection is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    user_id = str(user["_id"])
    
    # Verify session exists and belongs to user
    try:
        session = sessions_collection.find_one({"_id": ObjectId(session_id), "userId": user_id})
    except Exception:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Build update document
    update_fields = {"updatedAt": datetime.utcnow()}
    
    if "title" in body and body["title"]:
        update_fields["title"] = body["title"]
    
    if len(update_fields) == 1:  # Only updatedAt
        raise HTTPException(status_code=400, detail="No valid fields to update")
    
    # Update the session
    sessions_collection.update_one(
        {"_id": ObjectId(session_id)},
        {"$set": update_fields}
    )
    
    return {
        "id": session_id,
        "title": update_fields.get("title", session.get("title")),
        "updatedAt": update_fields["updatedAt"].isoformat(),
    }


@router.post("/sessions/{session_id}/messages")
async def add_message(
    session_id: str,
    message: dict,
    user: dict = Depends(get_current_user)
):
    """Add a message to a session (user message + get AI response)."""
    print("\n\nadd_message")
    print(message)
    if sessions_collection is None or messages_collection is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    user_id = str(user["_id"])
    
    # Verify session exists and belongs to user
    try:
        session = sessions_collection.find_one({"_id": ObjectId(session_id), "userId": user_id})
    except Exception:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get the user's message content
    user_message_content = message.get("content", "")
    if not user_message_content or not user_message_content.strip():
        raise HTTPException(status_code=400, detail="Message content required")
    
    now = datetime.utcnow()
    
    # Insert user message
    user_message = {
        "sessionId": session_id,
        "userId": user_id,
        "role": "user",
        "content": user_message_content,
        "createdAt": now,
    }
    messages_collection.insert_one(user_message)
    
    # Load settings for AI orchestration
    from ..config import Settings
    from ..workflows.workflow_factory import run_graph_workflow, run_sequential_goal
    from ..workflows.workflow_persistence import get_workflow_persistence
    
    cfg = Settings.load()
    
    # Get sources from Neo4j if using Neo4j search
    sources = []
    if cfg.use_neo4j_search:
        try:
            try:
                from ..tools.neo4j_graphrag import graphrag_search
            except ImportError:
                from src.news_reporter.tools.neo4j_graphrag import graphrag_search
            
            # Extract person names from query for keyword filtering
            person_names = extract_person_names(user_message_content)
            
            # Search Neo4j GraphRAG
            search_results = graphrag_search(
                query=user_message_content,
                top_k=12,
                similarity_threshold=0.75,
                keywords=person_names if person_names else None,
                keyword_match_type="any",
                keyword_boost=0.4
            )
            
            # Filter results to require exact name match or very high similarity
            filtered_results = filter_results_by_exact_match(
                search_results,
                user_message_content,
                min_similarity=0.7
            )
            
            # Limit to top 8 after filtering
            filtered_results = filtered_results[:8]
            
            if filtered_results:
                sources = [
                    {
                        "file_name": res.get("file_name"),
                        "file_path": res.get("file_path"),
                        "directory_name": res.get("directory_name"),
                        "text": res.get("text", "")[:500] if res.get("text") else None,
                        # "similarity": res.get("similarity"),
                        # "hybrid_score": res.get("hybrid_score"),
                        # "metadata": res.get("metadata")
                        "similarity": float(res.get("similarity", 0.0)) if res.get("similarity") is not None else None,
                        "hybrid_score": float(res.get("hybrid_score", 0.0)) if res.get("hybrid_score") is not None else None,
                        # CRITICAL: Ensure metadata values are serializable (convert ObjectId/datetime to str)
                        # Failure to do this causes 500 Internal Server Error during JSON response generation
                        "metadata": {k: str(v) if isinstance(v, (ObjectId, datetime)) else v 
                                   for k, v in res.get("metadata", {}).items()} if res.get("metadata") else None
                    }
                    for res in filtered_results
                ]
        except Exception as e:
            logging.error(f"Failed to get Neo4j sources: {e}", exc_info=True)
            sources = []
    
    # Run the agent workflow - try active workflow first, fallback to sequential
    try:
        # Check for active workflow (created in agent builder)
        persistence = get_workflow_persistence()
        active_workflow = None
        
        try:
            active_workflow = persistence.get_active_workflow()
        except Exception as persistence_error:
            logger.warning(f"Failed to retrieve active workflow: {persistence_error}, using sequential workflow", exc_info=True)
        
        if active_workflow:
            # Execute custom workflow from agent builder
            logger.info(f"Using active workflow: {active_workflow.workflow_id} ({active_workflow.name})")
            try:
                assistant_response = await run_graph_workflow(
                    cfg,
                    user_message_content,
                    workflow_definition=active_workflow.graph_definition
                )
                logger.info(f"Active workflow execution completed successfully")
                print(f"Assistant Response Type: {type(assistant_response)}")
                print(f"Assistant Response Preview: {str(assistant_response)[:100]}")
            except Exception as workflow_error:
                error_msg = str(workflow_error)
                logger.error(
                    f"Active workflow '{active_workflow.workflow_id}' failed: {error_msg}, "
                    f"falling back to sequential workflow",
                    exc_info=True
                )
                # Fallback to sequential workflow
                try:
                    assistant_response = await run_sequential_goal(cfg, user_message_content)
                    logger.info("Sequential workflow fallback completed successfully")
                    print(f"Assistant Response Type: {type(assistant_response)}")
                    print(f"Assistant Response Preview: {str(assistant_response)[:100]}")
                except Exception as sequential_error:
                    logger.error(f"Sequential workflow fallback also failed: {sequential_error}", exc_info=True)
                    raise
        else:
            # No active workflow set, use sequential fallback
            logger.info("No active workflow found, using sequential workflow")
            assistant_response = await run_sequential_goal(cfg, user_message_content)
            logger.info("Sequential workflow execution completed successfully")
            print(f"Assistant Response Type: {type(assistant_response)}")
            print(f"Assistant Response Preview: {str(assistant_response)[:100]}")
    except RuntimeError as e:
        error_msg = str(e)
        # Check if it's a Foundry access error
        if "Foundry" in error_msg or "foundry" in error_msg or "AZURE_AI_PROJECT" in error_msg:
            raise HTTPException(
                status_code=503,
                detail=f"Foundry access is required but not available. Error: {error_msg}"
            )
        raise HTTPException(status_code=500, detail=f"Agent workflow failed: {error_msg}")
    except Exception as e:
        import logging
        logging.exception("[add_message] Failed to process query: %s", e)
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")
    
    # Insert assistant message with sources
    assistant_message = {
        "sessionId": session_id,
        "userId": user_id,
        "role": "assistant",
        "content": assistant_response,
        "sources": sources if sources else None,
        "createdAt": datetime.utcnow(),
    }
    result = messages_collection.insert_one(assistant_message)
    
    # Update session timestamp
    sessions_collection.update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {"updatedAt": datetime.utcnow()}}
    )

    raw_response = {
        "response": assistant_response,
        "sources": sources,
        "conversation_id": session_id,
    }
    # # Test serialization before returning (to catch the error here)
    # try:
    #     from fastapi.encoders import jsonable_encoder
    #     jsonable_encoder(response_data)
    # except Exception as e:
    #     logging.error(f"Serialization Failed! Error: {e}")
    #     # Log specifically which part failed
    #     try:
    #         jsonable_encoder(sources)
    #     except:
    #         logging.error("Sources serialization failed")
    #         # If sources failed, fallback to sanitization (so the user gets a response)
    #         sources = [
    #             {
    #                 "file_name": res.get("file_name"),
    #                 "file_path": res.get("file_path"),
    #                 "directory_name": res.get("directory_name"),
    #                 "text": res.get("text", "")[:500] if res.get("text") else None,
    #                 "similarity": float(res.get("similarity", 0.0)) if res.get("similarity") is not None else None,
    #                 "hybrid_score": float(res.get("hybrid_score", 0.0)) if res.get("hybrid_score") is not None else None,
    #                 "metadata": {k: str(v) for k, v in res.get("metadata", {}).items()} if res.get("metadata") else None
    #             }
    #             for res in (sources or [])
    #         ]
    #         response_data["sources"] = sources
    # return response_data
    
    # Ensure COMPLETE serialization safety
    safe_response = recursive_serialize(raw_response)
    
    return safe_response


def recursive_serialize(obj):
    """Recursively convert Pymongo/Datetime objects to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: recursive_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_serialize(v) for v in obj]
    elif isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat() + 'Z'  # JavaScript expects ISO format with Z for UTC
    elif hasattr(obj, "isoformat"):
        return obj.isoformat()
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        return str(obj)

