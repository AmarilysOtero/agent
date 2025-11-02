"""Machine registration endpoints"""

import uuid
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, Request

from database import get_neo4j_connection
from utils import generate_fingerprint

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/register-machine")
async def register_machine(http_request: Request):
    """Register a machine and return a persistent machineId
    
    Uses fingerprint (IP + user-agent hash) to detect if this is a returning client.
    If fingerprint matches an existing machine, returns that machineId.
    Otherwise, creates a new machineId and stores it in Neo4j.
    """
    neo4j_conn = get_neo4j_connection()
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


