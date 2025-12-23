from fastapi import APIRouter, HTTPException, Depends

from typing import List, Optional
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId
import os
import logging

from .auth import get_current_user
from ..config import Settings

router = APIRouter(prefix="/api/workflows", tags=["workflows"])


from pydantic import BaseModel

class AgentCreateRequest(BaseModel):
    name: str
    model: Optional[str] = None
    instructions: Optional[str] = None
    description: Optional[str] = None

# Agent Management Endpoints
@router.get("/agent/list")
async def list_foundry_agents():
    """
    List all available workflow agents from Azure AI Foundry environment.
    
    Returns:
        List of agents with id, name, model, description, etc.
    """
    try:
        from ..agent_manager import list_agents
        agents = list_agents()
        
        print(f"Returned {len(agents)} workflow agents: {[a.get('name') for a in agents]}")
        return agents
    except ValueError as e:
        # Configuration error
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.exception("[agents/list] Failed to list workflow agents: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")

@router.post("/agent")
async def create_new_foundry_agent(request: AgentCreateRequest):
    """
    Create a new agent in Azure AI Foundry.
    """
    try:
        from ..agent_manager import create_agent
        agent = create_agent(
            name=request.name,
            model=request.model,
            instructions=request.instructions,
            description=request.description
        )
        return agent
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.exception("[agents/create] Failed to create agent: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")
