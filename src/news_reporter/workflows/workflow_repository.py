"""MongoDB repository for Workflow and WorkflowRun collections"""
from __future__ import annotations
import logging
from typing import List, Optional
from datetime import datetime, timezone
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection

from ..models.workflow import Workflow, WorkflowRun, WorkflowGraph, NodeResult

logger = logging.getLogger(__name__)


class WorkflowRepository:
    """Repository for workflow backend (MongoDB async)"""
    
    def __init__(self, workflow_url: str):
        """
        Initialize workflow repository with MongoDB connection
        
        Args:
            workflow_url: MongoDB connection string
        """
        self.client: AsyncIOMotorClient = AsyncIOMotorClient(workflow_url)
        self.db: AsyncIOMotorDatabase = self.client.workflow_db
        self.workflows: AsyncIOMotorCollection = self.db.workflows
        self.runs: AsyncIOMotorCollection = self.db.workflow_runs
        logger.info(f"Workflow repository initialized (mongodb: {workflow_url})")
    
    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("Workflow repository connection closed")
    
    # ===== Workflow CRUD =====
    
    async def create_workflow(self, workflow: Workflow) -> Workflow:
        """Create a new workflow"""
        doc = workflow.dict(exclude={"id"}, by_alias=False)
        result = await self.workflows.insert_one(doc)
        workflow.id = str(result.inserted_id)
        logger.info(f"Created workflow {workflow.id} for user {workflow.userId}")
        return workflow
    
    async def get_workflow(self, workflow_id: str, user_id: str) -> Optional[Workflow]:
        """Get workflow by ID with user scoping"""
        try:
            doc = await self.workflows.find_one({
                "_id": ObjectId(workflow_id),
                "userId": user_id
            })
            if not doc:
                return None
            doc["id"] = str(doc.pop("_id"))
            return Workflow(**doc)
        except Exception as e:
            logger.error(f"Failed to get workflow {workflow_id}: {e}")
            return None
    
    async def list_workflows(self, user_id: str) -> List[Workflow]:
        """List all workflows for a user"""
        try:
            cursor = self.workflows.find({"userId": user_id}).sort("updatedAt", -1)
            workflows = []
            async for doc in cursor:
                doc["id"] = str(doc.pop("_id"))
                workflows.append(Workflow(**doc))
            return workflows
        except Exception as e:
            logger.error(f"Failed to list workflows for user {user_id}: {e}")
            return []
    
    async def update_workflow(self, workflow_id: str, user_id: str, updates: dict) -> bool:
        """Update workflow fields"""
        try:
            updates["updatedAt"] = datetime.now(timezone.utc)  # Consistency patch: timezone-aware
            result = await self.workflows.update_one(
                {"_id": ObjectId(workflow_id), "userId": user_id},
                {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update workflow {workflow_id}: {e}")
            return False
    
    async def delete_workflow(self, workflow_id: str, user_id: str) -> bool:
        """Delete workflow"""
        try:
            result = await self.workflows.delete_one({
                "_id": ObjectId(workflow_id),
                "userId": user_id
            })
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Failed to delete workflow {workflow_id}: {e}")
            return False
    
    # ===== Workflow Run CRUD =====
    
    async def create_run(self, run: WorkflowRun) -> WorkflowRun:
        """Create a new workflow run"""
        doc = run.dict(exclude={"id"}, by_alias=False)
        # Convert NodeResult objects to dicts
        if "nodeResults" in doc:
            doc["nodeResults"] = {
                k: v.dict() if isinstance(v, NodeResult) else v
                for k, v in doc["nodeResults"].items()
            }
        result = await self.runs.insert_one(doc)
        run.id = str(result.inserted_id)
        logger.info(f"Created workflow run {run.id} for workflow {run.workflowId}")
        return run
    
    async def get_run(self, run_id: str, user_id: str) -> Optional[WorkflowRun]:
        """Get run by ID with user scoping"""
        try:
            doc = await self.runs.find_one({
                "_id": ObjectId(run_id),
                "userId": user_id
            })
            if not doc:
                return None
            doc["id"] = str(doc.pop("_id"))
            # Convert nodeResults dicts to NodeResult objects
            if "nodeResults" in doc:
                doc["nodeResults"] = {
                    k: NodeResult(**v) if isinstance(v, dict) else v
                    for k, v in doc["nodeResults"].items()
                }
            return WorkflowRun(**doc)
        except Exception as e:
            logger.error(f"Failed to get run {run_id}: {e}")
            return None
    
    async def update_run_status(self, run_id: str, user_id: str, status: str, **extra_fields) -> bool:
        """Update run status and optional fields (startedAt, completedAt, error)
        
        PR5 Fix 1: Enforce userId scoping on updates
        """
        try:
            updates = {"status": status}
            updates.update(extra_fields)
            result = await self.runs.update_one(
                {"_id": ObjectId(run_id), "userId": user_id},
                {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update run {run_id} status: {e}")
            return False
    
    async def update_run_heartbeat(self, run_id: str, user_id: str) -> bool:
        """Update run heartbeat timestamp
        
        PR5 Fix 1: Enforce userId scoping on updates
        """
        try:
            result = await self.runs.update_one(
                {"_id": ObjectId(run_id), "userId": user_id},
                {"$set": {"heartbeatAt": datetime.now(timezone.utc)}}  # Consistency patch: timezone-aware
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update run {run_id} heartbeat: {e}")
            return False
    
    async def persist_node_result(self, run_id: str, user_id: str, node_id: str, result: NodeResult) -> bool:
        """Persist node result to run document
        
        PR5 Fix 1: Enforce userId scoping on updates
        """
        try:
            result_dict = result.dict()
            update_result = await self.runs.update_one(
                {"_id": ObjectId(run_id), "userId": user_id},
                {"$set": {f"nodeResults.{node_id}": result_dict}}
            )
            return update_result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to persist node {node_id} result for run {run_id}: {e}")
            return False
    
    async def get_node_results(self, run_id: str, user_id: str) -> dict:
        """
        Get all node results for a run (PR5)
        
        Args:
            run_id: Run ID
            user_id: User ID for ownership validation
            
        Returns:
            Dictionary mapping nodeId -> NodeResult
        """
        try:
            run = await self.get_run(run_id, user_id)
            if not run or not run.nodeResults:
                return {}
            return run.nodeResults
        except Exception as e:
            logger.error(f"Failed to get node results for run {run_id}: {e}")
            return {}
