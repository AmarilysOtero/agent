"""MongoDB Backend for Workflow Persistence"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import os
import logging
from datetime import datetime
from urllib.parse import urlparse, parse_qs, unquote

# Optional MongoDB imports
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, OperationFailure, DuplicateKeyError
    _MONGO_AVAILABLE = True
except ImportError:
    MongoClient = None
    _MONGO_AVAILABLE = False

from .workflow_persistence import WorkflowRecord, ExecutionRecord, WorkflowStatus

logger = logging.getLogger(__name__)


class MongoWorkflowBackend:
    """MongoDB backend for workflow persistence"""
    
    def __init__(self, mongo_url: Optional[str] = None):
        """
        Initialize MongoDB backend.
        
        Args:
            mongo_url: MongoDB connection URL. If None, uses MONGO_WORKFLOW_URL env var.
                     Falls back to constructing from MONGO_AGENT_URL if not set.
        """
        self.mongo_url = mongo_url or os.getenv("MONGO_WORKFLOW_URL")
        self.client: Optional[MongoClient] = None
        self.db = None
        self.workflows_collection = None
        self.executions_collection = None
        self._connected = False
        
        # If MONGO_WORKFLOW_URL not set, try to construct from MONGO_AGENT_URL
        if not self.mongo_url:
            agent_url = os.getenv("MONGO_AGENT_URL")
            if agent_url:
                # Replace agent_db with workflow_db in the URL
                self.mongo_url = agent_url.replace("/agent_db?", "/workflow_db?").replace("/agent_db", "/workflow_db")
                if "workflow_db?" not in self.mongo_url and not self.mongo_url.endswith("/workflow_db"):
                    # Handle case where URL doesn't have query params
                    if "?" in agent_url:
                        self.mongo_url = agent_url.split("?")[0].replace("/agent_db", "/workflow_db") + "?" + agent_url.split("?")[1]
                    else:
                        self.mongo_url = agent_url.replace("/agent_db", "/workflow_db") + "?authSource=workflow_db"
                logger.info("Constructed MONGO_WORKFLOW_URL from MONGO_AGENT_URL")
    
    def connect(self) -> bool:
        """
        Establish MongoDB connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not _MONGO_AVAILABLE:
            logger.warning("pymongo not available - MongoDB backend disabled")
            return False
        
        if self._connected and self.client is not None:
            try:
                # Test connection
                self.client.admin.command('ping')
                return True
            except Exception:
                # Connection lost, reconnect
                self._connected = False
        
        if not self.mongo_url:
            logger.warning("MONGO_WORKFLOW_URL not set - MongoDB backend disabled")
            return False
        
        try:
            logger.info(f"Connecting to MongoDB workflow database...")
            # Parse connection string
            parsed = urlparse(self.mongo_url)
            db_name = parsed.path.lstrip('/').split('?')[0] if parsed.path else 'workflow_db'
            query_params = parse_qs(parsed.query)
            auth_source = query_params.get('authSource', [db_name])[0]
            
            # Extract password (handle URL encoding)
            password = unquote(parsed.password) if parsed.password else ""
            
            # Extract host and port
            mongo_host = parsed.hostname or "127.0.0.1"
            mongo_port = parsed.port or 27017
            
            # Create client with explicit parameters
            self.client = MongoClient(
                host=mongo_host,
                port=mongo_port,
                username=parsed.username,
                password=password,
                authSource=auth_source,
                authMechanism="SCRAM-SHA-256",
                serverSelectionTimeoutMS=5000
            )
            
            # Get database and collections
            self.db = self.client[db_name]
            
            # Test connection
            self.client.admin.command('ping')
            
            # Initialize collections
            self.workflows_collection = self.db["workflows"]
            self.executions_collection = self.db["workflow_executions"]
            
            self._connected = True
            logger.info(f"Successfully connected to MongoDB database: {self.db.name}")
            return True
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}", exc_info=True)
            self._connected = False
            return False
    
    def disconnect(self) -> None:
        """Close MongoDB connection"""
        if self.client:
            try:
                self.client.close()
                logger.info("Disconnected from MongoDB")
            except Exception as e:
                logger.warning(f"Error disconnecting from MongoDB: {e}")
            finally:
                self.client = None
                self.db = None
                self.workflows_collection = None
                self.executions_collection = None
                self._connected = False
    
    def _ensure_connected(self) -> bool:
        """Ensure MongoDB connection is active"""
        if not self._connected:
            return self.connect()
        return True
    
    def _workflow_to_dict(self, workflow: WorkflowRecord) -> Dict[str, Any]:
        """Convert WorkflowRecord to MongoDB document"""
        return {
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "description": workflow.description,
            "graph_definition": workflow.graph_definition,
            "version": workflow.version,
            "created_at": workflow.created_at,
            "updated_at": workflow.updated_at,
            "created_by": workflow.created_by,
            "tags": workflow.tags,
            "is_active": workflow.is_active
        }
    
    def _dict_to_workflow(self, doc: Dict[str, Any]) -> WorkflowRecord:
        """Convert MongoDB document to WorkflowRecord"""
        # Handle datetime conversion
        created_at = doc.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        elif created_at is None:
            created_at = None
        
        updated_at = doc.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
        elif updated_at is None:
            updated_at = None
        
        return WorkflowRecord(
            workflow_id=doc["workflow_id"],
            name=doc["name"],
            description=doc.get("description"),
            graph_definition=doc.get("graph_definition", {}),
            version=doc.get("version", "1.0.0"),
            created_at=created_at,
            updated_at=updated_at,
            created_by=doc.get("created_by"),
            tags=doc.get("tags", []),
            is_active=doc.get("is_active", True)
        )
    
    def _execution_to_dict(self, execution: ExecutionRecord) -> Dict[str, Any]:
        """Convert ExecutionRecord to MongoDB document"""
        return {
            "execution_id": execution.execution_id,
            "workflow_id": execution.workflow_id,
            "run_id": execution.run_id,
            "goal": execution.goal,
            "status": execution.status.value,  # Store enum as string
            "result": execution.result,
            "error": execution.error,
            "metrics": execution.metrics,
            "state_snapshot": execution.state_snapshot,
            "started_at": execution.started_at,
            "completed_at": execution.completed_at,
            "created_by": execution.created_by
        }
    
    def _dict_to_execution(self, doc: Dict[str, Any]) -> ExecutionRecord:
        """Convert MongoDB document to ExecutionRecord"""
        # Handle datetime conversion
        started_at = doc.get("started_at")
        if isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
        elif started_at is None:
            started_at = None
        
        completed_at = doc.get("completed_at")
        if isinstance(completed_at, str):
            completed_at = datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
        elif completed_at is None:
            completed_at = None
        
        # Handle status enum conversion
        status_str = doc.get("status", "pending")
        try:
            status = WorkflowStatus(status_str)
        except ValueError:
            logger.warning(f"Invalid status value: {status_str}, defaulting to PENDING")
            status = WorkflowStatus.PENDING
        
        return ExecutionRecord(
            execution_id=doc["execution_id"],
            workflow_id=doc["workflow_id"],
            run_id=doc["run_id"],
            goal=doc["goal"],
            status=status,
            result=doc.get("result"),
            error=doc.get("error"),
            metrics=doc.get("metrics"),
            state_snapshot=doc.get("state_snapshot"),
            started_at=started_at,
            completed_at=completed_at,
            created_by=doc.get("created_by")
        )
    
    def save_workflow(self, workflow: WorkflowRecord) -> bool:
        """
        Save a workflow (upsert by workflow_id).
        
        Args:
            workflow: WorkflowRecord to save
            
        Returns:
            True if successful, False otherwise
        """
        if not self._ensure_connected():
            return False
        
        try:
            workflow_dict = self._workflow_to_dict(workflow)
            # Upsert by workflow_id
            self.workflows_collection.update_one(
                {"workflow_id": workflow.workflow_id},
                {"$set": workflow_dict},
                upsert=True
            )
            logger.debug(f"Saved workflow to MongoDB: {workflow.workflow_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save workflow {workflow.workflow_id}: {e}", exc_info=True)
            return False
    
    def get_workflow(self, workflow_id: str) -> Optional[WorkflowRecord]:
        """
        Get a workflow by ID.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            WorkflowRecord if found, None otherwise
        """
        if not self._ensure_connected():
            return None
        
        try:
            doc = self.workflows_collection.find_one({"workflow_id": workflow_id})
            if doc:
                return self._dict_to_workflow(doc)
            return None
        except Exception as e:
            logger.error(f"Failed to get workflow {workflow_id}: {e}", exc_info=True)
            return None
    
    def list_workflows(
        self,
        tags: Optional[List[str]] = None,
        is_active: Optional[bool] = None
    ) -> List[WorkflowRecord]:
        """
        List workflows with optional filtering.
        
        Args:
            tags: Optional list of tags to filter by
            is_active: Optional filter by active status
            
        Returns:
            List of WorkflowRecord objects
        """
        if not self._ensure_connected():
            return []
        
        try:
            query = {}
            if tags:
                query["tags"] = {"$in": tags}
            if is_active is not None:
                query["is_active"] = is_active
            
            cursor = self.workflows_collection.find(query).sort("created_at", -1)
            workflows = []
            for doc in cursor:
                workflows.append(self._dict_to_workflow(doc))
            return workflows
        except Exception as e:
            logger.error(f"Failed to list workflows: {e}", exc_info=True)
            return []
    
    def delete_workflow(self, workflow_id: str) -> bool:
        """
        Soft delete a workflow (set is_active=False).
        
        Args:
            workflow_id: Workflow ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self._ensure_connected():
            return False
        
        try:
            result = self.workflows_collection.update_one(
                {"workflow_id": workflow_id},
                {"$set": {"is_active": False, "updated_at": datetime.now()}}
            )
            if result.modified_count > 0:
                logger.info(f"Soft deleted workflow: {workflow_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete workflow {workflow_id}: {e}", exc_info=True)
            return False
    
    def save_execution(self, execution: ExecutionRecord) -> bool:
        """
        Save an execution record.
        
        Args:
            execution: ExecutionRecord to save
            
        Returns:
            True if successful, False otherwise
        """
        if not self._ensure_connected():
            return False
        
        try:
            execution_dict = self._execution_to_dict(execution)
            # Insert execution (execution_id should be unique)
            self.executions_collection.insert_one(execution_dict)
            logger.debug(f"Saved execution to MongoDB: {execution.execution_id}")
            return True
        except DuplicateKeyError:
            # Execution already exists, update it instead
            logger.debug(f"Execution {execution.execution_id} already exists, updating...")
            try:
                self.executions_collection.update_one(
                    {"execution_id": execution.execution_id},
                    {"$set": execution_dict}
                )
                return True
            except Exception as e:
                logger.error(f"Failed to update execution {execution.execution_id}: {e}", exc_info=True)
                return False
        except Exception as e:
            logger.error(f"Failed to save execution {execution.execution_id}: {e}", exc_info=True)
            return False
    
    def get_execution(self, execution_id: str) -> Optional[ExecutionRecord]:
        """
        Get an execution by execution_id or run_id.
        
        Args:
            execution_id: Execution ID or run_id
            
        Returns:
            ExecutionRecord if found, None otherwise
        """
        if not self._ensure_connected():
            return None
        
        try:
            # Try execution_id first
            doc = self.executions_collection.find_one({"execution_id": execution_id})
            if not doc:
                # Try run_id
                doc = self.executions_collection.find_one({"run_id": execution_id})
            if doc:
                return self._dict_to_execution(doc)
            return None
        except Exception as e:
            logger.error(f"Failed to get execution {execution_id}: {e}", exc_info=True)
            return None
    
    def list_executions(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[WorkflowStatus] = None,
        limit: int = 100
    ) -> List[ExecutionRecord]:
        """
        List executions with optional filtering.
        
        Args:
            workflow_id: Optional filter by workflow_id
            status: Optional filter by status
            limit: Maximum number of results
            
        Returns:
            List of ExecutionRecord objects, sorted by started_at descending
        """
        if not self._ensure_connected():
            return []
        
        try:
            query = {}
            if workflow_id:
                query["workflow_id"] = workflow_id
            if status:
                query["status"] = status.value
            
            cursor = self.executions_collection.find(query).sort("started_at", -1).limit(limit)
            executions = []
            for doc in cursor:
                executions.append(self._dict_to_execution(doc))
            return executions
        except Exception as e:
            logger.error(f"Failed to list executions: {e}", exc_info=True)
            return []
    
    def update_execution_status(
        self,
        execution_id: str,
        status: WorkflowStatus,
        result: Optional[str] = None,
        error: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update execution status and results.
        
        Args:
            execution_id: Execution ID
            status: New status
            result: Optional result string
            error: Optional error string
            metrics: Optional metrics dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if not self._ensure_connected():
            return False
        
        try:
            update_fields = {
                "status": status.value,
                "completed_at": datetime.now()
            }
            if result is not None:
                update_fields["result"] = result
            if error is not None:
                update_fields["error"] = error
            if metrics is not None:
                update_fields["metrics"] = metrics
            
            result = self.executions_collection.update_one(
                {"execution_id": execution_id},
                {"$set": update_fields}
            )
            if result.modified_count > 0:
                logger.debug(f"Updated execution {execution_id} status to {status.value}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to update execution {execution_id}: {e}", exc_info=True)
            return False
