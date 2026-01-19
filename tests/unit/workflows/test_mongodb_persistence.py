"""Unit tests for MongoDB Workflow Persistence Integration"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from src.news_reporter.workflows.workflow_persistence import (
    WorkflowPersistence,
    WorkflowRecord,
    ExecutionRecord,
    WorkflowStatus,
    get_workflow_persistence
)
from src.news_reporter.workflows.mongo_backend import MongoWorkflowBackend


class TestMongoWorkflowBackend:
    """Test MongoDB backend functionality with mocks"""
    
    @pytest.fixture
    def mock_mongo_client(self):
        """Create a mock MongoDB client"""
        client = MagicMock()
        db = MagicMock()
        workflows_collection = MagicMock()
        executions_collection = MagicMock()
        
        db.__getitem__ = MagicMock(side_effect=lambda name: {
            "workflows": workflows_collection,
            "workflow_executions": executions_collection
        }[name])
        
        client.__getitem__ = MagicMock(return_value=db)
        client.admin.command = MagicMock(return_value={"ok": 1})
        
        return client, db, workflows_collection, executions_collection
    
    @pytest.fixture
    def backend(self, mock_mongo_client):
        """Create MongoDB backend with mocked client"""
        client, db, workflows_collection, executions_collection = mock_mongo_client
        
        with patch("src.news_reporter.workflows.mongo_backend.MongoClient", return_value=client):
            backend = MongoWorkflowBackend("mongodb://test:test@localhost:27017/workflow_db?authSource=workflow_db")
            backend.client = client
            backend.db = db
            backend.workflows_collection = workflows_collection
            backend.executions_collection = executions_collection
            backend._connected = True
            return backend
    
    @pytest.fixture
    def sample_workflow(self):
        """Create a sample workflow record"""
        return WorkflowRecord(
            workflow_id="test-workflow-1",
            name="Test Workflow",
            description="A test workflow",
            graph_definition={"nodes": [], "edges": []},
            version="1.0.0",
            tags=["test", "demo"],
            is_active=True
        )
    
    @pytest.fixture
    def sample_execution(self):
        """Create a sample execution record"""
        return ExecutionRecord(
            execution_id="exec-1",
            workflow_id="test-workflow-1",
            run_id="run-1",
            goal="Test goal",
            status=WorkflowStatus.COMPLETED,
            result="Success",
            started_at=datetime.now(),
            completed_at=datetime.now()
        )
    
    def test_save_workflow(self, backend, sample_workflow, mock_mongo_client):
        """Test saving a workflow to MongoDB"""
        _, _, workflows_collection, _ = mock_mongo_client
        workflows_collection.update_one.return_value = MagicMock(modified_count=1)
        
        result = backend.save_workflow(sample_workflow)
        
        assert result is True
        workflows_collection.update_one.assert_called_once()
        call_args = workflows_collection.update_one.call_args
        assert call_args[0][0]["workflow_id"] == "test-workflow-1"
        assert call_args[1]["upsert"] is True
    
    def test_get_workflow(self, backend, sample_workflow, mock_mongo_client):
        """Test retrieving a workflow from MongoDB"""
        _, _, workflows_collection, _ = mock_mongo_client
        workflows_collection.find_one.return_value = {
            "workflow_id": "test-workflow-1",
            "name": "Test Workflow",
            "description": "A test workflow",
            "graph_definition": {"nodes": [], "edges": []},
            "version": "1.0.0",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "tags": ["test", "demo"],
            "is_active": True
        }
        
        result = backend.get_workflow("test-workflow-1")
        
        assert result is not None
        assert result.workflow_id == "test-workflow-1"
        assert result.name == "Test Workflow"
        workflows_collection.find_one.assert_called_once_with({"workflow_id": "test-workflow-1"})
    
    def test_list_workflows_with_filters(self, backend, mock_mongo_client):
        """Test listing workflows with tag and is_active filters"""
        _, _, workflows_collection, _ = mock_mongo_client
        workflows_collection.find.return_value = [
            {
                "workflow_id": "wf-1",
                "name": "Workflow 1",
                "tags": ["test"],
                "is_active": True,
                "created_at": datetime.now(),
                "graph_definition": {},
                "version": "1.0.0"
            }
        ]
        
        results = backend.list_workflows(tags=["test"], is_active=True)
        
        assert len(results) == 1
        assert results[0].workflow_id == "wf-1"
        workflows_collection.find.assert_called_once()
    
    def test_save_execution(self, backend, sample_execution, mock_mongo_client):
        """Test saving an execution to MongoDB"""
        _, _, _, executions_collection = mock_mongo_client
        executions_collection.insert_one.return_value = MagicMock(inserted_id="exec-1")
        
        result = backend.save_execution(sample_execution)
        
        assert result is True
        executions_collection.insert_one.assert_called_once()
    
    def test_get_execution(self, backend, sample_execution, mock_mongo_client):
        """Test retrieving an execution from MongoDB"""
        _, _, _, executions_collection = mock_mongo_client
        executions_collection.find_one.return_value = {
            "execution_id": "exec-1",
            "workflow_id": "test-workflow-1",
            "run_id": "run-1",
            "goal": "Test goal",
            "status": "completed",
            "result": "Success",
            "started_at": datetime.now(),
            "completed_at": datetime.now()
        }
        
        result = backend.get_execution("exec-1")
        
        assert result is not None
        assert result.execution_id == "exec-1"
        assert result.status == WorkflowStatus.COMPLETED
    
    def test_list_executions_with_filters(self, backend, mock_mongo_client):
        """Test listing executions with workflow_id and status filters"""
        _, _, _, executions_collection = mock_mongo_client
        executions_collection.find.return_value = [
            {
                "execution_id": "exec-1",
                "workflow_id": "wf-1",
                "run_id": "run-1",
                "goal": "Test",
                "status": "completed",
                "started_at": datetime.now()
            }
        ]
        
        results = backend.list_executions(
            workflow_id="wf-1",
            status=WorkflowStatus.COMPLETED,
            limit=10
        )
        
        assert len(results) == 1
        assert results[0].execution_id == "exec-1"
        executions_collection.find.assert_called_once()
    
    def test_update_execution_status(self, backend, mock_mongo_client):
        """Test updating execution status"""
        _, _, _, executions_collection = mock_mongo_client
        executions_collection.update_one.return_value = MagicMock(modified_count=1)
        
        result = backend.update_execution_status(
            execution_id="exec-1",
            status=WorkflowStatus.FAILED,
            error="Test error",
            metrics={"duration": 100}
        )
        
        assert result is True
        executions_collection.update_one.assert_called_once()
        call_args = executions_collection.update_one.call_args
        assert call_args[0][0]["execution_id"] == "exec-1"
        update_fields = call_args[0][1]["$set"]
        assert update_fields["status"] == "failed"
        assert update_fields["error"] == "Test error"


class TestWorkflowPersistenceIntegration:
    """Test WorkflowPersistence with MongoDB backend integration"""
    
    @pytest.fixture
    def mock_backend(self):
        """Create a mock MongoDB backend"""
        backend = MagicMock(spec=MongoWorkflowBackend)
        backend.save_workflow = MagicMock(return_value=True)
        backend.get_workflow = MagicMock(return_value=None)
        backend.list_workflows = MagicMock(return_value=[])
        backend.delete_workflow = MagicMock(return_value=True)
        backend.save_execution = MagicMock(return_value=True)
        backend.get_execution = MagicMock(return_value=None)
        backend.list_executions = MagicMock(return_value=[])
        backend.update_execution_status = MagicMock(return_value=True)
        return backend
    
    def test_persistence_with_mongo_backend(self, mock_backend):
        """Test WorkflowPersistence uses MongoDB backend when available"""
        persistence = WorkflowPersistence(storage_backend=mock_backend)
        
        workflow = WorkflowRecord(
            workflow_id="test-1",
            name="Test",
            graph_definition={}
        )
        
        persistence.save_workflow(workflow)
        
        # Should call MongoDB backend
        mock_backend.save_workflow.assert_called_once()
        # Should also cache in memory
        assert "test-1" in persistence._workflows
    
    def test_persistence_fallback_to_memory(self):
        """Test WorkflowPersistence falls back to in-memory when backend fails"""
        mock_backend = MagicMock()
        mock_backend.save_workflow = MagicMock(return_value=False)  # Simulate failure
        
        persistence = WorkflowPersistence(storage_backend=mock_backend)
        
        workflow = WorkflowRecord(
            workflow_id="test-1",
            name="Test",
            graph_definition={}
        )
        
        persistence.save_workflow(workflow)
        
        # Should still save in memory even if MongoDB fails
        assert "test-1" in persistence._workflows
        assert persistence._workflows["test-1"].name == "Test"
    
    def test_persistence_without_backend(self):
        """Test WorkflowPersistence works with in-memory only"""
        persistence = WorkflowPersistence(storage_backend=None)
        
        workflow = WorkflowRecord(
            workflow_id="test-1",
            name="Test",
            graph_definition={}
        )
        
        persistence.save_workflow(workflow)
        retrieved = persistence.get_workflow("test-1")
        
        assert retrieved is not None
        assert retrieved.workflow_id == "test-1"
    
    def test_get_workflow_from_mongo_then_cache(self, mock_backend):
        """Test that workflows retrieved from MongoDB are cached"""
        workflow = WorkflowRecord(
            workflow_id="test-1",
            name="Test",
            graph_definition={}
        )
        mock_backend.get_workflow.return_value = workflow
        
        persistence = WorkflowPersistence(storage_backend=mock_backend)
        
        # First call - should query MongoDB
        result1 = persistence.get_workflow("test-1")
        assert result1 is not None
        mock_backend.get_workflow.assert_called_once_with("test-1")
        
        # Second call - should use cache
        mock_backend.get_workflow.reset_mock()
        result2 = persistence.get_workflow("test-1")
        assert result2 is not None
    
    def test_list_workflows_with_filters(self, mock_backend):
        """Test listing workflows with filters"""
        workflows = [
            WorkflowRecord(workflow_id="wf-1", name="WF1", tags=["test"], is_active=True, graph_definition={}),
            WorkflowRecord(workflow_id="wf-2", name="WF2", tags=["demo"], is_active=True, graph_definition={})
        ]
        mock_backend.list_workflows.return_value = workflows
        
        persistence = WorkflowPersistence(storage_backend=mock_backend)
        
        results = persistence.list_workflows(tags=["test"], is_active=True)
        
        assert len(results) == 2
        mock_backend.list_workflows.assert_called_once_with(tags=["test"], is_active=True)
    
    def test_save_execution_tracks_workflow_executions(self, mock_backend):
        """Test that saving execution tracks it per workflow"""
        mock_backend.save_execution.return_value = True
        
        persistence = WorkflowPersistence(storage_backend=mock_backend)
        
        execution = ExecutionRecord(
            execution_id="exec-1",
            workflow_id="wf-1",
            run_id="run-1",
            goal="Test",
            status=WorkflowStatus.RUNNING
        )
        
        persistence.save_execution(execution)
        
        # Should track execution per workflow
        assert "wf-1" in persistence._workflow_executions
        assert "exec-1" in persistence._workflow_executions["wf-1"]


class TestGetWorkflowPersistence:
    """Test get_workflow_persistence singleton"""
    
    def test_singleton_pattern(self):
        """Test that get_workflow_persistence returns the same instance"""
        # Reset global instance for testing
        import src.news_reporter.workflows.workflow_persistence as wp_module
        wp_module._global_persistence = None
        
        instance1 = get_workflow_persistence()
        instance2 = get_workflow_persistence()
        
        assert instance1 is instance2
    
    @patch("src.news_reporter.workflows.workflow_persistence.MongoWorkflowBackend")
    def test_initializes_mongo_on_first_call(self, mock_backend_class):
        """Test that MongoDB backend is initialized on first call"""
        # Reset global instance
        import src.news_reporter.workflows.workflow_persistence as wp_module
        wp_module._global_persistence = None
        
        mock_backend = MagicMock()
        mock_backend.connect.return_value = True
        mock_backend_class.return_value = mock_backend
        
        # Mock MONGO_BACKEND_AVAILABLE
        with patch("src.news_reporter.workflows.workflow_persistence._MONGO_BACKEND_AVAILABLE", True):
            persistence = get_workflow_persistence()
            
            # Should have attempted to initialize MongoDB
            mock_backend_class.assert_called_once()
            mock_backend.connect.assert_called_once()
    
    @patch("src.news_reporter.workflows.workflow_persistence.MongoWorkflowBackend")
    def test_fallback_when_mongo_unavailable(self, mock_backend_class):
        """Test fallback to in-memory when MongoDB unavailable"""
        # Reset global instance
        import src.news_reporter.workflows.workflow_persistence as wp_module
        wp_module._global_persistence = None
        
        mock_backend = MagicMock()
        mock_backend.connect.return_value = False  # Connection fails
        mock_backend_class.return_value = mock_backend
        
        # Mock MONGO_BACKEND_AVAILABLE
        with patch("src.news_reporter.workflows.workflow_persistence._MONGO_BACKEND_AVAILABLE", True):
            persistence = get_workflow_persistence()
            
            # Should have no backend (fell back to in-memory)
            assert persistence.storage_backend is None


class TestActiveWorkflowMongoBackend:
    """Test active workflow functionality in MongoDB backend"""
    
    @pytest.fixture
    def backend(self, mock_mongo_client):
        """Create MongoDB backend with mocked client"""
        client, db, workflows_collection, executions_collection = mock_mongo_client
        
        with patch("src.news_reporter.workflows.mongo_backend.MongoClient", return_value=client):
            backend = MongoWorkflowBackend("mongodb://test:test@localhost:27017/workflow_db?authSource=workflow_db")
            backend.client = client
            backend.db = db
            backend.workflows_collection = workflows_collection
            backend.executions_collection = executions_collection
            backend._connected = True
            return backend
    
    def test_get_active_workflow(self, backend, mock_mongo_client):
        """Test getting active workflow from MongoDB"""
        _, _, workflows_collection, _ = mock_mongo_client
        
        # Mock active workflow document
        active_doc = {
            "workflow_id": "active_workflow",
            "name": "Active Workflow",
            "is_active": True,
            "graph_definition": {},
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        workflows_collection.find_one = MagicMock(return_value=active_doc)
        
        active = backend.get_active_workflow()
        
        assert active is not None
        assert active.workflow_id == "active_workflow"
        assert active.is_active is True
        workflows_collection.find_one.assert_called_once_with(
            {"is_active": True},
            sort=[("updated_at", -1)]
        )
    
    def test_set_active_workflow(self, backend, mock_mongo_client):
        """Test setting a workflow as active in MongoDB"""
        _, _, workflows_collection, _ = mock_mongo_client
        
        # Mock workflow exists
        workflow_doc = {
            "workflow_id": "workflow1",
            "name": "Workflow 1",
            "is_active": False,
            "graph_definition": {}
        }
        workflows_collection.find_one = MagicMock(return_value=workflow_doc)
        workflows_collection.update_many = MagicMock(return_value=MagicMock(modified_count=1))
        workflows_collection.update_one = MagicMock(return_value=MagicMock(modified_count=1))
        
        success = backend.set_active_workflow("workflow1")
        
        assert success is True
        # Should deactivate other workflows
        workflows_collection.update_many.assert_called_once()
        # Should activate the specified workflow
        workflows_collection.update_one.assert_called_once()
    
    def test_set_active_workflow_not_found(self, backend, mock_mongo_client):
        """Test setting non-existent workflow as active"""
        _, _, workflows_collection, _ = mock_mongo_client
        
        workflows_collection.find_one = MagicMock(return_value=None)
        
        success = backend.set_active_workflow("non_existent")
        
        assert success is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
