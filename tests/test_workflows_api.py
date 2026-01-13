"""
PR5 Integration Tests: Workflow REST API

Tests for all 8 workflow endpoints including user scoping, validation gating,
and execution lifecycle.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timezone

from src.news_reporter.api import app
from src.news_reporter.models.workflow import Workflow, WorkflowRun, WorkflowGraph, NodeResult
from src.news_reporter.dependencies.auth import UserPrincipal


@pytest.fixture
def mock_repo():
    """Mock WorkflowRepository"""
    repo = AsyncMock()
    repo.create_workflow = AsyncMock()
    repo.list_workflows = AsyncMock(return_value=[])
    repo.get_workflow = AsyncMock()
    repo.update_workflow = AsyncMock(return_value=True)
    repo.create_run = AsyncMock()
    repo.get_run = AsyncMock()
    repo.update_run_status = AsyncMock(return_value=True)
    repo.get_node_results = AsyncMock(return_value={})
    return repo


@pytest.fixture
def mock_user():
    """Mock authenticated user"""
    return UserPrincipal(id="test_user_123", email="test@example.com")


@pytest.fixture
def test_client(mock_repo, mock_user):
    """TestClient with mocked dependencies"""
    
    def override_get_repo():
        return mock_repo
    
    def override_get_current_user():
        return mock_user
    
    # Override dependencies
    from src.news_reporter.routers import workflows
    app.dependency_overrides[workflows.get_repo] = override_get_repo
    app.dependency_overrides[workflows.get_current_user] = override_get_current_user
    
    yield TestClient(app)
    
    # Cleanup
    app.dependency_overrides = {}


@pytest.fixture
def sample_workflow_graph():
    """Sample workflow graph for testing"""
    return {
        "nodes": [
            {"id": "start", "type": "StartNode"},
            {"id": "msg", "type": "SendMessage", "config": {"message": "Test"}}
        ],
        "edges": [
            {"source": "start", "target": "msg"}
        ]
    }


class TestWorkflowCRUD:
    """Tests for workflow CRUD operations"""
    
    def test_create_workflow_returns_unvalidated(self, test_client, mock_repo, sample_workflow_graph):
        """POST /api/workflows should create workflow with validationStatus='unvalidated'"""
        created_workflow = Workflow(
            id="wf_123",
            userId="test_user_123",
            name="Test Workflow",
            graph=WorkflowGraph(**sample_workflow_graph),
            validationStatus="unvalidated",
            createdAt=datetime.now(timezone.utc),
            updatedAt=datetime.now(timezone.utc)
        )
        mock_repo.create_workflow.return_value = created_workflow
        
        response = test_client.post(
            "/api/workflows",
            json={
                "name": "Test Workflow",
                "graph": sample_workflow_graph
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["validationStatus"] == "unvalidated"
        assert data["name"] == "Test Workflow"
        
    def test_list_workflows(self, test_client, mock_repo):
        """GET /api/workflows should return user's workflows"""
        workflows = [
            Workflow(
                id=f"wf_{i}",
                userId="test_user_123",
                name=f"Workflow {i}",
                graph=WorkflowGraph(nodes=[], edges=[]),
                validationStatus="unvalidated",
                createdAt=datetime.now(timezone.utc),
                updatedAt=datetime.now(timezone.utc)
            )
            for i in range(3)
        ]
        mock_repo.list_workflows.return_value = workflows
        
        response = test_client.get("/api/workflows")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        
    def test_get_workflow(self, test_client, mock_repo, sample_workflow_graph):
        """GET /api/workflows/{id} should return workflow"""
        workflow = Workflow(
            id="wf_123",
            userId="test_user_123",
            name="Test Workflow",
            graph=WorkflowGraph(**sample_workflow_graph),
            validationStatus="valid",
            createdAt=datetime.now(timezone.utc),
            updatedAt=datetime.now(timezone.utc)
        )
        mock_repo.get_workflow.return_value = workflow
        
        response = test_client.get("/api/workflows/wf_123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "wf_123"
        
    def test_get_workflow_not_found_returns_404(self, test_client, mock_repo):
        """GET /api/workflows/{id} should return 404 if not found"""
        mock_repo.get_workflow.return_value = None
        
        response = test_client.get("/api/workflows/wf_nonexistent")
        
        assert response.status_code == 404
        
    def test_update_workflow_resets_validation(self, test_client, mock_repo, sample_workflow_graph):
        """PUT /api/workflows/{id} should reset validationStatus to 'unvalidated'"""
        existing = Workflow(
            id="wf_123",
            userId="test_user_123",
            name="Old Name",
            graph=WorkflowGraph(**sample_workflow_graph),
            validationStatus="valid",
            createdAt=datetime.now(timezone.utc),
            updatedAt=datetime.now(timezone.utc)
        )
        updated = Workflow(
            id="wf_123",
            userId="test_user_123",
            name="New Name",
            graph=WorkflowGraph(**sample_workflow_graph),
            validationStatus="unvalidated",  # Reset
            createdAt=existing.createdAt,
            updatedAt=datetime.now(timezone.utc)
        )
        
        mock_repo.get_workflow.side_effect = [existing, updated]
        mock_repo.update_workflow.return_value = True
        
        response = test_client.put(
            "/api/workflows/wf_123",
            json={"name": "New Name", "graph": sample_workflow_graph}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["validationStatus"] == "unvalidated"


class TestWorkflowValidation:
    """Tests for workflow validation endpoint"""
    
    def test_validate_workflow_valid(self, test_client, mock_repo, sample_workflow_graph):
        """POST /api/workflows/{id}/validate should return status='valid'"""
        workflow = Workflow(
            id="wf_123",
            userId="test_user_123",
            name="Test",
            graph=WorkflowGraph(**sample_workflow_graph),
            validationStatus="unvalidated",
            createdAt=datetime.now(timezone.utc),
            updatedAt=datetime.now(timezone.utc)
        )
        mock_repo.get_workflow.return_value = workflow
        
        with patch('src.news_reporter.routers.workflows.validate_workflow') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True, errors=[])
            
            response = test_client.post("/api/workflows/wf_123/validate")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "valid"
            assert data["errors"] is None
    
    def test_validate_workflow_invalid(self, test_client, mock_repo, sample_workflow_graph):
        """POST /api/workflows/{id}/validate should return errors if invalid"""
        workflow = Workflow(
            id="wf_123",
            userId="test_user_123",
            name="Test",
            graph=WorkflowGraph(**sample_workflow_graph),
            validationStatus="unvalidated",
            createdAt=datetime.now(timezone.utc),
            updatedAt=datetime.now(timezone.utc)
        )
        mock_repo.get_workflow.return_value = workflow
        
        with patch('src.news_reporter.routers.workflows.validate_workflow') as mock_validate:
            error = Mock(message="Missing required field", nodeId="msg", field="config.message")
            mock_validate.return_value = Mock(is_valid=False, errors=[error])
            
            response = test_client.post("/api/workflows/wf_123/validate")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "invalid"
            assert len(data["errors"]) == 1


class TestWorkflowExecution:
    """Tests for workflow execution endpoints"""
    
    def test_run_unvalidated_workflow_returns_409(self, test_client, mock_repo, sample_workflow_graph):
        """POST /api/workflows/{id}/runs should return 409 if workflow not validated"""
        workflow = Workflow(
            id="wf_123",
            userId="test_user_123",
            name="Test",
            graph=WorkflowGraph(**sample_workflow_graph),
            validationStatus="unvalidated",
            createdAt=datetime.now(timezone.utc),
            updatedAt=datetime.now(timezone.utc)
        )
        mock_repo.get_workflow.return_value = workflow
        
        response = test_client.post("/api/workflows/wf_123/runs")
        
        assert response.status_code == 409
        assert "must be validated" in response.json()["detail"].lower()
    
    def test_run_invalid_workflow_returns_409(self, test_client, mock_repo, sample_workflow_graph):
        """POST /api/workflows/{id}/runs should return 409 if workflow invalid"""
        workflow = Workflow(
            id="wf_123",
            userId="test_user_123",
            name="Test",
            graph=WorkflowGraph(**sample_workflow_graph),
            validationStatus="invalid",
            createdAt=datetime.now(timezone.utc),
            updatedAt=datetime.now(timezone.utc)
        )
        mock_repo.get_workflow.return_value = workflow
        
        response = test_client.post("/api/workflows/wf_123/runs")
        
        assert response.status_code == 409
        assert "invalid" in response.json()["detail"].lower()
    
    def test_run_valid_workflow_returns_run_id(self, test_client, mock_repo, sample_workflow_graph):
        """POST /api/workflows/{id}/runs should return runId for valid workflow"""
        workflow = Workflow(
            id="wf_123",
            userId="test_user_123",
            name="Test",
            graph=WorkflowGraph(**sample_workflow_graph),
            validationStatus="valid",
            createdAt=datetime.now(timezone.utc),
            updatedAt=datetime.now(timezone.utc)
        )
        run = WorkflowRun(
            id="run_456",
            workflowId="wf_123",
            userId="test_user_123",
            status="queued",
            createdAt=datetime.now(timezone.utc)
        )
        
        mock_repo.get_workflow.return_value = workflow
        mock_repo.create_run.return_value = run
        
        with patch('src.news_reporter.routers.workflows.BackgroundTasks'):
            response = test_client.post("/api/workflows/wf_123/runs")
            
            assert response.status_code == 201
            data = response.json()
            assert data["runId"] == "run_456"
            assert data["status"] == "queued"
    
    def test_get_run_status(self, test_client, mock_repo):
        """GET /api/workflows/{id}/runs/{runId} should return run status"""
        run = WorkflowRun(
            id="run_456",
            workflowId="wf_123",
            userId="test_user_123",
            status="succeeded",
            createdAt=datetime.now(timezone.utc),
            startedAt=datetime.now(timezone.utc),
            completedAt=datetime.now(timezone.utc)
        )
        mock_repo.get_run.return_value = run
        
        response = test_client.get("/api/workflows/wf_123/runs/run_456")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "succeeded"
    
    def test_get_results_returns_execution_order(self, test_client, mock_repo):
        """GET /api/workflows/{id}/runs/{runId}/results should return executionOrder"""
        start_time = datetime.now(timezone.utc)
        
        node_results = {
            "start": NodeResult(
                status="succeeded",
                output="{}",
                startedAt=start_time,
                completedAt=start_time,
                executionMs=1.0
            ),
            "msg": NodeResult(
                status="succeeded",
                output="Test",
                startedAt=start_time,
                completedAt=start_time,
                executionMs=2.0
            )
        }
        
        run = WorkflowRun(
            id="run_456",
            workflowId="wf_123",
            userId="test_user_123",
            status="succeeded",
            createdAt=start_time,
            nodeResults=node_results
        )
        mock_repo.get_run.return_value = run
        
        response = test_client.get("/api/workflows/wf_123/runs/run_456/results")
        
        assert response.status_code == 200
        data = response.json()
        assert "executionOrder" in data
        assert "resultsByNodeId" in data
        assert len(data["executionOrder"]) == 2


class TestUserIsolation:
    """Tests for user scoping and isolation"""
    
    def test_user_cannot_access_other_users_workflow(self, test_client, mock_repo):
        """User B should get 404 when accessing User A's workflow"""
        mock_repo.get_workflow.return_value = None  # Not found for this user
        
        response = test_client.get("/api/workflows/wf_other_user")
        
        assert response.status_code == 404
    
    def test_user_cannot_access_other_users_run(self, test_client, mock_repo):
        """User B should get 404 when accessing User A's run"""
        mock_repo.get_run.return_value = None  # Not found for this user
        
        response = test_client.get("/api/workflows/wf_123/runs/run_other_user")
        
        assert response.status_code == 404


class TestExecutionOrderSorting:
    """Tests for executionOrder sorting behavior"""
    
    def test_execution_order_sorts_none_last(self, test_client, mock_repo):
        """Nodes with startedAt=None should appear last in executionOrder"""
        start_time = datetime.now(timezone.utc)
        
        node_results = {
            "start": NodeResult(
                status="succeeded",
                output="{}",
                startedAt=start_time,
                completedAt=start_time,
                executionMs=1.0
            ),
            "pending": NodeResult(
                status="succeeded",  # Fix: NodeResult only allows succeeded/failed
                output=None,
                startedAt=None,  # Not started yet
                completedAt=None,
                executionMs=0.0
            ),
            "msg": NodeResult(
                status="succeeded",
                output="Test",
                startedAt=start_time,
                completedAt=start_time,
                executionMs=2.0
            )
        }
        
        run = WorkflowRun(
            id="run_456",
            workflowId="wf_123",
            userId="test_user_123",
            status="running",
            createdAt=start_time,
            nodeResults=node_results
        )
        mock_repo.get_run.return_value = run
        
        response = test_client.get("/api/workflows/wf_123/runs/run_456/results")
        
        assert response.status_code == 200
        data = response.json()
        exec_order = data["executionOrder"]
        
        # "pending" should be last because startedAt is None
        assert exec_order[-1] == "pending"
        assert "start" in exec_order[:-1]
        assert "msg" in exec_order[:-1]
