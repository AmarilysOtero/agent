"""Integration tests for Phase 3: Workflow Creation, Chat Execution, and Fallback"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from src.news_reporter.workflows.workflow_persistence import (
    WorkflowPersistence,
    WorkflowRecord,
    get_workflow_persistence
)
from src.news_reporter.workflows.workflow_factory import run_graph_workflow, run_sequential_goal
from src.news_reporter.workflows.graph_schema import GraphDefinition, NodeConfig, EdgeConfig
from src.news_reporter.config import Settings


class TestWorkflowCreationAndSaving:
    """Test workflow creation and saving functionality"""
    
    def test_create_and_save_workflow(self):
        """Test creating and saving a workflow"""
        persistence = WorkflowPersistence(storage_backend=None)  # Use in-memory
        
        workflow_def = {
            "nodes": [
                {
                    "id": "triage",
                    "type": "agent",
                    "agent_id": "test-triage-agent"
                },
                {
                    "id": "reporter",
                    "type": "agent",
                    "agent_id": "test-reporter-agent"
                }
            ],
            "edges": [
                {
                    "from_node": "triage",
                    "to_node": "reporter"
                }
            ],
            "entry_node_id": "triage",
            "name": "Test Workflow",
            "description": "A test workflow"
        }
        
        workflow = WorkflowRecord(
            workflow_id="test-workflow-1",
            name="Test Workflow",
            description="A test workflow",
            graph_definition=workflow_def,
            is_active=False
        )
        
        persistence.save_workflow(workflow)
        
        # Verify workflow was saved
        saved = persistence.get_workflow("test-workflow-1")
        assert saved is not None
        assert saved.workflow_id == "test-workflow-1"
        assert saved.name == "Test Workflow"
        assert saved.graph_definition == workflow_def
    
    def test_save_workflow_with_validation(self):
        """Test saving workflow with validation"""
        persistence = WorkflowPersistence(storage_backend=None)
        
        # Create a valid workflow
        workflow_def = {
            "nodes": [
                {
                    "id": "node1",
                    "type": "agent",
                    "agent_id": "agent1"
                },
                {
                    "id": "node2",
                    "type": "agent",
                    "agent_id": "agent2"
                }
            ],
            "edges": [
                {
                    "from_node": "node1",
                    "to_node": "node2"
                }
            ],
            "entry_node_id": "node1"
        }
        
        workflow = WorkflowRecord(
            workflow_id="valid-workflow",
            name="Valid Workflow",
            graph_definition=workflow_def
        )
        
        persistence.save_workflow(workflow)
        
        # Verify it can be retrieved
        saved = persistence.get_workflow("valid-workflow")
        assert saved is not None
        
        # Validate the graph definition structure
        from src.news_reporter.workflows.graph_schema import GraphDefinition, NodeConfig, EdgeConfig
        
        nodes = [NodeConfig(**node) for node in workflow_def["nodes"]]
        edges = [EdgeConfig(**edge) for edge in workflow_def["edges"]]
        graph_def = GraphDefinition(nodes=nodes, edges=edges, entry_node_id="node1")
        
        errors = graph_def.validate()
        # Should have no critical errors
        assert len([e for e in errors if "missing agent_id" in e]) == 0
    
    def test_list_saved_workflows(self):
        """Test listing saved workflows"""
        persistence = WorkflowPersistence(storage_backend=None)
        
        # Create multiple workflows
        for i in range(3):
            workflow = WorkflowRecord(
                workflow_id=f"workflow-{i}",
                name=f"Workflow {i}",
                graph_definition={},
                is_active=(i == 0)  # First one is active
            )
            persistence.save_workflow(workflow)
        
        # List all workflows
        all_workflows = persistence.list_workflows()
        assert len(all_workflows) == 3
        
        # List only active workflows
        active_workflows = persistence.list_workflows(is_active=True)
        assert len(active_workflows) == 1
        assert active_workflows[0].workflow_id == "workflow-0"


class TestChatExecutionWithCustomWorkflow:
    """Test chat execution with custom workflows"""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock config"""
        config = Mock(spec=Settings)
        config.agent_id_triage = "test-triage"
        config.agent_id_aisearch = "test-aisearch"
        config.agent_id_reviewer = "test-reviewer"
        config.reporter_ids = ["test-reporter"]
        config.checkpoint_dir = None
        return config
    
    @pytest.fixture
    def simple_workflow_definition(self):
        """Create a simple workflow definition for testing"""
        return {
            "nodes": [
                {
                    "id": "triage",
                    "type": "agent",
                    "agent_id": "test-triage",
                    "inputs": {},
                    "outputs": {"result": "triage.result"}
                }
            ],
            "edges": [],
            "entry_node_id": "triage"
        }
    
    @pytest.mark.asyncio
    async def test_run_graph_workflow_with_definition(self, mock_config, simple_workflow_definition):
        """Test running graph workflow with workflow definition"""
        with patch('src.news_reporter.workflows.graph_executor.GraphExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_executor.execute = AsyncMock(return_value="Test result from workflow")
            mock_executor.metrics_collector = Mock()
            mock_executor.metrics_collector.get_all_metrics = Mock(return_value=[])
            mock_executor_class.return_value = mock_executor
            
            result = await run_graph_workflow(
                mock_config,
                "test goal",
                workflow_definition=simple_workflow_definition
            )
            
            assert result == "Test result from workflow"
            mock_executor_class.assert_called_once()
            mock_executor.execute.assert_called_once_with("test goal")
    
    @pytest.mark.asyncio
    async def test_chat_execution_with_active_workflow(self, mock_config, simple_workflow_definition):
        """Test chat execution when active workflow is set"""
        persistence = WorkflowPersistence(storage_backend=None)
        
        # Create and save active workflow
        workflow = WorkflowRecord(
            workflow_id="active-workflow",
            name="Active Workflow",
            graph_definition=simple_workflow_definition,
            is_active=True
        )
        persistence.save_workflow(workflow)
        
        # Mock the workflow execution
        with patch('src.news_reporter.workflows.workflow_factory.run_graph_workflow') as mock_run_graph:
            mock_run_graph.return_value = AsyncMock(return_value="Result from active workflow")
            
            # Simulate chat execution logic
            active_workflow = persistence.get_active_workflow()
            assert active_workflow is not None
            assert active_workflow.workflow_id == "active-workflow"
            
            # Verify workflow definition is correct
            assert active_workflow.graph_definition == simple_workflow_definition
    
    @pytest.mark.asyncio
    async def test_chat_execution_without_active_workflow(self, mock_config):
        """Test chat execution when no active workflow is set"""
        persistence = WorkflowPersistence(storage_backend=None)
        
        # No active workflow
        active_workflow = persistence.get_active_workflow()
        assert active_workflow is None
        
        # Should fall back to sequential workflow
        with patch('src.news_reporter.workflows.workflow_factory.run_sequential_goal') as mock_sequential:
            mock_sequential.return_value = AsyncMock(return_value="Result from sequential workflow")
            
            # Simulate the fallback logic
            if active_workflow is None:
                # This is what chat_sessions.py does
                result = await run_sequential_goal(mock_config, "test goal")
                # In real scenario, this would be called
                pass


class TestFallbackToSequentialWorkflow:
    """Test fallback mechanism to sequential workflow"""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock config"""
        config = Mock(spec=Settings)
        config.agent_id_triage = "test-triage"
        config.agent_id_aisearch = "test-aisearch"
        config.agent_id_reviewer = "test-reviewer"
        config.reporter_ids = ["test-reporter"]
        config.checkpoint_dir = None
        return config
    
    @pytest.mark.asyncio
    async def test_fallback_on_workflow_execution_error(self, mock_config):
        """Test fallback when graph workflow execution fails"""
        invalid_workflow_def = {
            "nodes": [
                {
                    "id": "invalid-node",
                    "type": "agent",
                    "agent_id": "non-existent-agent"
                }
            ],
            "edges": [],
            "entry_node_id": "invalid-node"
        }
        
        # Mock run_graph_workflow to raise an exception
        with patch('src.news_reporter.workflows.graph_executor.GraphExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_executor.execute = AsyncMock(side_effect=Exception("Workflow execution failed"))
            mock_executor_class.return_value = mock_executor
            
            # run_graph_workflow should catch the exception and fall back
            with patch('src.news_reporter.workflows.workflow_factory.run_sequential_goal') as mock_sequential:
                mock_sequential.return_value = "Fallback result"
                
                result = await run_graph_workflow(
                    mock_config,
                    "test goal",
                    workflow_definition=invalid_workflow_def
                )
                
                # Should have fallen back to sequential
                assert result == "Fallback result"
                mock_sequential.assert_called_once_with(mock_config, "test goal")
    
    @pytest.mark.asyncio
    async def test_fallback_on_validation_errors(self, mock_config):
        """Test that validation errors don't prevent fallback"""
        # Workflow with validation errors but still executable
        workflow_with_warnings = {
            "nodes": [
                {
                    "id": "node1",
                    "type": "agent",
                    "agent_id": "agent1"
                }
            ],
            "edges": [
                {
                    "from_node": "node1",
                    "to_node": "node2"  # node2 doesn't exist - validation error
                }
            ],
            "entry_node_id": "node1"
        }
        
        with patch('src.news_reporter.workflows.graph_executor.GraphExecutor') as mock_executor_class:
            # Execution might fail due to validation errors
            mock_executor = Mock()
            mock_executor.execute = AsyncMock(side_effect=Exception("Invalid workflow"))
            mock_executor_class.return_value = mock_executor
            
            with patch('src.news_reporter.workflows.workflow_factory.run_sequential_goal') as mock_sequential:
                mock_sequential.return_value = "Fallback after validation error"
                
                result = await run_graph_workflow(
                    mock_config,
                    "test goal",
                    workflow_definition=workflow_with_warnings
                )
                
                # Should fall back even with validation errors
                assert result == "Fallback after validation error"
    
    @pytest.mark.asyncio
    async def test_fallback_on_missing_active_workflow(self, mock_config):
        """Test fallback when active workflow is not set"""
        persistence = WorkflowPersistence(storage_backend=None)
        
        # No active workflow
        active_workflow = persistence.get_active_workflow()
        assert active_workflow is None
        
        # Should use sequential workflow
        with patch('src.news_reporter.workflows.workflow_factory.run_sequential_goal') as mock_sequential:
            mock_sequential.return_value = "Sequential workflow result"
            
            # Simulate chat execution logic
            if active_workflow is None:
                result = await run_sequential_goal(mock_config, "test goal")
                # In real scenario, this would be the result
                pass


class TestErrorHandlingAndLogging:
    """Test error handling and logging"""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock config"""
        config = Mock(spec=Settings)
        config.agent_id_triage = "test-triage"
        config.agent_id_aisearch = "test-aisearch"
        config.agent_id_reviewer = "test-reviewer"
        config.reporter_ids = ["test-reporter"]
        return config
    
    def test_error_handling_workflow_not_found(self):
        """Test error handling when workflow is not found"""
        persistence = WorkflowPersistence(storage_backend=None)
        
        # Try to get non-existent workflow
        workflow = persistence.get_workflow("non-existent")
        assert workflow is None
        
        # Try to set non-existent workflow as active
        success = persistence.set_active_workflow("non-existent")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_workflow_definition(self, mock_config):
        """Test error handling with invalid workflow definition"""
        invalid_def = {
            "nodes": [],  # No nodes
            "edges": [],
            "entry_node_id": "non-existent"
        }
        
        # Should handle gracefully
        with patch('src.news_reporter.workflows.graph_executor.GraphExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_executor.execute = AsyncMock(side_effect=Exception("Invalid definition"))
            mock_executor_class.return_value = mock_executor
            
            with patch('src.news_reporter.workflows.workflow_factory.run_sequential_goal') as mock_sequential:
                mock_sequential.return_value = "Fallback result"
                
                # Should not raise exception, should fall back
                result = await run_graph_workflow(
                    mock_config,
                    "test goal",
                    workflow_definition=invalid_def
                )
                
                assert result == "Fallback result"
    
    def test_logging_workflow_activation(self):
        """Test that workflow activation is logged"""
        import logging
        
        persistence = WorkflowPersistence(storage_backend=None)
        
        workflow = WorkflowRecord(
            workflow_id="test-workflow",
            name="Test",
            graph_definition={}
        )
        persistence.save_workflow(workflow)
        
        # Set as active - should log
        with patch('src.news_reporter.workflows.workflow_persistence.logger') as mock_logger:
            persistence.set_active_workflow("test-workflow")
            # Verify logging was called (at least info level)
            assert mock_logger.info.called or mock_logger.debug.called
    
    @pytest.mark.asyncio
    async def test_logging_workflow_execution_failure(self, mock_config):
        """Test that workflow execution failures are logged"""
        with patch('src.news_reporter.workflows.workflow_factory.logger') as mock_logger:
            with patch('src.news_reporter.workflows.graph_executor.GraphExecutor') as mock_executor_class:
                mock_executor = Mock()
                mock_executor.execute = AsyncMock(side_effect=Exception("Execution failed"))
                mock_executor_class.return_value = mock_executor
                
                with patch('src.news_reporter.workflows.workflow_factory.run_sequential_goal') as mock_sequential:
                    mock_sequential.return_value = "Fallback"
                    
                    await run_graph_workflow(
                        mock_config,
                        "test goal",
                        workflow_definition={"nodes": [], "edges": []}
                    )
                    
                    # Should log the error
                    assert mock_logger.error.called
                    assert mock_logger.warning.called  # Fallback warning


class TestEndToEndWorkflow:
    """End-to-end tests for complete workflow lifecycle"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_lifecycle(self):
        """Test complete workflow lifecycle: create, save, activate, execute"""
        persistence = WorkflowPersistence(storage_backend=None)
        
        # 1. Create workflow
        workflow_def = {
            "nodes": [
                {
                    "id": "triage",
                    "type": "agent",
                    "agent_id": "test-triage"
                }
            ],
            "edges": [],
            "entry_node_id": "triage"
        }
        
        workflow = WorkflowRecord(
            workflow_id="e2e-workflow",
            name="E2E Test Workflow",
            graph_definition=workflow_def,
            is_active=False
        )
        
        # 2. Save workflow
        persistence.save_workflow(workflow)
        saved = persistence.get_workflow("e2e-workflow")
        assert saved is not None
        
        # 3. Set as active
        success = persistence.set_active_workflow("e2e-workflow")
        assert success is True
        
        # 4. Verify it's active
        active = persistence.get_active_workflow()
        assert active is not None
        assert active.workflow_id == "e2e-workflow"
        assert active.is_active is True
        
        # 5. Verify workflow definition is accessible
        assert active.graph_definition == workflow_def
        assert active.graph_definition["entry_node_id"] == "triage"
    
    @pytest.mark.asyncio
    async def test_workflow_switching(self):
        """Test switching between multiple workflows"""
        persistence = WorkflowPersistence(storage_backend=None)
        
        # Create multiple workflows
        for i in range(3):
            workflow = WorkflowRecord(
                workflow_id=f"workflow-{i}",
                name=f"Workflow {i}",
                graph_definition={"nodes": [], "edges": []},
                is_active=(i == 0)
            )
            persistence.save_workflow(workflow)
        
        # Verify first is active
        active = persistence.get_active_workflow()
        assert active.workflow_id == "workflow-0"
        
        # Switch to second workflow
        persistence.set_active_workflow("workflow-1")
        active = persistence.get_active_workflow()
        assert active.workflow_id == "workflow-1"
        
        # Verify first is deactivated
        wf0 = persistence.get_workflow("workflow-0")
        assert wf0.is_active is False
        
        # Switch to third workflow
        persistence.set_active_workflow("workflow-2")
        active = persistence.get_active_workflow()
        assert active.workflow_id == "workflow-2"
        
        # Verify second is deactivated
        wf1 = persistence.get_workflow("workflow-1")
        assert wf1.is_active is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
