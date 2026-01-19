"""Test Active Workflow Management - Phase 2 Implementation"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import pytest
from datetime import datetime
from src.news_reporter.workflows.workflow_persistence import (
    WorkflowPersistence,
    WorkflowRecord,
    get_workflow_persistence
)
from src.news_reporter.workflows.workflow_factory import run_graph_workflow, _substitute_agent_ids_in_dict
from src.news_reporter.config import Settings


class TestActiveWorkflowManagement:
    """Test active workflow management functionality"""
    
    def test_get_active_workflow_no_active(self):
        """Test getting active workflow when none is set"""
        persistence = WorkflowPersistence(storage_backend=None)  # Use in-memory
        
        active = persistence.get_active_workflow()
        assert active is None
    
    def test_set_active_workflow(self):
        """Test setting a workflow as active"""
        persistence = WorkflowPersistence(storage_backend=None)
        
        # Create two workflows
        workflow1 = WorkflowRecord(
            workflow_id="workflow1",
            name="Workflow 1",
            graph_definition={},
            is_active=False
        )
        workflow2 = WorkflowRecord(
            workflow_id="workflow2",
            name="Workflow 2",
            graph_definition={},
            is_active=False
        )
        
        persistence.save_workflow(workflow1)
        persistence.save_workflow(workflow2)
        
        # Set workflow1 as active
        success = persistence.set_active_workflow("workflow1")
        assert success is True
        
        # Verify workflow1 is active
        active = persistence.get_active_workflow()
        assert active is not None
        assert active.workflow_id == "workflow1"
        assert active.is_active is True
        
        # Verify workflow2 is not active
        wf2 = persistence.get_workflow("workflow2")
        assert wf2 is not None
        assert wf2.is_active is False
    
    def test_set_active_workflow_deactivates_others(self):
        """Test that setting a workflow as active deactivates others"""
        persistence = WorkflowPersistence(storage_backend=None)
        
        # Create and activate workflow1
        workflow1 = WorkflowRecord(
            workflow_id="workflow1",
            name="Workflow 1",
            graph_definition={},
            is_active=True
        )
        workflow2 = WorkflowRecord(
            workflow_id="workflow2",
            name="Workflow 2",
            graph_definition={},
            is_active=False
        )
        
        persistence.save_workflow(workflow1)
        persistence.save_workflow(workflow2)
        
        # Set workflow2 as active
        success = persistence.set_active_workflow("workflow2")
        assert success is True
        
        # Verify workflow2 is now active
        active = persistence.get_active_workflow()
        assert active.workflow_id == "workflow2"
        assert active.is_active is True
        
        # Verify workflow1 is deactivated
        wf1 = persistence.get_workflow("workflow1")
        assert wf1.is_active is False
    
    def test_set_active_workflow_not_found(self):
        """Test setting non-existent workflow as active"""
        persistence = WorkflowPersistence(storage_backend=None)
        
        success = persistence.set_active_workflow("non_existent")
        assert success is False


class TestWorkflowFactoryWithDefinition:
    """Test workflow factory with workflow_definition parameter"""
    
    def test_substitute_agent_ids_in_dict(self):
        """Test agent ID substitution in workflow definition"""
        from src.news_reporter.config import Settings
        
        config = Settings.load()
        
        workflow_def = {
            "nodes": [
                {
                    "id": "triage",
                    "type": "agent",
                    "agent_id": "${AGENT_ID_TRIAGE}"
                }
            ],
            "edges": []
        }
        
        result = _substitute_agent_ids_in_dict(workflow_def, config)
        
        # Agent ID should be substituted
        assert result["nodes"][0]["agent_id"] == config.agent_id_triage
        assert result["nodes"][0]["agent_id"] != "${AGENT_ID_TRIAGE}"
    
    def test_substitute_agent_ids_preserves_structure(self):
        """Test that substitution doesn't modify original structure"""
        from src.news_reporter.config import Settings
        
        config = Settings.load()
        
        workflow_def = {
            "nodes": [
                {
                    "id": "node1",
                    "type": "agent",
                    "agent_id": "fixed_agent_id"  # Not a placeholder
                }
            ],
            "edges": []
        }
        
        result = _substitute_agent_ids_in_dict(workflow_def, config)
        
        # Non-placeholder agent_id should remain unchanged
        assert result["nodes"][0]["agent_id"] == "fixed_agent_id"


class TestWorkflowDefinitionValidation:
    """Test workflow definition validation before execution"""
    
    def test_workflow_validation_errors(self):
        """Test that invalid workflows are detected"""
        from src.news_reporter.workflows.graph_schema import GraphDefinition, NodeConfig, EdgeConfig
        
        # Create workflow with invalid edge (references non-existent node)
        nodes = [NodeConfig(id="node1", type="agent", agent_id="agent1")]
        edges = [EdgeConfig(from_node="node1", to_node="node2")]  # node2 doesn't exist
        
        graph_def = GraphDefinition(nodes=nodes, edges=edges)
        errors = graph_def.validate()
        
        assert len(errors) > 0
        assert any("node2" in error for error in errors)
    
    def test_workflow_validation_success(self):
        """Test that valid workflows pass validation"""
        from src.news_reporter.workflows.graph_schema import GraphDefinition, NodeConfig, EdgeConfig
        
        # Create valid workflow
        nodes = [
            NodeConfig(id="node1", type="agent", agent_id="agent1"),
            NodeConfig(id="node2", type="agent", agent_id="agent2")
        ]
        edges = [EdgeConfig(from_node="node1", to_node="node2")]
        
        graph_def = GraphDefinition(nodes=nodes, edges=edges, entry_node_id="node1")
        errors = graph_def.validate()
        
        # Should have no errors (or only warnings for optional fields)
        # Agent nodes without agent_id would be an error, but we provided them
        assert len([e for e in errors if "missing agent_id" in e]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
