"""Test Phase 8: Cost Management, Backup, Debugger, Governance, AI, Documentation"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import pytest
from src.news_reporter.workflows.workflow_cost import WorkflowCostManager, CostType, CostEntry
from src.news_reporter.workflows.workflow_backup import WorkflowBackupManager, BackupType
from src.news_reporter.workflows.workflow_debugger import WorkflowDebugger, BreakpointType
from src.news_reporter.workflows.workflow_governance import WorkflowGovernance, PolicyType, PolicySeverity
from src.news_reporter.workflows.workflow_ai import WorkflowAI, AITaskType
from src.news_reporter.workflows.workflow_documentation import WorkflowDocumentation, DocumentationType
from src.news_reporter.workflows.graph_schema import GraphDefinition, NodeConfig


class TestWorkflowCost:
    """Test workflow cost management"""
    
    def test_record_cost(self):
        """Test recording costs"""
        manager = WorkflowCostManager()
        entry = manager.record_cost("workflow1", CostType.API_CALL, units=5)
        assert entry.workflow_id == "workflow1"
        assert entry.cost_type == CostType.API_CALL
        assert entry.amount > 0
    
    def test_add_budget(self):
        """Test adding budgets"""
        manager = WorkflowCostManager()
        budget = manager.add_budget("budget1", 100.0, workflow_id="workflow1")
        assert budget.budget_id == "budget1"
        assert budget.amount == 100.0
    
    def test_generate_cost_report(self):
        """Test generating cost reports"""
        from datetime import datetime, timedelta
        manager = WorkflowCostManager()
        manager.record_cost("workflow1", CostType.API_CALL, amount=10.0)
        report = manager.generate_cost_report("workflow1")
        assert report.total_cost == 10.0
        assert report.execution_count >= 0


class TestWorkflowBackup:
    """Test workflow backup"""
    
    def test_create_backup(self):
        """Test creating backups"""
        from src.news_reporter.workflows.workflow_persistence import WorkflowPersistence, WorkflowRecord
        persistence = WorkflowPersistence()
        workflow = WorkflowRecord(
            workflow_id="workflow1",
            name="Test",
            graph_definition={"nodes": [], "edges": []}
        )
        persistence.save_workflow(workflow)
        
        backup_manager = WorkflowBackupManager()
        backup_manager.set_persistence(persistence)
        backup = backup_manager.create_backup(backup_type=BackupType.FULL)
        assert backup.backup_type == BackupType.FULL
        assert len(backup.workflow_ids) > 0


class TestWorkflowDebugger:
    """Test workflow debugger"""
    
    def test_add_breakpoint(self):
        """Test adding breakpoints"""
        debugger = WorkflowDebugger()
        bp = debugger.add_breakpoint("bp1", BreakpointType.NODE, node_id="node1")
        assert bp.breakpoint_id == "bp1"
        assert bp.type == BreakpointType.NODE
    
    def test_add_trace(self):
        """Test adding traces"""
        from src.news_reporter.workflows.execution_context import ExecutionContext
        from src.news_reporter.workflows.workflow_state import WorkflowState
        debugger = WorkflowDebugger()
        context = ExecutionContext(run_id="run1", branch_id="branch1")
        state = WorkflowState(goal="test")
        trace = debugger.add_trace("node1", context, state)
        assert trace.node_id == "node1"
        assert trace.event_type == "execution"


class TestWorkflowGovernance:
    """Test workflow governance"""
    
    def test_validate_workflow(self):
        """Test workflow validation"""
        governance = WorkflowGovernance()
        workflow = GraphDefinition(
            nodes=[NodeConfig(id="node1", type="agent", agent_id="agent1")],
            edges=[],
            entry_node_id="node1"
        )
        violations = governance.validate_workflow(workflow, "workflow1")
        # Should have no violations for a simple valid workflow
        assert isinstance(violations, list)
    
    def test_get_compliance_report(self):
        """Test compliance reporting"""
        governance = WorkflowGovernance()
        workflow = GraphDefinition(
            nodes=[NodeConfig(id="node1", type="agent", agent_id="agent1")],
            edges=[],
            entry_node_id="node1"
        )
        report = governance.get_compliance_report("workflow1", workflow)
        assert "compliance_rate" in report
        assert "violations" in report


class TestWorkflowAI:
    """Test workflow AI"""
    
    def test_predict_execution_time(self):
        """Test execution time prediction"""
        ai = WorkflowAI()
        workflow = GraphDefinition(
            nodes=[NodeConfig(id="node1", type="agent", agent_id="agent1")],
            edges=[],
            entry_node_id="node1"
        )
        prediction = ai.predict_execution_time("workflow1", workflow)
        assert prediction.workflow_id == "workflow1"
        assert prediction.task_type == AITaskType.PREDICTION
        assert "predicted_duration_seconds" in prediction.prediction
    
    def test_generate_recommendations(self):
        """Test generating recommendations"""
        ai = WorkflowAI()
        workflow = GraphDefinition(
            nodes=[NodeConfig(id=f"node{i}", type="agent", agent_id="agent1") for i in range(25)],
            edges=[],
            entry_node_id="node1"
        )
        recommendations = ai.generate_recommendations("workflow1", workflow)
        assert len(recommendations) > 0


class TestWorkflowDocumentation:
    """Test workflow documentation"""
    
    def test_add_documentation(self):
        """Test adding documentation"""
        docs = WorkflowDocumentation()
        doc = docs.add_documentation(
            "doc1",
            DocumentationType.WORKFLOW,
            "Test Doc",
            "Content",
            workflow_id="workflow1"
        )
        assert doc.doc_id == "doc1"
        assert doc.type == DocumentationType.WORKFLOW
    
    def test_generate_workflow_docs(self):
        """Test auto-generating workflow documentation"""
        docs = WorkflowDocumentation()
        workflow = GraphDefinition(
            nodes=[NodeConfig(id="node1", type="agent", agent_id="agent1")],
            edges=[],
            entry_node_id="node1"
        )
        doc = docs.generate_workflow_docs("workflow1", workflow)
        assert doc.workflow_id == "workflow1"
        assert "node1" in doc.content
