"""Test Phase 5: Workflow Visualization, Versioning, Monitoring, Templates"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import pytest
from src.news_reporter.workflows.workflow_visualizer import WorkflowVisualizer
from src.news_reporter.workflows.workflow_versioning import WorkflowVersionManager
from src.news_reporter.workflows.execution_monitor import (
    ExecutionMonitor, ExecutionEvent, ExecutionEventType, get_execution_monitor
)
from src.news_reporter.workflows.workflow_templates import get_template_registry
from src.news_reporter.models.graph_schema import GraphDefinition, NodeConfig, EdgeConfig, GraphLimits


class TestWorkflowVisualizer:
    """Test workflow visualization"""
    
    def test_to_dot(self):
        """Test DOT format generation"""
        nodes = [
            NodeConfig(id="start", type="agent", agent_id="agent1"),
            NodeConfig(id="end", type="agent", agent_id="agent2")
        ]
        edges = [EdgeConfig(from_node="start", to_node="end")]
        graph_def = GraphDefinition(nodes=nodes, edges=edges, entry_node_id="start")
        
        visualizer = WorkflowVisualizer(graph_def)
        dot = visualizer.to_dot()
        
        assert "digraph workflow" in dot
        assert '"start"' in dot
        assert '"end"' in dot
        assert '"start" -> "end"' in dot
    
    def test_to_mermaid(self):
        """Test Mermaid format generation"""
        nodes = [
            NodeConfig(id="start", type="agent", agent_id="agent1"),
            NodeConfig(id="end", type="agent", agent_id="agent2")
        ]
        edges = [EdgeConfig(from_node="start", to_node="end")]
        graph_def = GraphDefinition(nodes=nodes, edges=edges, entry_node_id="start")
        
        visualizer = WorkflowVisualizer(graph_def)
        mermaid = visualizer.to_mermaid()
        
        assert "graph LR" in mermaid
        assert "start" in mermaid
        assert "end" in mermaid
    
    def test_to_json_graph(self):
        """Test JSON graph format"""
        nodes = [
            NodeConfig(id="start", type="agent", agent_id="agent1"),
            NodeConfig(id="end", type="agent", agent_id="agent2")
        ]
        edges = [EdgeConfig(from_node="start", to_node="end")]
        graph_def = GraphDefinition(nodes=nodes, edges=edges, entry_node_id="start")
        
        visualizer = WorkflowVisualizer(graph_def)
        json_graph = visualizer.to_json_graph()
        
        assert "nodes" in json_graph
        assert "edges" in json_graph
        assert len(json_graph["nodes"]) == 2
        assert len(json_graph["edges"]) == 1
        assert json_graph["nodes"][0]["id"] == "start"
    
    def test_to_summary(self):
        """Test workflow summary generation"""
        nodes = [
            NodeConfig(id="start", type="agent", agent_id="agent1"),
            NodeConfig(id="loop", type="loop", max_iters=3),
            NodeConfig(id="end", type="agent", agent_id="agent2")
        ]
        edges = [
            EdgeConfig(from_node="start", to_node="loop"),
            EdgeConfig(from_node="loop", to_node="end")
        ]
        graph_def = GraphDefinition(nodes=nodes, edges=edges, entry_node_id="start")
        
        visualizer = WorkflowVisualizer(graph_def)
        summary = visualizer.to_summary()
        
        assert summary["total_nodes"] == 3
        assert summary["total_edges"] == 2
        assert summary["entry_node"] == "start"
        assert summary["has_loops"] is True
        assert "agent" in summary["node_types"]


class TestWorkflowVersioning:
    """Test workflow versioning"""
    
    def test_save_and_load_version(self, tmp_path):
        """Test saving and loading workflow versions"""
        version_manager = WorkflowVersionManager(str(tmp_path / "versions"))
        
        nodes = [NodeConfig(id="start", type="agent", agent_id="agent1")]
        edges = []
        graph_def = GraphDefinition(nodes=nodes, edges=edges, entry_node_id="start")
        
        version = version_manager.save_version("workflow1", graph_def, version="v1.0")
        assert version == "v1.0"
        
        loaded = version_manager.load_version("workflow1", "v1.0")
        assert loaded is not None
        assert len(loaded.nodes) == 1
        assert loaded.nodes[0].id == "start"
    
    def test_list_versions(self, tmp_path):
        """Test listing workflow versions"""
        version_manager = WorkflowVersionManager(str(tmp_path / "versions"))
        
        nodes = [NodeConfig(id="start", type="agent", agent_id="agent1")]
        edges = []
        graph_def = GraphDefinition(nodes=nodes, edges=edges, entry_node_id="start")
        
        version_manager.save_version("workflow1", graph_def, version="v1.0")
        version_manager.save_version("workflow1", graph_def, version="v1.1")
        
        versions = version_manager.list_versions("workflow1")
        assert len(versions) == 2
        assert versions[0]["version"] == "v1.1"  # Most recent first
    
    def test_compare_versions(self, tmp_path):
        """Test comparing workflow versions"""
        version_manager = WorkflowVersionManager(str(tmp_path / "versions"))
        
        # Version 1
        nodes1 = [NodeConfig(id="start", type="agent", agent_id="agent1")]
        edges1 = []
        graph1 = GraphDefinition(nodes=nodes1, edges=edges1, entry_node_id="start")
        version_manager.save_version("workflow1", graph1, version="v1.0")
        
        # Version 2 - added node
        nodes2 = [
            NodeConfig(id="start", type="agent", agent_id="agent1"),
            NodeConfig(id="end", type="agent", agent_id="agent2")
        ]
        edges2 = [EdgeConfig(from_node="start", to_node="end")]
        graph2 = GraphDefinition(nodes=nodes2, edges=edges2, entry_node_id="start")
        version_manager.save_version("workflow1", graph2, version="v2.0")
        
        differences = version_manager.compare_versions("workflow1", "v1.0", "v2.0")
        
        assert "end" in differences["nodes_added"]
        assert len(differences["edges_added"]) == 1


class TestExecutionMonitor:
    """Test execution monitoring"""
    
    def test_workflow_events(self):
        """Test workflow lifecycle events"""
        monitor = ExecutionMonitor()
        events = []
        
        def capture_event(event: ExecutionEvent):
            events.append(event)
        
        monitor.subscribe(capture_event)
        
        monitor.workflow_started("run1", "test goal", "graph1")
        monitor.workflow_completed("run1", "result", 1000.0)
        
        assert len(events) == 2
        assert events[0].event_type == ExecutionEventType.WORKFLOW_STARTED
        assert events[1].event_type == ExecutionEventType.WORKFLOW_COMPLETED
    
    def test_node_events(self):
        """Test node execution events"""
        monitor = ExecutionMonitor()
        events = []
        
        def capture_event(event: ExecutionEvent):
            events.append(event)
        
        monitor.subscribe(capture_event)
        
        from src.news_reporter.workflows.execution_context import ExecutionContext
        from src.news_reporter.workflows.node_result import NodeResult, NodeStatus
        context = ExecutionContext(node_id="node1")
        
        monitor.node_started("run1", "node1", "agent", context)
        
        result = NodeResult.success(state_updates={}, artifacts={"test": "data"})
        monitor.node_completed("run1", "node1", result, context)
        
        assert len(events) >= 2
        assert events[0].event_type == ExecutionEventType.NODE_STARTED
        assert events[0].node_id == "node1"
        assert events[1].event_type == ExecutionEventType.NODE_COMPLETED
    
    def test_get_event_history(self):
        """Test retrieving event history"""
        monitor = ExecutionMonitor()
        
        monitor.workflow_started("run1", "test goal")
        monitor.workflow_completed("run1", "result", 1000.0)
        
        history = monitor.get_event_history("run1")
        assert len(history) == 2
    
    def test_get_active_runs(self):
        """Test getting active runs"""
        monitor = ExecutionMonitor()
        
        monitor.workflow_started("run1", "test goal")
        monitor.workflow_started("run2", "another goal")
        
        active = monitor.get_active_runs()
        assert "run1" in active
        assert "run2" in active
        assert active["run1"]["status"] == "running"


class TestWorkflowTemplates:
    """Test workflow templates"""
    
    def test_list_templates(self):
        """Test listing templates"""
        registry = get_template_registry()
        templates = registry.list_all()
        
        assert len(templates) > 0
        assert any(t["template_id"] == "simple_linear" for t in templates)
    
    def test_get_template(self):
        """Test getting a template"""
        registry = get_template_registry()
        template = registry.get("simple_linear")
        
        assert template is not None
        assert template.template_id == "simple_linear"
        assert template.name == "Simple Linear Workflow"
    
    def test_instantiate_template(self):
        """Test instantiating a template"""
        registry = get_template_registry()
        template = registry.get("simple_linear")
        
        assert template is not None
        graph_def = template.instantiate()
        
        assert graph_def is not None
        assert len(graph_def.nodes) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
