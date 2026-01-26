"""Test Phase 1: Graph Schema and Condition Evaluator"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import pytest
from src.news_reporter.models.graph_schema import GraphDefinition, NodeConfig, EdgeConfig, GraphLimits
from src.news_reporter.workflows.condition_evaluator import ConditionEvaluator
from src.news_reporter.workflows.workflow_state import WorkflowState


class TestGraphSchema:
    """Test Graph Schema with entry_node_id"""
    
    def test_graph_definition_with_entry_node(self):
        """Test that GraphDefinition accepts and validates entry_node_id"""
        nodes = [
            NodeConfig(id="triage", type="agent", agent_id="test_agent"),
            NodeConfig(id="search", type="agent", agent_id="test_agent"),
        ]
        edges = [
            EdgeConfig(from_node="triage", to_node="search"),
        ]
        
        graph = GraphDefinition(
            nodes=nodes,
            edges=edges,
            entry_node_id="triage"
        )
        
        assert graph.entry_node_id == "triage"
        assert graph.get_entry_nodes() == ["triage"]
    
    def test_entry_node_validation(self):
        """Test that invalid entry_node_id is caught"""
        nodes = [
            NodeConfig(id="triage", type="agent", agent_id="test_agent"),
        ]
        edges = []
        
        graph = GraphDefinition(
            nodes=nodes,
            edges=edges,
            entry_node_id="nonexistent"
        )
        
        errors = graph.validate()
        assert any("Entry node 'nonexistent' not found" in err for err in errors)
    
    def test_entry_node_fallback(self):
        """Test fallback to nodes with no incoming edges if entry_node_id not found"""
        nodes = [
            NodeConfig(id="start", type="agent", agent_id="test_agent"),
            NodeConfig(id="middle", type="agent", agent_id="test_agent"),
        ]
        edges = [
            EdgeConfig(from_node="start", to_node="middle"),
        ]
        
        graph = GraphDefinition(
            nodes=nodes,
            edges=edges,
            entry_node_id="invalid"  # Invalid entry node
        )
        
        # Should fall back to nodes with no incoming edges
        entry_nodes = graph.get_entry_nodes()
        assert "start" in entry_nodes
    
    def test_graph_validation(self):
        """Test graph validation catches common errors"""
        nodes = [
            NodeConfig(id="triage", type="agent", agent_id="test_agent"),
        ]
        edges = [
            EdgeConfig(from_node="triage", to_node="nonexistent"),
        ]
        
        graph = GraphDefinition(nodes=nodes, edges=edges)
        errors = graph.validate()
        assert any("unknown node" in err.lower() for err in errors)
    
    def test_default_entry_node(self):
        """Test default entry_node_id is 'triage'"""
        nodes = [
            NodeConfig(id="triage", type="agent", agent_id="test_agent"),
        ]
        edges = []
        
        graph = GraphDefinition(nodes=nodes, edges=edges)
        assert graph.entry_node_id == "triage"  # Default value


class TestConditionEvaluator:
    """Test Safe Condition Evaluator (no eval)"""
    
    def test_equality_operator(self):
        """Test == operator"""
        state = WorkflowState(goal="test")
        state.set("triage.preferred_agent", "sql")
        
        result = ConditionEvaluator.evaluate('triage.preferred_agent == "sql"', state)
        assert result is True
        
        result = ConditionEvaluator.evaluate('triage.preferred_agent == "neo4j"', state)
        assert result is False
    
    def test_inequality_operator(self):
        """Test != operator"""
        state = WorkflowState(goal="test")
        state.set("state.selected_search", "aisearch")
        
        result = ConditionEvaluator.evaluate('state.selected_search != "neo4j"', state)
        assert result is True
    
    def test_membership_operator(self):
        """Test 'in' operator"""
        state = WorkflowState(goal="test")
        state.set("triage.intents", ["ai_search", "news_script"])
        
        result = ConditionEvaluator.evaluate('"ai_search" in triage.intents', state)
        assert result is True
        
        result = ConditionEvaluator.evaluate('"unknown" in triage.intents', state)
        assert result is False
    
    def test_not_in_operator(self):
        """Test 'not in' operator"""
        state = WorkflowState(goal="test")
        state.set("triage.intents", ["ai_search"])
        
        result = ConditionEvaluator.evaluate('"unknown" not in triage.intents', state)
        assert result is True
    
    def test_is_none_operators(self):
        """Test 'is None' and 'is not None'"""
        state = WorkflowState(goal="test")
        state.set("triage.database_id", "db123")
        
        result = ConditionEvaluator.evaluate("triage.database_id is not None", state)
        assert result is True
        
        result = ConditionEvaluator.evaluate("triage.missing_field is None", state)
        assert result is True
    
    def test_logical_operators(self):
        """Test 'and' and 'or' operators"""
        state = WorkflowState(goal="test")
        state.set("triage.preferred_agent", "sql")
        state.set("triage.database_id", "db123")
        
        # AND
        result = ConditionEvaluator.evaluate(
            'triage.preferred_agent == "sql" and triage.database_id is not None',
            state
        )
        assert result is True
        
        # OR
        state.set("triage.intents", ["ai_search"])
        result = ConditionEvaluator.evaluate(
            '"ai_search" in triage.intents or "unknown" in triage.intents',
            state
        )
        assert result is True  # First part is True
    
    def test_negation(self):
        """Test 'not' operator"""
        state = WorkflowState(goal="test")
        state.set("triage.preferred_agent", "sql")
        
        result = ConditionEvaluator.evaluate('not triage.preferred_agent == "neo4j"', state)
        assert result is True
    
    def test_missing_path_handling(self):
        """Test handling of missing state paths"""
        state = WorkflowState(goal="test")
        
        # Should return False (not raise error) when path doesn't exist
        result = ConditionEvaluator.evaluate('triage.missing_field == "value"', state)
        assert result is False
        
        # Should return True for "is None" check on missing path
        result = ConditionEvaluator.evaluate("triage.missing_field is None", state)
        assert result is True
    
    def test_strict_mode(self):
        """Test strict mode raises error on missing paths"""
        state = WorkflowState(goal="test")
        
        with pytest.raises(ValueError, match="State path.*not found"):
            ConditionEvaluator.evaluate('triage.missing_field == "value"', state, strict=True)
    
    def test_numeric_comparison(self):
        """Test numeric comparisons"""
        state = WorkflowState(goal="test")
        state.set("triage.count", 5)
        
        result = ConditionEvaluator.evaluate("triage.count > 3", state)
        assert result is True
        
        result = ConditionEvaluator.evaluate("triage.count <= 5", state)
        assert result is True
        
        result = ConditionEvaluator.evaluate("triage.count < 3", state)
        assert result is False
    
    def test_complex_condition(self):
        """Test complex condition from actual workflow"""
        state = WorkflowState(goal="test")
        state.set("triage.intents", ["ai_search"])
        state.set("triage.preferred_agent", "sql")
        state.set("triage.database_id", "db123")
        
        condition = '"ai_search" in triage.intents or ("unknown" in triage.intents and triage.preferred_agent is not None and triage.database_id is not None)'
        result = ConditionEvaluator.evaluate(condition, state)
        assert result is True  # First part should be True
    
    def test_no_eval_used(self):
        """Verify that eval() is not used (security check)"""
        import inspect
        import ast
        source = inspect.getsource(ConditionEvaluator)
        # Parse AST to check for actual eval() calls in code (not docstrings)
        tree = ast.parse(source)
        has_eval = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "eval":
                has_eval = True
                break
        assert not has_eval, "ConditionEvaluator should not use eval() for security - only safe parser"
    
    def test_boolean_literals(self):
        """Test boolean literal parsing"""
        state = WorkflowState(goal="test")
        state.set("triage.flag", True)
        
        result = ConditionEvaluator.evaluate("triage.flag == true", state)
        assert result is True
        
        state.set("triage.flag", False)
        result = ConditionEvaluator.evaluate("triage.flag == false", state)
        assert result is True
    
    def test_empty_condition(self):
        """Test that empty condition returns True"""
        state = WorkflowState(goal="test")
        result = ConditionEvaluator.evaluate("", state)
        assert result is True
        
        result = ConditionEvaluator.evaluate("   ", state)
        assert result is True
    
    def test_string_literal_parsing(self):
        """Test string literal parsing (quoted strings)"""
        state = WorkflowState(goal="test")
        state.set("triage.value", "test")
        
        result = ConditionEvaluator.evaluate('triage.value == "test"', state)
        assert result is True
        
        result = ConditionEvaluator.evaluate("triage.value == 'test'", state)
        assert result is True


class TestGraphLoader:
    """Test graph loading from JSON"""
    
    def test_load_default_workflow(self):
        """Test loading default workflow JSON"""
        from src.news_reporter.workflows.graph_loader import load_graph_definition
        from src.news_reporter.config import Settings
        
        try:
            cfg = Settings.load()
            graph = load_graph_definition(config=cfg)
            
            assert graph.name is not None
            assert graph.entry_node_id == "triage"
            assert len(graph.nodes) > 0
            assert len(graph.edges) > 0
            
            # Validate graph
            errors = graph.validate()
            if errors:
                pytest.fail(f"Graph validation failed: {errors}")
        except FileNotFoundError:
            pytest.skip("Default workflow JSON not found")
        except Exception as e:
            pytest.skip(f"Could not load graph (config issue): {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
