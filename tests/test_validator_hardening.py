#!/usr/bin/env python3
"""
PR 2 Hardening Tests: Edge Case Validation
Tests for duplicate IDs, missing fields, duplicate edges
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.news_reporter.workflows.validator import validate_workflow


def test_validator_rejects_duplicate_node_ids():
    """Test that validator rejects graphs with duplicate node IDs"""
    graph = {
        "nodes": [
            {"id": "node1", "type": "StartNode"},
            {"id": "node1", "type": "SendMessage", "config": {"message": "Duplicate ID"}},  # Same ID
        ],
        "edges": []
    }
    
    result = validate_workflow(graph)
    assert result.valid == False
    assert any(e.code == "DUPLICATE_NODE_ID" for e in result.errors)
    # Should have exactly one duplicate error for "node1"
    duplicate_errors = [e for e in result.errors if e.code == "DUPLICATE_NODE_ID"]
    assert len(duplicate_errors) == 1
    assert "node1" in duplicate_errors[0].message


def test_validator_rejects_missing_node_id():
    """Test that validator rejects nodes without id field"""
    graph = {
        "nodes": [
            {"type": "StartNode"},  # Missing id
        ],
        "edges": []
    }
    
    result = validate_workflow(graph)
    assert result.valid == False
    assert any(e.code == "MISSING_NODE_ID" for e in result.errors)


def test_validator_rejects_empty_node_id():
    """Test that validator rejects nodes with empty id"""
    graph = {
        "nodes": [
            {"id": "", "type": "StartNode"},  # Empty id
        ],
        "edges": []
    }
    
    result = validate_workflow(graph)
    assert result.valid == False
    assert any(e.code == "MISSING_NODE_ID" for e in result.errors)


def test_validator_rejects_missing_node_type():
    """Test that validator rejects nodes without type field"""
    graph = {
        "nodes": [
            {"id": "node1"},  # Missing type
        ],
        "edges": []
    }
    
    result = validate_workflow(graph)
    assert result.valid == False
    assert any(e.code == "MISSING_NODE_TYPE" for e in result.errors)


def test_validator_rejects_empty_node_type():
    """Test that validator rejects nodes with empty type"""
    graph = {
        "nodes": [
            {"id": "node1", "type": ""},  # Empty type
        ],
        "edges": []
    }
    
    result = validate_workflow(graph)
    assert result.valid == False
    assert any(e.code == "MISSING_NODE_TYPE" for e in result.errors)


def test_validator_rejects_duplicate_edges():
    """Test that validator detects duplicate edges"""
    graph = {
        "nodes": [
            {"id": "start", "type": "StartNode"},
            {"id": "msg", "type": "SendMessage", "config": {"message": "Hello"}}
        ],
        "edges": [
            {"source": "start", "target": "msg"},
            {"source": "start", "target": "msg"},  # Duplicate
            {"source": "start", "target": "msg"}   # Duplicate
        ]
    }
    
    result = validate_workflow(graph)
    assert result.valid == False
    duplicate_errors = [e for e in result.errors if e.code == "DUPLICATE_EDGE"]
    assert len(duplicate_errors) == 2  # Two duplicates of the same edge


def test_validator_accepts_valid_graph_despite_duplicate_edges():
    """Test that validator still validates correctly after deduplicating edges"""
    graph = {
        "nodes": [
            {"id": "start", "type": "StartNode"},
            {"id": "msg", "type": "SendMessage", "config": {"message": "Hello"}}
        ],
        "edges": [
            {"source": "start", "target": "msg"},
            {"source": "start", "target": "msg"}  # Duplicate (will be deduplicated)
        ]
    }
    
    result = validate_workflow(graph)
    # Should have DUPLICATE_EDGE error but graph structure should still be validated
    assert any(e.code == "DUPLICATE_EDGE" for e in result.errors)
    # No other structural errors (like MULTIPLE_ROOTS) should be present
    # because deduplication should happen before in-degree calculation
    structural_errors = [e for e in result.errors if e.code not in ["DUPLICATE_EDGE"]]
    assert len(structural_errors) == 0


if __name__ == "__main__":
    # Run edge case tests
    print("Running PR2 hardening tests...")
    
    test_validator_rejects_duplicate_node_ids()
    print("[PASS] test_validator_rejects_duplicate_node_ids")
    
    test_validator_rejects_missing_node_id()
    print("[PASS] test_validator_rejects_missing_node_id")
    
    test_validator_rejects_empty_node_id()
    print("[PASS] test_validator_rejects_empty_node_id")
    
    test_validator_rejects_missing_node_type()
    print("[PASS] test_validator_rejects_missing_node_type")
    
    test_validator_rejects_empty_node_type()
    print("[PASS] test_validator_rejects_empty_node_type")
    
    test_validator_rejects_duplicate_edges()
    print("[PASS] test_validator_rejects_duplicate_edges")
    
    test_validator_accepts_valid_graph_despite_duplicate_edges()
    print("[PASS] test_validator_accepts_valid_graph_despite_duplicate_edges")
    
    print("\n=== All PR2 hardening tests PASSED ===")
