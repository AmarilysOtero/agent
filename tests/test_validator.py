#!/usr/bin/env python3
"""
PR 2 Unit Tests: Workflow Validator
Tests validation rules for workflow graphs.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.news_reporter.workflows.validator import validate_workflow


def test_validator_rejects_multiple_roots():
    """Test that validator rejects graphs with multiple root nodes"""
    graph = {
        "nodes": [
            {"id": "start1", "type": "StartNode"},
            {"id": "start2", "type": "StartNode"},
            {"id": "msg", "type": "SendMessage", "config": {"message": "Hello"}}
        ],
        "edges": [
            {"source": "start1", "target": "msg"},
            {"source": "start2", "target": "msg"}
        ]
    }
    
    result = validate_workflow(graph)
    assert result.valid == False
    assert any(e.code == "MULTIPLE_ROOTS" for e in result.errors)


def test_validator_rejects_cycle():
    """Test that validator rejects graphs with cycles"""
    graph = {
        "nodes": [
            {"id": "start", "type": "StartNode"},
            {"id": "a", "type": "SendMessage", "config": {"message": "A"}},
            {"id": "b", "type": "SendMessage", "config": {"message": "B"}}
        ],
        "edges": [
            {"source": "start", "target": "a"},
            {"source": "a", "target": "b"},
            {"source": "b", "target": "a"}  # Cycle: a <-> b
        ]
    }
    
    result = validate_workflow(graph)
    assert result.valid == False
    assert any(e.code == "CYCLE_DETECTED" for e in result.errors)


def test_validator_rejects_orphan_node():
    """Test that validator rejects graphs with unreachable orphan nodes"""
    graph = {
        "nodes": [
            {"id": "start", "type": "StartNode"},
            {"id": "msg", "type": "SendMessage", "config": {"message": "Hello"}},
            {"id": "orphan_a", "type": "SendMessage", "config": {"message": "Orphan A"}},
            {"id": "orphan_b", "type": "SendMessage", "config": {"message": "Orphan B"}}
        ],
        "edges": [
            {"source": "start", "target": "msg"},
            # Disconnected subgraph: orphan_a -> orphan_b (not reachable from start)
            {"source": "orphan_a", "target": "orphan_b"}
        ]
    }
    
    result = validate_workflow(graph)
    assert result.valid == False
    # orphan_a is a second root (has in-degree 0)
    # This MUST trigger MULTIPLE_ROOTS error
    assert any(e.code == "MULTIPLE_ROOTS" for e in result.errors)


def test_validator_rejects_unknown_node_type():
    """Test that validator rejects unknown node types"""
    graph = {
        "nodes": [
            {"id": "start", "type": "StartNode"},
            {"id": "bad", "type": "UnknownType", "config": {}}
        ],
        "edges": [
            {"source": "start", "target": "bad"}
        ]
    }
    
    result = validate_workflow(graph)
    assert result.valid == False
    assert any(e.code == "UNKNOWN_NODE_TYPE" and e.nodeId == "bad" for e in result.errors)


def test_validator_rejects_missing_sendmessage_message():
    """Test that validator rejects SendMessage without message field"""
    graph = {
        "nodes": [
            {"id": "start", "type": "StartNode"},
            {"id": "msg", "type": "SendMessage", "config": {}}  # Missing message
        ],
        "edges": [
            {"source": "start", "target": "msg"}
        ]
    }
    
    result = validate_workflow(graph)
    assert result.valid == False
    assert any(e.code == "MISSING_MESSAGE" and e.nodeId == "msg" for e in result.errors)


def test_validator_rejects_missing_invokeagent_agentId():
    """Test that validator rejects InvokeAgent without agentId field"""
    graph = {
        "nodes": [
            {"id": "start", "type": "StartNode"},
            {"id": "agent", "type": "InvokeAgent", "config": {}}  # Missing agentId
        ],
        "edges": [
            {"source": "start", "target": "agent"}
        ]
    }
    
    result = validate_workflow(graph)
    assert result.valid == False
    assert any(e.code == "MISSING_AGENT_ID" and e.nodeId == "agent" for e in result.errors)


def test_validator_accepts_valid_diamond_graph():
    """Test that validator accepts a valid diamond graph (fan-out/fan-in)"""
    graph = {
        "nodes": [
            {"id": "start", "type": "StartNode"},
            {"id": "b", "type": "SendMessage", "config": {"message": "Hello from B"}},
            {"id": "c", "type": "SendMessage", "config": {"message": "Hello from C"}},
            {"id": "d", "type": "InvokeAgent", "config": {"agentId": "test-agent-id", "input": "optional prefix"}}
        ],
        "edges": [
            {"source": "start", "target": "b"},
            {"source": "start", "target": "c"},
            {"source": "b", "target": "d"},
            {"source": "c", "target": "d"}
        ]
    }
    
    result = validate_workflow(graph)
    assert result.valid == True
    assert len(result.errors) == 0


def test_validator_rejects_non_startnode_root():
    """Test that validator rejects root node that is not StartNode"""
    graph = {
        "nodes": [
            {"id": "msg", "type": "SendMessage", "config": {"message": "Hello"}},
            {"id": "agent", "type": "InvokeAgent", "config": {"agentId": "test-agent"}}
        ],
        "edges": [
            {"source": "msg", "target": "agent"}
        ]
    }
    
    result = validate_workflow(graph)
    assert result.valid == False
    assert any(e.code == "INVALID_ROOT_TYPE" for e in result.errors)


def test_validator_rejects_self_loop():
    """Test that validator rejects self-loop edges"""
    graph = {
        "nodes": [
            {"id": "start", "type": "StartNode"},
            {"id": "loop", "type": "SendMessage", "config": {"message": "Loop"}}
        ],
        "edges": [
            {"source": "start", "target": "loop"},
            {"source": "loop", "target": "loop"}  # Self-loop
        ]
    }
    
    result = validate_workflow(graph)
    assert result.valid == False
    assert any(e.code == "SELF_LOOP" for e in result.errors)


if __name__ == "__main__":
    # Run tests manually
    print("Running validator tests...")
    
    test_validator_rejects_multiple_roots()
    print("[PASS] test_validator_rejects_multiple_roots")
    
    test_validator_rejects_cycle()
    print("[PASS] test_validator_rejects_cycle")
    
    test_validator_rejects_orphan_node()
    print("[PASS] test_validator_rejects_orphan_node")
    
    test_validator_rejects_unknown_node_type()
    print("[PASS] test_validator_rejects_unknown_node_type")
    
    test_validator_rejects_missing_sendmessage_message()
    print("[PASS] test_validator_rejects_missing_sendmessage_message")
    
    test_validator_rejects_missing_invokeagent_agentId()
    print("[PASS] test_validator_rejects_missing_invokeagent_agentId")
    
    test_validator_accepts_valid_diamond_graph()
    print("[PASS] test_validator_accepts_valid_diamond_graph")
    
    test_validator_rejects_non_startnode_root()
    print("[PASS] test_validator_rejects_non_startnode_root")
    
    test_validator_rejects_self_loop()
    print("[PASS] test_validator_rejects_self_loop")
    
    print("\n=== All validator tests PASSED ===")

