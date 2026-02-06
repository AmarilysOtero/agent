"""Quick test script for Phase 1 - Run this to verify Phase 1 works

Usage:
    # With venv (Python 3.11)
    .venv\Scripts\activate
    python test_phase1_quick.py
    
    # Or with pytest
    pytest tests/unit/workflows/test_phase1.py -v
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.news_reporter.models.graph_schema import GraphDefinition, NodeConfig, EdgeConfig
    from src.news_reporter.workflows.condition_evaluator import ConditionEvaluator
    from src.news_reporter.workflows.workflow_state import WorkflowState
    from src.news_reporter.workflows.graph_loader import load_graph_definition
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nMake sure you're using Python 3.9+ (preferably 3.11)")
    print("Activate venv: .venv\\Scripts\\activate")
    sys.exit(1)

def test_graph_schema():
    """Test Graph Schema"""
    print("=" * 60)
    print("TEST 1: Graph Schema with entry_node_id")
    print("=" * 60)
    
    nodes = [
        NodeConfig(id="triage", type="agent", agent_id="test_agent"),
        NodeConfig(id="search", type="agent", agent_id="test_agent"),
    ]
    edges = [
        EdgeConfig(from_node="triage", to_node="search"),
    ]
    
    graph = GraphDefinition(nodes=nodes, edges=edges, entry_node_id="triage")
    
    print(f"[OK] Entry node ID: {graph.entry_node_id}")
    print(f"[OK] Entry nodes: {graph.get_entry_nodes()}")
    
    errors = graph.validate()
    if errors:
        print(f"[FAIL] Validation errors: {errors}")
        return False
    else:
        print("[OK] Graph validation passed")
        return True


def test_condition_evaluator():
    """Test Condition Evaluator"""
    print("\n" + "=" * 60)
    print("TEST 2: Condition Evaluator (Safe Parser)")
    print("=" * 60)
    
    state = WorkflowState(goal="test")
    state.set("triage.preferred_agent", "sql")
    state.set("triage.intents", ["ai_search"])
    state.set("triage.database_id", "db123")
    
    tests = [
        ('triage.preferred_agent == "sql"', True, "Equality"),
        ('"ai_search" in triage.intents', True, "Membership"),
        ('triage.database_id is not None', True, "Is not None"),
        ('triage.missing_field is None', True, "Missing field is None"),
        ('triage.preferred_agent == "neo4j"', False, "Inequality"),
        ('"ai_search" in triage.intents and triage.preferred_agent == "sql"', True, "AND operator"),
    ]
    
    all_passed = True
    for condition, expected, description in tests:
        try:
            result = ConditionEvaluator.evaluate(condition, state)
            status = "[OK]" if result == expected else "[FAIL]"
            print(f"{status} {description}: {condition} = {result} (expected {expected})")
            if result != expected:
                all_passed = False
        except Exception as e:
            print(f"[FAIL] {description}: Error - {e}")
            all_passed = False
    
    # Security check - look for eval( in actual code, not docstrings
    import inspect
    import ast
    source = inspect.getsource(ConditionEvaluator)
    # Parse AST to check for actual eval() calls in code
    try:
        tree = ast.parse(source)
        has_eval = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "eval":
                has_eval = True
                break
        if has_eval:
            print("[FAIL] SECURITY ISSUE: eval() function call found in ConditionEvaluator code!")
            all_passed = False
        else:
            print("[OK] Security: No eval() function calls found - safe parser only")
    except Exception as e:
        print(f"[WARN] Could not parse AST for security check: {e}")
        # Fallback: check if "eval(" appears outside of strings/comments
        # Simple heuristic: check if it's in a call pattern
        import re
        # Look for eval( as a function call (not in strings)
        if re.search(r'\beval\s*\(', source):
            # Check if it's in a string literal
            in_string = False
            for line in source.split('\n'):
                if '"""' in line or "'''" in line:
                    in_string = not in_string
                if not in_string and re.search(r'\beval\s*\(', line):
                    print("[FAIL] SECURITY ISSUE: eval() found in ConditionEvaluator code!")
                    all_passed = False
                    break
            else:
                print("[OK] Security: No eval() function calls found - safe parser only")
        else:
            print("[OK] Security: No eval() function calls found - safe parser only")
    
    return all_passed


def test_complex_condition():
    """Test complex condition from actual workflow"""
    print("\n" + "=" * 60)
    print("TEST 3: Complex Condition (Real Workflow Example)")
    print("=" * 60)
    
    state = WorkflowState(goal="list the names")
    state.triage = {
        "preferred_agent": "sql",
        "database_id": "db123",
        "intents": ["ai_search"]
    }
    
    # Complex condition from default_workflow.json
    condition = '"ai_search" in triage.intents or ("unknown" in triage.intents and triage.preferred_agent is not None and triage.database_id is not None)'
    
    try:
        result = ConditionEvaluator.evaluate(condition, state)
        print(f"[OK] Complex condition evaluated: {result}")
        print(f"   Condition: {condition}")
        return True
    except Exception as e:
        print(f"[FAIL] Complex condition failed: {e}")
        return False


def test_graph_loading():
    """Test loading default workflow JSON"""
    print("\n" + "=" * 60)
    print("TEST 4: Load Default Workflow JSON")
    print("=" * 60)
    
    try:
        from src.news_reporter.config import Settings
        cfg = Settings.load()
        graph = load_graph_definition(config=cfg)
        
        print(f"[OK] Graph loaded: {graph.name}")
        print(f"[OK] Entry node: {graph.entry_node_id}")
        print(f"[OK] Nodes: {len(graph.nodes)}")
        print(f"[OK] Edges: {len(graph.edges)}")
        
        errors = graph.validate()
        if errors:
            print(f"[WARN] Validation warnings: {errors}")
        else:
            print("[OK] Graph validation passed")
        
        return True
    except FileNotFoundError as e:
        print(f"[WARN] Could not find default workflow JSON: {e}")
        return False
    except Exception as e:
        print(f"[WARN] Could not load graph (config issue): {e}")
        return False


def main():
    """Run all Phase 1 tests"""
    print("\n" + "=" * 60)
    print("PHASE 1 TEST SUITE")
    print("=" * 60)
    
    results = []
    
    results.append(("Graph Schema", test_graph_schema()))
    results.append(("Condition Evaluator", test_condition_evaluator()))
    results.append(("Complex Condition", test_complex_condition()))
    results.append(("Graph Loading", test_graph_loading()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n[SUCCESS] All Phase 1 tests passed!")
        return 0
    else:
        print("\n[WARNING] Some tests failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
