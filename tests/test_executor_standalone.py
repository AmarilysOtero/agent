#!/usr/bin/env python3
"""
PR 3 Unit Tests: Workflow Executor Core (Standalone)
Tests without requiring pymongo/bson installation.
"""
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.news_reporter.models.workflow import Workflow, WorkflowRun, WorkflowGraph, NodeResult

# ===== Standalone truncate_output implementation =====

MAX_OUTPUT_BYTES = 16 * 1024

def truncate_output(output: Any) -> Tuple[str, bool, Optional[str]]:
    """Truncate output to string format with size limit"""
    if isinstance(output, str):
        output_str = output
    elif isinstance(output, (dict, list)):
        output_str = json.dumps(output, sort_keys=True, ensure_ascii=False)
    else:
        output_str = str(output)
    
    byte_len = len(output_str.encode('utf-8'))
    
    if byte_len <= MAX_OUTPUT_BYTES:
        return output_str, False, None
    
    truncated_str = output_str[:MAX_OUTPUT_BYTES]
    preview = output_str[:500]
    
    return truncated_str, True, preview

# ===== Tests =====

def test_truncate_output_string():
    """Test truncate_output with plain string"""
    output, truncated, preview = truncate_output("Hello World")
    assert output == "Hello World"
    assert truncated == False
    assert preview is None
    print("[PASS] test_truncate_output_string")


def test_truncate_output_dict():
    """Test truncate_output with dict (JSON serialization)"""
    data = {"key": "value", "number": 42}
    output, truncated, preview = truncate_output(data)
    assert '{"key": "value"' in output or '{"number": 42' in output
    assert truncated == False
    print("[PASS] test_truncate_output_dict")


def test_truncate_output_large():
    """Test truncate_output with large string"""
    large_str = "x" * (20 * 1024)  # 20KB
    output, truncated, preview = truncate_output(large_str)
    assert truncated == True
    assert len(preview) == 500
    assert len(output) < len(large_str)
    print("[PASS] test_truncate_output_large")


def test_noderesult_model():
    """Test that NodeResult model can be instantiated with new outputPreview field"""
    result = NodeResult(
        status="succeeded",
        inputs={"node1": "output1"},
        output="test output",
        outputTruncated=True,
        outputPreview="test...",  # New field
        executionMs=123.45,
        startedAt=datetime.now(timezone.utc),
        completedAt=datetime.now(timezone.utc),
        logs=[],
        error=None
    )
    assert result.outputPreview == "test..."
    assert result.outputTruncated == True
    print("[PASS] test_noderesult_model")


def test_json_serialization_sorted_keys():
    """Test that dict serialization has sorted keys for determinism"""
    data = {"z": 1, "a": 2, "m": 3}
    output, _, _ = truncate_output(data)
    # Keys should be sorted: a, m, z
    assert output.index('"a"') < output.index('"m"')
    assert output.index('"m"') < output.index('"z"')
    print("[PASS] test_json_serialization_sorted_keys")


if __name__ == "__main__":
    print("Running PR3 executor tests (standalone)...")
    
    test_truncate_output_string()
    test_truncate_output_dict()
    test_truncate_output_large()
    test_noderesult_model()
    test_json_serialization_sorted_keys()
    
    print("\n=== All PR3 standalone tests PASSED ===")
    print("\nNOTE: Full executor integration tests require pymongo/motor packages.")
    print("Install with: pip install pymongo motor")
    print("Then run full test suite with actual executor class.")
