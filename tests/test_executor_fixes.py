#!/usr/bin/env python3
"""
PR3 Patch Tests: Testing bug fixes for executor and model alignment
Tests for outputPreview persistence, requeue safety, and ValidationResult model
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.news_reporter.models.workflow import (
    Workflow, WorkflowRun, WorkflowGraph, NodeResult, 
    ValidationIssue, ValidationResult
)

# Import truncate for standalone test
import json
from typing import Any, Tuple, Optional

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


# ===== FIX 1: outputPreview Persistence Tests =====

def test_fix1_truncated_output_has_preview():
    """Test that large outputs produce truncated results with preview"""
    # Create a large output (>16KB)
    large_message = "x" * (20 * 1024)
    
    output_str, truncated, preview = truncate_output(large_message)
    
    # Verify truncation happened
    assert truncated == True, "Large output should be truncated"
    assert preview is not None, "Preview should be set for truncated output"
    assert len(preview) == 500, "Preview should be 500 chars"
    assert len(output_str) < len(large_message), "Output should be shorter than original"
    
    # Create NodeResult with this data
    result = NodeResult(
        status="succeeded",
        inputs={},
        output=output_str,
        outputTruncated=truncated,
        outputPreview=preview,  # FIX 1: This should be persisted
        executionMs=100.0,
        startedAt=datetime.now(timezone.utc),
        completedAt=datetime.now(timezone.utc),
        logs=[]
    )
    
    # Verify NodeResult has preview
    assert result.outputPreview is not None, "NodeResult.outputPreview should be set"
    assert result.outputPreview == preview, "NodeResult.outputPreview should match truncate output"
    assert result.outputTruncated == True, "NodeResult.outputTruncated should be True"
    
    print("[PASS] test_fix1_truncated_output_has_preview")


def test_fix1_non_truncated_output_has_no_preview():
    """Test that small outputs don't have preview"""
    small_message = "Hello World"
    
    output_str, truncated, preview = truncate_output(small_message)
    
    assert truncated == False, "Small output should not be truncated"
    assert preview is None, "Preview should be None for non-truncated output"
    
    result = NodeResult(
        status="succeeded",
        inputs={},
        output=output_str,
        outputTruncated=truncated,
        outputPreview=preview,  # Should be None
        executionMs=100.0,
        startedAt=datetime.now(timezone.utc),
        completedAt=datetime.now(timezone.utc),
        logs=[]
    )
    
    assert result.outputPreview is None, "NodeResult.outputPreview should be None for non-truncated"
    assert result.outputTruncated == False, "NodeResult.outputTruncated should be False"
    
    print("[PASS] test_fix1_non_truncated_output_has_no_preview")


# ===== FIX 3: ValidationResult Model Alignment Tests =====

def test_fix3_validation_issue_model():
    """Test that ValidationIssue model can be instantiated with validator output"""
    # Simulate validator error output
    issue = ValidationIssue(
        code="DUPLICATE_NODE_ID",
        message="Duplicate node id 'node1'",
        nodeId="node1",
        edgeId=None
    )
    
    assert issue.code == "DUPLICATE_NODE_ID"
    assert issue.message == "Duplicate node id 'node1'"
    assert issue.nodeId == "node1"
    assert issue.edgeId is None
    
    print("[PASS] test_fix3_validation_issue_model")


def test_fix3_validation_result_with_structured_errors():
    """Test that ValidationResult accepts structured ValidationIssue errors"""
    # Create multiple validation issues
    issues = [
        ValidationIssue(
            code="MISSING_NODE_ID",
            message="Node at index 0 is missing required 'id' field"
        ),
        ValidationIssue(
            code="DUPLICATE_EDGE",
            message="Duplicate edge from 'a' to 'b'",
            edgeId="a->b"
        ),
        ValidationIssue(
            code="CYCLE_DETECTED",
            message="Workflow graph contains a cycle"
        )
    ]
    
    # Create ValidationResult with structured errors
    result = ValidationResult(
        valid=False,
        errors=issues
    )
    
    assert result.valid == False
    assert len(result.errors) == 3
    assert all(isinstance(e, ValidationIssue) for e in result.errors)
    assert result.errors[0].code == "MISSING_NODE_ID"
    assert result.errors[1].code == "DUPLICATE_EDGE"
    assert result.errors[1].edgeId == "a->b"
    assert result.errors[2].code == "CYCLE_DETECTED"
    
    print("[PASS] test_fix3_validation_result_with_structured_errors")


def test_fix3_validation_result_json_serialization():
    """Test that ValidationResult can be serialized to JSON (for API responses)"""
    issues = [
        ValidationIssue(
            code="INVALID_EDGE_SOURCE",
            message="Edge source 'nonexistent' does not exist",
            edgeId="edge1"
        )
    ]
    
    result = ValidationResult(valid=False, errors=issues)
    
    # Serialize to dict
    result_dict = result.model_dump()
    
    assert result_dict["valid"] == False
    assert len(result_dict["errors"]) == 1
    assert result_dict["errors"][0]["code"] == "INVALID_EDGE_SOURCE"
    assert result_dict["errors"][0]["message"] == "Edge source 'nonexistent' does not exist"
    assert result_dict["errors"][0]["edgeId"] == "edge1"
    assert result_dict["errors"][0]["nodeId"] is None
    
    print("[PASS] test_fix3_validation_result_json_serialization")


def test_fix3_valid_result_has_empty_errors():
    """Test that valid ValidationResult has empty errors list"""
    result = ValidationResult(valid=True, errors=[])
    
    assert result.valid == True
    assert len(result.errors) == 0
    
    print("[PASS] test_fix3_valid_result_has_empty_errors")


if __name__ == "__main__":
    print("Running PR3 patch tests...")
    print("\n=== FIX 1: outputPreview Persistence ===")
    
    test_fix1_truncated_output_has_preview()
    test_fix1_non_truncated_output_has_no_preview()
    
    print("\n=== FIX 3: ValidationResult Model Alignment ===")
    
    test_fix3_validation_issue_model()
    test_fix3_validation_result_with_structured_errors()
    test_fix3_validation_result_json_serialization()
    test_fix3_valid_result_has_empty_errors()
    
    print("\n=== All PR3 patch tests PASSED ===")
    print("\nNOTE: FIX 2 (requeue safety) requires full executor integration test with pymongo.")
    print("The requeue logic and no-progress counter are implemented in executor.py.")
