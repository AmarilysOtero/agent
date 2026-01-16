"""Workflow Tester - Testing framework for workflows"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Callable
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum

from .graph_schema import GraphDefinition
from .workflow_state import WorkflowState
from .node_result import NodeResult, NodeStatus

logger = logging.getLogger(__name__)


class TestStatus(str, Enum):
    """Test execution status"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestCase:
    """A test case for a workflow"""
    test_id: str
    name: str
    description: str
    goal: str
    expected_output: Optional[str] = None
    expected_nodes: Optional[List[str]] = None
    expected_state: Optional[Dict[str, Any]] = None
    timeout_seconds: float = 60.0
    enabled: bool = True


@dataclass
class TestResult:
    """Result of a test execution"""
    test_id: str
    status: TestStatus
    duration_ms: float
    actual_output: Optional[str] = None
    error: Optional[str] = None
    executed_nodes: List[str] = field(default_factory=list)
    state_snapshot: Optional[Dict[str, Any]] = None
    assertions: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "test_id": self.test_id,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "actual_output": self.actual_output,
            "error": self.error,
            "executed_nodes": self.executed_nodes,
            "state_snapshot": self.state_snapshot,
            "assertions": self.assertions
        }


class WorkflowTester:
    """Testing framework for workflows"""
    
    def __init__(self, graph_def: GraphDefinition):
        self.graph_def = graph_def
        self.test_cases: Dict[str, TestCase] = {}
    
    def add_test_case(self, test_case: TestCase) -> None:
        """Add a test case"""
        self.test_cases[test_case.test_id] = test_case
        logger.info(f"Added test case: {test_case.test_id}")
    
    async def run_test(
        self,
        test_id: str,
        execute_workflow: Callable[[str], Any]
    ) -> TestResult:
        """
        Run a single test case.
        
        Args:
            test_id: Test case identifier
            execute_workflow: Function to execute the workflow
        
        Returns:
            TestResult
        """
        test_case = self.test_cases.get(test_id)
        if not test_case:
            return TestResult(
                test_id=test_id,
                status=TestStatus.ERROR,
                duration_ms=0.0,
                error=f"Test case {test_id} not found"
            )
        
        if not test_case.enabled:
            return TestResult(
                test_id=test_id,
                status=TestStatus.SKIPPED,
                duration_ms=0.0
            )
        
        import time
        start_time = time.time()
        assertions = []
        
        try:
            # Execute workflow
            actual_output = await asyncio.wait_for(
                execute_workflow(test_case.goal),
                timeout=test_case.timeout_seconds
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Run assertions
            status = TestStatus.PASSED
            
            # Check output
            if test_case.expected_output:
                if test_case.expected_output not in str(actual_output):
                    assertions.append({
                        "type": "output_match",
                        "status": "failed",
                        "expected": test_case.expected_output,
                        "actual": str(actual_output)[:100]
                    })
                    status = TestStatus.FAILED
                else:
                    assertions.append({
                        "type": "output_match",
                        "status": "passed"
                    })
            
            # Check executed nodes (would need to track this during execution)
            if test_case.expected_nodes:
                # This would require integration with executor
                assertions.append({
                    "type": "nodes_executed",
                    "status": "info",
                    "note": "Node execution tracking requires executor integration"
                })
            
            return TestResult(
                test_id=test_id,
                status=status,
                duration_ms=duration_ms,
                actual_output=str(actual_output),
                executed_nodes=[],
                assertions=assertions
            )
        
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                test_id=test_id,
                status=TestStatus.FAILED,
                duration_ms=duration_ms,
                error=f"Test timed out after {test_case.timeout_seconds}s"
            )
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                test_id=test_id,
                status=TestStatus.ERROR,
                duration_ms=duration_ms,
                error=str(e)
            )
    
    async def run_all_tests(
        self,
        execute_workflow: Callable[[str], Any]
    ) -> Dict[str, TestResult]:
        """
        Run all test cases.
        
        Returns:
            Dictionary of test_id -> TestResult
        """
        results = {}
        
        for test_id in self.test_cases.keys():
            result = await self.run_test(test_id, execute_workflow)
            results[test_id] = result
        
        return results
    
    def get_test_summary(self, results: Dict[str, TestResult]) -> Dict[str, Any]:
        """Get summary of test results"""
        total = len(results)
        passed = sum(1 for r in results.values() if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results.values() if r.status == TestStatus.FAILED)
        skipped = sum(1 for r in results.values() if r.status == TestStatus.SKIPPED)
        errors = sum(1 for r in results.values() if r.status == TestStatus.ERROR)
        
        total_duration = sum(r.duration_ms for r in results.values())
        avg_duration = total_duration / total if total > 0 else 0.0
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "errors": errors,
            "pass_rate": passed / total if total > 0 else 0.0,
            "avg_duration_ms": avg_duration,
            "total_duration_ms": total_duration
        }
