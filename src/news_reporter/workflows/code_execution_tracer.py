"""Code Execution Tracer - Captures exact Python code executed during graph workflow"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import time
import inspect
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CodeExecutionTracer:
    """Tracks and records exact Python code execution during workflow"""
    
    def __init__(self):
        self.code_trace: List[Dict[str, Any]] = []
        self.step_counter = 0
    
    def record_execution(
        self,
        method_name: str,
        class_name: str = "GraphExecutor",
        args: Optional[Dict[str, Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        result: Any = None,
        result_type: Optional[str] = None,
        condition: Optional[str] = None,
        condition_result: Optional[bool] = None
    ) -> None:
        """Record a code execution step"""
        self.step_counter += 1
        
        # Get source code if possible
        source_preview = None
        try:
            # Try to get source from the class
            if class_name == "GraphExecutor":
                from .graph_executor import GraphExecutor
                method = getattr(GraphExecutor, method_name, None)
                if method:
                    try:
                        source = inspect.getsource(method)
                        source_lines = source.split('\n')
                        # Get first 10 lines of method
                        source_preview = '\n'.join(source_lines[:10])
                    except (OSError, TypeError):
                        pass
        except Exception:
            pass
        
        trace_entry = {
            "step": self.step_counter,
            "class": class_name,
            "method": method_name,
            "args": args or {},
            "kwargs": kwargs or {},
            "result_type": result_type,
            "result_preview": self._preview_value(result),
            "condition": condition,
            "condition_result": condition_result,
            "source_preview": source_preview,
            "timestamp": time.time()
        }
        
        self.code_trace.append(trace_entry)
    
    def _preview_value(self, value: Any, max_length: int = 200) -> Optional[str]:
        """Create a preview of a value for display"""
        if value is None:
            return None
        if isinstance(value, str):
            return value[:max_length] + ("..." if len(value) > max_length else "")
        if isinstance(value, (dict, list)):
            return str(value)[:max_length] + ("..." if len(str(value)) > max_length else "")
        return str(value)[:max_length]
    
    def _serialize_for_code(self, obj: Any, indent: int = 0) -> str:
        """Serialize object for Python code representation"""
        indent_str = " " * indent
        
        if isinstance(obj, dict):
            if not obj:
                return "{}"
            items = []
            for k, v in obj.items():
                key_str = f'"{k}"' if isinstance(k, str) else str(k)
                val_str = self._serialize_for_code(v, indent + 2)
                items.append(f"{indent_str}    {key_str}: {val_str}")
            return "{\n" + ",\n".join(items) + f"\n{indent_str}}}"
        elif isinstance(obj, list):
            if not obj:
                return "[]"
            if len(obj) <= 3:
                items = [self._serialize_for_code(item, indent + 2) for item in obj]
                return "[" + ", ".join(items) + "]"
            else:
                items = [self._serialize_for_code(item, indent + 2) for item in obj[:3]]
                return "[" + ", ".join(items) + f", ... # {len(obj) - 3} more items]"
        elif isinstance(obj, str):
            # Escape and quote strings
            escaped = obj.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
            if len(escaped) > 100:
                escaped = escaped[:100] + "..."
            return f'"{escaped}"'
        elif obj is None:
            return "None"
        elif isinstance(obj, bool):
            return str(obj)
        elif isinstance(obj, (int, float)):
            return str(obj)
        else:
            # For other types, use repr but truncate
            repr_str = repr(obj)
            if len(repr_str) > 100:
                return repr_str[:100] + "..."
            return repr_str
    
    def write_execution_script(
        self,
        output_dir: Path,
        run_id: str,
        goal: str,
        start_time: float,
        duration_ms: float,
        final_output: str,
        state_snapshot: Dict[str, Any]
    ) -> Path:
        """Write a Python script showing the exact code that was executed"""
        timestamp = datetime.fromtimestamp(start_time).strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"executed_code_{timestamp}_{run_id[:8]}.py"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # Write header
                f.write('"""\n')
                f.write('Python Code Execution Trace - Graph Workflow\n')
                f.write('=' * 80 + '\n')
                f.write(f'Run ID: {run_id}\n')
                f.write(f'Goal: {goal}\n')
                f.write(f'Start Time: {datetime.fromtimestamp(start_time).isoformat()}\n')
                f.write(f'Duration: {duration_ms:.2f}ms\n')
                f.write('=' * 80 + '\n\n')
                f.write('This file shows the exact Python code that was executed during the workflow.\n')
                f.write('Each step shows the method called, arguments, and results.\n')
                f.write('You can use this to verify and debug the workflow operation.\n')
                f.write('"""\n\n')
                
                # Write imports
                f.write('# Imports\n')
                f.write('from typing import Dict, List, Any, Optional\n')
                f.write('from collections import deque\n')
                f.write('import asyncio\n')
                f.write('from news_reporter.workflows.workflow_state import WorkflowState\n')
                f.write('from news_reporter.workflows.node_result import NodeResult, NodeStatus\n')
                f.write('from news_reporter.workflows.execution_context import ExecutionContext\n')
                f.write('from news_reporter.workflows.condition_evaluator import ConditionEvaluator\n\n')
                
                # Write initial state
                f.write('# Initial State\n')
                f.write('# ' + '=' * 76 + '\n')
                f.write(f'goal = "{goal}"\n')
                f.write('state = WorkflowState(goal=goal)\n\n')
                
                # Write execution trace
                f.write('# Code Execution Trace\n')
                f.write('# ' + '=' * 76 + '\n\n')
                
                for trace in self.code_trace:
                    step = trace["step"]
                    class_name = trace["class"]
                    method = trace["method"]
                    args = trace.get("args", {})
                    kwargs = trace.get("kwargs", {})
                    result_type = trace.get("result_type")
                    result_preview = trace.get("result_preview")
                    condition = trace.get("condition")
                    condition_result = trace.get("condition_result")
                    source_preview = trace.get("source_preview")
                    timestamp_str = datetime.fromtimestamp(trace["timestamp"]).strftime("%H:%M:%S.%f")[:-3]
                    
                    f.write(f'# Step {step}: {class_name}.{method}\n')
                    f.write(f'# Timestamp: {timestamp_str}\n')
                    
                    # Write condition evaluation if present
                    if condition is not None:
                        f.write(f'# Condition: {condition}\n')
                        f.write(f'# Condition Result: {condition_result}\n')
                        f.write(f'# Code: condition_result = ConditionEvaluator.evaluate("{condition}", state)\n')
                        f.write(f'#       # Result: {condition_result}\n\n')
                        continue
                    
                    # Write source preview
                    if source_preview:
                        f.write('# Source code:\n')
                        for line in source_preview.split('\n'):
                            f.write(f'# {line}\n')
                        f.write('\n')
                    
                    # Write arguments
                    if args or kwargs:
                        f.write('# Arguments:\n')
                        all_args = {**(args or {}), **(kwargs or {})}
                        for key, value in all_args.items():
                            serialized = self._serialize_for_code(value)
                            f.write(f'#   {key} = {serialized}\n')
                        f.write('\n')
                    
                    # Write actual code call
                    f.write('# Code executed:\n')
                    if args or kwargs:
                        all_args = {**(args or {}), **(kwargs or {})}
                        args_list = []
                        for key, value in all_args.items():
                            serialized = self._serialize_for_code(value)
                            args_list.append(f'{key}={serialized}')
                        args_str = ", ".join(args_list)
                        f.write(f'# result = executor.{method}({args_str})\n')
                    else:
                        f.write(f'# result = executor.{method}()\n')
                    
                    # Write result
                    if result_type:
                        f.write(f'# Result type: {result_type}\n')
                    if result_preview:
                        f.write(f'# Result preview: {result_preview}\n')
                    
                    f.write('\n')
                
                # Write final state
                f.write('\n# Final State\n')
                f.write('# ' + '=' * 76 + '\n')
                f.write(f'# Goal: {state_snapshot.get("goal", "None")}\n')
                f.write(f'# Final output length: {len(final_output) if final_output else 0}\n')
                if final_output:
                    f.write(f'# Final output (first 500 chars): {final_output[:500]}...\n')
                f.write(f'# Triage: {state_snapshot.get("triage", "None")}\n')
                if state_snapshot.get("latest"):
                    latest = state_snapshot["latest"]
                    f.write(f'# Latest (first 200 chars): {latest[:200] if isinstance(latest, str) else latest}...\n')
                
                # Show outputs namespace (key state for chaining)
                outputs = state_snapshot.get("outputs", {})
                if outputs:
                    f.write(f'# Outputs: {list(outputs.keys())}\n')
                    for node_id, output in outputs.items():
                        output_preview = str(output)[:100] if output else "None"
                        f.write(f'#   {node_id}: {output_preview}...\n')
                else:
                    f.write('# Outputs: {}\n')
                    
                f.write(f'# Conditional results: {state_snapshot.get("conditional", {})}\n')
                f.write(f'# Loop state: {state_snapshot.get("loop_state", {})}\n')
            
            logger.info(f"✅ Code execution script written: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error writing code execution script: {e}", exc_info=True)
            raise
