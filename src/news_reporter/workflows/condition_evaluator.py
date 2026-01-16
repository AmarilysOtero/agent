"""Condition Expression Evaluator for edge routing - Safe parser (no eval)"""

from __future__ import annotations
from typing import Any, Optional
import re
import logging

from .workflow_state import WorkflowState

logger = logging.getLogger(__name__)


class ConditionEvaluator:
    """
    Safe condition expression evaluator for graph edge routing.
    
    NO eval() - uses parser instead for security.
    
    Supports:
    - Operators: ==, !=, in, not in, is None, is not None, and, or
    - Types: string literals (quoted), numbers, booleans
    - State paths: triage.preferred_agent, state.selected_search
    - Missing path behavior: treat as None (or raise error based on strictness)
    """
    
    @staticmethod
    def evaluate(condition: str, state: WorkflowState, strict: bool = False) -> bool:
        """
        Evaluate a condition expression against workflow state.
        
        Args:
            condition: Condition expression string
            state: WorkflowState to evaluate against
            strict: If True, raise error on missing paths. If False, treat as None.
        
        Returns:
            True if condition is met, False otherwise
        """
        if not condition or not condition.strip():
            return True  # Empty condition is always true
        
        condition = condition.strip()
        
        # Handle logical operators (and, or) - simple precedence: and before or
        if " or " in condition:
            parts = condition.split(" or ", 1)
            left = ConditionEvaluator.evaluate(parts[0].strip(), state, strict)
            right = ConditionEvaluator.evaluate(parts[1].strip(), state, strict)
            return left or right
        
        if " and " in condition:
            parts = condition.split(" and ", 1)
            left = ConditionEvaluator.evaluate(parts[0].strip(), state, strict)
            right = ConditionEvaluator.evaluate(parts[1].strip(), state, strict)
            return left and right
        
        # Handle negation
        if condition.startswith("not "):
            return not ConditionEvaluator.evaluate(condition[4:], state, strict)
        
        # Handle "is not None" / "is None"
        if " is not None" in condition:
            path = condition.replace(" is not None", "").strip()
            value = ConditionEvaluator._get_state_value(path, state, strict)
            return value is not None
        elif " is None" in condition:
            path = condition.replace(" is None", "").strip()
            value = ConditionEvaluator._get_state_value(path, state, strict)
            return value is None
        
        # Handle "not in" operator (must check before "in")
        if " not in " in condition:
            parts = condition.split(" not in ", 1)
            if len(parts) == 2:
                search_value = ConditionEvaluator._parse_literal(parts[0].strip())
                container_path = parts[1].strip()
                container = ConditionEvaluator._get_state_value(container_path, state, strict)
                return not ConditionEvaluator._check_membership(search_value, container)
        
        # Handle "in" operator (membership)
        if " in " in condition:
            parts = condition.split(" in ", 1)
            if len(parts) == 2:
                search_value = ConditionEvaluator._parse_literal(parts[0].strip())
                container_path = parts[1].strip()
                container = ConditionEvaluator._get_state_value(container_path, state, strict)
                return ConditionEvaluator._check_membership(search_value, container)
        
        # Handle equality operators
        for op in ["==", "!=", ">=", "<=", ">", "<"]:
            if f" {op} " in condition:
                parts = condition.split(f" {op} ", 1)
                if len(parts) == 2:
                    left_expr = parts[0].strip()
                    right_expr = parts[1].strip()
                    
                    # Parse left side (state path or literal)
                    left_value = ConditionEvaluator._parse_expression(left_expr, state, strict)
                    
                    # Parse right side (literal or state path)
                    right_value = ConditionEvaluator._parse_expression(right_expr, state, strict)
                    
                    # Compare
                    return ConditionEvaluator._compare_values(left_value, right_value, op)
        
        # If no operator matched, check if path exists and is truthy
        value = ConditionEvaluator._get_state_value(condition, state, strict)
        if value is not None:
            return bool(value)
        
        logger.warning(f"Could not evaluate condition: {condition}")
        return False
    
    @staticmethod
    def _parse_expression(expr: str, state: WorkflowState, strict: bool) -> Any:
        """Parse an expression - either a state path or a literal value"""
        expr = expr.strip()
        
        # Check if it's a state path (contains dot or starts with known prefixes)
        if "." in expr or expr.startswith("state."):
            # State path
            if expr.startswith("state."):
                expr = expr[6:]  # Remove "state." prefix
            return ConditionEvaluator._get_state_value(expr, state, strict)
        else:
            # Try as literal
            return ConditionEvaluator._parse_literal(expr)
    
    @staticmethod
    def _parse_literal(value: str) -> Any:
        """Parse a literal value (string, number, boolean) - NO eval()"""
        value = value.strip()
        
        # Remove quotes if present
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        
        # Try boolean
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        
        # Try number
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    @staticmethod
    def _get_state_value(path: str, state: WorkflowState, strict: bool) -> Any:
        """Get value from state by path, handling missing paths"""
        value = state.get(path)
        
        if value is None and strict:
            raise ValueError(f"State path '{path}' not found and strict mode enabled")
        
        return value
    
    @staticmethod
    def _check_membership(search_value: Any, container: Any) -> bool:
        """Check if search_value is in container"""
        if container is None:
            return False
        
        if isinstance(container, (list, tuple)):
            return search_value in container
        elif isinstance(container, str):
            return str(search_value) in container
        elif isinstance(container, dict):
            return search_value in container.values() or search_value in container.keys()
        
        return False
    
    @staticmethod
    def _compare_values(left: Any, right: Any, op: str) -> bool:
        """Compare two values using the given operator"""
        if op == "==":
            return str(left) == str(right)
        elif op == "!=":
            return str(left) != str(right)
        elif op in [">=", "<=", ">", "<"]:
            try:
                left_num = float(left) if left is not None else 0
                right_num = float(right) if right is not None else 0
                if op == ">=":
                    return left_num >= right_num
                elif op == "<=":
                    return left_num <= right_num
                elif op == ">":
                    return left_num > right_num
                elif op == "<":
                    return left_num < right_num
            except (ValueError, TypeError):
                logger.warning(f"Cannot compare non-numeric values: {left} {op} {right}")
                return False
        
        return False
