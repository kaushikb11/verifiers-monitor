"""
JSON serialization utilities for Verifiers Monitor

Provides consistent serialization for complex objects including OpenAI types,
Pydantic models, and nested structures.
"""

import ast
import json
from typing import Any, Dict


def to_serializable(obj: Any) -> Any:
    """
    Recursively convert objects to JSON-serializable types.

    Handles:
    - Pydantic models (model_dump)
    - Objects with dict() method
    - Objects with __dict__
    - Lists and dicts recursively
    - None and primitives

    Args:
        obj: Object to serialize

    Returns:
        JSON-serializable representation
    """
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(item) for item in obj]
    elif hasattr(obj, "model_dump"):
        # Pydantic v2
        return to_serializable(obj.model_dump())
    elif hasattr(obj, "dict"):
        # Pydantic v1 or similar
        return to_serializable(obj.dict())
    elif hasattr(obj, "__dict__"):
        return to_serializable(obj.__dict__)
    else:
        return str(obj)


def json_serializer(obj: Any) -> Any:
    """
    Custom JSON serializer for complex objects.

    Used as the default parameter in json.dumps() calls.
    Converts objects to strings or dicts for JSON compatibility.

    Args:
        obj: Object to serialize

    Returns:
        JSON-compatible representation
    """
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    else:
        return str(obj)


def normalize_metrics(metrics_data: Any) -> Dict[str, Any] | None:
    """
    Normalize metrics data to ensure it's a proper dict.

    Handles cases where metrics might be:
    - A proper dict: {"key": "value"}
    - A string representation: "{'key': 'value'}" or '{"key": "value"}'
    - None or empty

    Args:
        metrics_data: Raw metrics data (dict, str, or None)

    Returns:
        Normalized dict or None if parsing fails
    """
    if metrics_data is None:
        return None

    # If it's already a dict, return as is
    if isinstance(metrics_data, dict):
        return metrics_data

    # If it's a string, try to parse it
    if isinstance(metrics_data, str):
        try:
            # First try JSON parsing (handles double quotes)
            return json.loads(metrics_data)
        except (json.JSONDecodeError, ValueError):
            try:
                # Fall back to ast.literal_eval for Python dict strings (single quotes)
                return ast.literal_eval(metrics_data)
            except (ValueError, SyntaxError):
                return None

    # For other types, return as is (might fail later, but at least we tried)
    return metrics_data


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Safely serialize object to JSON string using custom serializer.

    Args:
        obj: Object to serialize
        **kwargs: Additional arguments for json.dumps()

    Returns:
        JSON string representation
    """
    return json.dumps(obj, default=json_serializer, **kwargs)


def safe_json_loads(json_str: str) -> Any:
    """
    Safely deserialize JSON string with fallback to ast.literal_eval.

    Args:
        json_str: JSON string to parse

    Returns:
        Parsed object or None if parsing fails
    """
    if not json_str:
        return None

    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        try:
            return ast.literal_eval(json_str)
        except (ValueError, SyntaxError):
            return None
