"""
Input validation utilities for security and data integrity
"""

import re
from typing import Any, Optional


def sanitize_string(
    value: Any, max_length: int = 255, allow_special_chars: bool = False
) -> str:
    """
    Sanitize string input to prevent injection and ensure valid data

    Args:
        value: Input value to sanitize
        max_length: Maximum allowed length
        allow_special_chars: Whether to allow special characters

    Returns:
        Sanitized string

    Raises:
        ValueError: If input is invalid
    """
    if value is None:
        return "unknown"

    str_value = str(value).strip()

    if len(str_value) > max_length:
        raise ValueError(f"String exceeds maximum length of {max_length}")

    if not str_value:
        return "unknown"

    if not allow_special_chars:
        if not re.match(r"^[a-zA-Z0-9_\-\.\/:\s]+$", str_value):
            raise ValueError(
                f"String contains invalid characters: {str_value[:50]}... "
                "(only alphanumeric, _, -, ., /, :, and spaces allowed)"
            )

    return str_value


def validate_session_id(session_id: Optional[str]) -> str:
    """
    Validate session ID format

    Args:
        session_id: Session ID to validate

    Returns:
        Validated session ID

    Raises:
        ValueError: If session ID is invalid
    """
    if not session_id:
        return "unknown"

    if not re.match(r"^[a-zA-Z0-9_\-]+$", session_id):
        raise ValueError(
            f"Invalid session ID format: {session_id[:50]}... "
            "(only alphanumeric, _, and - allowed)"
        )

    if len(session_id) > 100:
        raise ValueError("Session ID too long (max 100 characters)")

    return session_id


def validate_model_name(model_name: Optional[str]) -> str:
    """
    Validate model name

    Args:
        model_name: Model name to validate

    Returns:
        Validated model name
    """
    if not model_name:
        return "unknown"

    return sanitize_string(model_name, max_length=255, allow_special_chars=True)


def validate_positive_int(value: Any, name: str = "value") -> int:
    """
    Validate positive integer

    Args:
        value: Value to validate
        name: Name of the value for error messages

    Returns:
        Validated integer

    Raises:
        ValueError: If value is not a positive integer
    """
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")

    if int_value < -1:
        raise ValueError(f"{name} must be >= -1, got {int_value}")

    return int_value


def validate_float_range(
    value: Any, min_val: float = 0.0, max_val: float = float("inf"), name: str = "value"
) -> float:
    """
    Validate float is in range

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the value for error messages

    Returns:
        Validated float

    Raises:
        ValueError: If value is out of range
    """
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")

    if not (min_val <= float_value <= max_val):
        raise ValueError(
            f"{name} must be between {min_val} and {max_val}, got {float_value}"
        )

    return float_value
