"""
Custom exception classes for Verifiers Monitor

This module defines specific exceptions for different error scenarios,
enabling better error handling and debugging.
"""

from typing import Any, Dict, Optional


class VerifiersMonitorError(Exception):
    """Base exception for all Verifiers Monitor errors"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def __str__(self):
        base_msg = self.message
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            base_msg += f" (Details: {details_str})"
        if self.cause:
            base_msg += f" (Caused by: {self.cause})"
        return base_msg


# =============================================================================
# Storage Exceptions
# =============================================================================


class StorageError(VerifiersMonitorError):
    """Base exception for storage-related errors"""

    pass


class StorageConnectionError(StorageError):
    """Database connection failed"""

    pass


class StorageTimeoutError(StorageError):
    """Database operation timed out"""

    pass


class StorageIntegrityError(StorageError):
    """Data integrity violation in storage"""

    pass


class StorageMigrationError(StorageError):
    """Database migration failed"""

    pass


class StorageConfigurationError(StorageError):
    """Storage backend configuration error"""

    pass


# =============================================================================
# Dashboard Exceptions
# =============================================================================


class DashboardError(VerifiersMonitorError):
    """Base exception for dashboard-related errors"""

    pass


class DashboardStartupError(DashboardError):
    """Dashboard failed to start"""

    pass


class DashboardPortError(DashboardError):
    """Port is already in use or unavailable"""

    pass


class DashboardTemplateError(DashboardError):
    """Dashboard template not found or invalid"""

    pass


class DashboardAPIError(DashboardError):
    """API endpoint error"""

    pass


class WebSocketError(DashboardError):
    """WebSocket connection error"""

    pass


# =============================================================================
# =============================================================================


class MonitoringError(VerifiersMonitorError):
    """Base exception for monitoring-related errors"""

    pass


class MetricsCollectionError(MonitoringError):
    """Error collecting metrics from components"""

    pass


class SessionError(MonitoringError):
    """Evaluation session error"""

    pass


class WrapperError(MonitoringError):
    """Error wrapping Verifiers components"""

    pass


class DataExtractionError(MonitoringError):
    """Error extracting data from Verifiers objects"""

    pass


# =============================================================================
# Configuration Exceptions
# =============================================================================


class ConfigurationError(VerifiersMonitorError):
    """Configuration-related errors"""

    pass


class EnvironmentConfigError(ConfigurationError):
    """Environment configuration error"""

    pass


class TunnelError(ConfigurationError):
    """Tunnel setup error"""

    pass


# =============================================================================
# Validation Exceptions
# =============================================================================


class ValidationError(VerifiersMonitorError):
    """Data validation errors"""

    pass


class SchemaValidationError(ValidationError):
    """Schema validation failed"""

    pass


class DataFormatError(ValidationError):
    """Data format is invalid"""

    pass


# =============================================================================
# Recovery Exceptions
# =============================================================================


class RecoverableError(VerifiersMonitorError):
    """Base class for errors that can be recovered from"""

    def __init__(
        self, message: str, retry_after: float = 1.0, max_retries: int = 3, **kwargs
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        self.max_retries = max_retries


class TemporaryStorageError(RecoverableError, StorageError):
    """Temporary storage error that can be retried"""

    pass


class TemporaryNetworkError(RecoverableError, DashboardError):
    """Temporary network error that can be retried"""

    pass


class TemporaryResourceError(RecoverableError, MonitoringError):
    """Temporary resource unavailability"""

    pass


# =============================================================================
# Utility Functions
# =============================================================================


def wrap_exception(
    original_exception: Exception,
    new_exception_class: type,
    message: Optional[str] = None,
    **kwargs,
) -> VerifiersMonitorError:
    """
    Wrap a generic exception in a specific Verifiers Monitor exception

    Args:
        original_exception: The original exception
        new_exception_class: The new exception class to use
        message: Optional custom message
        **kwargs: Additional arguments for the new exception

    Returns:
        New exception instance with the original as cause
    """
    if message is None:
        message = f"Operation failed: {str(original_exception)}"

    return new_exception_class(message=message, cause=original_exception, **kwargs)


def is_recoverable_error(exception: Exception) -> bool:
    """
    Check if an exception is recoverable

    Args:
        exception: Exception to check

    Returns:
        True if the exception can be recovered from
    """
    return isinstance(exception, RecoverableError)


def get_retry_info(exception: Exception) -> tuple[float, int]:
    """
    Get retry information from a recoverable exception

    Args:
        exception: Exception to get retry info from

    Returns:
        Tuple of (retry_after, max_retries)
    """
    if isinstance(exception, RecoverableError):
        return exception.retry_after, exception.max_retries
    return 1.0, 0


def create_error_context(operation: str, **context) -> Dict[str, Any]:
    """
    Create standardized error context for logging

    Args:
        operation: Operation that failed
        **context: Additional context information

    Returns:
        Dictionary with error context
    """
    return {"operation": operation, "timestamp": __import__("time").time(), **context}
