"""
Centralized logging system for Verifiers Monitor

This module provides structured logging with proper error handling,
performance monitoring, and debugging capabilities.
"""

import logging
import os
import sys
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Union


class VerifiersMonitorFormatter(logging.Formatter):
    """Custom formatter for Verifiers Monitor logs"""

    # Color codes for different log levels
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
        "RESET": "\033[0m",
    }

    def format(self, record):
        """Format log record with colors and structure"""
        # Add timestamp
        record.timestamp = datetime.fromtimestamp(record.created).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Add color for console output
        if hasattr(record, "levelname"):
            color = self.COLORS.get(record.levelname, "")
            reset = self.COLORS["RESET"]
            record.colored_levelname = f"{color}{record.levelname}{reset}"

        # Format: [TIMESTAMP] LEVEL | MODULE | MESSAGE
        return f"[{record.timestamp}] {record.colored_levelname:8} | {record.name:20} | {record.getMessage()}"


class PerformanceLogger:
    """Logger for performance monitoring"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_duration(self, operation: str, duration: float, threshold: float = 1.0):
        """Log operation duration with performance warnings"""
        if duration > threshold:
            self.logger.warning(
                f"Slow operation: {operation} took {duration:.2f}s (threshold: {threshold}s)"
            )
        else:
            self.logger.debug(f"Operation: {operation} completed in {duration:.2f}s")

    def log_memory_usage(self, operation: str, memory_mb: float):
        """Log memory usage for operations"""
        if memory_mb > 100:
            self.logger.warning(
                f"High memory usage: {operation} used {memory_mb:.1f}MB"
            )
        else:
            self.logger.debug(f"Memory usage: {operation} used {memory_mb:.1f}MB")


class VerbosityFilter(logging.Filter):
    """Filter to suppress verbose logs from verifiers.* modules"""

    def filter(self, record):
        """Filter log records based on module and level"""
        # For verifiers.* modules (not verifiers_monitor), only show WARNING and above
        if record.name.startswith("verifiers.") and record.levelno < logging.WARNING:
            return False

        return True


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
    format_type: str = "structured",
    suppress_verbose: bool = True,
) -> logging.Logger:
    """
    Setup centralized logging for Verifiers Monitor

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        console: Whether to log to console
        format_type: Format type ("structured", "simple", "json")
        suppress_verbose: Whether to suppress verbose logs from verifiers.* modules (default: True)

    Returns:
        Configured logger instance
    """

    # Create root logger for verifiers_monitor
    logger = logging.getLogger("verifiers_monitor")
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter: logging.Formatter
    if format_type == "structured":
        formatter = VerifiersMonitorFormatter()
    elif format_type == "simple":
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
        )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        # Add verbosity filter to suppress noisy logs
        if suppress_verbose:
            console_handler.addFilter(VerbosityFilter())

        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Also configure root logger to suppress verbose logs from verifiers.*
    if suppress_verbose:
        root_logger = logging.getLogger("verifiers")
        # Only show WARNING+ from verifiers.* modules
        root_logger.setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"verifiers_monitor.{name}")


def log_performance(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function performance

    Args:
        logger: Optional logger instance

    Usage:
        @log_performance()
        def my_function():
            pass
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or get_logger(func.__module__)

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                _logger.debug(f"{func.__name__} completed in {duration:.3f}s")
                return result

            except Exception as e:
                duration = time.time() - start_time
                _logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
                raise

        return wrapper

    return decorator


def log_errors(logger: Optional[logging.Logger] = None, reraise: bool = True):
    """
    Decorator to log function errors with context

    Args:
        logger: Optional logger instance
        reraise: Whether to reraise the exception

    Usage:
        @log_errors()
        def my_function():
            pass
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or get_logger(func.__module__)

            try:
                return func(*args, **kwargs)
            except Exception as e:
                _logger.error(
                    f"Error in {func.__name__}: {type(e).__name__}: {e}",
                    exc_info=True,
                    extra={
                        "function": func.__name__,
                        "args": str(args)[:100],
                        "kwargs": str(kwargs)[:100],
                    },
                )

                if reraise:
                    raise
                return None

        return wrapper

    return decorator


class ErrorContext:
    """Context manager for error logging with additional context"""

    def __init__(self, logger: logging.Logger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(f"Starting: {self.operation}", extra=self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else 0

        if exc_type is None:
            self.logger.debug(
                f"Completed: {self.operation} in {duration:.3f}s", extra=self.context
            )
        else:
            self.logger.error(
                f"Failed: {self.operation} after {duration:.3f}s - {exc_type.__name__}: {exc_val}",
                exc_info=True,
                extra=self.context,
            )

        return False


# Initialize default logger
_default_logger = None


def init_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Initialize the default logging configuration"""
    global _default_logger
    _default_logger = setup_logging(
        level=getattr(logging, level.upper()),
        log_file=log_file,
        console=True,
        format_type="structured",
    )
    return _default_logger


def get_default_logger() -> logging.Logger:
    """Get the default logger, initializing if needed"""
    global _default_logger
    if _default_logger is None:
        _default_logger = init_logging()
    return _default_logger
