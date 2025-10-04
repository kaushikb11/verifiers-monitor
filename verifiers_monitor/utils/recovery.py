"""
Error recovery and retry mechanisms for Verifiers Monitor

This module provides robust error recovery, retry logic, and fallback mechanisms
to handle transient failures gracefully.
"""

import asyncio
import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union

from ..core.exceptions import RecoverableError, get_retry_info, is_recoverable_error
from .logging import get_logger

logger = get_logger(__name__)


class RetryConfig:
    """Configuration for retry behavior"""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_multiplier: float = 2.0,
        jitter: bool = True,
        recoverable_exceptions: Optional[List[Type[Exception]]] = None,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter
        self.recoverable_exceptions = recoverable_exceptions or [RecoverableError]

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number"""
        delay = self.base_delay * (self.backoff_multiplier ** (attempt - 1))
        delay = min(delay, self.max_delay)

        if self.jitter:
            import random

            delay *= 0.5 + random.random() * 0.5

        return delay

    def is_recoverable(self, exception: Exception) -> bool:
        """Check if exception is recoverable according to config"""
        return any(
            isinstance(exception, exc_type) for exc_type in self.recoverable_exceptions
        )


def retry_on_failure(
    config: Optional[RetryConfig] = None,
    logger_instance: Optional[logging.Logger] = None,
):
    """
    Decorator to retry function calls on recoverable failures

    Args:
        config: Retry configuration
        logger_instance: Logger to use for retry messages

    Usage:
        @retry_on_failure()
        def unreliable_function():
            pass
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger_instance or logger

            last_exception = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    result = func(*args, **kwargs)

                    if attempt > 1:
                        _logger.info(f"{func.__name__} succeeded on attempt {attempt}")

                    return result

                except Exception as e:
                    last_exception = e

                    # Check if this is a recoverable error
                    if not (config.is_recoverable(e) or is_recoverable_error(e)):
                        _logger.error(
                            f"{func.__name__} failed with non-recoverable error: {e}"
                        )
                        raise

                    if attempt == config.max_attempts:
                        _logger.error(
                            f"{func.__name__} failed after {attempt} attempts: {e}"
                        )
                        raise

                    # Calculate delay
                    if is_recoverable_error(e):
                        delay, _ = get_retry_info(e)
                    else:
                        delay = config.calculate_delay(attempt)

                    _logger.warning(
                        f"{func.__name__} failed on attempt {attempt}/{config.max_attempts}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    time.sleep(delay)

            # This should never be reached, but just in case
            raise last_exception

        return wrapper

    return decorator


async def async_retry_on_failure(
    config: Optional[RetryConfig] = None,
    logger_instance: Optional[logging.Logger] = None,
):
    """
    Async version of retry decorator

    Args:
        config: Retry configuration
        logger_instance: Logger to use for retry messages
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            _logger = logger_instance or logger

            last_exception = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    result = await func(*args, **kwargs)

                    if attempt > 1:
                        _logger.info(f"{func.__name__} succeeded on attempt {attempt}")

                    return result

                except Exception as e:
                    last_exception = e

                    if not (config.is_recoverable(e) or is_recoverable_error(e)):
                        _logger.error(
                            f"{func.__name__} failed with non-recoverable error: {e}"
                        )
                        raise

                    if attempt == config.max_attempts:
                        _logger.error(
                            f"{func.__name__} failed after {attempt} attempts: {e}"
                        )
                        raise

                    # Calculate delay
                    if is_recoverable_error(e):
                        delay, _ = get_retry_info(e)
                    else:
                        delay = config.calculate_delay(attempt)

                    _logger.warning(
                        f"{func.__name__} failed on attempt {attempt}/{config.max_attempts}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    await asyncio.sleep(delay)

            raise last_exception

        return wrapper

    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for preventing cascade failures
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if time.time() - self.last_failure_time < self.recovery_timeout:
                    raise Exception(
                        f"Circuit breaker is OPEN. Try again in {self.recovery_timeout}s"
                    )
                else:
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker transitioning to HALF_OPEN")

            try:
                result = func(*args, **kwargs)

                # Success - reset circuit breaker
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info("Circuit breaker reset to CLOSED")

                return result

            except self.expected_exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.error(
                        f"Circuit breaker opened after {self.failure_count} failures"
                    )

                raise

        return wrapper


class FallbackHandler:
    """
    Fallback mechanism for when primary operations fail
    """

    def __init__(
        self, fallback_func: Callable, logger_instance: Optional[logging.Logger] = None
    ):
        self.fallback_func = fallback_func
        self.logger = logger_instance or logger

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.warning(f"{func.__name__} failed: {e}. Using fallback.")

                try:
                    return self.fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    self.logger.error(f"Fallback also failed: {fallback_error}")
                    raise e

        return wrapper


class HealthChecker:
    """
    Health checking mechanism for system components
    """

    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.components: Dict[str, Callable[[], bool]] = {}
        self.last_check: Dict[str, Optional[float]] = {}

    def register_component(self, name: str, health_check_func: Callable[[], bool]):
        """Register a component for health checking"""
        self.components[name] = health_check_func
        self.last_check[name] = None

    def check_component(self, name: str) -> bool:
        """Check health of a specific component"""
        if name not in self.components:
            logger.warning(f"Component '{name}' not registered for health checking")
            return False

        try:
            is_healthy = self.components[name]()
            self.last_check[name] = time.time()

            if is_healthy:
                logger.debug(f"Component '{name}' is healthy")
            else:
                logger.warning(f"Component '{name}' health check failed")

            return is_healthy

        except Exception as e:
            logger.error(f"Health check for '{name}' raised exception: {e}")
            return False

    def check_all_components(self) -> dict[str, bool]:
        """Check health of all registered components"""
        results = {}
        for name in self.components:
            results[name] = self.check_component(name)
        return results

    def get_unhealthy_components(self) -> List[str]:
        """Get list of unhealthy components"""
        results = self.check_all_components()
        return [name for name, is_healthy in results.items() if not is_healthy]


# Global health checker instance
health_checker = HealthChecker()


def with_fallback(fallback_func: Callable):
    """Decorator to add fallback functionality"""
    return FallbackHandler(fallback_func)


def circuit_breaker(failure_threshold: int = 5, recovery_timeout: float = 60.0):
    """Decorator to add circuit breaker functionality"""
    return CircuitBreaker(failure_threshold, recovery_timeout)


def safe_execute(
    func: Callable, *args, default_return=None, logger_instance=None, **kwargs
):
    """
    Safely execute a function with error handling

    Args:
        func: Function to execute
        *args: Function arguments
        default_return: Value to return on error
        logger_instance: Logger to use
        **kwargs: Function keyword arguments

    Returns:
        Function result or default_return on error
    """
    _logger = logger_instance or logger

    try:
        return func(*args, **kwargs)
    except Exception as e:
        _logger.error(f"Safe execution of {func.__name__} failed: {e}")
        return default_return
