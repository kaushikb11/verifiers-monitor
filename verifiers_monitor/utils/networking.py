"""
Network utilities for Verifiers Monitor

Provides port resolution, availability checking, and socket utilities.
"""

import socket
from contextlib import closing
from typing import Tuple

from .logging import get_logger

logger = get_logger(__name__)


def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """
    Check if a port is available for binding.

    Args:
        port: Port number to check
        host: Host address (default: localhost)

    Returns:
        True if port is available, False otherwise
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            return True
        except OSError:
            return False


def find_available_port(
    preferred: int, host: str = "127.0.0.1", search_range: int = 100
) -> int:
    """
    Find an available port, starting with preferred.

    Strategy:
    1. Try preferred port first
    2. Search nearby ports (±1, ±2, ±3, ...)
    3. Let OS choose random port as last resort

    Args:
        preferred: Preferred port number (0 = let OS choose)
        host: Host to bind to (default: localhost)
        search_range: Number of ports to search in each direction

    Returns:
        Available port number

    Raises:
        RuntimeError: If no port found (very unlikely)
    """
    # If preferred is 0, let the OS choose
    if preferred == 0:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.bind((host, 0))
            return sock.getsockname()[1]

    # Try preferred port first
    if is_port_available(preferred, host):
        return preferred

    logger.debug(f"Port {preferred} not available, searching for alternative...")

    # Try nearby ports
    for offset in range(1, search_range):
        for port in [preferred + offset, preferred - offset]:
            if port < 1024 or port > 65535:
                continue
            if is_port_available(port, host):
                logger.info(f"Using port {port} instead of {preferred}")
                return port

    # Last resort: let OS choose
    logger.warning(f"No ports available in range {preferred}±{search_range}")
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind((host, 0))
        chosen_port = sock.getsockname()[1]
        logger.info(f"OS chose port {chosen_port}")
        return chosen_port


def resolve_dashboard_port(
    preferred_port: int, host: str = "127.0.0.1"
) -> Tuple[int, bool]:
    """
    Resolve dashboard port, finding an available one if preferred is taken.

    Args:
        preferred_port: Desired port number
        host: Host to bind to (default: localhost)

    Returns:
        Tuple of (resolved_port, port_changed)
    """
    # Try preferred port first
    if is_port_available(preferred_port, host):
        return preferred_port, False

    # Find alternative
    resolved_port = find_available_port(preferred_port, host, search_range=100)
    return resolved_port, True
