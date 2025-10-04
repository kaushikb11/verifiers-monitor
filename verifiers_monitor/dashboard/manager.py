"""
Dashboard manager for singleton dashboard instances
"""

import threading
from typing import Any, Dict, Optional

from ..utils import cli_output
from ..utils.logging import get_logger
from .server import DashboardServer

logger = get_logger(__name__)


class DashboardManager:
    """Manages singleton dashboard instances per port"""

    def __init__(self) -> None:
        self._dashboards: Dict[int, DashboardServer] = {}
        self._lock = threading.Lock()
        self._session_counts: Dict[int, int] = {}

    def get_or_create_dashboard(
        self,
        storage_backend,
        port: int,
        tunnel: bool = False,
        env_info: Optional[Dict[str, Any]] = None,
    ) -> DashboardServer:
        """Get existing dashboard or create new one, with automatic port fallback"""
        with self._lock:
            if port in self._dashboards:
                dashboard = self._dashboards[port]
                self._session_counts[port] += 1
                cli_output.print_session_reuse(port, self._session_counts[port])

                if env_info:
                    dashboard.update_env_info(env_info)
                return dashboard

            original_port = port
            host = "127.0.0.1"
            available_port = self._find_available_port(port, host=host)

            if available_port != original_port:
                logger.info(
                    f"Port {original_port} in use, using port {available_port} instead"
                )
                port = available_port

            dashboard = DashboardServer(storage_backend, port, tunnel, host=host)
            if env_info:
                dashboard._env_info = env_info

            try:
                dashboard.start()
                self._dashboards[port] = dashboard
                self._session_counts[port] = 1
                logger.debug(f"Dashboard started on port {port}")
                return dashboard
            except Exception as e:
                logger.warning(f"Failed to start dashboard on port {port}: {e}")
                fallback_port = self._find_available_port(0, host="127.0.0.1")
                logger.info(f"Retrying with port {fallback_port}")
                dashboard = DashboardServer(
                    storage_backend, fallback_port, tunnel, host="127.0.0.1"
                )
                if env_info:
                    dashboard.update_env_info(env_info)
                dashboard.start()
                self._dashboards[fallback_port] = dashboard
                self._session_counts[fallback_port] = 1
                logger.debug(f"Dashboard started on fallback port {fallback_port}")
                return dashboard

    def _find_available_port(self, preferred: int, host: str = "127.0.0.1") -> int:
        """
        Find an available port, starting with preferred

        Args:
            preferred: Preferred port number (0 = let OS choose)
            host: Host to bind to (default: 127.0.0.1)

        Returns:
            Available port number
        """
        import socket
        from contextlib import closing

        if preferred == 0:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                sock.bind((host, 0))
                return sock.getsockname()[1]

        try:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, preferred))
                return preferred
        except OSError:
            logger.debug(
                f"Port {preferred} not available, searching for alternative..."
            )

        for offset in range(1, 100):
            for port in [preferred + offset, preferred - offset]:
                if port < 1024 or port > 65535:
                    continue
                try:
                    with closing(
                        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    ) as sock:
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        sock.bind((host, port))
                        return port
                except OSError:
                    continue

        # Last resort: let OS choose
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.bind((host, 0))
            return sock.getsockname()[1]

    def release_dashboard(self, port: int) -> None:
        """Release a session from the dashboard"""
        with self._lock:
            if port not in self._dashboards:
                return

            self._session_counts[port] -= 1

            # Keep dashboard running even when sessions end
            # Users can manually stop with Ctrl+C when they're done viewing
            if self._session_counts[port] <= 0:
                logger.info(
                    f"All sessions ended for port {port}. Dashboard still running - press Ctrl+C to stop."
                )

    def stop_dashboard(self, port: int) -> None:
        """Stop dashboard on specific port"""
        with self._lock:
            if port in self._dashboards:
                self._dashboards[port].stop()
                del self._dashboards[port]
                if port in self._session_counts:
                    del self._session_counts[port]

    def stop_all_dashboards(self) -> None:
        """Stop all active dashboards"""
        with self._lock:
            for dashboard in self._dashboards.values():
                dashboard.stop()
            self._dashboards.clear()
            self._session_counts.clear()

    def get_active_ports(self) -> list[int]:
        """Get list of ports with active dashboards"""
        with self._lock:
            return list(self._dashboards.keys())


# Global singleton manager
_dashboard_manager = DashboardManager()


def get_dashboard_manager() -> DashboardManager:
    """Get the global dashboard manager"""
    return _dashboard_manager
