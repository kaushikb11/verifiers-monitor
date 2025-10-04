"""
Modular FastAPI dashboard server for real-time monitoring visualization

This is the new, refactored server that uses a clean modular architecture
with separated concerns for routes, services, and business logic.
"""

import asyncio
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..utils.logging import get_logger
from .routes import create_api_routes, create_websocket_routes
from .services import DataService

logger = get_logger(__name__)


class DashboardServer:
    """Modular real-time web dashboard server"""

    def __init__(
        self,
        storage_backend,
        port: int = 8080,
        tunnel: bool = False,
        host: str = "127.0.0.1",
    ):
        """
        Initialize dashboard server with modular architecture

        Args:
            storage_backend: Storage backend for retrieving metrics
            port: Port to serve dashboard on
            tunnel: Whether to create ngrok tunnel for remote access
            host: Host to bind to (default: 127.0.0.1 for localhost only)
                  Use 0.0.0.0 to expose to all network interfaces (NOT RECOMMENDED)
        """
        self.storage = storage_backend
        self.port = port
        self.tunnel = tunnel
        self.host = host
        self.tunnel_url = None

        if host == "0.0.0.0":
            logger.warning(
                "⚠️  Dashboard binding to 0.0.0.0 - accessible from ALL network interfaces. "
                "This may expose sensitive data. Use host='127.0.0.1' for localhost only."
            )

        # Initialize FastAPI app
        self.app = FastAPI(
            title="Verifiers Monitor Dashboard",
            description="Verifiers Monitor - Modular Architecture",
            version="2.0.0",
        )

        # Initialize services
        self.data_service = DataService(storage_backend)
        self._env_info: dict = {}

        # Setup modular routes
        self._setup_modular_routes()

        # Server management
        self.server_thread = None
        self._server_started = False
        self._stop_event = threading.Event()
        self._uvicorn_server = None

    def _setup_modular_routes(self):
        """Setup routes using the new modular architecture"""

        # Mount static files
        static_path = Path(__file__).parent / "static"
        if static_path.exists():
            self.app.mount(
                "/static", StaticFiles(directory=str(static_path)), name="static"
            )

        # Create API routes with dependency injection
        api_router = create_api_routes(self.storage, lambda: self._env_info)
        self.app.include_router(api_router)

        # Create WebSocket routes
        ws_router = create_websocket_routes(self.storage)
        self.app.include_router(ws_router)

        # Store WebSocket router for broadcasting
        self.ws_router = ws_router

        # Add main dashboard route (this could be moved to routes later)
        @self.app.get("/")
        async def dashboard():
            """Main dashboard page with cache busting"""
            import time

            cache_version = str(int(time.time()))
            html = self.data_service.get_dashboard_html()
            # Replace template variable with actual timestamp
            html = html.replace("{{ cache_version }}", cache_version)
            return HTMLResponse(html)

    def update_env_info(self, env_info: Dict[str, Any]):
        """Update environment information"""
        self._env_info.update(env_info)
        self.data_service.env_info.update(env_info)

    async def broadcast_update(self, metrics_data: Dict[str, Any]):
        """Broadcast metrics update to WebSocket clients"""
        if hasattr(self.ws_router, "broadcast_metrics"):
            await self.ws_router.broadcast_metrics(metrics_data)

    def start(self):
        """Start the dashboard server"""
        if self._server_started:
            logger.info(f"Dashboard already running on port {self.port}")
            return

        def run_server():
            """Run the uvicorn server in a separate thread"""
            config = uvicorn.Config(
                app=self.app,
                host=self.host,
                port=self.port,
                log_level="warning",
                access_log=False,
            )

            self._uvicorn_server = uvicorn.Server(config)

            # Run server
            try:
                asyncio.run(self._uvicorn_server.serve())
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error(f"Dashboard server error: {e}")

        # Start server in background thread
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

        # Wait for server to start
        max_wait = 10
        for _ in range(max_wait * 10):
            try:
                import requests

                response = requests.get(
                    f"http://localhost:{self.port}/api/status", timeout=1
                )
                if response.status_code == 200:
                    break
            except:
                pass
                time.sleep(0.1)

        self._server_started = True

        # Setup tunnel if requested
        if self.tunnel:
            self._setup_tunnel()

        logger.info(f"Dashboard started: http://localhost:{self.port}")
        if self.tunnel_url:
            logger.info(f"Remote access: {self.tunnel_url}")

    def _setup_tunnel(self):
        """Setup ngrok tunnel for remote access"""
        try:
            from ..utils.tunnel import create_tunnel

            self.tunnel_url = create_tunnel(self.port)
        except ImportError:
            logger.warning("Tunnel setup requires pyngrok: pip install pyngrok")
        except Exception as e:
            logger.warning(f"Tunnel setup failed: {e}")

    def stop(self):
        """Stop the dashboard server"""
        if not self._server_started:
            return

        self._stop_event.set()

        if self._uvicorn_server:
            self._uvicorn_server.should_exit = True

        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5)

        self._server_started = False
        logger.info(f"Dashboard stopped (port {self.port})")

    def get_url(self) -> str:
        """Get dashboard URL"""
        if self.tunnel_url:
            return self.tunnel_url
        return f"http://localhost:{self.port}"

    def is_running(self) -> bool:
        """Check if server is running"""
        return self._server_started and (
            not self.server_thread or self.server_thread.is_alive()
        )

    # Static method for backward compatibility
    @staticmethod
    def _get_dashboard_html_static():
        """Static method to get dashboard HTML (for backward compatibility)"""
        dashboard_path = Path(__file__).parent / "templates" / "dashboard.html"

        try:
            with open(dashboard_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return """
            <!DOCTYPE html>
            <html>
            <head><title>Dashboard Not Found</title></head>
<body>
                <h1>Dashboard HTML file not found</h1>
                <p>Expected location: {}</p>
</body>
</html>
            """.format(
                dashboard_path
            )
