"""
Dashboard routes module

This module provides a clean separation of concerns for different types of routes:
- API routes: REST endpoints for data retrieval
- WebSocket routes: Real-time streaming endpoints
"""

from .api import create_api_routes
from .websocket import create_websocket_routes

__all__ = ["create_api_routes", "create_websocket_routes"]
