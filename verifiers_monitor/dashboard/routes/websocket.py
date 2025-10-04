"""
WebSocket routes for real-time dashboard updates

This module handles WebSocket connections for live metric streaming
and real-time dashboard updates.
"""

import asyncio
import json
from typing import List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ...core.exceptions import WebSocketError
from ...utils.logging import get_logger

logger = get_logger(__name__)


def create_websocket_routes(storage) -> APIRouter:
    """
    Create WebSocket routes with dependency injection

    Args:
        storage: Storage backend instance

    Returns:
        APIRouter: Configured router with WebSocket endpoints
    """
    router = APIRouter()

    # Store active connections with proper lifecycle management
    active_connections: List[WebSocket] = []

    @router.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """WebSocket endpoint for real-time metrics streaming"""
        await websocket.accept()
        active_connections.append(websocket)
        logger.info(
            f"WebSocket client connected (total connections: {len(active_connections)})"
        )

        try:
            while True:
                # Send latest metrics every 2 seconds
                try:
                    latest_metrics = storage.get_latest()
                    await websocket.send_text(json.dumps(latest_metrics))
                except WebSocketDisconnect:
                    logger.info("WebSocket client disconnected")
                    break
                except ConnectionResetError:
                    logger.debug("WebSocket connection reset by client")
                    break
                except Exception as e:
                    # Check if it's a connection-related error (normal during shutdown)
                    if "connection" in str(e).lower() or "closed" in str(e).lower():
                        logger.debug(
                            f"WebSocket connection closed during shutdown: {e}"
                        )
                    else:
                        logger.error(f"Unexpected WebSocket error: {e}")
                    break

                await asyncio.sleep(2)

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"WebSocket error in main loop: {e}")
        finally:
            if websocket in active_connections:
                active_connections.remove(websocket)
            try:
                await websocket.close()
            except Exception:
                pass
            logger.info(
                f"WebSocket client cleaned up (remaining connections: {len(active_connections)})"
            )

    async def broadcast_metrics(metrics_data):
        """Broadcast metrics to all connected WebSocket clients"""
        if not active_connections:
            return

        message = json.dumps(metrics_data)
        disconnected = []

        for connection in active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)

        # Remove disconnected clients
        for connection in disconnected:
            if connection in active_connections:
                active_connections.remove(connection)

        if disconnected:
            logger.debug(
                f"Cleaned up {len(disconnected)} disconnected WebSocket clients"
            )

    def cleanup_connections():
        """Cleanup all active connections"""
        logger.info(
            f"Cleaning up {len(active_connections)} active WebSocket connections"
        )
        disconnected = []
        for connection in active_connections:
            try:
                asyncio.create_task(connection.close())
            except Exception:
                pass
            disconnected.append(connection)

        for connection in disconnected:
            if connection in active_connections:
                active_connections.remove(connection)

    # Attach functions to router for external access
    router.broadcast_metrics = broadcast_metrics
    router.cleanup_connections = cleanup_connections
    router.active_connections = active_connections

    return router
