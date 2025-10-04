"""
Tunnel utilities for remote dashboard access
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def create_tunnel(port: int) -> Optional[str]:
    """
    Create an ngrok tunnel for the dashboard

    Args:
        port: Local port to tunnel

    Returns:
        Public tunnel URL or None if failed
    """
    try:
        from pyngrok import ngrok

        # Create tunnel
        tunnel = ngrok.connect(port, "http")
        public_url = tunnel.public_url

        logger.info(f"Tunnel created: {public_url} -> localhost:{port}")
        return public_url

    except ImportError:
        logger.error("pyngrok not available - install with: pip install pyngrok")
        return None
    except Exception as e:
        logger.error(f"Failed to create tunnel: {e}")
        return None


def close_tunnels():
    """Close all active ngrok tunnels"""
    try:
        from pyngrok import ngrok

        ngrok.disconnect_all()
        logger.info("All tunnels closed")
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"Failed to close tunnels: {e}")
