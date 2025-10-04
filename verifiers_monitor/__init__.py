"""
Verifiers Monitor
Observability for Verifiers RL training and evaluation
"""

__version__ = "0.1.0"

import logging
from typing import TYPE_CHECKING, TypeVar, Union

from .core.models import ExampleSummary
from .core.wrappers import auto_wrap
from .data_access import MonitorData
from .utils import cli_output

if TYPE_CHECKING:
    from verifiers.envs import Environment
    from verifiers.trainers import GRPOTrainer


logging.getLogger("verifiers").setLevel(logging.WARNING)

T = TypeVar("T")


def monitor(
    component: T,
    dashboard: bool = True,
    port: int = 8080,
    tunnel: bool = False,
) -> T:
    """
    Add monitoring to Verifiers.

    Args:
        component: A Verifiers component (GRPOTrainer or Environment)
        dashboard: Whether to start web dashboard (default: True)
        port: Dashboard port (default: 8080)
        tunnel: Create ngrok tunnel for remote access (default: False)

    Returns:
        Wrapped component with identical API + monitoring capabilities

    Raises:
        ValueError: If component is not a supported type (GRPOTrainer or Environment)

    Example:
        >>> import verifiers as vf
        >>> from verifiers_monitor import monitor
        >>>
        >>> # Simple monitoring (default port 8080)
        >>> env = monitor(vf.load_environment("gsm8k"))
        >>> results = env.evaluate(client, model="gpt-4o-mini")
        >>> # Dashboard automatically available at http://localhost:8080 (or next available port)
        >>>
        >>> # Remote monitoring with tunnel
        >>> env = monitor(env, tunnel=True)
        >>> # Dashboard at https://abc123.ngrok.io
        >>>
        >>> # Multiple environments (automatic port management)
        >>> env1 = monitor(vf.load_environment("gsm8k"))      # Port 8080
        >>> env2 = monitor(vf.load_environment("math"))       # Port 8081 (auto)
        >>> # Both dashboards work independently
    """
    return auto_wrap(component, dashboard=dashboard, port=port, tunnel=tunnel)


__all__ = ["monitor", "MonitorData", "ExampleSummary"]
