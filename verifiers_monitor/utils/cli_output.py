"""
Styled CLI output for Verifiers Monitor using Rich library

This module provides beautiful, user-friendly terminal output with colors,
panels, and formatting to make the monitoring experience more polished.
"""

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Global console instance
console = Console()

# Track if banner has been printed
_BANNER_PRINTED = False


def print_banner():
    """Print startup banner (only once per session)"""
    global _BANNER_PRINTED

    if _BANNER_PRINTED:
        return

    console.print()  # Add newline before banner
    console.print(
        "[bold cyan]Verifiers Monitor[/bold cyan] [dim]─ Real-time monitoring[/dim]"
    )
    _BANNER_PRINTED = True


def print_dashboard_url(url: str, port: int, tunnel_url: Optional[str] = None):
    """Print dashboard URL"""
    console.print(f"[cyan]→[/cyan] Dashboard: [bold cyan]{url}[/bold cyan]")

    if tunnel_url:
        console.print(
            f"[magenta]→[/magenta] Remote:    [bold magenta]{tunnel_url}[/bold magenta]"
        )

    console.print()


def print_port_info(original_port: int, new_port: int):
    """Print port fallback info"""
    console.print(
        f"[dim]Port [cyan]{original_port}[/cyan] in use, using [cyan]{new_port}[/cyan][/dim]"
    )


def print_evaluation_complete(
    url: str,
    port: int,
    total_rollouts: Optional[int] = None,
    duration: Optional[float] = None,
    rate: Optional[float] = None,
):
    """Print evaluation completion summary"""
    console.print()
    console.print("[bold green]Evaluation complete[/bold green]")
    console.print(f"[cyan]→[/cyan] Dashboard: [bold cyan]{url}[/bold cyan]")
    console.print()


def print_session_reuse(port: int, session_num: int):
    """Print session reuse message"""
    console.print(
        f"[dim]Reusing dashboard on port [cyan]{port}[/cyan] (session #{session_num})[/dim]"
    )


def print_dashboard_started(url: str):
    """Print dashboard started message (for internal logging)"""
    console.print(f"[dim]Dashboard started: [cyan]{url}[/cyan][/dim]")


def print_dashboard_stopped(port: int):
    """Print dashboard stopped message"""
    console.print(f"[dim]Dashboard stopped (port {port})[/dim]")


def print_info(message: str):
    """Print info message"""
    console.print(f"[dim]{message}[/dim]")


def print_warning(message: str):
    """Print warning message"""
    console.print(f"[yellow]{message}[/yellow]")


def print_error(message: str):
    """Print error message"""
    console.print(f"[bold red]{message}[/bold red]")


def print_success(message: str):
    """Print success message"""
    console.print(f"[green]{message}[/green]")
