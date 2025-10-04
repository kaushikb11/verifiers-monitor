"""
Test dashboard functionality
"""

import pytest

from verifiers_monitor.core.sqlmodel_storage import SQLModelStorage
from verifiers_monitor.dashboard.server import DashboardServer


def test_dashboard_creation(sqlite_storage):
    """Test dashboard server creation"""
    dashboard = DashboardServer(sqlite_storage, port=8888)

    assert dashboard.storage == sqlite_storage
    assert dashboard.port == 8888
    assert dashboard.app is not None


def test_dashboard_html_generation(sqlite_storage):
    """Test dashboard HTML generation"""
    dashboard = DashboardServer(sqlite_storage, port=8888)

    html = dashboard._get_dashboard_html_static()

    # Check for key components
    assert "Verifiers Monitor Dashboard" in html or "Verifiers Pro" in html
    assert "<!DOCTYPE html>" in html
    assert "</html>" in html


def test_dashboard_routes_setup(sqlite_storage):
    """Test that dashboard routes are properly configured"""
    dashboard = DashboardServer(sqlite_storage, port=8888)

    # Check that FastAPI app has routes configured
    routes = [route.path for route in dashboard.app.routes]

    expected_routes = ["/", "/api/metrics", "/api/status"]
    for expected in expected_routes:
        assert any(expected in route for route in routes)
