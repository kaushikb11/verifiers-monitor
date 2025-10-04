"""
Test storage backend functionality
"""

import time

import pytest

from verifiers_monitor.core.sqlmodel_storage import SQLModelStorage


def test_sqlite_storage_basic(sqlite_storage):
    """Test basic SQLite storage operations with rollout data"""
    rollout_data = {
        "session_id": "test_session",
        "example_number": 1,
        "rollout_number": 1,
        "prompt": [{"role": "user", "content": "test"}],
        "completion": [{"role": "assistant", "content": "response"}],
        "reward": 0.8,
        "rollout_time": 0.5,
        "timestamp": time.time(),
    }

    # Test store and retrieve
    sqlite_storage.store("rollout", rollout_data)
    data = sqlite_storage.get_recent("rollout")

    assert len(data) == 1
    assert data[0]["reward"] == 0.8


def test_sqlite_storage_session(sqlite_storage):
    """Test SQLite storage with evaluation sessions"""
    session_data = {
        "session_id": "test_session_123",
        "model_name": "gpt-4o-mini",
        "env_id": "math-python",
        "num_examples": 10,
        "timestamp": time.time(),
    }

    # Store session
    sqlite_storage.store("evaluation_session", session_data)

    # Verify session was stored
    sessions = sqlite_storage.get_sessions()
    assert len(sessions) > 0
    assert any(s.session_id == "test_session_123" for s in sessions)


def test_storage_get_stats(sqlite_storage):
    """Test get_stats functionality"""
    # Store some rollout data first
    rollout_data = {
        "session_id": "test_session",
        "example_number": 1,
        "rollout_number": 1,
        "prompt": [{"role": "user", "content": "test"}],
        "completion": [{"role": "assistant", "content": "response"}],
        "reward": 0.9,
        "rollout_time": 0.5,
        "timestamp": time.time(),
    }
    sqlite_storage.store("rollout", rollout_data)

    stats = sqlite_storage.get_stats()
    assert "total_rollouts" in stats
    assert stats["total_rollouts"] >= 1
