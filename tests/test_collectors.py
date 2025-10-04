"""
Test metrics collection functionality
"""

import time

import pytest

from verifiers_monitor.core.collectors import MetricsCollector


def test_metrics_collector_creation(sqlite_storage):
    """Test metrics collector creation"""
    collector = MetricsCollector(sqlite_storage)

    assert collector.storage == sqlite_storage
    assert collector._evaluation_count == 0


def test_training_step_collection(sqlite_storage):
    """Test training step metrics collection"""
    collector = MetricsCollector(sqlite_storage)

    # Collect training metrics
    metrics = {"step": 1, "loss": 2.5, "compute_time": 0.1, "timestamp": time.time()}

    collector.collect_training_step(metrics)

    # Verify storage
    stored_data = sqlite_storage.get_recent("training_step")
    assert len(stored_data) == 1
    assert stored_data[0]["step"] == 1
    assert stored_data[0]["loss"] == 2.5


def test_evaluation_session(sqlite_storage):
    """Test evaluation session management"""
    collector = MetricsCollector(sqlite_storage)

    # Start evaluation session
    collector.start_evaluation_session("test_model", 10)

    # Verify session started
    session_data = sqlite_storage.get_recent("evaluation_session")
    assert len(session_data) == 1
    assert session_data[0]["model"] == "test_model"
    assert session_data[0]["num_examples"] == 10


def test_rollout_collection(sqlite_storage):
    """Test rollout metrics collection"""
    collector = MetricsCollector(sqlite_storage)

    # Start session first
    collector.start_evaluation_session("test_model", 3)

    # Collect rollout data
    rollout_data = {
        "prompt": "test prompt",
        "completion": "test completion",
        "reward": 0.8,
        "rollout_time": 0.05,
    }

    collector.collect_rollout(rollout_data)

    # Verify storage
    stored_data = sqlite_storage.get_recent("rollout")
    assert len(stored_data) == 1
    assert stored_data[0]["reward"] == 0.8
    assert stored_data[0]["example_number"] == 1


def test_advantages_collection(sqlite_storage):
    """Test advantages metrics collection"""
    collector = MetricsCollector(sqlite_storage)

    # Mock advantages with statistics
    advantages = {"mean": 0.1, "std": 0.2, "min": -0.1, "max": 0.3}

    metrics = {
        "step": 1,
        "loss": 2.0,
        "advantages": advantages,
        "timestamp": time.time(),
    }

    collector.collect_training_step(metrics)

    # Verify advantages are stored
    stored_data = sqlite_storage.get_recent("training_step")
    assert stored_data[0]["advantages_mean"] == 0.1
    assert stored_data[0]["advantages_std"] == 0.2
