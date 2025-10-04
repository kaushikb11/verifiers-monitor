"""
Test core Verifiers Monitor functionality
"""

import time

import pytest

from verifiers_monitor import monitor


def test_import():
    """Test that Verifiers Monitor can be imported successfully"""
    from verifiers_monitor import monitor

    assert monitor is not None


def test_trainer_monitoring(mock_trainer):
    """Test basic trainer monitoring functionality"""
    monitored_trainer = monitor(mock_trainer, dashboard=False)

    # Test that the wrapper preserves the original interface
    assert hasattr(monitored_trainer, "train")
    assert hasattr(monitored_trainer, "state")

    # Test training execution
    result = monitored_trainer.train()
    assert result == "success"
    assert monitored_trainer.state.global_step == 3


def test_environment_monitoring(mock_environment):
    """Test basic environment monitoring functionality"""
    monitored_env = monitor(mock_environment, dashboard=False)

    # Test that the wrapper preserves the original interface
    assert hasattr(monitored_env, "evaluate")
    assert hasattr(monitored_env, "rollout")

    # Test evaluation execution
    results = monitored_env.evaluate("client", "model")
    assert len(results.reward) == 3
    assert all(isinstance(r, float) for r in results.reward)


def test_monitor_invalid_object():
    """Test that monitor raises error for invalid objects"""
    with pytest.raises(ValueError, match="Cannot monitor object"):
        monitor("invalid_object", dashboard=False)
