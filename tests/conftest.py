"""
Pytest configuration and fixtures for Verifiers Monitor tests
"""

import tempfile
from pathlib import Path

import pytest

from verifiers_monitor.core.sqlmodel_storage import SQLModelStorage


@pytest.fixture
def sqlite_storage():
    """Provide a clean SQLite storage instance for testing"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        storage = SQLModelStorage(database_url=f"sqlite:///{tmp.name}")
        yield storage
        # Cleanup
        Path(tmp.name).unlink(missing_ok=True)


@pytest.fixture
def mock_trainer():
    """Mock trainer for testing"""

    class MockTrainer:
        def __init__(self):
            self.state = type("State", (), {"global_step": 0})()
            self.env = "test_env"

        def train(self):
            for i in range(3):
                self.state.global_step = i + 1
            return "success"

        def compute_loss(self, model, inputs, **kwargs):
            return 2.5 - (self.state.global_step * 0.1)

        def log(self, logs, **kwargs):
            pass

    return MockTrainer()


@pytest.fixture
def mock_environment():
    """Mock environment for testing"""

    class MockEnvironment:
        async def rollout(self, client, model, prompt, answer="", **kwargs):
            import asyncio

            await asyncio.sleep(0.01)
            return f"completion for {prompt}", {"reward": 0.8}

        def evaluate(self, client, model, num_examples=3, **kwargs):
            return type(
                "Results",
                (),
                {"reward": [0.7, 0.8, 0.9], "completion": ["comp1", "comp2", "comp3"]},
            )()

    return MockEnvironment()
