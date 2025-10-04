"""
Data Access API for Verifiers Monitor

This module provides a high-level API for researchers and engineers to fetch
and analyze evaluation data from the monitoring database.

Example:
    >>> from verifiers_monitor import MonitorData
    >>> data = MonitorData()
    >>>
    >>> # Get latest session
    >>> session = data.get_latest_session()
    >>> print(f"Session: {session.model_name} on {session.env_id}")
    >>>
    >>> # Get failed examples
    >>> failures = data.get_failed_examples(session.session_id, threshold=0.5)
    >>> print(f"Found {len(failures)} failures")
    >>>
    >>> # Export to pandas
    >>> df = data.to_dataframe(session.session_id)
    >>> print(df.head())
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import desc
from sqlmodel import Session, col, select

from .core.models import (
    EvaluationSession,
    ExampleSummary,
    RolloutMetric,
    RolloutResponse,
    SessionResponse,
    create_database_engine,
)
from .utils.logging import get_logger

logger = get_logger(__name__)


class MonitorData:
    """
    High-level API for accessing and analyzing Verifiers monitoring data.

    This class provides convenient methods for researchers to fetch evaluation
    results, analyze metrics, and export data for further analysis.

    Args:
        db_path: Path to SQLite database file (default: "~/.verifiers_monitor/vmf_metrics.db")

    Example:
        >>> data = MonitorData()
        >>> sessions = data.list_sessions()
        >>> latest = data.get_latest_session(env_id="math-python")
        >>> rollouts = data.get_rollouts(latest.session_id)
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize data access with database connection."""
        import os

        # Use same default location as SQLModelStorage
        if db_path is None:
            db_path = os.path.join(
                os.path.expanduser("~"), ".verifiers_monitor", "vmf_metrics.db"
            )

        self.db_path = Path(db_path)

        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database not found at {db_path}. "
                "Make sure you've run an evaluation with monitoring enabled."
            )

        self.engine = create_database_engine(f"sqlite:///{self.db_path}")
        logger.info(f"Connected to database: {self.db_path}")

    # =========================================================================
    # Session Management
    # =========================================================================

    def list_sessions(self) -> List[SessionResponse]:
        """
        List all evaluation sessions in the database.

        Returns:
            List of sessions, ordered by most recent first

        Example:
            >>> data = MonitorData()
            >>> sessions = data.list_sessions()
            >>> for session in sessions:
            ...     print(f"{session.model_name} - {session.env_id}")
        """
        with Session(self.engine) as session:
            statement = select(EvaluationSession).order_by(
                desc(col(EvaluationSession.started_at))
            )
            db_sessions = session.exec(statement).all()
            return [SessionResponse.from_db_model(s) for s in db_sessions]

    def get_latest_session(self, env_id: Optional[str] = None) -> SessionResponse:
        """
        Get the most recent evaluation session.

        Args:
            env_id: Optional filter by environment ID (e.g., "math-python", "gsm8k")

        Returns:
            Most recent session matching the criteria

        Raises:
            ValueError: If no sessions found

        Example:
            >>> data = MonitorData()
            >>> session = data.get_latest_session(env_id="math-python")
            >>> print(session.session_id)
        """
        with Session(self.engine) as session:
            statement = select(EvaluationSession).order_by(
                desc(col(EvaluationSession.started_at))
            )

            if env_id:
                statement = statement.where(EvaluationSession.env_id == env_id)

            db_session = session.exec(statement).first()

            if not db_session:
                filter_msg = f" for environment '{env_id}'" if env_id else ""
                raise ValueError(f"No sessions found{filter_msg}")

            return SessionResponse.from_db_model(db_session)

    def get_session(self, session_id: str) -> SessionResponse:
        """
        Get a specific session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session information

        Raises:
            ValueError: If session not found

        Example:
            >>> data = MonitorData()
            >>> session = data.get_session("session_123")
        """
        with Session(self.engine) as session:
            statement = select(EvaluationSession).where(
                EvaluationSession.session_id == session_id
            )
            db_session = session.exec(statement).first()

            if not db_session:
                raise ValueError(f"Session not found: {session_id}")

            return SessionResponse.from_db_model(db_session)

    def get_sessions_by_env(self, env_id: str) -> List[SessionResponse]:
        """
        Get all sessions for a specific environment.

        Args:
            env_id: Environment ID (e.g., "math-python", "gsm8k")

        Returns:
            List of sessions for this environment, ordered by most recent first

        Example:
            >>> data = MonitorData()
            >>> math_sessions = data.get_sessions_by_env("math-python")
            >>> print(f"Found {len(math_sessions)} math-python sessions")
        """
        with Session(self.engine) as session:
            statement = (
                select(EvaluationSession)
                .where(EvaluationSession.env_id == env_id)
                .order_by(desc(col(EvaluationSession.started_at)))
            )
            db_sessions = session.exec(statement).all()
            return [SessionResponse.from_db_model(s) for s in db_sessions]

    def filter_sessions(
        self,
        env_id: Optional[str] = None,
        model_name: Optional[str] = None,
        status: Optional[str] = None,
        min_examples: Optional[int] = None,
    ) -> List[SessionResponse]:
        """
        Filter sessions by multiple criteria.

        Args:
            env_id: Filter by environment ID (e.g., "math-python", "gsm8k")
            model_name: Filter by model name
            status: Filter by status ("running", "completed", "failed")
            min_examples: Minimum number of examples

        Returns:
            List of sessions matching all criteria

        Example:
            >>> data = MonitorData()
            >>> sessions = data.filter_sessions(
            ...     env_id="math-python",
            ...     model_name="gpt-4o-mini",
            ...     status="completed"
            ... )
        """
        with Session(self.engine) as session:
            statement = select(EvaluationSession).order_by(
                desc(col(EvaluationSession.started_at))
            )

            if env_id:
                statement = statement.where(EvaluationSession.env_id == env_id)
            if model_name:
                statement = statement.where(EvaluationSession.model_name == model_name)
            if status:
                statement = statement.where(EvaluationSession.status == status)
            if min_examples is not None:
                statement = statement.where(
                    EvaluationSession.num_examples >= min_examples
                )

            db_sessions = session.exec(statement).all()
            return [SessionResponse.from_db_model(s) for s in db_sessions]

    # =========================================================================
    # Rollout Retrieval
    # =========================================================================

    def get_rollouts(self, session_id: str) -> List[RolloutResponse]:
        """
        Get all rollouts for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of all rollouts in the session

        Example:
            >>> data = MonitorData()
            >>> rollouts = data.get_rollouts("session_123")
            >>> print(f"Total rollouts: {len(rollouts)}")
        """
        with Session(self.engine) as session:
            statement = (
                select(RolloutMetric)
                .where(RolloutMetric.session_id == session_id)
                .order_by(
                    col(RolloutMetric.example_number), col(RolloutMetric.rollout_number)
                )
            )
            db_rollouts = session.exec(statement).all()
            return [RolloutResponse.from_db_model(r) for r in db_rollouts]

    def get_example(self, session_id: str, example_num: int) -> List[RolloutResponse]:
        """
        Get all rollouts for a specific example.

        Args:
            session_id: Session identifier
            example_num: Example number

        Returns:
            List of rollouts for this example

        Example:
            >>> data = MonitorData()
            >>> rollouts = data.get_example("session_123", example_num=5)
            >>> for r in rollouts:
            ...     print(f"Rollout {r.rollout_number}: reward={r.reward}")
        """
        with Session(self.engine) as session:
            statement = (
                select(RolloutMetric)
                .where(
                    RolloutMetric.session_id == session_id,
                    RolloutMetric.example_number == example_num,
                )
                .order_by(col(RolloutMetric.rollout_number))
            )
            db_rollouts = session.exec(statement).all()
            return [RolloutResponse.from_db_model(r) for r in db_rollouts]

    def filter_rollouts(
        self,
        session_id: str,
        min_reward: Optional[float] = None,
        max_reward: Optional[float] = None,
        example_numbers: Optional[List[int]] = None,
    ) -> List[RolloutResponse]:
        """
        Filter rollouts by criteria.

        Args:
            session_id: Session identifier
            min_reward: Minimum reward threshold
            max_reward: Maximum reward threshold
            example_numbers: List of specific example numbers

        Returns:
            Filtered list of rollouts

        Example:
            >>> data = MonitorData()
            >>> failures = data.filter_rollouts(
            ...     "session_123",
            ...     max_reward=0.5
            ... )
        """
        with Session(self.engine) as session:
            statement = select(RolloutMetric).where(
                RolloutMetric.session_id == session_id
            )

            if min_reward is not None:
                statement = statement.where(RolloutMetric.reward >= min_reward)
            if max_reward is not None:
                statement = statement.where(RolloutMetric.reward <= max_reward)
            if example_numbers is not None:
                statement = statement.where(
                    col(RolloutMetric.example_number).in_(example_numbers)
                )

            statement = statement.order_by(
                col(RolloutMetric.example_number), col(RolloutMetric.rollout_number)
            )

            db_rollouts = session.exec(statement).all()
            return [RolloutResponse.from_db_model(r) for r in db_rollouts]

    # =========================================================================
    # Example-Level Analysis
    # =========================================================================

    def get_examples(self, session_id: str) -> List[ExampleSummary]:
        """
        Get example-level summary (aggregated across rollouts).

        Returns one entry per example with aggregated statistics.

        Args:
            session_id: Session identifier

        Returns:
            List of example summaries with stats

        Example:
            >>> data = MonitorData()
            >>> examples = data.get_examples("session_123")
            >>> for ex in examples:
            ...     print(f"Example {ex.example_number}: "
            ...           f"avg_reward={ex.mean_reward:.2f}")
        """
        rollouts = self.get_rollouts(session_id)

        examples_dict: Dict[int, List[RolloutResponse]] = {}
        for rollout in rollouts:
            ex_num = rollout.example_number
            if ex_num not in examples_dict:
                examples_dict[ex_num] = []
            examples_dict[ex_num].append(rollout)

        # Aggregate stats per example
        examples = []
        for ex_num, ex_rollouts in sorted(examples_dict.items()):
            rewards = [r.reward for r in ex_rollouts]
            mean_reward = sum(rewards) / len(rewards) if rewards else 0

            # Calculate standard deviation
            if len(rewards) > 1:
                variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
                std_reward = variance**0.5
            else:
                std_reward = 0.0

            examples.append(
                ExampleSummary(
                    example_number=ex_num,
                    num_rollouts=len(ex_rollouts),
                    mean_reward=mean_reward,
                    max_reward=max(rewards) if rewards else 0,
                    min_reward=min(rewards) if rewards else 0,
                    std_reward=std_reward,
                    rollouts=ex_rollouts,
                )
            )

        return examples

    def get_failed_examples(
        self, session_id: str, threshold: float = 0.5
    ) -> List[ExampleSummary]:
        """
        Get examples that failed (reward below threshold).

        Args:
            session_id: Session identifier
            threshold: Reward threshold (default: 0.5)

        Returns:
            List of failed examples with their rollouts

        Example:
            >>> data = MonitorData()
            >>> failures = data.get_failed_examples("session_123", threshold=0.5)
            >>> print(f"Found {len(failures)} failures")
        """
        examples = self.get_examples(session_id)
        return [ex for ex in examples if ex.is_failure(threshold)]

    def get_top_failures(self, session_id: str, n: int = 10) -> List[ExampleSummary]:
        """
        Get the worst performing examples.

        Args:
            session_id: Session identifier
            n: Number of failures to return

        Returns:
            Top N worst examples, sorted by mean reward (ascending)

        Example:
            >>> data = MonitorData()
            >>> worst = data.get_top_failures("session_123", n=5)
            >>> for ex in worst:
            ...     print(f"Example {ex.example_number}: {ex.mean_reward:.2f}")
        """
        examples = self.get_examples(session_id)
        return sorted(examples, key=lambda x: x.mean_reward)[:n]

    # =========================================================================
    # Data Export
    # =========================================================================

    def to_dataframe(self, session_id: str):
        """
        Export session rollouts to pandas DataFrame.

        Args:
            session_id: Session identifier

        Returns:
            pandas.DataFrame with all rollouts

        Raises:
            ImportError: If pandas is not installed

        Example:
            >>> data = MonitorData()
            >>> df = data.to_dataframe("session_123")
            >>> print(df.describe())
            >>> print(df.groupby('example_number')['reward'].mean())
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for DataFrame export. "
                "Install with: pip install 'verifiers-monitor[analysis]'"
            )

        rollouts = self.get_rollouts(session_id)

        data = []
        for r in rollouts:
            row = {
                "example_number": r.example_number,
                "rollout_number": r.rollout_number,
                "reward": r.reward,
                "rollout_time": r.rollout_time,
                "completion_length": r.completion_length,
                "model_name": r.model_name,
                "env_type": r.env_type,
                "task": r.task,
                "parser_type": r.parser_type,
                "timestamp": r.timestamp,
            }

            if r.metrics:
                for metric_name, metric_value in r.metrics.items():
                    row[f"metric_{metric_name}"] = metric_value

            data.append(row)

        return pd.DataFrame(data)

    def get_summary_table(self, session_id: str):
        """
        Get example-level summary as pandas DataFrame.

        Args:
            session_id: Session identifier

        Returns:
            pandas.DataFrame with one row per example

        Example:
            >>> data = MonitorData()
            >>> summary = data.get_summary_table("session_123")
            >>> print(summary[['example_number', 'mean_reward', 'max_reward']])
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for DataFrame export. "
                "Install with: pip install 'verifiers-monitor[analysis]'"
            )

        examples = self.get_examples(session_id)

        data = [
            {
                "example_number": ex.example_number,
                "num_rollouts": ex.num_rollouts,
                "mean_reward": ex.mean_reward,
                "max_reward": ex.max_reward,
                "min_reward": ex.min_reward,
                "std_reward": ex.std_reward,
            }
            for ex in examples
        ]

        return pd.DataFrame(data)

    # =========================================================================
    # Environment Discovery
    # =========================================================================

    def list_environments(self) -> List[str]:
        """
        List all environment IDs in the database.

        Returns:
            List of unique environment IDs (e.g., ["math-python", "gsm8k"])

        Example:
            >>> data = MonitorData()
            >>> envs = data.list_environments()
            >>> print(f"Environments: {', '.join(envs)}")
        """
        with Session(self.engine) as session:
            statement = select(EvaluationSession.env_id).distinct()
            env_ids = session.exec(statement).all()
            return sorted([env for env in env_ids if env])

    def get_environment_info(self, env_id: str) -> Dict[str, Any]:
        """
        Get metadata about an environment.

        Args:
            env_id: Environment ID (e.g., "math-python", "gsm8k")

        Returns:
            Dictionary with environment statistics

        Example:
            >>> data = MonitorData()
            >>> info = data.get_environment_info("math-python")
            >>> print(f"Sessions: {info['num_sessions']}")
        """
        sessions = self.get_sessions_by_env(env_id)

        if not sessions:
            return {
                "env_id": env_id,
                "num_sessions": 0,
                "models_used": [],
            }

        models_used = list(set(s.model_name for s in sessions))

        return {
            "env_id": env_id,
            "num_sessions": len(sessions),
            "models_used": models_used,
            "first_session": sessions[-1].started_at if sessions else None,
            "latest_session": sessions[0].started_at if sessions else None,
        }
