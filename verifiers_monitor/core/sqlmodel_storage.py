"""
Modern SQLModel-based storage backend

This demonstrates best practices for:
- Clean architecture with proper separation of concerns
- Type-safe database operations
- Proper error handling and logging
- Efficient querying with pagination
- Connection management
"""

import ast
import json
import sqlite3
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlmodel import Session, and_, desc, func, select

from ..utils.logging import get_logger
from ..utils.recovery import RetryConfig, retry_on_failure
from .constants import (
    DEFAULT_EMPTY_STRING,
    DEFAULT_ENV_TYPE,
    DEFAULT_MODEL_NAME,
    DEFAULT_PARSER_TYPE,
    DEFAULT_SESSION_ID,
    DEFAULT_TASK,
)
from .exceptions import (
    StorageConnectionError,
    StorageError,
    StorageIntegrityError,
    StorageTimeoutError,
    ValidationError,
)
from .models import (
    ErrorEvent,
    EvaluationProgress,
    EvaluationSession,
    ProgressResponse,
    RolloutMetric,
    RolloutResponse,
    SessionResponse,
    TrainingStep,
    create_database_engine,
    create_tables,
    get_session,
)

logger = get_logger(__name__)


class SQLModelStorage:
    """
    Modern SQLModel-based storage backend

    Demonstrates best practices:
    - Proper ORM usage with SQLModel
    - Type safety with Pydantic validation
    - Clean separation between database models and API responses
    - Efficient querying with proper indexing
    - Connection pooling and error handling
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        pool_size: int = 5,
        max_overflow: int = 10,
    ):
        """
        Initialize SQLModel storage with enhanced connection management

        Args:
            database_url: Database connection string (SQLite by default)
            pool_size: Number of connections to maintain in pool
            max_overflow: Maximum overflow connections beyond pool_size
        """
        # Use absolute path for consistent database location
        if database_url is None:
            import os

            db_path = os.path.join(
                os.path.expanduser("~"), ".verifiers_monitor", "vmf_metrics.db"
            )
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            database_url = f"sqlite:///{db_path}"

        self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow

        self.engine = self._create_enhanced_engine()

        create_tables(self.engine)

        logger.info(f"SQLModel storage initialized: {database_url}")
        logger.info(
            f"Connection pool: {pool_size} connections, {max_overflow} overflow"
        )

    def _create_enhanced_engine(self):
        """Create database engine with enhanced connection pooling and configuration"""
        from sqlmodel import create_engine

        # Enhanced engine configuration
        engine_kwargs = {
            "echo": False,
            "future": True,
        }

        if not self.database_url.startswith("sqlite"):
            engine_kwargs.update(
                {
                    "pool_size": self.pool_size,
                    "max_overflow": self.max_overflow,
                    "pool_pre_ping": True,
                    "pool_recycle": 3600,
                }
            )
        else:
            # SQLite-specific optimizations
            engine_kwargs.update(
                {
                    "connect_args": {
                        "check_same_thread": False,
                        "timeout": 30,
                    }
                }
            )

        return create_engine(self.database_url, **engine_kwargs)

    # =============================================================================
    # Core Storage Interface (Legacy Compatibility)
    # =============================================================================

    @retry_on_failure(RetryConfig(max_attempts=3))
    def store(self, metric_type: str, data: Dict[str, Any]) -> None:
        """
        Store metric data (legacy interface for backward compatibility)

        Routes different metric types to appropriate SQLModel tables
        """
        try:
            if metric_type == "rollout":
                self._store_rollout(data)
            elif metric_type == "training_step":
                self._store_training_step(data)
            elif metric_type == "evaluation_progress":
                self._store_progress(data)
            elif metric_type == "evaluation_session":
                self._store_evaluation_session(data)
            elif metric_type == "evaluation_session_end":
                self._store_evaluation_session_end(data)
            elif metric_type == "error_event":
                self._store_error_event(data)
            else:
                logger.warning(f"Unknown metric type: {metric_type}")
                raise ValidationError(f"Unsupported metric type: {metric_type}")

        except ValidationError:
            raise
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower():
                raise StorageConnectionError(f"Database locked: {e}") from e
            elif "no such table" in str(e).lower():
                raise StorageIntegrityError(f"Database schema error: {e}") from e
            else:
                raise StorageError(f"Database operation failed: {e}") from e
        except sqlite3.IntegrityError as e:
            raise StorageIntegrityError(f"Data integrity violation: {e}") from e
        except Exception as e:
            logger.error(
                f"Unexpected storage error for {metric_type}: {e}", exc_info=True
            )
            raise StorageError(f"Storage operation failed: {e}") from e

    def get_recent(self, metric_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent metrics"""
        try:
            with get_session(self.engine) as session:
                if metric_type == "rollout":
                    rollouts = session.exec(
                        select(RolloutMetric)
                        .order_by(desc(RolloutMetric.timestamp))
                        .limit(limit)
                    ).all()
                    return [self._rollout_to_dict(rollout) for rollout in rollouts]
                elif metric_type == "training_step":
                    steps = session.exec(
                        select(TrainingStep)
                        .order_by(desc(TrainingStep.timestamp))
                        .limit(limit)
                    ).all()
                    return [self._training_step_to_dict(step) for step in steps]
                elif metric_type == "evaluation_progress":
                    progress_records = session.exec(
                        select(EvaluationProgress)
                        .order_by(desc(EvaluationProgress.timestamp))
                        .limit(limit)
                    ).all()
                    return [
                        {
                            "timestamp": p.timestamp,
                            "completed": p.completed,
                            "total": p.total,
                            "percentage": p.percentage,
                            "eta_seconds": p.eta_seconds,
                            "throughput": p.throughput,
                            "elapsed_time": p.elapsed_time,
                            "session_id": p.session_id,
                        }
                        for p in progress_records
                    ]
                elif metric_type == "evaluation_session":
                    sessions = session.exec(
                        select(EvaluationSession)
                        .order_by(desc(EvaluationSession.started_at))
                        .limit(limit)
                    ).all()
                    return [
                        {
                            "session_id": s.session_id,
                            "model": s.model_name,  # Backward compatibility
                            "model_name": s.model_name,
                            "environment_type": s.environment_type,
                            "env_type": s.environment_type,  # Backward compatibility
                            "env_id": s.env_id,
                            "num_examples": s.num_examples,
                            "started_at": s.started_at,
                            "completed_at": s.completed_at,
                            "status": s.status,
                            "final_reward_mean": s.final_reward_mean,
                            "total_examples": s.total_examples,
                        }
                        for s in sessions
                    ]
                else:
                    return []
        except Exception as e:
            logger.error(f"Query error for {metric_type}: {e}")
            return []

    def get_latest(self) -> Dict[str, Any]:
        """Get latest metrics across all types (legacy interface)"""
        try:
            with get_session(self.engine) as session:
                latest_rollout = session.exec(
                    select(RolloutMetric)
                    .order_by(desc(RolloutMetric.timestamp))
                    .limit(1)
                ).first()

                latest_training = session.exec(
                    select(TrainingStep).order_by(desc(TrainingStep.timestamp)).limit(1)
                ).first()

                result = {}
                if latest_rollout:
                    result["rollout"] = self._rollout_to_dict(latest_rollout)
                if latest_training:
                    result["training_step"] = self._training_step_to_dict(
                        latest_training
                    )

                return result

        except Exception as e:
            logger.error(f"Latest query error: {e}")
            return {}

    # =============================================================================
    # Modern SQLModel Methods (Best Practices)
    # =============================================================================

    def create_session(
        self,
        session_id: str,
        model_name: str,
        environment_type: str,
        num_examples: int,
        env_config: Optional[Dict[str, Any]] = None,
        env_id: Optional[str] = None,
    ) -> EvaluationSession:
        """
        Create a new evaluation session with proper validation

        Returns:
            Created session model
        """
        with get_session(self.engine) as session:
            db_session = EvaluationSession(
                session_id=session_id,
                model_name=model_name,
                environment_type=environment_type,
                env_id=env_id,
                num_examples=num_examples,
                env_config=json.dumps(env_config) if env_config else None,
                status="running",
            )

            session.add(db_session)
            session.commit()
            session.refresh(db_session)

            return db_session

    def store_rollout(self, rollout_data: Dict[str, Any]) -> RolloutMetric:
        """
        Store rollout with proper validation and type safety

        Args:
            rollout_data: Rollout metrics dictionary

        Returns:
            Created rollout model
        """
        return self._store_rollout(rollout_data)

    def get_progress(self, session_id: str) -> Optional[ProgressResponse]:
        """Get latest progress for a session"""
        with get_session(self.engine) as session:
            progress = session.exec(
                select(EvaluationProgress)
                .where(EvaluationProgress.session_id == session_id)
                .order_by(desc(EvaluationProgress.timestamp))
                .limit(1)
            ).first()

            if progress:
                return ProgressResponse(
                    session_id=progress.session_id,
                    completed=progress.completed,
                    total=progress.total,
                    percentage=progress.percentage,
                    throughput=progress.throughput,
                    eta_seconds=progress.eta_seconds,
                    timestamp=progress.timestamp,
                )
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        with get_session(self.engine) as session:
            rollout_count = session.exec(select(func.count(RolloutMetric.id))).one()
            session_count = session.exec(select(func.count(EvaluationSession.id))).one()
            training_count = session.exec(select(func.count(TrainingStep.id))).one()

            # Recent activity
            recent_sessions = session.exec(
                select(EvaluationSession)
                .order_by(desc(EvaluationSession.started_at))
                .limit(5)
            ).all()

            return {
                "total_rollouts": rollout_count,
                "total_sessions": session_count,
                "total_training_steps": training_count,
                "recent_sessions": [
                    {
                        "session_id": s.session_id,
                        "model": s.model_name,
                        "environment": s.environment_type,
                        "started_at": s.started_at.isoformat(),
                        "status": s.status,
                    }
                    for s in recent_sessions
                ],
            }

    # =============================================================================
    # Private Helper Methods
    # =============================================================================

    def _json_serializer(self, obj):
        """Custom JSON serializer for complex objects"""
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        elif hasattr(obj, "__dict__"):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        else:
            return str(obj)

    def _normalize_metrics(self, metrics_data):
        """
        Normalize metrics data to ensure it's a proper dict, not a string representation.

        Handles cases where metrics might be:
        - A proper dict: {"key": "value"}
        - A string representation: "{'key': 'value'}" or '{"key": "value"}'
        - None or empty
        """
        if metrics_data is None:
            return None

        if isinstance(metrics_data, dict):
            return metrics_data

        if isinstance(metrics_data, str):
            try:
                # First try JSON parsing (handles double quotes)
                return json.loads(metrics_data)
            except (json.JSONDecodeError, ValueError):
                try:
                    # Fall back to ast.literal_eval for Python dict strings (single quotes)
                    return ast.literal_eval(metrics_data)
                except (ValueError, SyntaxError) as e:
                    logger.warning(
                        f"Failed to parse metrics string: {e}, raw value: {metrics_data[:100]}"
                    )
                    return None

        return metrics_data

    def _store_error_event(self, data: Dict[str, Any]) -> ErrorEvent:
        """Store error event"""
        try:
            with get_session(self.engine) as session:
                error_event = ErrorEvent(
                    session_id=data.get("session_id", DEFAULT_SESSION_ID),
                    timestamp=data.get("timestamp", time.time()),
                    error_type=data.get("error_type", "unknown"),
                    error_message=data.get("error_message", ""),
                    error_context=data.get("error_context"),
                    example_number=data.get("example_number"),
                    rollout_number=data.get("rollout_number"),
                    model_name=data.get("model_name"),
                    retry_count=data.get("retry_count", 0),
                    recovered=data.get("recovered", False),
                )

                session.add(error_event)
                session.commit()
                session.refresh(error_event)

                logger.debug(
                    f"Stored error event: {error_event.error_type} at {error_event.timestamp}"
                )
                return error_event
        except Exception as e:
            logger.error(f"Error storing error event: {e}")
            raise StorageError(f"Failed to store error event: {e}") from e

    def _rollout_to_dict(self, rollout: RolloutMetric) -> Dict[str, Any]:
        """Convert rollout model to dictionary (legacy compatibility)"""
        result = {
            "timestamp": rollout.timestamp,
            "example_number": rollout.example_number,
            "rollout_number": rollout.rollout_number,
            "reward": rollout.reward,
            "rollout_time": rollout.rollout_time,
            "completion_length": rollout.completion_length,
            "session_id": rollout.session_id,
        }

        if rollout.prompt:
            result["prompt"] = rollout.prompt
        if rollout.completion:
            result["completion"] = rollout.completion
        if rollout.reward_breakdown:
            try:
                result["reward_breakdown"] = json.loads(rollout.reward_breakdown)
            except json.JSONDecodeError:
                pass

        return result

    def _training_step_to_dict(self, step: TrainingStep) -> Dict[str, Any]:
        """Convert training step model to dictionary (legacy compatibility)"""
        return {
            "timestamp": step.timestamp,
            "step": step.step,
            "loss": step.loss,
            "compute_time": step.compute_time,
            "advantages_mean": step.advantages_mean,
            "advantages_std": step.advantages_std,
            "advantages_min": step.advantages_min,
            "advantages_max": step.advantages_max,
        }

    # =============================================================================
    # Storage Methods
    # =============================================================================

    def _store_rollout(self, data: Dict[str, Any]) -> RolloutMetric:
        """Store rollout data using SQLModel"""
        try:
            with get_session(self.engine) as session:
                rollout_number = data.get("rollout_number")
                if rollout_number is None:
                    session_id = data.get("session_id", DEFAULT_SESSION_ID)
                    max_rollout = session.exec(
                        select(func.max(RolloutMetric.rollout_number))
                        .where(RolloutMetric.session_id == session_id)
                        .with_for_update()
                    ).first()
                    rollout_number = (max_rollout or 0) + 1

                rollout = RolloutMetric(
                    timestamp=data.get("timestamp", time.time()),
                    example_number=data.get("example_number", 0),
                    rollout_number=rollout_number,
                    reward=data.get("reward", 0.0),
                    rollout_time=data.get("rollout_time", 0.0),
                    session_id=data.get("session_id", DEFAULT_SESSION_ID),
                    prompt=self._json_serializer(data.get("prompt")),
                    completion=self._json_serializer(data.get("completion")),
                    answer=data.get("answer", DEFAULT_EMPTY_STRING),
                    completion_length=len(
                        str(data.get("completion", DEFAULT_EMPTY_STRING))
                    ),
                    env_type=data.get("env_type", DEFAULT_ENV_TYPE),
                    parser_type=data.get("parser_type", DEFAULT_PARSER_TYPE),
                    task=data.get("task", DEFAULT_TASK),
                    prompt_hash=data.get("prompt_hash", DEFAULT_EMPTY_STRING),
                    rollout_id=data.get("rollout_id", DEFAULT_EMPTY_STRING),
                    reward_breakdown=self._json_serializer(
                        data.get("reward_breakdown")
                    ),
                    parsing_analysis=self._json_serializer(
                        data.get("parsing_analysis")
                    ),
                    response_analysis=self._json_serializer(
                        data.get("response_analysis")
                    ),
                    metrics=self._json_serializer(
                        self._normalize_metrics(data.get("metrics"))
                    ),
                )
                session.add(rollout)
                session.commit()
                session.refresh(rollout)
                return rollout
        except Exception as e:
            logger.error(f"Error storing rollout: {e}")
            if session:
                session.rollback()
            raise StorageError(f"Failed to store rollout: {e}") from e

    def _store_training_step(self, data: Dict[str, Any]) -> None:
        """Store training step data using SQLModel"""
        try:
            with get_session(self.engine) as session:
                step = TrainingStep(
                    session_id=data.get("session_id", DEFAULT_SESSION_ID),
                    timestamp=data.get("timestamp", time.time()),
                    step=data.get("step", 0),
                    loss=data.get("loss", 0.0),
                    compute_time=data.get("compute_time", 0.0),
                    advantages_mean=data.get("advantages_mean", 0.0),
                    advantages_std=data.get("advantages_std", 0.0),
                    advantages_min=data.get("advantages_min", 0.0),
                    advantages_max=data.get("advantages_max", 0.0),
                )
                session.add(step)
                session.commit()
        except Exception as e:
            logger.error(f"Error storing training step: {e}")
            if session:
                session.rollback()
            raise StorageError(f"Failed to store training step: {e}") from e

    def _store_progress(self, data: Dict[str, Any]) -> None:
        """Store evaluation progress data using SQLModel"""
        try:
            with get_session(self.engine) as session:
                progress = EvaluationProgress(
                    timestamp=data.get("timestamp", time.time()),
                    completed=data.get("completed", 0),
                    total=data.get("total", 0),
                    percentage=data.get("percentage", 0.0),
                    eta_seconds=data.get("eta_seconds"),
                    throughput=data.get("throughput", 0.0),
                    elapsed_time=data.get("elapsed_time", 0.0),
                    session_id=data.get("session_id", DEFAULT_SESSION_ID),
                )
                session.add(progress)
                session.commit()
        except Exception as e:
            logger.error(f"Error storing progress: {e}")
            if session:
                session.rollback()
            raise StorageError(f"Failed to store progress: {e}") from e

    def _store_evaluation_session(self, data: Dict[str, Any]) -> None:
        """Store evaluation session data using SQLModel"""
        try:
            with get_session(self.engine) as session:
                existing = session.exec(
                    select(EvaluationSession).where(
                        EvaluationSession.session_id == data.get("session_id")
                    )
                ).first()

                if existing:
                    for key, value in data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                else:
                    eval_session = EvaluationSession(
                        session_id=data.get("session_id", DEFAULT_SESSION_ID),
                        model_name=data.get("model", DEFAULT_MODEL_NAME),
                        environment_type=data.get("env_type", DEFAULT_ENV_TYPE),
                        env_id=data.get("env_id"),
                        num_examples=data.get("num_examples", 0),
                        status=data.get("status", "running"),
                        env_config=self._json_serializer(data.get("env_config")),
                    )
                    session.add(eval_session)

                session.commit()
        except Exception as e:
            logger.error(f"Error storing evaluation session: {e}")
            if session:
                session.rollback()
            raise StorageError(f"Failed to store evaluation session: {e}") from e

    def _store_evaluation_session_end(self, data: Dict[str, Any]) -> None:
        """Update evaluation session with completion data"""
        try:
            with get_session(self.engine) as session:
                latest_session = session.exec(
                    select(EvaluationSession)
                    .order_by(desc(EvaluationSession.started_at))
                    .limit(1)
                ).first()

                if latest_session:
                    latest_session.status = "completed"
                    latest_session.completed_at = datetime.utcnow()
                    latest_session.final_reward_mean = data.get("final_reward_mean")
                    latest_session.total_examples = data.get("total_examples")
                    session.commit()
                else:
                    logger.warning("No session found to update with completion data")
        except Exception as e:
            logger.error(f"Error updating session completion: {e}")
            if session:
                session.rollback()
            raise StorageError(f"Failed to update session completion: {e}") from e

    # =============================================================================
    # Query Methods
    # =============================================================================

    def get_rollouts_paginated(
        self, session_id: Optional[str] = None, page: int = 1, per_page: int = 50
    ) -> Tuple[List[RolloutResponse], int]:
        """Get paginated rollouts with optional session filtering"""
        try:
            with get_session(self.engine) as session:
                query = select(RolloutMetric)

                if session_id:
                    query = query.where(RolloutMetric.session_id == session_id)

                query = query.order_by(desc(RolloutMetric.timestamp))

                count_query = select(func.count(RolloutMetric.id))
                if session_id:
                    count_query = count_query.where(
                        RolloutMetric.session_id == session_id
                    )
                total_count = session.exec(count_query).first() or 0

                offset = (page - 1) * per_page
                query = query.offset(offset).limit(per_page)

                rollouts = session.exec(query).all()

                rollout_responses = [
                    RolloutResponse.from_db_model(rollout) for rollout in rollouts
                ]

                return rollout_responses, total_count

        except Exception as e:
            logger.error(f"Error getting paginated rollouts: {e}")
            return [], 0

    def get_sessions(self) -> List[SessionResponse]:
        """Get all evaluation sessions"""
        try:
            with get_session(self.engine) as session:
                sessions = session.exec(
                    select(EvaluationSession).order_by(
                        desc(EvaluationSession.started_at)
                    )
                ).all()

                return [SessionResponse.from_db_model(sess) for sess in sessions]

        except Exception as e:
            logger.error(f"Error getting sessions: {e}")
            return []

    def get_progress_for_session(self, session_id: str) -> Optional[ProgressResponse]:
        """Get latest progress for a specific session"""
        try:
            with get_session(self.engine) as session:
                progress = session.exec(
                    select(EvaluationProgress)
                    .where(EvaluationProgress.session_id == session_id)
                    .order_by(desc(EvaluationProgress.timestamp))
                    .limit(1)
                ).first()

                if progress:
                    return ProgressResponse.from_db_model(progress)
                return None

        except Exception as e:
            logger.error(f"Error getting progress for session {session_id}: {e}")
            return None

    def get_session_by_id(self, session_id: str) -> Optional[SessionResponse]:
        """Get a specific session by ID"""
        try:
            with get_session(self.engine) as session:
                sess = session.exec(
                    select(EvaluationSession).where(
                        EvaluationSession.session_id == session_id
                    )
                ).first()

                if sess:
                    return SessionResponse.from_db_model(sess)
                return None

        except Exception as e:
            logger.error(f"Error getting session {session_id}: {e}")
            return None
