"""
SQLModel schemas for verifiers monitoring data

This module defines the database models and API schemas using SQLModel,
demonstrating best practices for modern Python frameworks.
"""

import ast
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from sqlmodel import Field, Session, SQLModel, create_engine, select

from ..utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def _parse_json_field(value: Optional[str], field_name: str) -> Any:
    """Parse JSON/dict field with multiple fallback strategies"""
    if not value:
        return None

    # Try JSON first (handles double quotes)
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        pass

    # Fall back to ast.literal_eval (handles Python dict strings)
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        logger.warning(f"Failed to parse {field_name}, returning raw value")
        return value


# =============================================================================
# Database Models (SQLModel with table=True)
# =============================================================================


class EvaluationSession(SQLModel, table=True):  # type: ignore[call-arg]
    """Represents an evaluation session"""

    __tablename__ = "evaluation_sessions"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True, unique=True)
    model_name: str
    environment_type: str
    env_id: Optional[str] = (
        None  # Inferred environment name (e.g., "math-python", "gsm8k")
    )
    num_examples: int
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    status: str = Field(default="running")

    # Final results
    final_reward_mean: Optional[float] = None
    total_examples: Optional[int] = None

    # Environment configuration (JSON)
    env_config: Optional[str] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class RolloutMetric(SQLModel, table=True):  # type: ignore[call-arg]
    """Individual rollout metrics"""

    __tablename__ = "rollout_metrics"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key="evaluation_sessions.session_id", index=True)

    # Core metrics
    timestamp: float = Field(index=True)
    example_number: int
    rollout_number: int
    reward: float
    rollout_time: float
    model_name: Optional[str] = Field(default=None, index=True)

    # Content
    prompt: Optional[str] = None
    completion: Optional[str] = None
    answer: Optional[str] = None
    completion_length: int = 0

    # Verifiers-specific
    env_type: Optional[str] = Field(default=None, index=True)
    parser_type: Optional[str] = None
    task: Optional[str] = Field(default=None, index=True)
    prompt_hash: Optional[str] = None
    rollout_id: Optional[str] = None

    # Analysis (JSON strings)
    reward_breakdown: Optional[str] = None
    parsing_analysis: Optional[str] = None
    response_analysis: Optional[str] = None
    metrics: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)


class TrainingStep(SQLModel, table=True):  # type: ignore[call-arg]
    """Training step metrics"""

    __tablename__ = "training_steps"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)

    timestamp: float = Field(index=True)
    step: int
    loss: float
    compute_time: float

    # Advantage statistics
    advantages_mean: Optional[float] = None
    advantages_std: Optional[float] = None
    advantages_min: Optional[float] = None
    advantages_max: Optional[float] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)


class EvaluationProgress(SQLModel, table=True):  # type: ignore[call-arg]
    """Real-time evaluation progress tracking"""

    __tablename__ = "evaluation_progress"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key="evaluation_sessions.session_id", index=True)

    timestamp: float = Field(index=True)
    completed: int
    total: int
    percentage: float
    throughput: float
    eta_seconds: Optional[float] = None
    elapsed_time: Optional[float] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)


class ErrorEvent(SQLModel, table=True):  # type: ignore[call-arg]
    """Track errors and failures during evaluation/training"""

    __tablename__ = "error_events"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)

    # Error identification
    timestamp: float = Field(index=True)
    error_type: str = Field(index=True)
    error_message: str
    error_context: Optional[str] = None

    # Location tracking
    example_number: Optional[int] = None
    rollout_number: Optional[int] = None
    model_name: Optional[str] = None

    # Recovery info
    retry_count: int = 0
    recovered: bool = False

    created_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# API Response Models (SQLModel with table=False)
# =============================================================================


class RolloutResponse(SQLModel):
    """API response model for rollout data"""

    id: int
    session_id: str
    timestamp: float
    example_number: int
    rollout_number: int
    reward: float
    rollout_time: float
    model_name: Optional[str] = None
    prompt: Optional[str] = None
    completion: Optional[str] = None
    answer: Optional[str] = None
    completion_length: int
    env_type: Optional[str] = None
    prompt_hash: Optional[str] = None
    rollout_id: Optional[str] = None

    reward_breakdown: Optional[Dict[str, Any]] = None
    parsing_analysis: Optional[Dict[str, Any]] = None
    response_analysis: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    parser_type: Optional[str] = None
    task: Optional[str] = None

    @classmethod
    def from_db_model(cls, db_model: RolloutMetric) -> "RolloutResponse":
        """Convert database model to API response"""
        # Parse JSON fields using helper
        reward_breakdown = _parse_json_field(
            db_model.reward_breakdown, "reward_breakdown"
        )
        parsing_analysis = _parse_json_field(
            db_model.parsing_analysis, "parsing_analysis"
        )
        response_analysis = _parse_json_field(
            db_model.response_analysis, "response_analysis"
        )
        metrics = _parse_json_field(db_model.metrics, "metrics")

        # Parse prompt/completion and serialize back to JSON string if needed
        def _serialize_messages(value: Optional[str], field_name: str) -> Optional[str]:
            if not value:
                return None
            parsed = _parse_json_field(value, field_name)
            if isinstance(parsed, list):
                return json.dumps(parsed)
            return parsed if isinstance(parsed, str) else str(parsed)

        prompt = _serialize_messages(db_model.prompt, "prompt")
        completion = _serialize_messages(db_model.completion, "completion")

        return cls(
            id=db_model.id,
            session_id=db_model.session_id,
            timestamp=db_model.timestamp,
            example_number=db_model.example_number,
            rollout_number=db_model.rollout_number,
            reward=db_model.reward,
            rollout_time=db_model.rollout_time,
            model_name=db_model.model_name,
            prompt=prompt,
            completion=completion,
            answer=db_model.answer,
            completion_length=db_model.completion_length,
            env_type=db_model.env_type,
            prompt_hash=db_model.prompt_hash,
            rollout_id=db_model.rollout_id,
            parser_type=db_model.parser_type,
            task=db_model.task,
            reward_breakdown=reward_breakdown,
            parsing_analysis=parsing_analysis,
            response_analysis=response_analysis,
            metrics=metrics,
        )

    @property
    def prompt_messages(self) -> List[Dict[str, str]]:
        """Get prompt as parsed message list"""
        if not self.prompt:
            return []
        parsed = _parse_json_field(self.prompt, "prompt")
        return parsed if isinstance(parsed, list) else []

    @property
    def completion_messages(self) -> List[Dict[str, str]]:
        """Get completion as parsed message list"""
        if not self.completion:
            return []
        parsed = _parse_json_field(self.completion, "completion")
        return parsed if isinstance(parsed, list) else []

    @property
    def has_tool_calls(self) -> bool:
        """Check if completion includes tool calls"""
        return any(m.get("role") == "tool" for m in self.completion_messages)


class SessionResponse(SQLModel):
    """API response model for session data"""

    session_id: str
    model_name: str
    environment_type: str
    env_id: Optional[str] = (
        None  # Inferred environment name (e.g., "math-python", "gsm8k")
    )
    num_examples: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str

    @classmethod
    def from_db_model(cls, db_model: EvaluationSession) -> "SessionResponse":
        """Convert database model to API response"""
        return cls(
            session_id=db_model.session_id,
            model_name=db_model.model_name,
            environment_type=db_model.environment_type,
            env_id=db_model.env_id,
            num_examples=db_model.num_examples,
            started_at=db_model.started_at,
            completed_at=db_model.completed_at,
            status=db_model.status,
        )


class ProgressResponse(SQLModel):
    """API response model for progress data"""

    session_id: str
    completed: int
    total: int
    percentage: float
    throughput: float
    eta_seconds: Optional[float] = None
    elapsed_time: Optional[float] = None
    timestamp: float

    @classmethod
    def from_db_model(cls, db_model: EvaluationProgress) -> "ProgressResponse":
        """Convert database model to API response"""
        return cls(
            session_id=db_model.session_id,
            completed=db_model.completed,
            total=db_model.total,
            percentage=db_model.percentage,
            throughput=db_model.throughput,
            eta_seconds=db_model.eta_seconds,
            elapsed_time=db_model.elapsed_time,
            timestamp=db_model.timestamp,
        )


class PaginatedResponse(BaseModel):
    """Generic paginated response wrapper"""

    data: List[Any]
    total: int
    page: int
    per_page: int
    has_next: bool
    has_prev: bool


@dataclass
class ExampleSummary:
    """Summary statistics for a single example across multiple rollouts"""

    example_number: int
    num_rollouts: int
    mean_reward: float
    max_reward: float
    min_reward: float
    std_reward: float
    rollouts: List[RolloutResponse]

    def is_failure(self, threshold: float = 0.5) -> bool:
        """Check if example failed (max reward below threshold)"""
        return self.max_reward < threshold

    def is_unstable(self, threshold: float = 0.3) -> bool:
        """Check if example has high variance across rollouts"""
        return self.std_reward > threshold

    def get_best_rollout(self) -> Optional[RolloutResponse]:
        """Get the rollout with highest reward"""
        if not self.rollouts:
            return None
        return max(self.rollouts, key=lambda r: r.reward)

    def get_worst_rollout(self) -> Optional[RolloutResponse]:
        """Get the rollout with lowest reward"""
        if not self.rollouts:
            return None
        return min(self.rollouts, key=lambda r: r.reward)


# =============================================================================
# Database Helper Functions
# =============================================================================


def create_database_engine(database_url: str = "sqlite:///vmf_metrics.db"):
    """Create SQLModel database engine with proper configuration"""
    engine = create_engine(
        database_url,
        echo=False,
        connect_args={"check_same_thread": False} if "sqlite" in database_url else {},
    )
    return engine


def create_tables(engine):
    """Create all database tables"""
    SQLModel.metadata.create_all(engine)


def get_session(engine):
    """Get database session (context manager)"""
    return Session(engine)
