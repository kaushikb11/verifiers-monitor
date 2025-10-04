"""
Metrics collection system for RL-specific training and evaluation metrics
"""

import time
from collections import defaultdict
from typing import Any, Dict, Optional

from ..utils.logging import get_logger
from ..utils.validation import (
    validate_model_name,
    validate_positive_int,
    validate_session_id,
)

logger = get_logger(__name__)


class MetricsCollector:
    """Collects RL-specific metrics during training and evaluation"""

    def __init__(self, storage_backend):
        self.storage = storage_backend
        self.session_data = {
            "training": defaultdict(list),
            "evaluation": defaultdict(list),
        }
        self._evaluation_count = 0

    def collect_training_step(self, metrics: Dict[str, Any]):
        """
        Collect metrics from a training step

        Args:
            metrics: Dictionary containing step metrics
                - step: Training step number
                - loss: Training loss value
                - compute_time: Time taken for loss computation
                - advantages: Dictionary with advantage statistics (optional)
                - timestamp: Unix timestamp
        """

        # Core training metrics
        step_metrics = {
            "timestamp": metrics.get("timestamp", time.time()),
            "step": metrics.get("step", 0),
            "loss": metrics.get("loss", 0.0),
            "compute_time": metrics.get("compute_time", 0.0),
            "session_id": getattr(self, "_session_id", "default_training_session"),
        }

        if metrics.get("advantages"):
            adv = metrics["advantages"]
            step_metrics.update(
                {
                    "advantages_mean": adv.get("mean", 0.0),
                    "advantages_std": adv.get("std", 0.0),
                    "advantages_min": adv.get("min", 0.0),
                    "advantages_max": adv.get("max", 0.0),
                }
            )

        self.storage.store("training_step", step_metrics)
        self.session_data["training"]["steps"].append(step_metrics)

        step_num = step_metrics["step"]
        if step_num > 0 and step_num % 10 == 0:
            loss_val = step_metrics["loss"]
            logger.info(f"Step {step_num}, Loss: {loss_val:.4f}")

    def collect_logged_metrics(self, logs: Dict[str, Any], step: int):
        """
        Collect metrics from trainer.log() calls

        Args:
            logs: Dictionary of logged metrics from trainer
            step: Current training step
        """

        logged_metrics = {
            "timestamp": time.time(),
            "step": step,
        }

        for key, value in logs.items():
            if isinstance(value, (int, float)) and not key.startswith("_"):
                # Clean up metric names
                clean_key = key.replace("train_", "").replace("eval_", "")
                logged_metrics[clean_key] = float(value)

        if len(logged_metrics) > 2:
            self.storage.store("logged_metrics", logged_metrics)

    def start_evaluation_session(
        self,
        model: str,
        num_examples: int,
        rollouts_per_example: int = 1,
        env_type: Optional[str] = None,
        env_id: Optional[str] = None,
    ):
        """
        Start evaluation monitoring session

        Args:
            model: Model name being evaluated
            num_examples: Number of examples to evaluate (-1 for all)
            rollouts_per_example: Number of rollouts per example
            env_type: Environment type/class name (e.g., "ToolEnv", "SingleTurnEnv")
            env_id: Inferred environment name (e.g., "math-python", "gsm8k")
        """

        validated_model = validate_model_name(model)
        validated_num_examples = validate_positive_int(num_examples, "num_examples")
        validated_rollouts = validate_positive_int(
            rollouts_per_example, "rollouts_per_example"
        )

        import uuid
        from datetime import datetime

        self._session_id = f"{datetime.now():%Y%m%d_%H%M%S}_{str(uuid.uuid4())[:8]}"
        self._completed_examples: set[int] = set()
        self._rollouts_per_example = validated_rollouts
        self._model_name = validated_model

        # Verifiers uses round-robin pattern, not sequential
        self._prompt_hash_map: Dict[str, int] = {}  # prompt_hash -> example_number
        self._example_rollout_counts: Dict[int, int] = (
            {}
        )  # example_number -> rollout_count
        self._next_example_number = 1

        session_info = {
            "timestamp": time.time(),
            "model": validated_model,
            "num_examples": validated_num_examples,
            "env_type": env_type or "unknown",
            "env_id": env_id,  # Inferred environment name
            "status": "started",
            "start_time": time.time(),
            "session_id": self._session_id,
            "progress": {
                "completed": 0,
                "total": validated_num_examples,
                "percentage": 0.0,
                "eta_seconds": None,
                "throughput": 0.0,
                "session_id": self._session_id,
            },
        }

        self.storage.store("evaluation_session", session_info)
        self.storage.store("evaluation_progress", session_info["progress"])
        self.session_data["evaluation"]["session_info"] = session_info
        self._evaluation_count = 0
        self._session_start_time = time.time()

        examples_text = f"{num_examples}" if num_examples > 0 else "all"
        logger.info(
            f"Evaluation started - {examples_text} examples (session: {self._session_id})"
        )

    def _get_or_assign_example_number(
        self, rollout_data: Dict[str, Any]
    ) -> tuple[int, int]:
        """
        Get or assign example_number and rollout_number based on prompt_hash.

        Verifiers uses round-robin pattern for multiple rollouts per example:
        [Ex1, Ex2, Ex3, Ex1, Ex2, Ex3, Ex1, Ex2, Ex3]

        We track by prompt_hash to correctly group rollouts of the same example.

        Returns:
            (example_number, rollout_number) tuple
        """
        import hashlib

        prompt_hash = rollout_data.get("prompt_hash")

        if not prompt_hash or prompt_hash == "" or prompt_hash == "0":
            if "answer" in rollout_data and rollout_data["answer"]:
                prompt_hash = hashlib.sha256(
                    str(rollout_data["answer"]).encode()
                ).hexdigest()[:16]
            elif "prompt" in rollout_data and rollout_data["prompt"]:
                prompt_hash = hashlib.sha256(
                    str(rollout_data["prompt"]).encode()
                ).hexdigest()[:16]
            else:
                # Last resort: use evaluation count (falls back to old behavior)
                prompt_hash = f"fallback_{self._evaluation_count}"

        prompt_hash = str(prompt_hash)

        # Look up or create example number
        if prompt_hash not in self._prompt_hash_map:
            # New example
            example_number = self._next_example_number
            self._prompt_hash_map[prompt_hash] = example_number
            self._example_rollout_counts[example_number] = 1
            rollout_number = 1
            self._next_example_number += 1

            self._completed_examples.add(example_number)
        else:
            # Existing example - another rollout
            example_number = self._prompt_hash_map[prompt_hash]
            self._example_rollout_counts[example_number] += 1
            rollout_number = self._example_rollout_counts[example_number]

        return example_number, rollout_number

    def collect_rollout(self, rollout_data: Dict[str, Any]):
        """
        Collect metrics from individual rollout

        Args:
            rollout_data: Dictionary containing rollout information
                - prompt: Input prompt
                - completion: Model completion
                - reward: Reward score
                - rollout_time: Time taken for rollout
                - timestamp: Unix timestamp
        """

        self._evaluation_count += 1

        # This handles verifiers' round-robin pattern correctly
        example_number, rollout_number = self._get_or_assign_example_number(
            rollout_data
        )

        rollout_metrics = {
            "timestamp": rollout_data.get("timestamp", time.time()),
            "reward": rollout_data.get("reward", 0.0),
            "rollout_time": rollout_data.get("rollout_time", 0.0),
            "completion_length": rollout_data.get(
                "completion_length", len(str(rollout_data.get("completion", "")))
            ),
            "example_number": example_number,
            "rollout_number": rollout_number,
            "session_id": getattr(self, "_session_id", "unknown"),
            "model_name": getattr(self, "_model_name", "unknown"),
        }

        # Include enhanced verifiers-specific data if available
        enhanced_fields = [
            "prompt",
            "completion",
            "answer",
            "prompt_hash",
            "rollout_id",
            "env_type",
            "parser_type",
            "task",
            "metrics",
            "reward_breakdown",
            "parsing_analysis",
            "response_analysis",
        ]
        for field in enhanced_fields:
            if field in rollout_data:
                rollout_metrics[field] = rollout_data[field]

        self.storage.store("rollout", rollout_metrics)
        self.session_data["evaluation"]["rollouts"].append(rollout_metrics)

        self._update_progress()

        if self._evaluation_count % 10 == 0:
            reward = rollout_metrics["reward"]
            logger.info(
                f"Evaluated {self._evaluation_count} examples, latest reward: {reward:.3f}"
            )

    def end_evaluation_session(self, results):
        """
        End evaluation session with final results

        Args:
            results: Final evaluation results from environment
        """

        final_reward_mean = 0.0
        total_examples = self._evaluation_count

        if hasattr(results, "reward") and len(results.reward) > 0:
            if hasattr(results.reward, "mean"):
                final_reward_mean = float(results.reward.mean())
            else:
                final_reward_mean = sum(results.reward) / len(results.reward)
            total_examples = len(results.reward)

        session_end = {
            "timestamp": time.time(),
            "status": "completed",
            "final_reward_mean": final_reward_mean,
            "total_examples": total_examples,
        }

        self.storage.store("evaluation_session_end", session_end)

        logger.info(
            f"Evaluation completed - {total_examples} examples, "
            f"avg reward: {final_reward_mean:.3f}"
        )

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current monitoring session"""

        summary = {
            "training_steps": len(self.session_data["training"]["steps"]),
            "evaluation_rollouts": len(self.session_data["evaluation"]["rollouts"]),
            "latest_metrics": self.storage.get_latest(),
        }

        return summary

    def _update_progress(self):
        """Update real-time progress tracking"""
        if not hasattr(self, "_session_start_time"):
            return

        current_time = time.time()
        elapsed_time = current_time - self._session_start_time

        session_info = self.session_data["evaluation"].get("session_info", {})
        total_examples = session_info.get("num_examples", 0)

        if total_examples <= 0:
            return

        completed_examples = (
            len(self._completed_examples) if hasattr(self, "_completed_examples") else 0
        )
        percentage = (
            (completed_examples / total_examples) * 100 if total_examples > 0 else 0
        )

        throughput = completed_examples / elapsed_time if elapsed_time > 0 else 0
        eta_seconds = (
            (total_examples - completed_examples) / throughput
            if throughput > 0
            else None
        )

        progress_data = {
            "timestamp": current_time,
            "completed": completed_examples,
            "total": total_examples,
            "percentage": percentage,
            "eta_seconds": eta_seconds,
            "throughput": throughput,
            "elapsed_time": elapsed_time,
            "session_id": getattr(self, "_session_id", "unknown"),
        }

        self.storage.store("evaluation_progress", progress_data)

    def collect_error(
        self,
        error_type: str,
        error_message: str,
        error_context: Optional[dict] = None,
        example_number: Optional[int] = None,
        rollout_number: Optional[int] = None,
        retry_count: int = 0,
        recovered: bool = False,
    ):
        """
        Collect error events during evaluation or training

        Args:
            error_type: Type of error (parsing_error, api_timeout, rollout_failure, etc.)
            error_message: Error message
            error_context: Additional context as dictionary
            example_number: Example number where error occurred
            rollout_number: Rollout number where error occurred
            retry_count: Number of retry attempts
            recovered: Whether the error was recovered from
        """
        import json

        error_data = {
            "timestamp": time.time(),
            "session_id": getattr(self, "_session_id", "unknown"),
            "error_type": error_type,
            "error_message": str(error_message),
            "error_context": json.dumps(error_context) if error_context else None,
            "example_number": example_number,
            "rollout_number": rollout_number or self._evaluation_count,
            "model_name": getattr(self, "_model_name", "unknown"),
            "retry_count": retry_count,
            "recovered": recovered,
        }

        self.storage.store("error_event", error_data)

        logger.error(
            f"Error captured: {error_type} - {error_message} "
            f"(example={example_number}, rollout={rollout_number})"
        )
