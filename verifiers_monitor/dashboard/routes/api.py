"""
API routes for the dashboard server

This module contains all REST API endpoints for metrics, evaluation data,
and configuration information.
"""

import json
import sqlite3
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from ...core.exceptions import (
    DashboardAPIError,
    StorageConnectionError,
    StorageError,
    StorageTimeoutError,
    ValidationError,
)
from ...utils.logging import get_logger
from ...utils.recovery import RetryConfig, retry_on_failure

logger = get_logger(__name__)


def create_api_routes(storage, get_env_info_func):
    """
    Create API routes with dependency injection

    Args:
        storage: Storage backend instance
        get_env_info_func: Function that returns current environment information

    Returns:
        APIRouter: Configured router with all API endpoints
    """
    router = APIRouter()

    # NOTE: The "/" route is handled in server.py _setup_modular_routes()
    # to properly support dashboard versioning (V1 vs V2)

    @router.get("/api/metrics")
    @retry_on_failure(RetryConfig(max_attempts=2))
    async def get_current_metrics():
        """Get current/latest metrics"""
        try:
            return storage.get_latest()
        except (StorageConnectionError, StorageTimeoutError) as e:
            logger.error(f"Storage error getting metrics: {e}")
            raise HTTPException(
                status_code=503, detail="Storage temporarily unavailable"
            )
        except StorageError as e:
            logger.error(f"Storage error getting metrics: {e}")
            raise HTTPException(status_code=500, detail="Storage error")
        except Exception as e:
            logger.error(f"Unexpected error getting metrics: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")

    @router.get("/api/training_history")
    @retry_on_failure(RetryConfig(max_attempts=2))
    async def get_training_history(limit: int = 1000):
        """Get training step history"""
        try:
            if limit <= 0 or limit > 10000:
                raise ValidationError("Limit must be between 1 and 10000")

            return storage.get_recent("training_step", limit)
        except ValidationError as e:
            logger.warning(f"Invalid training history request: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except (StorageConnectionError, StorageTimeoutError) as e:
            logger.error(f"Storage error getting training history: {e}")
            raise HTTPException(
                status_code=503, detail="Storage temporarily unavailable"
            )
        except StorageError as e:
            logger.error(f"Storage error getting training history: {e}")
            raise HTTPException(status_code=500, detail="Storage error")
        except Exception as e:
            logger.error(
                f"Unexpected error getting training history: {e}", exc_info=True
            )
            raise HTTPException(status_code=500, detail="Internal server error")

    @router.get("/api/evaluation_history")
    async def get_evaluation_history(
        limit: int = 100, offset: int = 0, session_id: str = "current"
    ):
        """Get evaluation rollout history with pagination and session filtering"""
        try:
            # Determine which session to query
            session_id_to_use: Optional[str]
            if session_id == "current":
                sessions = storage.get_sessions()
                if sessions:
                    session_id_to_use = sessions[0].session_id
                else:
                    session_id_to_use = None
            else:
                session_id_to_use = session_id

            rollouts, total_count = storage.get_rollouts_paginated(
                session_id=session_id_to_use,
                page=(offset // limit) + 1,
                per_page=limit,
            )

            rollout_dicts = []
            for rollout in rollouts:
                rollout_dict = rollout.dict(exclude_none=False)

                json_fields = [
                    "prompt",
                    "completion",
                    "metrics",
                    "reward_breakdown",
                    "parsing_analysis",
                    "response_analysis",
                ]
                for field in json_fields:
                    field_value = rollout_dict.get(field)
                    if field_value and isinstance(field_value, str):
                        try:
                            rollout_dict[field] = json.loads(field_value)
                        except (json.JSONDecodeError, TypeError):
                            try:
                                import ast

                                rollout_dict[field] = ast.literal_eval(field_value)
                            except (ValueError, SyntaxError):
                                pass

                rollout_dicts.append(rollout_dict)

            return {
                "data": rollout_dicts,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": total_count,
                    "has_more": offset + limit < total_count,
                },
            }
        except Exception as e:
            return {"error": str(e)}

    @router.get("/api/environment_info")
    async def get_environment_info():
        """Get environment configuration and details"""
        return get_env_info_func()

    @router.get("/api/progress")
    async def get_progress():
        """Get real-time evaluation progress"""
        try:
            progress_data = storage.get_recent("evaluation_progress", 1)
            if progress_data:
                return progress_data[0]
            else:
                return {"error": "No progress data available"}
        except Exception as e:
            return {"error": str(e)}

    @router.get("/api/sessions")
    async def get_sessions():
        """Get all evaluation sessions"""
        try:
            if hasattr(storage, "get_sessions"):
                # SQLModel storage
                sessions = storage.get_sessions()
                return [session.dict() for session in sessions]
            else:
                # Legacy storage - get unique sessions from rollout data
                rollouts = storage.get_recent("rollout", 10000)
                sessions = {}

                for rollout in rollouts:
                    session_id = rollout.get("session_id")
                    if session_id and session_id not in sessions:
                        sessions[session_id] = {
                            "session_id": session_id,
                            "timestamp": rollout.get("timestamp", 0),
                            "model": "unknown",
                            "num_examples": 0,
                        }

                return list(sessions.values())
        except Exception as e:
            return {"error": str(e)}

    @router.get("/api/reward_breakdown")
    async def get_reward_breakdown(session_id: str = "current"):
        """Get reward function breakdown analysis"""
        try:
            if session_id == "current":
                rollouts = storage.get_recent("rollout", 1000)
            else:
                rollout_responses, _ = storage.get_rollouts_paginated(
                    session_id, page=1, per_page=1000
                )
                rollouts = [rollout.dict() for rollout in rollout_responses]

            if not rollouts:
                return {"error": "No rollout data available"}

            # Analyze reward breakdown
            breakdown: Dict[str, Any] = {
                "total_rollouts": len(rollouts),
                "reward_functions": {},
                "average_rewards": {},
                "reward_distribution": {},
            }

            for rollout in rollouts:
                reward_breakdown = rollout.get("reward_breakdown")
                if reward_breakdown and isinstance(reward_breakdown, dict):
                    reward_functions = reward_breakdown.get("reward_functions", [])
                    contributions = reward_breakdown.get("contributions", {})

                    for func_name in reward_functions:
                        if func_name not in breakdown["reward_functions"]:
                            breakdown["reward_functions"][func_name] = {
                                "count": 0,
                                "total_contribution": 0.0,
                                "scores": [],
                            }

                        breakdown["reward_functions"][func_name]["count"] += 1

                        if func_name in contributions:
                            contrib = contributions[func_name].get("contribution", 0)
                            score = contributions[func_name].get("score", 0)
                            breakdown["reward_functions"][func_name][
                                "total_contribution"
                            ] += contrib
                            breakdown["reward_functions"][func_name]["scores"].append(
                                score
                            )

            for func_name, data in breakdown["reward_functions"].items():
                if data["count"] > 0:
                    breakdown["average_rewards"][func_name] = (
                        data["total_contribution"] / data["count"]
                    )

            return breakdown
        except Exception as e:
            return {"error": str(e)}

    @router.get("/api/multi_rollout_analysis")
    async def get_multi_rollout_analysis(session_id: str = "current"):
        """Get multi-rollout consistency analysis"""
        try:
            session_id_to_use: Optional[str]
            if session_id == "current":
                sessions = storage.get_sessions()
                if sessions:
                    session_id_to_use = sessions[0].session_id
                else:
                    session_id_to_use = None
            else:
                session_id_to_use = session_id

            rollout_responses, _ = storage.get_rollouts_paginated(
                session_id_to_use, page=1, per_page=1000
            )
            rollouts = [rollout.dict() for rollout in rollout_responses]

            if not rollouts:
                return {"error": "No rollout data available"}

            # Group rollouts by prompt_hash
            prompt_groups: Dict[str, List[Any]] = {}
            for rollout in rollouts:
                prompt_hash = rollout.get("prompt_hash")
                if prompt_hash is None or prompt_hash == "":
                    prompt_hash = f"unknown_{rollout.get('example_number', 0)}"

                if prompt_hash not in prompt_groups:
                    prompt_groups[prompt_hash] = []
                prompt_groups[prompt_hash].append(rollout)

            # Analyze consistency for groups with multiple rollouts
            consistency_stats: Dict[str, int] = {
                "high_variance": 0,
                "medium_variance": 0,
                "low_variance": 0,
            }
            prompt_analysis: List[Dict[str, Any]] = []

            analysis: Dict[str, Any] = {
                "total_prompts": len(prompt_groups),
                "multi_rollout_prompts": 0,
                "consistency_stats": consistency_stats,
                "prompt_analysis": prompt_analysis,
                "best_of_n_improvement": 0.0,
            }

            total_first_rewards = []
            total_best_rewards = []

            for prompt_hash, rollout_group in prompt_groups.items():
                if len(rollout_group) > 1:
                    analysis["multi_rollout_prompts"] += 1

                    rewards = [r.get("reward", 0) for r in rollout_group]
                    avg_reward = sum(rewards) / len(rewards)
                    variance = sum((r - avg_reward) ** 2 for r in rewards) / len(
                        rewards
                    )
                    std_dev = variance**0.5

                    # Categorize variance
                    if std_dev > 0.3:
                        consistency_stats["high_variance"] += 1
                    elif std_dev > 0.1:
                        consistency_stats["medium_variance"] += 1
                    else:
                        consistency_stats["low_variance"] += 1

                    # Best-of-N analysis
                    first_reward = rewards[0]
                    best_reward = max(rewards)
                    total_first_rewards.append(first_reward)
                    total_best_rewards.append(best_reward)

                    prompt_analysis.append(
                        {
                            "prompt_hash": prompt_hash,
                            "rollout_count": len(rollout_group),
                            "rewards": rewards,
                            "avg_reward": avg_reward,
                            "std_dev": std_dev,
                            "min_reward": min(rewards),
                            "max_reward": max(rewards),
                            "answer": rollout_group[0].get("answer", ""),
                        }
                    )

            if total_first_rewards and total_best_rewards:
                avg_first = sum(total_first_rewards) / len(total_first_rewards)
                avg_best = sum(total_best_rewards) / len(total_best_rewards)
                analysis["best_of_n_improvement"] = avg_best - avg_first

            return analysis

        except Exception as e:
            return {"error": str(e)}

    @router.get("/api/status")
    async def get_status():
        """Get dashboard and storage status"""
        try:
            status = {
                "dashboard": "running",
                "storage_type": storage.__class__.__name__,
                "timestamp": __import__("time").time(),
            }

            if hasattr(storage, "get_stats"):
                status["storage_stats"] = storage.get_stats()

            return status
        except Exception as e:
            return {"error": str(e)}

    @router.get("/api/environment/info")
    async def get_environment_info():
        """Get comprehensive environment configuration and metadata"""
        try:
            env_info = get_env_info_func()
            if not env_info:
                return {"error": "No environment information available"}

            response = {
                **env_info,
                "timestamp": __import__("time").time(),
            }

            return response
        except Exception as e:
            logger.error(f"Error fetching environment info: {e}")
            return {"error": str(e)}

    @router.get("/api/reward-system/info")
    async def get_reward_system_info():
        """Get detailed reward system configuration"""
        try:
            env_info = get_env_info_func()
            if not env_info:
                return {"error": "No environment information available"}

            reward_info = {
                "rubric_type": env_info.get("rubric_type", "unknown"),
                "reward_functions": env_info.get("reward_functions", []),
                "reward_weights": env_info.get("reward_weights", []),
                "reward_function_details": env_info.get("reward_function_details", {}),
                "max_possible_reward": sum(env_info.get("reward_weights", []))
                if env_info.get("reward_weights")
                else None,
            }

            return reward_info
        except Exception as e:
            logger.error(f"Error fetching reward system info: {e}")
            return {"error": str(e)}

    @router.get("/api/reward-system/statistics")
    async def get_reward_system_statistics(session_id: Optional[str] = None):
        """Get per-function performance statistics"""
        try:
            rollouts = storage.get_recent(metric_type="rollout", limit=10000)

            if session_id and session_id != "current":
                rollouts = [r for r in rollouts if r.get("session_id") == session_id]

            if not rollouts:
                return {"error": "No rollout data available"}

            env_info = get_env_info_func()
            reward_functions = env_info.get("reward_functions", [])

            stats = {}
            for func_name in reward_functions:
                func_scores = []
                for rollout in rollouts:
                    metrics = rollout.get("metrics", {})
                    if func_name in metrics:
                        func_scores.append(metrics[func_name])

                if func_scores:
                    stats[func_name] = {
                        "pass_rate": sum(1 for s in func_scores if s > 0.5)
                        / len(func_scores),
                        "avg_score": sum(func_scores) / len(func_scores),
                        "total_evaluations": len(func_scores),
                    }

            return {
                "statistics": stats,
                "total_rollouts": len(rollouts),
            }
        except Exception as e:
            logger.error(f"Error calculating reward statistics: {e}")
            return {"error": str(e)}

    return router
