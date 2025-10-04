"""
Data transformation and business logic service

This module contains data processing, transformation, and business logic
that was previously embedded in the monolithic server.
"""

from pathlib import Path
from typing import Any, Dict, Optional


class DataService:
    """Service for data transformation and business logic"""

    def __init__(self, storage_backend, env_info: Optional[Dict[str, Any]] = None):
        """
        Initialize data service

        Args:
            storage_backend: Storage backend instance
            env_info: Environment information dictionary
        """
        self.storage = storage_backend
        self.env_info = env_info or {}

    def get_dashboard_html(self) -> str:
        """
        Get dashboard HTML content from template

        Returns:
            str: HTML content for the dashboard
        """
        template_path = Path(__file__).parent.parent / "templates" / "dashboard.html"

        try:
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return """
                <!DOCTYPE html>
                <html>
                <head><title>Dashboard Not Found</title></head>
                <body>
                    <h1>Dashboard Template Not Found</h1>
                    <p>Expected location: {}</p>
                </body>
                </html>
                """.format(
                template_path
            )

    def process_metrics_for_display(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw metrics for dashboard display

        Args:
            metrics: Raw metrics from storage

        Returns:
            Dict: Processed metrics ready for display
        """
        processed = metrics.copy()

        # Add computed fields, formatting, etc.
        if "rollout" in processed:
            rollout_data = processed["rollout"]
            if isinstance(rollout_data, dict):
                # Add computed reward statistics
                reward = rollout_data.get("reward", 0)
                processed["rollout"]["reward_category"] = (
                    "high" if reward > 0.8 else "medium" if reward > 0.5 else "low"
                )

        return processed

    def calculate_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for a session

        Args:
            session_id: Session identifier

        Returns:
            Dict: Session statistics
        """
        try:
            # Get rollouts for this session
            if hasattr(self.storage, "get_rollouts_paginated"):
                rollouts, total_count = self.storage.get_rollouts_paginated(
                    session_id, page=1, per_page=10000
                )
                rollouts = [rollout.dict() for rollout in rollouts]
            else:
                rollouts = self.storage.get_filtered(
                    "rollout", {"session_id": session_id}, limit=10000
                )
                total_count = len(rollouts)

            if not rollouts:
                return {"error": "No data available for session"}

            # Calculate statistics
            rewards = [r.get("reward", 0) for r in rollouts]
            response_times = [
                r.get("rollout_time", 0) for r in rollouts if r.get("rollout_time")
            ]

            stats = {
                "total_rollouts": total_count,
                "unique_examples": len(
                    set(
                        r.get("example_number")
                        for r in rollouts
                        if r.get("example_number")
                    )
                ),
                "average_reward": sum(rewards) / len(rewards) if rewards else 0,
                "success_rate": sum(1 for r in rewards if r > 0) / len(rewards)
                if rewards
                else 0,
                "average_response_time": sum(response_times) / len(response_times)
                if response_times
                else 0,
                "reward_distribution": {
                    "min": min(rewards) if rewards else 0,
                    "max": max(rewards) if rewards else 0,
                    "std": self._calculate_std(rewards) if len(rewards) > 1 else 0,
                },
            }

            return stats

        except Exception as e:
            return {"error": str(e)}

    def _calculate_std(self, values):
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance**0.5

    def get_environment_summary(self) -> Dict[str, Any]:
        """
        Get a summary of environment configuration

        Returns:
            Dict: Environment summary
        """
        summary = {
            "environment_type": self.env_info.get("env_type", "Unknown"),
            "has_system_prompt": bool(self.env_info.get("system_prompt")),
            "num_reward_functions": len(self.env_info.get("reward_functions", [])),
            "parser_type": self.env_info.get("parser_type", "Unknown"),
            "dataset_size": self.env_info.get("dataset_size", 0),
            "has_tools": len(self.env_info.get("tools", [])) > 0,
        }

        return summary

    def validate_session_data(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean session data

        Args:
            session_data: Raw session data

        Returns:
            Dict: Validated and cleaned session data
        """
        validated = {}

        # Required fields with defaults
        validated["session_id"] = session_data.get("session_id", "unknown")
        validated["model_name"] = session_data.get("model_name", "unknown")
        validated["started_at"] = session_data.get("started_at", 0)
        validated["status"] = session_data.get("status", "unknown")

        # Optional fields
        if "num_examples" in session_data:
            validated["num_examples"] = max(0, int(session_data["num_examples"]))

        if "completed_at" in session_data:
            validated["completed_at"] = session_data["completed_at"]

        return validated
