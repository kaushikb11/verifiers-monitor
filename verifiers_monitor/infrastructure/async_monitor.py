"""
AsyncBatchGenerator pipeline monitoring for Verifiers training
"""

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class AsyncPipelineMetrics:
    """Async pipeline metrics snapshot"""

    timestamp: float
    queue_depth: int
    pending_batches: int
    completed_batches: int
    is_generating: bool
    avg_batch_time: float
    pipeline_utilization: float
    error_count: int


class AsyncPipelineMonitor:
    """Monitor Verifiers AsyncBatchGenerator pipeline"""

    def __init__(self, check_interval: float = 2.0):
        """
        Initialize async pipeline monitor

        Args:
            check_interval: How often to collect metrics (seconds)
        """
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)

        self._monitoring = False
        self._monitor_task = None
        self._metrics_history: List[AsyncPipelineMetrics] = []
        self._max_history = 200
        self._monitored_trainer = None

    def attach_to_trainer(self, trainer):
        """Attach monitor to a trainer with AsyncBatchGenerator"""
        self._monitored_trainer = trainer

        if hasattr(trainer, "async_generator"):
            self.logger.info("Attached to AsyncBatchGenerator")
            return True
        else:
            self.logger.warning("Trainer has no AsyncBatchGenerator")
            return False

    async def start_monitoring(self, storage_backend=None):
        """Start continuous async pipeline monitoring"""
        if self._monitoring or not self._monitored_trainer:
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(storage_backend))
        self.logger.info("Started async pipeline monitoring")

    async def stop_monitoring(self):
        """Stop async pipeline monitoring"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped async pipeline monitoring")

    async def _monitor_loop(self, storage_backend):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                metrics = await self.collect_metrics()

                if metrics:
                    self._metrics_history.append(metrics)
                    if len(self._metrics_history) > self._max_history:
                        self._metrics_history.pop(0)

                    if storage_backend:
                        metrics_dict = {
                            "timestamp": metrics.timestamp,
                            "async_queue_depth": metrics.queue_depth,
                            "async_pending_batches": metrics.pending_batches,
                            "async_completed_batches": metrics.completed_batches,
                            "async_is_generating": metrics.is_generating,
                            "async_avg_batch_time": metrics.avg_batch_time,
                            "async_pipeline_utilization": metrics.pipeline_utilization,
                            "async_error_count": metrics.error_count,
                        }

                        storage_backend.store("async_pipeline", metrics_dict)

                await asyncio.sleep(self.check_interval)

            except Exception as e:
                self.logger.error(f"Async pipeline monitoring error: {e}")
                await asyncio.sleep(self.check_interval * 2)

    async def collect_metrics(self) -> Optional[AsyncPipelineMetrics]:
        """Collect current async pipeline metrics"""
        if not self._monitored_trainer or not hasattr(
            self._monitored_trainer, "async_generator"
        ):
            return None

        try:
            async_gen = self._monitored_trainer.async_generator
            timestamp = time.time()

            queue_depth = 0
            pending_batches = 0
            completed_batches = 0
            is_generating = False
            avg_batch_time = 0.0
            error_count = 0

            if hasattr(async_gen, "request_queue"):
                try:
                    queue_depth = async_gen.request_queue.qsize()
                except:
                    pass

            if hasattr(async_gen, "pending_batches"):
                try:
                    pending_batches = len(async_gen.pending_batches)
                except:
                    pass

            if hasattr(async_gen, "completed_batches"):
                try:
                    completed_batches = len(async_gen.completed_batches)
                except:
                    pass

            if hasattr(async_gen, "is_generating"):
                try:
                    is_generating = bool(async_gen.is_generating)
                except:
                    pass

            if hasattr(async_gen, "generation_times"):
                try:
                    times = list(async_gen.generation_times)
                    if times:
                        avg_batch_time = sum(times) / len(times)
                except:
                    pass

            pipeline_utilization = 0.0
            if pending_batches > 0 or is_generating:
                # Pipeline is active
                max_concurrent = getattr(
                    self._monitored_trainer, "num_batches_ahead", 1
                )
                utilization = min(
                    1.0,
                    (pending_batches + (1 if is_generating else 0))
                    / max(1, max_concurrent),
                )
                pipeline_utilization = utilization * 100

            return AsyncPipelineMetrics(
                timestamp=timestamp,
                queue_depth=queue_depth,
                pending_batches=pending_batches,
                completed_batches=completed_batches,
                is_generating=is_generating,
                avg_batch_time=avg_batch_time,
                pipeline_utilization=pipeline_utilization,
                error_count=error_count,
            )

        except Exception as e:
            self.logger.error(f"Error collecting async pipeline metrics: {e}")
            return None

    def get_current_metrics(self) -> Optional[AsyncPipelineMetrics]:
        """Get most recent metrics"""
        return self._metrics_history[-1] if self._metrics_history else None

    def get_metrics_history(self, limit: int = 50) -> List[AsyncPipelineMetrics]:
        """Get recent metrics history"""
        return self._metrics_history[-limit:]

    def get_avg_batch_time(self, minutes: int = 5) -> float:
        """Get average batch time over last N minutes"""
        cutoff_time = time.time() - (minutes * 60)
        recent_metrics = [
            m
            for m in self._metrics_history
            if m.timestamp > cutoff_time and m.avg_batch_time > 0
        ]

        if not recent_metrics:
            return 0.0

        return sum(m.avg_batch_time for m in recent_metrics) / len(recent_metrics)

    def get_avg_utilization(self, minutes: int = 5) -> float:
        """Get average pipeline utilization over last N minutes"""
        cutoff_time = time.time() - (minutes * 60)
        recent_metrics = [m for m in self._metrics_history if m.timestamp > cutoff_time]

        if not recent_metrics:
            return 0.0

        return sum(m.pipeline_utilization for m in recent_metrics) / len(recent_metrics)

    def get_throughput(self, minutes: int = 5) -> float:
        """Get batch processing throughput (batches per minute)"""
        if len(self._metrics_history) < 2:
            return 0.0

        cutoff_time = time.time() - (minutes * 60)
        recent_metrics = [m for m in self._metrics_history if m.timestamp > cutoff_time]

        if len(recent_metrics) < 2:
            return 0.0

        first = recent_metrics[0]
        last = recent_metrics[-1]

        time_diff_minutes = (last.timestamp - first.timestamp) / 60
        batch_diff = last.completed_batches - first.completed_batches

        if time_diff_minutes <= 0:
            return 0.0

        return batch_diff / time_diff_minutes

    def is_pipeline_healthy(self) -> bool:
        """Check if async pipeline is healthy"""
        current = self.get_current_metrics()
        if not current:
            return False

        # Pipeline is unhealthy if:
        # 1. Queue depth is very high (>10)
        # 2. No progress in batch completion for a while
        # 3. High error count

        if current.queue_depth > 10:
            return False

        if current.error_count > 5:
            return False

        recent_throughput = self.get_throughput(2)
        if self._monitored_trainer and hasattr(self._monitored_trainer, "state"):
            if (
                hasattr(self._monitored_trainer.state, "global_step")
                and self._monitored_trainer.state.global_step > 10
            ):
                if recent_throughput == 0 and current.pending_batches == 0:
                    return False

        return True

    def get_pipeline_status(self) -> str:
        """Get human-readable pipeline status"""
        current = self.get_current_metrics()
        if not current:
            return "unknown"

        if not self.is_pipeline_healthy():
            return "unhealthy"

        if current.is_generating or current.pending_batches > 0:
            return "active"

        if current.completed_batches > 0:
            return "idle"

        return "starting"

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of async pipeline metrics"""
        current = self.get_current_metrics()
        if not current:
            return {"status": "no_data"}

        return {
            "timestamp": current.timestamp,
            "status": self.get_pipeline_status(),
            "is_healthy": self.is_pipeline_healthy(),
            "queue_depth": current.queue_depth,
            "pending_batches": current.pending_batches,
            "completed_batches": current.completed_batches,
            "is_generating": current.is_generating,
            "avg_batch_time": current.avg_batch_time,
            "avg_batch_time_5min": self.get_avg_batch_time(5),
            "pipeline_utilization": current.pipeline_utilization,
            "avg_utilization_5min": self.get_avg_utilization(5),
            "throughput_batches_per_min": self.get_throughput(5),
            "error_count": current.error_count,
        }
