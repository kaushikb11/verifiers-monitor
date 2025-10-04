"""
vLLM server monitoring for Verifiers training infrastructure
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp
import psutil


@dataclass
class VLLMHealth:
    """vLLM server health status"""

    is_alive: bool
    response_time: float
    queue_size: int
    model_name: str
    gpu_memory_used: float
    error_message: Optional[str] = None


class VLLMMonitor:
    """Monitor vLLM server health and performance"""

    def __init__(
        self, host: str = "localhost", port: int = 8000, check_interval: float = 5.0
    ):
        """
        Initialize vLLM monitor

        Args:
            host: vLLM server host
            port: vLLM server port
            check_interval: How often to check health (seconds)
        """
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)

        self._monitoring = False
        self._monitor_task = None
        self._health_history: List[VLLMHealth] = []
        self._max_history = 100

    async def start_monitoring(self, storage_backend=None):
        """Start continuous vLLM monitoring"""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(storage_backend))
        self.logger.info(f"Started vLLM monitoring ({self.base_url})")

    async def stop_monitoring(self):
        """Stop vLLM monitoring"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped vLLM monitoring")

    async def _monitor_loop(self, storage_backend):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                health = await self.check_health()

                # Store health data
                self._health_history.append(health)
                if len(self._health_history) > self._max_history:
                    self._health_history.pop(0)

                # Store in backend if available
                if storage_backend:
                    metrics = {
                        "timestamp": time.time(),
                        "vllm_alive": health.is_alive,
                        "vllm_response_time": health.response_time,
                        "vllm_queue_size": health.queue_size,
                        "vllm_gpu_memory": health.gpu_memory_used,
                        "vllm_model": health.model_name,
                    }

                    if health.error_message:
                        metrics["vllm_error"] = health.error_message

                    storage_backend.store("vllm_health", metrics)

                await asyncio.sleep(self.check_interval)

            except Exception as e:
                self.logger.error(f"vLLM monitoring error: {e}")
                await asyncio.sleep(self.check_interval * 2)

    async def check_health(self) -> VLLMHealth:
        """Check vLLM server health"""
        start_time = time.time()

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session:
                # Check health endpoint
                health_url = f"{self.base_url}/health"
                async with session.get(health_url) as response:
                    response_time = time.time() - start_time

                    if response.status == 200:
                        # Try to get model info
                        model_name = await self._get_model_name(session)
                        queue_size = await self._get_queue_size(session)
                        gpu_memory = await self._get_gpu_memory()

                        return VLLMHealth(
                            is_alive=True,
                            response_time=response_time,
                            queue_size=queue_size,
                            model_name=model_name,
                            gpu_memory_used=gpu_memory,
                        )
                    else:
                        return VLLMHealth(
                            is_alive=False,
                            response_time=response_time,
                            queue_size=0,
                            model_name="unknown",
                            gpu_memory_used=0.0,
                            error_message=f"HTTP {response.status}",
                        )

        except Exception as e:
            return VLLMHealth(
                is_alive=False,
                response_time=time.time() - start_time,
                queue_size=0,
                model_name="unknown",
                gpu_memory_used=0.0,
                error_message=str(e),
            )

    async def _get_model_name(self, session) -> str:
        """Get model name from vLLM server"""
        try:
            models_url = f"{self.base_url}/v1/models"
            async with session.get(models_url) as response:
                if response.status == 200:
                    data = await response.json()
                    if "data" in data and len(data["data"]) > 0:
                        return data["data"][0].get("id", "unknown")
        except:
            pass
        return "unknown"

    async def _get_queue_size(self, session) -> int:
        """Get current queue size (if available)"""
        try:
            # Try to get queue info from metrics endpoint
            metrics_url = f"{self.base_url}/metrics"
            async with session.get(metrics_url) as response:
                if response.status == 200:
                    text = await response.text()
                    # Parse prometheus metrics for queue size
                    for line in text.split("\n"):
                        if "vllm_request_queue_size" in line and not line.startswith(
                            "#"
                        ):
                            return int(float(line.split()[-1]))
        except:
            pass
        return 0

    async def _get_gpu_memory(self) -> float:
        """Get GPU memory usage"""
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryUtil * 100
        except ImportError:
            # Fallback to nvidia-ml-py if available
            try:
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return (info.used / info.total) * 100
            except:
                pass
        except:
            pass
        return 0.0

    def get_current_health(self) -> Optional[VLLMHealth]:
        """Get most recent health check"""
        return self._health_history[-1] if self._health_history else None

    def get_health_history(self, limit: int = 50) -> List[VLLMHealth]:
        """Get recent health history"""
        return self._health_history[-limit:]

    def get_avg_response_time(self, minutes: int = 5) -> float:
        """Get average response time over last N minutes"""
        cutoff_time = time.time() - (minutes * 60)
        recent_checks = [
            h
            for h in self._health_history
            if h.is_alive and time.time() - cutoff_time < minutes * 60
        ]

        if not recent_checks:
            return 0.0

        return sum(h.response_time for h in recent_checks) / len(recent_checks)

    def is_healthy(self) -> bool:
        """Check if vLLM server is currently healthy"""
        current = self.get_current_health()
        return current is not None and current.is_alive

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of vLLM metrics"""
        current = self.get_current_health()
        if not current:
            return {"status": "unknown", "message": "No health data available"}

        return {
            "status": "healthy" if current.is_alive else "unhealthy",
            "response_time": current.response_time,
            "queue_size": current.queue_size,
            "model_name": current.model_name,
            "gpu_memory_percent": current.gpu_memory_used,
            "avg_response_time_5min": self.get_avg_response_time(5),
            "error_message": current.error_message,
        }
