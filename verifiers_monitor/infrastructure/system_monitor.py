"""
System resource monitoring for GPU, CPU, memory, and network
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import psutil


@dataclass
class SystemMetrics:
    """System resource metrics snapshot"""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    gpu_count: int
    gpu_memory_used: List[float]
    gpu_utilization: List[float]
    network_bytes_sent: int
    network_bytes_recv: int
    disk_usage_percent: float


class SystemMonitor:
    """Monitor system resources (CPU, GPU, memory, network)"""

    def __init__(self, check_interval: float = 5.0):
        """
        Initialize system monitor

        Args:
            check_interval: How often to collect metrics (seconds)
        """
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)

        self._monitoring = False
        self._monitor_task = None
        self._metrics_history: List[SystemMetrics] = []
        self._max_history = 200
        self._baseline_network = None

        self._gpu_available = self._check_gpu_availability()

    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available"""
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            if gpus:
                self.logger.info(f"Found {len(gpus)} GPU(s) for monitoring")
                return True
        except ImportError:
            try:
                import pynvml

                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    self.logger.info(f"Found {device_count} GPU(s) via pynvml")
                    return True
            except:
                pass

        self.logger.info("No GPU monitoring available")
        return False

    async def start_monitoring(self, storage_backend=None):
        """Start continuous system monitoring"""
        if self._monitoring:
            return

        self._monitoring = True
        self._baseline_network = self._get_network_stats()
        self._monitor_task = asyncio.create_task(self._monitor_loop(storage_backend))
        self.logger.info("Started system resource monitoring")

    async def stop_monitoring(self):
        """Stop system monitoring"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped system monitoring")

    async def _monitor_loop(self, storage_backend):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                metrics = await self.collect_metrics()

                self._metrics_history.append(metrics)
                if len(self._metrics_history) > self._max_history:
                    self._metrics_history.pop(0)

                if storage_backend:
                    metrics_dict = {
                        "timestamp": metrics.timestamp,
                        "cpu_percent": metrics.cpu_percent,
                        "memory_percent": metrics.memory_percent,
                        "memory_used_gb": metrics.memory_used_gb,
                        "memory_total_gb": metrics.memory_total_gb,
                        "gpu_count": metrics.gpu_count,
                        "network_bytes_sent": metrics.network_bytes_sent,
                        "network_bytes_recv": metrics.network_bytes_recv,
                        "disk_usage_percent": metrics.disk_usage_percent,
                    }

                    if metrics.gpu_memory_used:
                        metrics_dict["gpu_memory_used"] = metrics.gpu_memory_used
                        metrics_dict["gpu_utilization"] = metrics.gpu_utilization
                        metrics_dict["gpu_memory_avg"] = sum(
                            metrics.gpu_memory_used
                        ) / len(metrics.gpu_memory_used)
                        metrics_dict["gpu_util_avg"] = sum(
                            metrics.gpu_utilization
                        ) / len(metrics.gpu_utilization)

                    storage_backend.store("system_metrics", metrics_dict)

                await asyncio.sleep(self.check_interval)

            except Exception as e:
                self.logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(self.check_interval * 2)

    async def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        timestamp = time.time()

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)

        # GPU metrics
        gpu_count = 0
        gpu_memory_used: List[float] = []
        gpu_utilization: List[float] = []

        if self._gpu_available:
            gpu_count, gpu_memory_used, gpu_utilization = await self._get_gpu_metrics()

        # Network metrics
        network_stats = self._get_network_stats()
        network_bytes_sent = network_stats["bytes_sent"]
        network_bytes_recv = network_stats["bytes_recv"]

        # Disk metrics
        disk_usage = psutil.disk_usage("/")
        disk_usage_percent = (disk_usage.used / disk_usage.total) * 100

        return SystemMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            memory_total_gb=memory_total_gb,
            gpu_count=gpu_count,
            gpu_memory_used=gpu_memory_used,
            gpu_utilization=gpu_utilization,
            network_bytes_sent=network_bytes_sent,
            network_bytes_recv=network_bytes_recv,
            disk_usage_percent=disk_usage_percent,
        )

    async def _get_gpu_metrics(self) -> tuple[int, List[float], List[float]]:
        """Get GPU metrics"""
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            if gpus:
                memory_used = [gpu.memoryUtil * 100 for gpu in gpus]
                utilization = [gpu.load * 100 for gpu in gpus]
                return len(gpus), memory_used, utilization
        except ImportError:
            try:
                import pynvml

                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                memory_used = []
                utilization = []

                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                    # Memory usage
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_percent = (mem_info.used / mem_info.total) * 100
                    memory_used.append(mem_percent)

                    # GPU utilization
                    util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization.append(util_info.gpu)

                return device_count, memory_used, utilization
            except:
                pass
        except:
            pass

        return 0, [], []

    def _get_network_stats(self) -> Dict[str, int]:
        """Get network I/O statistics"""
        try:
            net_io = psutil.net_io_counters()
            return {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
            }
        except:
            return {
                "bytes_sent": 0,
                "bytes_recv": 0,
                "packets_sent": 0,
                "packets_recv": 0,
            }

    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get most recent metrics"""
        return self._metrics_history[-1] if self._metrics_history else None

    def get_metrics_history(self, limit: int = 50) -> List[SystemMetrics]:
        """Get recent metrics history"""
        return self._metrics_history[-limit:]

    def get_avg_cpu_usage(self, minutes: int = 5) -> float:
        """Get average CPU usage over last N minutes"""
        cutoff_time = time.time() - (minutes * 60)
        recent_metrics = [m for m in self._metrics_history if m.timestamp > cutoff_time]

        if not recent_metrics:
            return 0.0

        return sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)

    def get_avg_gpu_usage(self, minutes: int = 5) -> Dict[str, float]:
        """Get average GPU usage over last N minutes"""
        cutoff_time = time.time() - (minutes * 60)
        recent_metrics = [
            m
            for m in self._metrics_history
            if m.timestamp > cutoff_time and m.gpu_memory_used
        ]

        if not recent_metrics:
            return {"memory": 0.0, "utilization": 0.0}

        all_memory = []
        all_utilization = []

        for metrics in recent_metrics:
            all_memory.extend(metrics.gpu_memory_used)
            all_utilization.extend(metrics.gpu_utilization)

        return {
            "memory": sum(all_memory) / len(all_memory) if all_memory else 0.0,
            "utilization": (
                sum(all_utilization) / len(all_utilization) if all_utilization else 0.0
            ),
        }

    def get_network_throughput(self, minutes: int = 1) -> Dict[str, float]:
        """Get network throughput over last N minutes (bytes per second)"""
        if len(self._metrics_history) < 2:
            return {"sent_bps": 0.0, "recv_bps": 0.0}

        cutoff_time = time.time() - (minutes * 60)
        recent_metrics = [m for m in self._metrics_history if m.timestamp > cutoff_time]

        if len(recent_metrics) < 2:
            return {"sent_bps": 0.0, "recv_bps": 0.0}

        first = recent_metrics[0]
        last = recent_metrics[-1]
        time_diff = last.timestamp - first.timestamp

        if time_diff <= 0:
            return {"sent_bps": 0.0, "recv_bps": 0.0}

        sent_bps = (last.network_bytes_sent - first.network_bytes_sent) / time_diff
        recv_bps = (last.network_bytes_recv - first.network_bytes_recv) / time_diff

        return {
            "sent_bps": max(0, sent_bps),
            "recv_bps": max(0, recv_bps),
        }

    def is_resource_constrained(self) -> Dict[str, bool]:
        """Check if any resources are constrained"""
        current = self.get_current_metrics()
        if not current:
            return {"cpu": False, "memory": False, "gpu": False, "disk": False}

        return {
            "cpu": current.cpu_percent > 90,
            "memory": current.memory_percent > 90,
            "gpu": (
                any(util > 95 for util in current.gpu_utilization)
                if current.gpu_utilization
                else False
            ),
            "disk": current.disk_usage_percent > 90,
        }

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of system metrics"""
        current = self.get_current_metrics()
        if not current:
            return {"status": "no_data"}

        constraints = self.is_resource_constrained()
        avg_cpu = self.get_avg_cpu_usage(5)
        avg_gpu = self.get_avg_gpu_usage(5)
        network = self.get_network_throughput(1)

        return {
            "timestamp": current.timestamp,
            "cpu_percent": current.cpu_percent,
            "cpu_avg_5min": avg_cpu,
            "memory_percent": current.memory_percent,
            "memory_used_gb": current.memory_used_gb,
            "memory_total_gb": current.memory_total_gb,
            "gpu_count": current.gpu_count,
            "gpu_memory_avg": avg_gpu["memory"],
            "gpu_util_avg": avg_gpu["utilization"],
            "network_sent_bps": network["sent_bps"],
            "network_recv_bps": network["recv_bps"],
            "disk_usage_percent": current.disk_usage_percent,
            "resource_constraints": constraints,
            "status": "constrained" if any(constraints.values()) else "healthy",
        }
