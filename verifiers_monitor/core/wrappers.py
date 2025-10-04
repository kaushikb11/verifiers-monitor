"""
Smart wrapper system for auto-detecting and monitoring Verifiers components
"""

import atexit
import hashlib
import signal
import socket
import sys
import threading
import time
from contextlib import closing
from typing import Any, Dict, List, Optional, Union

from ..dashboard.manager import get_dashboard_manager
from ..dashboard.server import DashboardServer
from ..infrastructure.async_monitor import AsyncPipelineMonitor
from ..infrastructure.system_monitor import SystemMonitor
from ..infrastructure.vllm_monitor import VLLMMonitor
from ..utils import cli_output
from ..utils.logging import get_logger
from ..utils.recovery import RetryConfig, retry_on_failure, safe_execute
from .collectors import MetricsCollector
from .exceptions import (
    ConfigurationError,
    DashboardStartupError,
    DataExtractionError,
    MonitoringError,
    WrapperError,
)
from .sqlmodel_storage import SQLModelStorage

logger = get_logger(__name__)


class AsyncCleanupStderrFilter:
    """Filter that suppresses 'Event loop is closed' errors from async cleanup"""

    def __init__(self, stream):
        self.stream = stream
        self.buffer = ""
        self.in_traceback = False

    def write(self, text):
        """Filter out Event loop is closed errors from AsyncClient cleanup"""
        if not text:
            return 0

        self.buffer += text

        # Start buffering when we see "Task exception was never retrieved"
        if (
            "Task exception was never retrieved" in self.buffer
            and not self.in_traceback
        ):
            self.in_traceback = True

        # If we're in a traceback, check if it's complete
        if self.in_traceback:
            # Check if the traceback is complete (ends with the final error line)
            if "RuntimeError: Event loop is closed\n" in self.buffer:
                # Check if this is from AsyncClient - suppress if it is
                if (
                    "AsyncClient.aclose" in self.buffer
                    or "httpx/_client.py" in self.buffer
                ):
                    # Suppress this entire traceback
                    self.buffer = ""
                    self.in_traceback = False
                    return len(text)
                else:
                    # Not from AsyncClient, output it
                    self.stream.write(self.buffer)
                    self.buffer = ""
                    self.in_traceback = False
                    return len(text)

            # If buffer is getting too large, something is wrong - flush it
            if len(self.buffer) > 10000:
                self.stream.write(self.buffer)
                self.buffer = ""
                self.in_traceback = False
                return len(text)

        # If not in traceback, write immediately
        if not self.in_traceback and self.buffer:
            self.stream.write(self.buffer)
            self.buffer = ""

        return len(text)

    def flush(self):
        """Flush any remaining buffer"""
        if self.buffer:
            self.stream.write(self.buffer)
            self.buffer = ""
        self.stream.flush()

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped stream"""
        return getattr(self.stream, name)


# Install global stderr filter once
_stderr_filter_installed = False


def _install_stderr_filter():
    """Install global stderr filter to suppress async cleanup warnings"""
    global _stderr_filter_installed
    if not _stderr_filter_installed:
        sys.stderr = AsyncCleanupStderrFilter(sys.stderr)
        _stderr_filter_installed = True


def auto_wrap(obj: Any, **config) -> Any:
    """
    Auto-detect object type and apply appropriate wrapper

    Args:
        obj: Verifiers component to wrap
        **config: Configuration options (dashboard, port, tunnel)

    Returns:
        Wrapped component with monitoring capabilities
    """

    # Detect GRPOTrainer or similar trainer
    if hasattr(obj, "train") and hasattr(obj, "env"):
        logger.info(f"Detected trainer ({type(obj).__name__})")
        return TrainerWrapper(obj, **config)

    # Detect Environment or similar
    elif hasattr(obj, "evaluate") and hasattr(obj, "rollout"):
        logger.info(f"Detected environment ({type(obj).__name__})")
        return EnvironmentWrapper(obj, **config)

    else:
        raise ValueError(
            f"Cannot monitor object of type {type(obj).__name__}. "
            f"Verifiers Monitor supports GRPOTrainer and Environment objects."
        )


def _setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""

    def shutdown_handler(signum, frame):
        print("\n")  # Clean newline after ^C
        print("Shutting down...")
        get_dashboard_manager().stop_all_dashboards()
        import sys

        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)


def _ensure_signal_handlers():
    """Ensure signal handlers are set up (called once)"""
    if not hasattr(_ensure_signal_handlers, "_setup_done"):
        _setup_signal_handlers()
        atexit.register(lambda: get_dashboard_manager().stop_all_dashboards())
        _ensure_signal_handlers._setup_done = True


def _resolve_dashboard_port(preferred_port: int) -> tuple[int, bool]:
    """
    Resolve dashboard port, finding an available one if preferred is taken

    Args:
        preferred_port: Desired port number

    Returns:
        Tuple of (resolved_port, port_changed)
    """

    def is_port_available(port: int) -> bool:
        """Check if a port is available"""
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            try:
                sock.bind(("", port))
                return True
            except OSError:
                return False

    if is_port_available(preferred_port):
        return preferred_port, False

    for port in range(preferred_port + 1, preferred_port + 100):
        if is_port_available(port):
            return port, True

    raise DashboardStartupError(
        f"Could not find available port in range {preferred_port}-{preferred_port + 99}"
    )


def _setup_dashboard_common(
    storage_backend, port: int, tunnel: bool, env_info: Optional[Dict[str, Any]] = None
) -> tuple[Optional[DashboardServer], int]:
    """
    Common dashboard setup logic shared between TrainerWrapper and EnvironmentWrapper

    Args:
        storage_backend: Storage backend instance
        port: Preferred port number
        tunnel: Whether to enable tunnel
        env_info: Environment information dict

    Returns:
        Tuple of (dashboard, resolved_port)
    """
    _ensure_signal_handlers()
    manager = get_dashboard_manager()

    if port in manager.get_active_ports():
        dashboard = manager.get_or_create_dashboard(
            storage_backend, port, tunnel, env_info
        )
        return dashboard, port

    # Resolve new port if no dashboard exists
    resolved_port, port_changed = _resolve_dashboard_port(port)
    if port_changed:
        cli_output.print_port_info(port, resolved_port)

    dashboard = manager.get_or_create_dashboard(
        storage_backend, resolved_port, tunnel, env_info
    )
    return dashboard, resolved_port


class TrainerWrapper:
    """Transparent wrapper for GRPOTrainer with monitoring capabilities"""

    def __init__(self, trainer, dashboard=True, port=8080, tunnel=False):
        self._trainer = trainer
        self._storage = SQLModelStorage()
        self._collector = MetricsCollector(self._storage)
        self._dashboard = None
        self._original_methods = {}
        self._monitoring_loop = None
        self._monitoring_thread = None

        # Infrastructure monitoring
        self._vllm_monitor = VLLMMonitor()
        self._system_monitor = SystemMonitor()
        self._async_monitor = AsyncPipelineMonitor()

        if dashboard:
            cli_output.print_banner()

            self._dashboard, self._dashboard_port = _setup_dashboard_common(
                self._storage, port, tunnel
            )

            url = f"http://localhost:{self._dashboard_port}"
            tunnel_url = (
                self._dashboard.tunnel_url
                if hasattr(self._dashboard, "tunnel_url")
                else None
            )
            cli_output.print_dashboard_url(url, self._dashboard_port, tunnel_url)
        else:
            self._dashboard_port = port

        logger.info("Training monitoring ready")

    def __del__(self):
        """Cleanup monitoring on object deletion"""
        try:
            self._stop_infrastructure_monitoring()
        except Exception:
            pass  # Ignore errors during cleanup

    def __getattr__(self, name):
        """Delegate all attributes to wrapped trainer"""
        attr = getattr(self._trainer, name)

        # Intercept key training methods
        if name == "train":
            return self._monitored_train
        elif name == "evaluate":
            return self._monitored_evaluate
        else:
            return attr

    def _monitored_train(self):
        """Wrap train() method with metrics collection"""
        logger.info("Starting training monitoring...")

        # Patch trainer methods for monitoring
        self._patch_trainer_methods()

        self._start_infrastructure_monitoring()

        try:
            result = self._trainer.train()
            logger.info("Training completed")
            return result
        finally:
            self._unpatch_trainer_methods()
            self._stop_infrastructure_monitoring()

    def _monitored_evaluate(self, *args, **kwargs):
        """Wrap evaluate() method with metrics collection"""
        logger.info("Starting evaluation monitoring...")

        try:
            result = self._trainer.evaluate(*args, **kwargs)
            logger.info("Evaluation completed")
            return result
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    def _patch_trainer_methods(self):
        """Patch key trainer methods to collect metrics"""
        self._original_methods["compute_loss"] = getattr(
            self._trainer, "compute_loss", None
        )
        self._original_methods["log"] = getattr(self._trainer, "log", None)

        if self._original_methods["compute_loss"]:
            self._trainer.compute_loss = self._create_patched_compute_loss()

        if self._original_methods["log"]:
            self._trainer.log = self._create_patched_log()

    def _unpatch_trainer_methods(self):
        """Restore original trainer methods"""
        for method_name, original_method in self._original_methods.items():
            if original_method:
                setattr(self._trainer, method_name, original_method)

        self._original_methods.clear()

    def _create_patched_compute_loss(self):
        """Create patched version of compute_loss method"""
        original_compute_loss = self._original_methods["compute_loss"]

        def patched_compute_loss(
            model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            start_time = time.time()

            loss = original_compute_loss(
                model, inputs, return_outputs, num_items_in_batch
            )

            metrics = {
                "step": getattr(self._trainer.state, "global_step", 0),
                "loss": float(loss) if hasattr(loss, "item") else float(loss),
                "compute_time": time.time() - start_time,
                "timestamp": time.time(),
            }

            if "advantages" in inputs:
                advantages = inputs["advantages"]
                if hasattr(advantages, "mean"):
                    metrics["advantages"] = {
                        "mean": float(advantages.mean()),
                        "std": float(advantages.std()),
                        "min": float(advantages.min()),
                        "max": float(advantages.max()),
                    }

            self._collector.collect_training_step(metrics)

            return loss

        return patched_compute_loss

    def _create_patched_log(self):
        """Create patched version of log method"""
        original_log = self._original_methods["log"]

        def patched_log(logs, start_time=None):
            step = getattr(self._trainer.state, "global_step", 0)

            self._collector.collect_logged_metrics(logs, step)

            return original_log(logs, start_time)

        return patched_log

    def _start_infrastructure_monitoring(self):
        """Start infrastructure monitoring components"""
        import asyncio

        try:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, start monitoring directly
                    loop.create_task(
                        self._system_monitor.start_monitoring(self._storage)
                    )
                    loop.create_task(self._vllm_monitor.start_monitoring(self._storage))

                    if self._async_monitor.attach_to_trainer(self._trainer):
                        loop.create_task(
                            self._async_monitor.start_monitoring(self._storage)
                        )
                        logger.info("Async pipeline monitoring started")

                    logger.info("Infrastructure monitoring started")
                else:
                    self._start_monitoring_thread()
            except RuntimeError:
                # No event loop, start in background thread
                self._start_monitoring_thread()

        except Exception as e:
            logger.warning(f"Infrastructure monitoring setup failed: {e}")

    def _start_monitoring_thread(self):
        """Start infrastructure monitoring in background thread"""

        def run_monitoring():
            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._monitoring_loop = loop  # Store reference for cleanup

            try:
                loop.create_task(self._system_monitor.start_monitoring(self._storage))
                loop.create_task(self._vllm_monitor.start_monitoring(self._storage))

                if self._async_monitor.attach_to_trainer(self._trainer):
                    loop.create_task(
                        self._async_monitor.start_monitoring(self._storage)
                    )
                    logger.info("Async pipeline monitoring started")

                logger.info("Infrastructure monitoring started (background thread)")

                loop.run_forever()

            except Exception as e:
                logger.warning(f"Background monitoring error: {e}")
            finally:
                # Cancel all pending tasks
                try:
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    # Give tasks a chance to handle cancellation
                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception as e:
                    logger.debug(f"Task cleanup error: {e}")
                finally:
                    loop.close()

        self._monitoring_thread = threading.Thread(target=run_monitoring, daemon=True)
        self._monitoring_thread.start()

    def _stop_infrastructure_monitoring(self):
        """Stop infrastructure monitoring components"""
        try:
            import asyncio

            # Stop the monitoring thread if it exists
            if (
                hasattr(self, "_monitoring_thread")
                and self._monitoring_thread
                and self._monitoring_thread.is_alive()
            ):
                # Signal the event loop to stop
                if hasattr(self, "_monitoring_loop") and self._monitoring_loop:
                    try:
                        self._monitoring_loop.call_soon_threadsafe(
                            self._monitoring_loop.stop
                        )
                    except Exception:
                        pass

                # Wait for thread to finish with timeout
                self._monitoring_thread.join(timeout=2.0)

            logger.info("Infrastructure monitoring stopped")
        except Exception as e:
            logger.warning(f"Infrastructure monitoring cleanup failed: {e}")

    def wait(self):
        """
        Keep the dashboard running until interrupted (Ctrl+C).
        Call this after training to keep the dashboard accessible.

        Example:
            trainer = monitor(GRPOTrainer(...))
            trainer.train()
            trainer.wait()
        """
        if self._dashboard:
            logger.info(
                f"Dashboard running at: http://localhost:{self._dashboard_port}"
            )
            logger.info("Press Ctrl+C to stop.")
            try:
                signal.pause()
            except (AttributeError, KeyboardInterrupt):
                logger.info("Shutting down dashboard...")

    @property
    def monitor(self):
        """Access to monitoring functionality"""
        return MonitorInterface(
            self._storage,
            self._collector,
            {
                "vllm": self._vllm_monitor,
                "system": self._system_monitor,
                "async": self._async_monitor,
            },
            dashboard=self._dashboard,
        )


class EnvironmentWrapper:
    """Transparent wrapper for Environment with evaluation monitoring"""

    def __init__(self, env, dashboard=True, port=8080, tunnel=False):
        self._env = env
        self._storage = SQLModelStorage()
        self._collector = MetricsCollector(self._storage)
        self._dashboard = None
        self._dashboard_port = port

        # Install global stderr filter to suppress harmless async cleanup warnings
        _install_stderr_filter()

        self._env_info = self._extract_env_info(env)

        if dashboard:
            cli_output.print_banner()

            self._dashboard, self._dashboard_port = _setup_dashboard_common(
                self._storage, port, tunnel, self._env_info
            )

            url = f"http://localhost:{self._dashboard_port}"
            tunnel_url = (
                self._dashboard.tunnel_url
                if hasattr(self._dashboard, "tunnel_url")
                else None
            )
            cli_output.print_dashboard_url(url, self._dashboard_port, tunnel_url)
        else:
            self._dashboard_port = port

    def __getattr__(self, name):
        """Delegate all attributes to wrapped environment"""
        return getattr(self._env, name)

    def evaluate(self, client, model, num_examples=-1, wait=True, **kwargs):
        """
        Monitored evaluation with progress tracking

        Args:
            client: OpenAI client for API calls
            model: Model name to use
            num_examples: Number of examples to evaluate (-1 for all)
            wait: If True (default), blocks after evaluation to keep dashboard running.
                  Set to False to return immediately.
            **kwargs: Additional arguments passed to env.evaluate()
        """
        logger.info(f"Starting evaluation ({num_examples} examples)")

        # Patch both rollout and scoring methods for monitoring
        original_rollout = getattr(self._env, "rollout", None)
        original_score_rollout = None

        if original_rollout:
            self._env.rollout = self._create_monitored_rollout(original_rollout)

        # Also patch the rubric's score_rollout method to capture rewards
        if hasattr(self._env, "rubric") and hasattr(self._env.rubric, "score_rollout"):
            logger.debug(f"Patching rubric score_rollout for enhanced monitoring")
            original_score_rollout = self._env.rubric.score_rollout
            self._env.rubric.score_rollout = self._create_monitored_score_rollout(
                original_score_rollout
            )

        try:
            rollouts_per_example = kwargs.get("rollouts_per_example", 1)

            env_type = (
                self._env_info.get("env_type", "unknown")
                if hasattr(self, "_env_info") and self._env_info
                else "unknown"
            )

            env_id = (
                self._env_info.get("env_id")
                if hasattr(self, "_env_info") and self._env_info
                else None
            )

            self._collector.start_evaluation_session(
                model, num_examples, rollouts_per_example, env_type, env_id
            )

            results = self._env.evaluate(
                client, model, num_examples=num_examples, **kwargs
            )

            # End evaluation session
            self._collector.end_evaluation_session(results)

            logger.info("Evaluation completed")
            if self._dashboard:
                url = f"http://localhost:{self._dashboard_port}"
                cli_output.print_evaluation_complete(url, self._dashboard_port)

                if wait:
                    self.wait()
                else:
                    cli_output.print_info(
                        "Dashboard will remain available while your script is running."
                    )
                    cli_output.print_info(
                        "Call env.wait() if you want to block and keep it accessible."
                    )

            return results

        except KeyboardInterrupt:
            # User interrupted evaluation - show clean message
            print("\n")  # New line after ^C
            logger.info("⚠️  Evaluation interrupted by user")

            try:
                self._collector.end_evaluation_session(None)
            except Exception:
                pass  # Ignore errors during cleanup

            print("Shutting down...")

            # Re-raise to trigger signal handler and clean shutdown
            raise

        finally:
            if original_rollout:
                self._env.rollout = original_rollout
            if original_score_rollout:
                self._env.rubric.score_rollout = original_score_rollout

            # Ensure all async operations from verifiers library complete cleanup
            # This prevents "Event loop is closed" errors from httpx AsyncClient
            import asyncio
            import gc
            import io
            import sys

            try:
                # First, force garbage collection while we still might have an event loop
                gc.collect()

                # Check if there's an existing event loop we can use for cleanup
                try:
                    existing_loop = asyncio.get_event_loop()
                    if existing_loop and not existing_loop.is_closed():
                        # Run a brief sleep to allow pending cleanup tasks to complete
                        if not existing_loop.is_running():
                            existing_loop.run_until_complete(asyncio.sleep(0.05))
                except RuntimeError:
                    pass

                # Force another garbage collection to trigger __del__ methods
                # while an event loop might still be available
                gc.collect()

                # Create a temporary event loop for any remaining cleanup
                cleanup_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(cleanup_loop)

                # Give async cleanup tasks time to complete
                cleanup_loop.run_until_complete(asyncio.sleep(0.1))

                # Final garbage collection - suppress stderr during this
                # to hide harmless "Event loop is closed" warnings
                original_stderr = sys.stderr
                sys.stderr = io.StringIO()
                try:
                    gc.collect()
                finally:
                    sys.stderr = original_stderr

                # Close the cleanup loop
                cleanup_loop.close()
                asyncio.set_event_loop(None)

            except Exception as e:
                logger.debug(f"Async cleanup error (safe to ignore): {e}")

    def _create_monitored_rollout(self, original_rollout):
        """Create monitored version of rollout method"""

        async def monitored_rollout(
            client,
            model,
            prompt,
            answer="",
            task="default",
            info=None,
            sampling_args=None,
            **kwargs,
        ):
            start_time = time.time()

            try:
                completion, state = await original_rollout(
                    client,
                    model,
                    prompt,
                    answer,
                    task,
                    info,
                    sampling_args,
                    **kwargs,
                )

                if "timing" not in state:
                    state["timing"] = {}
                state["timing"]["generation_ms"] = (time.time() - start_time) * 1000
                state["timing"]["total_ms"] = state["timing"]["generation_ms"]

                # Note: We don't collect rollout data here anymore since we don't have the reward yet
                # The monitored_score_rollout will collect the complete data with actual rewards

                return completion, state

            except TimeoutError as e:
                # Capture API timeout errors
                self._collector.collect_error(
                    error_type="api_timeout",
                    error_message=str(e),
                    error_context={"model": model, "task": task},
                )
                raise
            except Exception as e:
                # Capture general rollout failures
                self._collector.collect_error(
                    error_type="rollout_failure",
                    error_message=str(e),
                    error_context={"model": model, "task": task},
                )
                raise

        return monitored_rollout

    def _create_monitored_score_rollout(self, original_score_rollout):
        """Create monitored version of score_rollout method to capture rewards"""

        async def monitored_score_rollout(
            prompt, completion, answer, state, task="default", info=None, **kwargs
        ):
            logger.debug(f"Enhanced monitoring: scoring rollout for task '{task}'")
            rollout_score = await original_score_rollout(
                prompt, completion, answer, state, task, info, **kwargs
            )

            # Now we have the actual reward! Collect detailed verifiers-specific data
            logger.debug(f"RolloutScore type: {type(rollout_score)}")
            logger.debug(
                f"RolloutScore.metrics: {rollout_score.metrics if hasattr(rollout_score, 'metrics') else 'NO METRICS ATTR'}"
            )
            logger.debug(f"RolloutScore.reward: {rollout_score.reward}")

            def to_serializable(obj):
                """Recursively convert objects to JSON-serializable dicts"""
                if obj is None:
                    return None
                elif isinstance(obj, dict):
                    return {k: to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [to_serializable(item) for item in obj]
                elif hasattr(obj, "model_dump"):
                    return obj.model_dump()
                elif hasattr(obj, "dict"):
                    return obj.dict()
                elif hasattr(obj, "__dict__"):
                    return to_serializable(obj.__dict__)
                else:
                    return obj

            rollout_data = {
                "prompt": to_serializable(prompt),
                "completion": to_serializable(completion),
                "answer": answer,
                "reward": rollout_score.reward,
                "rollout_time": state.get("timing", {}).get("total_ms", 0) / 1000.0,
                "timestamp": time.time(),
                "completion_length": len(str(completion)),
                "task": task,
                "metrics": rollout_score.metrics
                if hasattr(rollout_score, "metrics")
                else {},
                "prompt_hash": hashlib.sha256(
                    str(prompt).encode() if prompt else str(answer).encode()
                ).hexdigest()[:16],
                "rollout_id": state.get("id", 0),
                "env_type": self._env_info.get("env_type", "unknown"),
                "parser_type": self._env_info.get("parser_type", "unknown"),
                "reward_breakdown": self._analyze_reward_breakdown(rollout_score),
                "parsing_analysis": self._analyze_parsing(completion, answer, state),
                "response_analysis": self._analyze_response(completion, state),
            }

            logger.info(
                f"Collecting rollout with metrics: {rollout_data.get('metrics')}"
            )

            self._collector.collect_rollout(rollout_data)

            return rollout_score

        return monitored_score_rollout

    # Class-level cache for env_id inference
    _env_id_cache: Dict[str, str] = {}

    def _infer_env_id(self, env):
        """
        Infer environment ID by inspecting the calling script.

        Strategy:
        1. Check cache first (based on dataset fingerprint)
        2. Walk up call stack to find user code
        3. Parse source file for load_environment("env-name") patterns
        4. Match variable names if multiple environments
        5. Fallback to dataset.info.dataset_name if available

        This is fast (<5ms) vs loading environments (~5-15 seconds)
        """
        import inspect
        import re

        # Check cache first
        env_fingerprint = (
            getattr(env.dataset, "_fingerprint", None)
            if hasattr(env, "dataset")
            else None
        )
        if env_fingerprint and env_fingerprint in self._env_id_cache:
            return self._env_id_cache[env_fingerprint]

        env_id = None

        # Try 1: Stack inspection (fast, works for most cases)
        try:
            stack = inspect.stack()

            for frame_info in stack:
                filename = frame_info.filename
                lineno = frame_info.lineno

                # Skip our own code and standard library
                if "verifiers_monitor" in filename or "site-packages" in filename:
                    continue

                # Skip if not a real file (e.g., <stdin>, <console>)
                if not filename or filename.startswith("<"):
                    continue

                # Try to read the source file
                try:
                    with open(filename, "r") as f:
                        source_lines = f.readlines()

                    # Look for patterns like:
                    # env = load_environment("math-python")
                    # env = vf.load_environment('gsm8k')
                    # env = verifiers.load_environment("tool-test")

                    # Find all load_environment calls with their line numbers
                    load_pattern = r'load_environment\s*\(\s*["\']([^"\']+)["\']\s*\)'
                    load_matches = []
                    for i, line in enumerate(source_lines, 1):
                        match = re.search(load_pattern, line)
                        if match:
                            load_matches.append((i, match.group(1)))

                    if load_matches:
                        # If only one match, use it
                        if len(load_matches) == 1:
                            env_id = load_matches[0][1]
                            break

                        # Multiple matches: only consider calls BEFORE current line
                        # (to avoid matching future calls that haven't executed yet)
                        load_matches_before = [
                            (line_num, env_name)
                            for line_num, env_name in load_matches
                            if line_num < lineno
                        ]

                        if load_matches_before:
                            # Use the most recent load_environment call before this line
                            env_id = load_matches_before[-1][1]
                            break

                        # Fallback: use the first match (shouldn't happen in normal code)
                        env_id = load_matches[0][1]
                        break

                except (IOError, OSError):
                    continue

        except Exception:
            pass

        # Try 2: Check dataset.info.dataset_name
        if not env_id:
            try:
                if hasattr(env, "dataset") and hasattr(env.dataset, "info"):
                    dataset_name = env.dataset.info.dataset_name
                    if dataset_name and dataset_name not in (None, "None"):
                        env_id = dataset_name
            except Exception:
                pass

        # Cache the result if we found something
        if env_id and env_fingerprint:
            self._env_id_cache[env_fingerprint] = env_id

        return env_id

    def _extract_env_info(self, env):
        """Extract comprehensive verifiers environment configuration"""

        # Infer environment ID by matching against loaded environment modules
        env_id = self._infer_env_id(env)

        info = {
            # Environment basics
            "env_id": env_id,  # Inferred ID (math-python, etc.) or None
            "env_type": type(env).__name__,  # Class name (ToolEnv, SingleTurnEnv, etc.)
            "message_type": getattr(env, "message_type", "unknown"),
            # Prompt configuration
            "system_prompt": getattr(env, "system_prompt", None),
            "few_shot": getattr(env, "few_shot", None),
            "parser_type": type(getattr(env, "parser", None)).__name__
            if hasattr(env, "parser")
            else "unknown",
            "parser_config": self._get_parser_config(env),
            # Rubric configuration
            "rubric_type": type(getattr(env, "rubric", None)).__name__
            if hasattr(env, "rubric")
            else "unknown",
            "reward_functions": [],
            "reward_weights": [],
            "reward_function_details": {},
            # Tool configuration (for ToolEnv)
            "tools": [],
            "tool_details": {},
            # Dataset information
            "dataset_size": 0,
            "eval_dataset_size": 0,
            "dataset_sample": None,
            "dataset_columns": [],
            # Sampling configuration
            "sampling_args": getattr(env, "sampling_args", {}),
            "max_workers": getattr(env, "max_workers", 0),
        }

        if hasattr(env, "rubric") and env.rubric:
            rubric = env.rubric
            if hasattr(rubric, "get_reward_func_names"):
                info["reward_functions"] = rubric.get_reward_func_names()
            if hasattr(rubric, "get_reward_weights"):
                info["reward_weights"] = rubric.get_reward_weights()
                logger.info(f"Extracted reward weights: {info['reward_weights']}")
                logger.info(
                    f"Sum of weights (max_possible_reward): {sum(info['reward_weights']) if info['reward_weights'] else 0}"
                )

            if hasattr(rubric, "reward_funcs"):
                for i, func in enumerate(rubric.reward_funcs):
                    func_name = func.__name__
                    weight = (
                        info["reward_weights"][i]
                        if i < len(info["reward_weights"])
                        else 1.0
                    )
                    info["reward_function_details"][func_name] = {
                        "weight": weight,
                        "doc": getattr(func, "__doc__", "No description"),
                        "module": getattr(func, "__module__", "unknown"),
                    }

        if hasattr(env, "tools") and env.tools:
            info["tools"] = [tool.__name__ for tool in env.tools]
            for tool in env.tools:
                info["tool_details"][tool.__name__] = {
                    "doc": getattr(tool, "__doc__", "No description"),
                    "module": getattr(tool, "__module__", "unknown"),
                    "signature": str(getattr(tool, "__annotations__", {})),
                }

        if hasattr(env, "dataset") and env.dataset:
            info["dataset_size"] = len(env.dataset)
            info["dataset_columns"] = (
                env.dataset.column_names if hasattr(env.dataset, "column_names") else []
            )
            if len(env.dataset) > 0:
                sample = env.dataset[0]
                # Sanitize sample (remove large fields, keep structure)
                info["dataset_sample"] = self._sanitize_dataset_sample(sample)

        if hasattr(env, "eval_dataset") and env.eval_dataset:
            info["eval_dataset_size"] = len(env.eval_dataset)

        return info

    def _get_parser_config(self, env):
        """Extract parser configuration details"""
        if not hasattr(env, "parser") or not env.parser:
            return {}

        parser = env.parser
        config = {
            "type": type(parser).__name__,
            "extract_function": getattr(parser.extract_fn, "__name__", "unknown")
            if hasattr(parser, "extract_fn")
            else None,
        }

        if hasattr(parser, "__dict__"):
            for key, value in parser.__dict__.items():
                if not key.startswith("_") and not callable(value):
                    config[key] = (
                        str(value)
                        if not isinstance(value, (str, int, float, bool, type(None)))
                        else value
                    )

        return config

    def _sanitize_dataset_sample(self, sample):
        """Sanitize dataset sample for display (remove large fields)"""
        if isinstance(sample, dict):
            sanitized = {}
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 200:
                    sanitized[key] = value[:200] + "..."
                elif isinstance(value, list) and len(value) > 3:
                    sanitized[key] = value[:3] + ["..."]
                else:
                    sanitized[key] = value
            return sanitized
        return sample

    def _analyze_reward_breakdown(self, rollout_score):
        """Analyze individual reward function contributions"""
        breakdown = {
            "total_reward": rollout_score.reward,
            "individual_scores": rollout_score.metrics,
            "reward_functions": self._env_info.get("reward_functions", []),
            "weights": self._env_info.get("reward_weights", []),
        }

        if breakdown["individual_scores"] and breakdown["weights"]:
            contributions = {}
            for func_name, score in breakdown["individual_scores"].items():
                if func_name in breakdown["reward_functions"]:
                    idx = breakdown["reward_functions"].index(func_name)
                    weight = (
                        breakdown["weights"][idx]
                        if idx < len(breakdown["weights"])
                        else 1.0
                    )
                    contributions[func_name] = {
                        "score": score,
                        "weight": weight,
                        "contribution": score * weight,
                    }
            breakdown["contributions"] = contributions

        return breakdown

    def _analyze_parsing(self, completion, answer, state):
        """Analyze parsing success and extraction quality"""
        analysis = {
            "parsing_attempted": False,
            "parsing_successful": False,
            "extracted_answer": None,
            "expected_answer": answer,
            "format_compliance": False,
        }

        if hasattr(self._env, "parser") and self._env.parser:
            try:
                analysis["parsing_attempted"] = True
                extracted = self._env.parser.parse_answer(completion)
                analysis["extracted_answer"] = extracted
                analysis["parsing_successful"] = extracted is not None
                analysis["format_compliance"] = (
                    extracted == answer if extracted else False
                )
            except Exception as e:
                analysis["parsing_error"] = str(e)
                self._collector.collect_error(
                    error_type="parsing_error",
                    error_message=str(e),
                    error_context={
                        "completion_preview": str(completion)[:200],
                        "expected_answer": str(answer),
                    },
                )

        return analysis

    def _analyze_response(self, completion, state):
        """Analyze response characteristics"""
        analysis = {
            "response_length": len(str(completion)),
            "response_type": "chat" if isinstance(completion, list) else "completion",
            "tool_calls_made": 0,
            "tool_calls_successful": 0,
            "response_time_ms": state.get("timing", {}).get("generation_ms", 0),
        }

        # Analyze tool calls for ToolEnv
        if isinstance(completion, list):
            for message in completion:
                if isinstance(message, dict) and message.get("role") == "assistant":
                    tool_calls = message.get("tool_calls", [])
                    if tool_calls:
                        analysis["tool_calls_made"] += len(tool_calls)

        if "responses" in state:
            for response in state["responses"]:
                if hasattr(response, "choices") and response.choices:
                    choice = response.choices[0]
                    if hasattr(choice, "message") and hasattr(
                        choice.message, "tool_calls"
                    ):
                        if choice.message.tool_calls:
                            analysis["tool_calls_made"] += len(
                                choice.message.tool_calls
                            )

        return analysis

    def wait(self):
        """
        Keep the dashboard running until interrupted (Ctrl+C).
        Call this after evaluation to keep the dashboard accessible.

        Example:
            env = monitor(vf.load_environment("gsm8k"))
            results = env.evaluate(client, model="gpt-4")
            env.wait()
        """
        if self._dashboard:
            logger.info(
                f"Dashboard running at: http://localhost:{self._dashboard_port}"
            )
            logger.info("Press Ctrl+C to stop.")
            try:
                import signal

                signal.pause()
            except (AttributeError, KeyboardInterrupt):
                # Windows doesn't have signal.pause() or user pressed Ctrl+C
                logger.info("Shutting down dashboard...")

    @property
    def monitor(self):
        """Access to monitoring functionality"""
        return MonitorInterface(
            self._storage,
            self._collector,
            dashboard=self._dashboard,
        )


class MonitorInterface:
    """Interface for accessing monitoring data and export functionality"""

    def __init__(
        self, storage, collector, infrastructure_monitors=None, dashboard=None
    ):
        self.storage = storage
        self.collector = collector
        self.infrastructure = infrastructure_monitors or {}
        self.dashboard = dashboard

    def wait(self):
        """Block until dashboard is stopped"""
        if self.dashboard:
            self.dashboard.wait()

    def stop(self):
        """Stop dashboard server"""
        if self.dashboard:
            self.dashboard.stop()

    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get latest metrics across all types"""
        return self.storage.get_latest()

    def get_training_history(self, limit: int = 1000) -> list:
        """Get training step history"""
        return self.storage.get_recent("training_step", limit)

    def get_evaluation_history(self, limit: int = 1000) -> list:
        """Get evaluation rollout history"""
        return self.storage.get_recent("rollout", limit)

    def get_infrastructure_summary(self) -> Dict[str, Any]:
        """Get summary of infrastructure metrics"""
        summary = {}

        # vLLM metrics
        if "vllm" in self.infrastructure:
            summary["vllm"] = self.infrastructure["vllm"].get_metrics_summary()

        # System metrics
        if "system" in self.infrastructure:
            summary["system"] = self.infrastructure["system"].get_metrics_summary()

        # Async pipeline metrics
        if "async" in self.infrastructure:
            summary["async_pipeline"] = self.infrastructure[
                "async"
            ].get_metrics_summary()

        return summary

    def get_system_health(self) -> Dict[str, Union[str, List[str]]]:
        """Get overall system health status"""
        issues: List[str] = []
        health: Dict[str, Union[str, List[str]]] = {
            "overall": "healthy",
            "issues": issues,
        }

        if "vllm" in self.infrastructure:
            if not self.infrastructure["vllm"].is_healthy():
                issues.append("vLLM server unhealthy")

        if "system" in self.infrastructure:
            constraints = self.infrastructure["system"].is_resource_constrained()
            if any(constraints.values()):
                constrained_resources = [k for k, v in constraints.items() if v]
                issues.append(
                    f"Resource constraints: {', '.join(constrained_resources)}"
                )

        if "async" in self.infrastructure:
            if not self.infrastructure["async"].is_pipeline_healthy():
                issues.append("Async pipeline unhealthy")

        if issues:
            health["overall"] = "degraded" if len(issues) == 1 else "unhealthy"

        return health
