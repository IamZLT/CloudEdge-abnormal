import threading
from time import monotonic
from typing import Any, Dict, Optional


class CircuitOpenError(ConnectionError):
    """云端断路器打开时的快速失败，不发起网络请求。"""


class CloudCircuitBreaker:
    def __init__(
        self,
        enabled: bool = True,
        failure_threshold: int = 1,
        recovery_timeout_seconds: float = 10.0,
    ):
        self.enabled = enabled
        self.failure_threshold = max(1, int(failure_threshold))
        self.recovery_timeout_seconds = max(0.1, float(recovery_timeout_seconds))
        self._state = "closed"
        self._consecutive_failures = 0
        self._opened_at: Optional[float] = None
        self._last_error: Optional[str] = None
        self._lock = threading.Lock()

    def before_request(self) -> str:
        """返回调用前状态；打开/半开占用时直接抛出，避免阻塞实时任务。"""
        if not self.enabled:
            return "disabled"
        now = monotonic()
        with self._lock:
            if self._state == "closed":
                return "closed"
            if self._state == "half_open":
                raise CircuitOpenError("云端恢复探测正在进行，当前任务使用边缘兜底")
            assert self._opened_at is not None
            remaining = self.recovery_timeout_seconds - (now - self._opened_at)
            if remaining > 0:
                raise CircuitOpenError(
                    f"云端断路器已打开，{remaining:.3f}s 后允许恢复探测"
                )
            self._state = "half_open"
            return "half_open"

    def record_success(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._state = "closed"
            self._consecutive_failures = 0
            self._opened_at = None
            self._last_error = None

    def record_failure(self, error: object, force_open: bool = False) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._consecutive_failures += 1
            self._last_error = str(error)
            if (
                force_open
                or self._state == "half_open"
                or self._consecutive_failures >= self.failure_threshold
            ):
                self._state = "open"
                self._opened_at = monotonic()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            retry_after_seconds = None
            if self._state == "open" and self._opened_at is not None:
                retry_after_seconds = max(
                    0.0,
                    self.recovery_timeout_seconds - (monotonic() - self._opened_at),
                )
            return {
                "enabled": self.enabled,
                "state": self._state if self.enabled else "disabled",
                "failure_threshold": self.failure_threshold,
                "consecutive_failures": self._consecutive_failures,
                "recovery_timeout_seconds": self.recovery_timeout_seconds,
                "retry_after_seconds": retry_after_seconds,
                "last_error": self._last_error,
            }
