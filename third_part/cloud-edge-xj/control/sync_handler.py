import threading
from time import perf_counter_ns
from typing import Any, Dict, Optional, Protocol

from common.schemas import DetectionResult
from control.circuit_breaker import CloudCircuitBreaker, CircuitOpenError


class CloudReviewer(Protocol):
    def review_result(self, result: DetectionResult, image_path: str) -> DetectionResult:
        ...


class SyncHandler:
    def __init__(
        self,
        cloud_service: CloudReviewer,
        circuit_breaker: Optional[CloudCircuitBreaker] = None,
    ):
        self.cloud_service = cloud_service
        self.circuit_breaker = circuit_breaker or CloudCircuitBreaker(enabled=False)
        self._thread_state = threading.local()

    def _set_last_event(
        self,
        *,
        cloud_attempted: bool,
        cloud_success: bool,
        circuit_short_circuited: bool,
        state_before: str,
        error: Optional[object],
        started_ns: int,
    ) -> None:
        self._thread_state.last_call_metrics = (
            dict(getattr(self.cloud_service, "last_call_metrics", {}) or {})
            if cloud_attempted
            else {}
        )
        self._thread_state.last_resilience_event = {
            "cloud_attempted": cloud_attempted,
            "cloud_success": cloud_success,
            "circuit_short_circuited": circuit_short_circuited,
            "circuit_state_before": state_before,
            "circuit_state_after": self.circuit_breaker.snapshot()["state"],
            "cloud_wait_ms": (perf_counter_ns() - started_ns) / 1_000_000,
            "cloud_error": None if error is None else str(error),
        }

    def upload_result(self, result: DetectionResult, image_path: str) -> DetectionResult:
        started_ns = perf_counter_ns()
        try:
            state_before = self.circuit_breaker.before_request()
        except CircuitOpenError as exc:
            self._set_last_event(
                cloud_attempted=False,
                cloud_success=False,
                circuit_short_circuited=True,
                state_before="open",
                error=exc,
                started_ns=started_ns,
            )
            raise

        try:
            cloud_result = self.cloud_service.review_result(result, image_path)
        except Exception as exc:
            self.circuit_breaker.record_failure(exc)
            self._set_last_event(
                cloud_attempted=True,
                cloud_success=False,
                circuit_short_circuited=False,
                state_before=state_before,
                error=exc,
                started_ns=started_ns,
            )
            raise

        self.circuit_breaker.record_success()
        self._set_last_event(
            cloud_attempted=True,
            cloud_success=True,
            circuit_short_circuited=False,
            state_before=state_before,
            error=None,
            started_ns=started_ns,
        )
        return cloud_result

    def mark_cloud_unavailable(self, error: object) -> None:
        """启动探测失败时立即打开断路器，避免每个任务重复等待网络超时。"""
        self.circuit_breaker.record_failure(error, force_open=True)

    @property
    def last_resilience_event(self) -> Dict[str, Any]:
        return dict(getattr(self._thread_state, "last_resilience_event", {}) or {})

    @property
    def last_call_metrics(self):
        return dict(getattr(self._thread_state, "last_call_metrics", {}) or {})

    @property
    def circuit_state(self) -> Dict[str, Any]:
        return self.circuit_breaker.snapshot()
