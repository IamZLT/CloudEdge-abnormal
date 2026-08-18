import threading
from typing import Any, Dict, Optional
from time import perf_counter_ns
import requests

from cloud.gpu_monitor import GpuSampler


class CloudClient:
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 60,
        gpu_index: Optional[int] = None,
        gpu_sample_interval: float = 0.25,
    ):
        self.api_url = f"{base_url.rstrip('/')}/chat/completions"
        self.api_key = api_key
        self.timeout = timeout
        self.gpu_index = gpu_index
        self.gpu_sample_interval = gpu_sample_interval
        # 网关使用 ThreadingHTTPServer；每个请求线程拥有独立 Session 和调用指标。
        self._thread_state = threading.local()
        self.last_call_metrics = {}

    @property
    def session(self) -> requests.Session:
        session = getattr(self._thread_state, "session", None)
        if session is None:
            session = requests.Session()
            # 云端模型部署在 localhost 时，不应继承系统 HTTP(S) 代理。
            session.trust_env = False
            self._thread_state.session = session
        return session

    @property
    def last_call_metrics(self) -> Dict[str, Any]:
        return getattr(self._thread_state, "last_call_metrics", {})

    @last_call_metrics.setter
    def last_call_metrics(self, value: Dict[str, Any]) -> None:
        self._thread_state.last_call_metrics = value

    def post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        started_ns = perf_counter_ns()
        response = None
        self.last_call_metrics = {}
        gpu_sampler = (
            GpuSampler(self.gpu_index, self.gpu_sample_interval)
            if self.gpu_index is not None
            else None
        )
        if gpu_sampler is not None:
            gpu_sampler.start()
        try:
            response = self.session.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            request_body = response.request.body or b""
            if isinstance(request_body, str):
                request_body = request_body.encode("utf-8")
            self.last_call_metrics = {
                "request_body_bytes": len(request_body),
                "response_body_bytes": len(response.content),
                "http_round_trip_ms": (perf_counter_ns() - started_ns) / 1_000_000,
                "http_status": response.status_code,
            }
            response.raise_for_status()
            return response.json()
        except Exception:
            elapsed_ms = (perf_counter_ns() - started_ns) / 1_000_000
            if not self.last_call_metrics:
                self.last_call_metrics = {
                    "request_body_bytes": None,
                    "response_body_bytes": len(response.content) if response is not None else None,
                    "http_round_trip_ms": elapsed_ms,
                    "http_status": response.status_code if response is not None else None,
                }
            raise
        finally:
            if gpu_sampler is not None:
                self.last_call_metrics["gpu"] = gpu_sampler.stop()
