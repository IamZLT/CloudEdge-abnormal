import json
import mimetypes
import threading
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Dict, Optional
from uuid import uuid4

import requests

from common.schemas import DetectionResult


class CloudGatewayClient:
    """边缘侧客户端：只访问云端业务网关，不接触大模型 API。"""

    def __init__(
        self,
        gateway_url: str,
        device_id: str,
        run_id: str,
        timeout: float = 60,
        connect_timeout: Optional[float] = None,
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self.inspect_url = f"{self.gateway_url}/v1/inspect"
        self.start_run_url = f"{self.gateway_url}/v1/runs/start"
        self.finish_run_url = f"{self.gateway_url}/v1/runs/finish"
        self.device_id = device_id
        self.run_id = run_id
        self.timeout = timeout
        self.connect_timeout = timeout if connect_timeout is None else connect_timeout
        self.request_timeout = (self.connect_timeout, self.timeout)
        self.session = requests.Session()
        self.session.trust_env = False
        self.last_call_metrics: Dict[str, Any] = {}
        self.run_registered = False
        self._run_lock = threading.Lock()

    @staticmethod
    def _parse_result(payload: Dict[str, Any]) -> DetectionResult:
        try:
            result = payload["result"]
        except (KeyError, TypeError) as exc:
            raise ValueError("云端网关响应缺少 result") from exc
        if not isinstance(result, dict):
            raise ValueError("云端网关 result 必须是 JSON 对象")
        label = str(result.get("label", "")).strip().lower()
        if label not in {"normal", "anomaly"}:
            raise ValueError(f"云端网关返回了不支持的 label：{label!r}")
        return DetectionResult(
            label=label,
            confidence=float(result.get("confidence", 0.0)),
            defect_category=result.get("defect_category"),
            bbox=result.get("bbox"),
            timestamp=result.get("timestamp"),
            source=result.get("source", "cloud"),
            metadata=result.get("metadata") or {},
        )

    def inspect_image(
        self,
        image_path: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DetectionResult:
        if not self.run_registered:
            self.start_run()
        path = Path(image_path)
        image_bytes = path.read_bytes()
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        request_id = uuid4().hex
        metadata_payload = {
            "schema_version": "1.0",
            "request_id": request_id,
            "run_id": self.run_id,
            "device_id": self.device_id,
            "context": context or {},
        }
        started_ns = perf_counter_ns()
        response = None
        self.last_call_metrics = {}
        try:
            response = self.session.post(
                self.inspect_url,
                data={
                    "metadata": json.dumps(
                        metadata_payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                },
                files={"image": (path.name, image_bytes, mime_type)},
                timeout=self.request_timeout,
            )
        except requests.RequestException as exc:
            request = getattr(exc, "request", None)
            request_body = getattr(request, "body", b"") or b""
            if isinstance(request_body, str):
                request_body = request_body.encode("utf-8")
            self.last_call_metrics = {
                "protocol": "http_multipart",
                "request_id": request_id,
                "original_image_bytes": len(image_bytes),
                "request_body_bytes": len(request_body),
                "response_body_bytes": None,
                "edge_gateway_round_trip_ms": (
                    perf_counter_ns() - started_ns
                )
                / 1_000_000,
                "http_status": None,
                "success": False,
            }
            raise
        request_body = response.request.body or b""
        if isinstance(request_body, str):
            request_body = request_body.encode("utf-8")
        self.last_call_metrics = {
            "protocol": "http_multipart",
            "request_id": request_id,
            "original_image_bytes": len(image_bytes),
            "request_body_bytes": len(request_body),
            "response_body_bytes": len(response.content),
            "edge_gateway_round_trip_ms": (
                perf_counter_ns() - started_ns
            )
            / 1_000_000,
            "http_status": response.status_code,
            "success": response.ok,
        }
        if response.status_code == 422 and "run_id" in response.text:
            # 网关重启后内存中的 run 注册会丢失；下次半开探测先自动重注册。
            self.run_registered = False
        response.raise_for_status()
        response_payload = response.json()
        if response_payload.get("request_id") != request_id:
            raise ValueError("云端网关响应 request_id 与请求不一致")
        if response_payload.get("run_id") != self.run_id:
            raise ValueError("云端网关响应 run_id 与请求不一致")
        result = self._parse_result(response_payload)
        metadata = dict(result.metadata or {})
        metadata["communication_metrics"] = dict(self.last_call_metrics)
        result.metadata = metadata
        return result

    def start_run(self) -> Dict[str, Any]:
        with self._run_lock:
            if self.run_registered:
                return {"status": "already_started", "run_id": self.run_id}
            response = self.session.post(
                self.start_run_url,
                json={"run_id": self.run_id, "device_id": self.device_id},
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            payload = response.json()
            self.run_registered = True
            return payload

    def finish_run(self) -> Dict[str, Any]:
        response = self.session.post(
            self.finish_run_url,
            json={"run_id": self.run_id},
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        self.run_registered = False
        return payload

    def review_result(self, result: DetectionResult, image_path: str) -> DetectionResult:
        # result 参数仅用于兼容协同接口；默认不把边缘检测结果上传到云端。
        return self.inspect_image(image_path)

    def close(self) -> None:
        self.session.close()
