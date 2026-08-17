import argparse
import json
import threading
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from time import perf_counter_ns
from typing import Any, Dict, Type

import cv2
import numpy as np

from cloud.cloud_client import CloudClient
from cloud.cloud_service import CloudService
from cloud.gpu_monitor import ContinuousGpuMonitor
from cloud.model_api import LargeModelAPI
from common.config import Config
from common.utils import current_timestamp, log_info


class CloudGatewayApplication:
    def __init__(
        self,
        cloud_service: CloudService,
        max_upload_bytes: int,
        gpu_monitor: ContinuousGpuMonitor = None,
    ):
        self.cloud_service = cloud_service
        self.max_upload_bytes = max_upload_bytes
        self.gpu_monitor = gpu_monitor
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._runs_condition = threading.Condition()

    def handle_run_control(self, action: str, run_id: str) -> Dict[str, Any]:
        if not run_id:
            raise ValueError("run_id 不能为空")
        if action == "start":
            with self._runs_condition:
                existing = self._runs.get(run_id)
                if existing is not None:
                    if existing["status"] == "finished":
                        return {
                            "status": "finished",
                            "run_id": run_id,
                            "metrics": existing["finished_metrics"],
                        }
                    return {"status": "already_started", "run_id": run_id}
            sampled_ns = (
                self.gpu_monitor.capture_sample()
                if self.gpu_monitor is not None
                else None
            )
            started_ns = sampled_ns or perf_counter_ns()
            with self._runs_condition:
                # 并发重复 start 必须幂等，不能清空已经累计的请求。
                if run_id in self._runs:
                    return {"status": "already_started", "run_id": run_id}
                self._runs[run_id] = {
                    "status": "active",
                    "started_ns": started_ns,
                    "in_flight": 0,
                    "request_count": 0,
                    "successful_request_count": 0,
                    "failed_request_count": 0,
                    "request_body_bytes": 0,
                    "image_bytes": 0,
                }
            return {"status": "started", "run_id": run_id}
        if action != "finish":
            raise ValueError(f"不支持的 run 控制动作：{action}")
        with self._runs_condition:
            run = self._runs.get(run_id)
            if run is None:
                raise ValueError(f"未找到 run_id：{run_id}")
            if run["status"] == "finished":
                return {
                    "status": "finished",
                    "run_id": run_id,
                    "metrics": run["finished_metrics"],
                }
            if run["status"] == "finishing":
                while run["status"] != "finished":
                    self._runs_condition.wait()
                return {
                    "status": "finished",
                    "run_id": run_id,
                    "metrics": run["finished_metrics"],
                }
            run["status"] = "finishing"
            while run["in_flight"]:
                self._runs_condition.wait()
            started_ns = int(run["started_ns"])
            counters = {
                key: run[key]
                for key in (
                    "request_count",
                    "successful_request_count",
                    "failed_request_count",
                    "request_body_bytes",
                    "image_bytes",
                )
            }

        sampled_ns = (
            self.gpu_monitor.capture_sample()
            if self.gpu_monitor is not None
            else None
        )
        finished_ns = sampled_ns or perf_counter_ns()
        counters["elapsed_seconds"] = (finished_ns - started_ns) / 1_000_000_000
        counters["gpu"] = (
            self.gpu_monitor.summary(started_ns, finished_ns)
            if self.gpu_monitor is not None
            else {
                "sample_count": 0,
                "unavailable_reason": "云端网关未启用 GPU 采样",
            }
        )
        with self._runs_condition:
            run = self._runs[run_id]
            run["status"] = "finished"
            run["finished_metrics"] = counters
            self._runs_condition.notify_all()
        return {"status": "finished", "run_id": run_id, "metrics": counters}

    def _begin_request(
        self,
        run_id: str,
        request_body_bytes: int,
        image_bytes: int,
    ) -> None:
        with self._runs_condition:
            run = self._runs.get(run_id)
            if run is None:
                raise ValueError(f"未注册的 run_id：{run_id}")
            if run["status"] != "active":
                raise ValueError(f"run_id {run_id} 当前状态不接受新请求：{run['status']}")
            run["in_flight"] += 1
            run["request_count"] += 1
            run["request_body_bytes"] += request_body_bytes
            run["image_bytes"] += image_bytes

    def _finish_request(self, run_id: str, success: bool) -> None:
        with self._runs_condition:
            run = self._runs[run_id]
            key = "successful_request_count" if success else "failed_request_count"
            run[key] += 1
            run["in_flight"] -= 1
            self._runs_condition.notify_all()

    @staticmethod
    def _parse_multipart(request_body: bytes, content_type: str) -> Dict[str, Any]:
        if not content_type.lower().startswith("multipart/form-data"):
            raise TypeError("Content-Type 必须是 multipart/form-data")
        message = BytesParser(policy=policy.default).parsebytes(
            (
                f"Content-Type: {content_type}\r\n"
                "MIME-Version: 1.0\r\n\r\n"
            ).encode("ascii")
            + request_body
        )
        parts: Dict[str, Any] = {}
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if name:
                parts[str(name)] = part
        if "image" not in parts or "metadata" not in parts:
            raise ValueError("multipart 请求必须包含 image 和 metadata 字段")
        image_part = parts["image"]
        metadata_part = parts["metadata"]
        try:
            metadata = json.loads(metadata_part.get_content())
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("metadata 不是有效 JSON") from exc
        return {
            "metadata": metadata,
            "image_bytes": image_part.get_payload(decode=True) or b"",
            "mime_type": image_part.get_content_type(),
            "filename": image_part.get_filename(),
        }

    def inspect(self, request_body: bytes, content_type: str) -> Dict[str, Any]:
        started_ns = perf_counter_ns()
        parsed = self._parse_multipart(request_body, content_type)
        metadata = parsed["metadata"]
        image_bytes = parsed["image_bytes"]
        mime_type = str(parsed["mime_type"])
        if not isinstance(metadata, dict):
            raise ValueError("metadata 必须是 JSON 对象")
        request_id = str(metadata.get("request_id", ""))
        run_id = str(metadata.get("run_id", ""))
        if not request_id or not run_id:
            raise ValueError("metadata 必须包含 request_id 和 run_id")
        self._begin_request(run_id, len(request_body), len(image_bytes))
        success = False
        try:
            if not mime_type.startswith("image/"):
                raise TypeError(f"不支持的图像 MIME 类型：{mime_type!r}")
            if not image_bytes:
                raise ValueError("上传图像不能为空")
            if len(image_bytes) > self.max_upload_bytes:
                raise ValueError(
                    f"上传图像超过限制：{len(image_bytes)} > "
                    f"{self.max_upload_bytes} bytes"
                )
            decoded = cv2.imdecode(
                np.frombuffer(image_bytes, dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
            if decoded is None:
                raise ValueError("上传内容无法解码为图像")
            context = metadata.get("context") or {}
            if not isinstance(context, dict):
                raise ValueError("context 必须是 JSON 对象")
            result = self.cloud_service.inspect_image_bytes(image_bytes, mime_type)
            gateway_metrics = {
                "received_request_body_bytes": len(request_body),
                "received_image_bytes": len(image_bytes),
                "request_id": request_id,
                "run_id": run_id,
                "context_keys": sorted(str(key) for key in context),
                "processing_ms": (perf_counter_ns() - started_ns) / 1_000_000,
                "processed_at": current_timestamp(),
            }
            result_metadata = dict(result.metadata or {})
            result_metadata["gateway_metrics"] = gateway_metrics
            result.metadata = result_metadata
            success = True
            return {
                "schema_version": "1.0",
                "request_id": request_id,
                "run_id": run_id,
                "result": result.to_dict(),
            }
        finally:
            self._finish_request(run_id, success)


def create_handler(
    application: CloudGatewayApplication,
    max_request_body_bytes: int,
) -> Type[BaseHTTPRequestHandler]:
    class GatewayHandler(BaseHTTPRequestHandler):
        server_version = "CollabCloudGateway/1.0"

        def _send_json(self, status: HTTPStatus, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(
                    HTTPStatus.OK,
                    {"status": "ok", "service": "cloud-gateway"},
                )
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:
            if self.path in {"/v1/runs/start", "/v1/runs/finish"}:
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                    result = application.handle_run_control(
                        self.path.rsplit("/", 1)[-1],
                        str(payload["run_id"]),
                    )
                    self._send_json(HTTPStatus.OK, result)
                except Exception as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            if self.path != "/v1/inspect":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_content_length"})
                return
            if content_length <= 0 or content_length > max_request_body_bytes:
                self._send_json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {
                        "error": "request_body_too_large",
                        "max_request_body_bytes": max_request_body_bytes,
                    },
                )
                return
            request_body = self.rfile.read(content_length)
            try:
                response_payload = application.inspect(
                    request_body,
                    self.headers.get("Content-Type", ""),
                )
                self._send_json(HTTPStatus.OK, response_payload)
            except TypeError as exc:
                self._send_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": str(exc)})
            except ValueError as exc:
                self._send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(exc)})
            except Exception as exc:
                log_info("Cloud gateway processing failed", {"error": str(exc)})
                self._send_json(
                    HTTPStatus.BAD_GATEWAY,
                    {"error": "cloud_processing_failed", "detail": str(exc)},
                )

        def log_message(self, format: str, *args: object) -> None:
            log_info("Cloud gateway HTTP", {"message": format % args})

    return GatewayHandler


def build_application(config: Config) -> CloudGatewayApplication:
    model_client = CloudClient(
        config.cloud_model_api_base_url,
        config.cloud_api_key,
        config.cloud_timeout,
        None,
        config.cloud_gpu_sample_interval,
    )
    model_api = LargeModelAPI(
        model_client,
        config.cloud_model,
        config.cloud_max_tokens,
        config.cloud_temperature,
    )
    gpu_monitor = (
        ContinuousGpuMonitor(
            config.cloud_gpu_index,
            config.cloud_gpu_sample_interval,
        )
        if config.cloud_gpu_index is not None
        else None
    )
    if gpu_monitor is not None:
        gpu_monitor.start()
    return CloudGatewayApplication(
        CloudService(model_api),
        config.cloud_gateway_max_upload_bytes,
        gpu_monitor,
    )


def serve(config_path: str, host: str = None, port: int = None) -> None:
    config = Config.load(config_path)
    application = build_application(config)
    # 图像在边缘到网关这一段按原始二进制 multipart 上传，不需要 Base64 的 4/3 裕量。
    request_limit = config.cloud_gateway_max_upload_bytes + 1024 * 1024
    handler = create_handler(application, request_limit)
    bind_host = config.cloud_gateway_host if host is None else host
    bind_port = config.cloud_gateway_port if port is None else port
    server = ThreadingHTTPServer((bind_host, bind_port), handler)
    log_info(
        "Cloud gateway started",
        {
            "host": bind_host,
            "port": bind_port,
            "model": config.cloud_model,
            "model_api_scope": "cloud-only",
        },
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if application.gpu_monitor is not None:
            application.gpu_monitor.close()
        log_info("Cloud gateway stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="启动云端工业缺陷检测 HTTP 网关")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    serve(args.config, args.host, args.port)


if __name__ == "__main__":
    main()
