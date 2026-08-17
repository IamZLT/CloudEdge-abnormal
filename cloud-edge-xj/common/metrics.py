import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Dict, List, Optional
from uuid import uuid4

from common.config import Config, DatasetConfig
from common.ground_truth import infer_ground_truth
from common.schemas import DetectionResult


PDF_NAME = "XH-202606_面向云边协同场景的分布式人工智能感知与决策关键技术研究.pdf"
E2E_TARGET_MS = 200.0
CONFLICT_TARGET_PERCENT = 5.0
RESOLUTION_TARGET_PERCENT = 90.0


def _percentile(values: List[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _safe_percent(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator * 100.0


def _round(value: Optional[float], digits: int = 3) -> Optional[float]:
    return None if value is None else round(value, digits)


def _latency_summary(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "std_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    return {
        "count": len(values),
        "mean_ms": _round(statistics.fmean(values)),
        "median_ms": _round(statistics.median(values)),
        "p95_ms": _round(_percentile(values, 95)),
        "p99_ms": _round(_percentile(values, 99)),
        "std_ms": _round(statistics.pstdev(values)),
        "min_ms": _round(min(values)),
        "max_ms": _round(max(values)),
    }


def _resolve_output_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class MetricsCollector:
    def __init__(self, config: Config, run_id: Optional[str] = None):
        self.enabled = config.evaluation_enabled
        self.config = config
        self.run_id = run_id or uuid4().hex
        self.gateway_run_metrics: Dict[str, Any] = {}
        self.summary_path = _resolve_output_path(config.metrics_summary_path)
        self.detail_path = _resolve_output_path(config.metrics_detail_path)
        self.report_path = _resolve_output_path(config.metrics_report_path)
        self.started_ns = perf_counter_ns()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._detail_file = (
            self.detail_path.open("w", encoding="utf-8") if self.enabled else None
        )
        self.success_count = 0
        self.error_count = 0
        self.total_source_bytes = 0
        self.e2e_latencies: List[float] = []
        self.edge_only_latencies: List[float] = []
        self.cloud_routed_latencies: List[float] = []
        self.sla_success_count = 0
        self.cloud_attempt_count = 0
        self.cloud_success_count = 0
        self.cloud_uploaded_source_bytes = 0
        self.cloud_request_body_bytes = 0
        self.cloud_response_body_bytes = 0
        self.cloud_wire_size_samples = 0
        self.request_overhead_bytes: List[int] = []
        self.cloud_round_trip_ms: List[float] = []
        self.cloud_total_ms: List[float] = []
        self.gateway_round_trip_ms: List[float] = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.gpu_utilization_samples: List[float] = []
        self.gpu_memory_used_samples_mib: List[float] = []
        self.gpu_memory_total_mib: Optional[float] = None
        self.gpu_monitor_errors: List[str] = []
        self.business_completed_count = 0
        self.deadline_retained_count = 0
        self.cloud_required_business_count = 0
        self.cloud_required_deadline_retained_count = 0
        self.fallback_count = 0
        self.provisional_count = 0
        self.circuit_short_circuit_count = 0
        self.network_failure_count = 0
        self.cloud_wait_ms: List[float] = []
        self.ground_truth_task_count = 0
        self.ground_truth_correct_count = 0
        self.multi_edge_task_count = 0
        self.conflict_count = 0
        self.resolution_completion_count = 0
        self.resolution_ground_truth_count = 0
        self.resolution_correct_count = 0
        self.cloud_arbitration_success_count = 0
        self.node_latencies: Dict[str, List[float]] = defaultdict(list)
        self.node_labels: Dict[str, Counter] = defaultdict(Counter)
        self.dataset_metrics: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "success": 0,
                "errors": 0,
                "latencies": [],
                "cloud_attempts": 0,
                "multi_edge_tasks": 0,
                "conflicts": 0,
                "resolution_success": 0,
                "resolution_gt": 0,
                "resolution_correct": 0,
            }
        )

    def _write_detail(self, record: Dict[str, Any]) -> None:
        if self._detail_file is None:
            return
        self._detail_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._detail_file.flush()

    def set_gateway_run_metrics(self, metrics: Optional[Dict[str, Any]]) -> None:
        self.gateway_run_metrics = dict(metrics or {})

    def record_success(
        self,
        dataset: DatasetConfig,
        image_path: str,
        result: DetectionResult,
        e2e_latency_ms: float,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {}

        source_bytes = Path(image_path).stat().st_size
        ground_truth = infer_ground_truth(dataset, image_path)
        metadata = result.metadata or {}
        resilience = metadata.get("resilience") or {}
        multi_edge = metadata.get("multi_edge") or {}
        cloud_metrics = metadata.get("cloud_metrics") or {}
        communication_metrics = metadata.get("communication_metrics") or {}
        if not communication_metrics:
            communication_metrics = multi_edge.get("failed_communication_metrics") or {}
        cloud_required = bool(
            resilience.get(
                "cloud_required",
                multi_edge.get("cloud_attempted", result.source == "cloud"),
            )
        )
        cloud_attempted = bool(
            resilience.get(
                "cloud_attempted",
                multi_edge.get("cloud_attempted", result.source == "cloud"),
            )
        )
        cloud_success = bool(
            resilience.get(
                "cloud_success",
                multi_edge.get("cloud_success", result.source == "cloud"),
            )
        )
        conflict = bool(multi_edge.get("conflict", False))
        is_multi_edge = bool(multi_edge)
        resolution_completed = conflict and result.label in {"normal", "anomaly"}
        resolution_correct = (
            conflict and ground_truth is not None and result.label == ground_truth
        )
        business_completed = bool(
            resilience.get(
                "business_completed",
                result.label in {"normal", "anomaly"},
            )
        )
        retained_within_deadline = (
            business_completed
            and e2e_latency_ms <= self.config.business_deadline_ms
        )

        detail = {
            "run_id": self.run_id,
            "status": "success",
            "dataset": dataset.name,
            "dataset_type": dataset.dataset_type,
            "image_path": image_path,
            "source_image_bytes": source_bytes,
            "ground_truth": ground_truth,
            "final_label": result.label,
            "final_source": result.source,
            "e2e_latency_ms": e2e_latency_ms,
            "within_200ms": e2e_latency_ms <= E2E_TARGET_MS,
            "cloud_attempted": cloud_attempted,
            "cloud_required": cloud_required,
            "cloud_success": cloud_success,
            "image_quality": metadata.get("image_quality"),
            "routing": metadata.get("routing"),
            "cloud_metrics": cloud_metrics or None,
            "communication_metrics": communication_metrics or None,
            "resilience": resilience or None,
            "business_completed": business_completed,
            "business_deadline_ms": self.config.business_deadline_ms,
            "business_retained_within_deadline": retained_within_deadline,
            "multi_edge": multi_edge or None,
            "conflict": conflict if is_multi_edge else None,
            "resolution_completed": resolution_completed if conflict else None,
            "resolution_correct": resolution_correct if conflict and ground_truth else None,
        }
        self._write_detail(detail)

        self.success_count += 1
        self.total_source_bytes += source_bytes
        self.e2e_latencies.append(e2e_latency_ms)
        self.sla_success_count += int(e2e_latency_ms <= E2E_TARGET_MS)
        self.business_completed_count += int(business_completed)
        self.deadline_retained_count += int(retained_within_deadline)
        if cloud_required:
            self.cloud_required_business_count += 1
            self.cloud_required_deadline_retained_count += int(
                retained_within_deadline
            )
        self.fallback_count += int(bool(resilience.get("fallback_used", False)))
        self.provisional_count += int(bool(resilience.get("provisional", False)))
        self.circuit_short_circuit_count += int(
            bool(resilience.get("circuit_short_circuited", False))
        )
        self.network_failure_count += int(
            bool(resilience.get("cloud_error"))
            or bool(resilience.get("circuit_short_circuited", False))
        )
        if resilience.get("cloud_wait_ms") is not None:
            self.cloud_wait_ms.append(float(resilience["cloud_wait_ms"]))
        if ground_truth is not None:
            self.ground_truth_task_count += 1
            self.ground_truth_correct_count += int(result.label == ground_truth)
        bucket = self.dataset_metrics[dataset.name]
        bucket["success"] += 1
        bucket["latencies"].append(e2e_latency_ms)

        if cloud_attempted:
            self.cloud_attempt_count += 1
            self.cloud_uploaded_source_bytes += source_bytes
            self.cloud_routed_latencies.append(e2e_latency_ms)
            bucket["cloud_attempts"] += 1
        else:
            self.edge_only_latencies.append(e2e_latency_ms)
        if cloud_success:
            self.cloud_success_count += 1

        request_bytes = communication_metrics.get("request_body_bytes")
        response_bytes = communication_metrics.get("response_body_bytes")
        if request_bytes is not None:
            self.cloud_request_body_bytes += int(request_bytes)
            self.cloud_wire_size_samples += 1
            self.request_overhead_bytes.append(max(0, int(request_bytes) - source_bytes))
        if response_bytes is not None:
            self.cloud_response_body_bytes += int(response_bytes)
        if cloud_metrics.get("http_round_trip_ms") is not None:
            self.cloud_round_trip_ms.append(float(cloud_metrics["http_round_trip_ms"]))
        elif cloud_attempted and multi_edge.get("cloud_attempt_latency_ms") is not None:
            self.cloud_round_trip_ms.append(float(multi_edge["cloud_attempt_latency_ms"]))
        if cloud_metrics.get("cloud_total_ms") is not None:
            self.cloud_total_ms.append(float(cloud_metrics["cloud_total_ms"]))
        if communication_metrics.get("edge_gateway_round_trip_ms") is not None:
            self.gateway_round_trip_ms.append(
                float(communication_metrics["edge_gateway_round_trip_ms"])
            )
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = cloud_metrics.get(field)
            if value is not None:
                setattr(self, field, getattr(self, field) + int(value))
        gpu_metrics = cloud_metrics.get("gpu") or {}
        self.gpu_utilization_samples.extend(
            float(value)
            for value in gpu_metrics.get("utilization_samples_percent", [])
        )
        self.gpu_memory_used_samples_mib.extend(
            float(value) for value in gpu_metrics.get("memory_used_samples_mib", [])
        )
        if gpu_metrics.get("memory_total_mib") is not None:
            self.gpu_memory_total_mib = float(gpu_metrics["memory_total_mib"])
        for error in gpu_metrics.get("errors", []):
            if error not in self.gpu_monitor_errors:
                self.gpu_monitor_errors.append(str(error))

        if is_multi_edge:
            self.multi_edge_task_count += 1
            bucket["multi_edge_tasks"] += 1
            for node in multi_edge.get("nodes", []):
                node_id = str(node.get("node_id", "unknown"))
                if node.get("latency_ms") is not None:
                    self.node_latencies[node_id].append(float(node["latency_ms"]))
                self.node_labels[node_id][str(node.get("label", "unknown"))] += 1
            if conflict:
                self.conflict_count += 1
                bucket["conflicts"] += 1
                self.resolution_completion_count += int(resolution_completed)
                self.cloud_arbitration_success_count += int(
                    bool(multi_edge.get("resolution_success"))
                )
                bucket["resolution_success"] += int(
                    bool(multi_edge.get("resolution_success"))
                )
                if ground_truth is not None:
                    self.resolution_ground_truth_count += 1
                    bucket["resolution_gt"] += 1
                    self.resolution_correct_count += int(resolution_correct)
                    bucket["resolution_correct"] += int(resolution_correct)
        return detail

    def record_error(
        self,
        dataset: DatasetConfig,
        image_path: Optional[str],
        error: str,
        elapsed_ms: float,
    ) -> None:
        if not self.enabled:
            return
        source_bytes = 0
        if image_path:
            try:
                source_bytes = Path(image_path).stat().st_size
            except OSError:
                pass
        self.total_source_bytes += source_bytes
        self.error_count += 1
        if image_path and infer_ground_truth(dataset, image_path) is not None:
            self.ground_truth_task_count += 1
        self.dataset_metrics[dataset.name]["errors"] += 1
        self._write_detail(
            {
                "run_id": self.run_id,
                "status": "error",
                "dataset": dataset.name,
                "dataset_type": dataset.dataset_type,
                "image_path": image_path,
                "source_image_bytes": source_bytes,
                "elapsed_before_error_ms": elapsed_ms,
                "business_completed": False,
                "business_deadline_ms": self.config.business_deadline_ms,
                "business_retained_within_deadline": False,
                "error": error,
            }
        )

    def _dataset_summaries(self) -> Dict[str, Any]:
        summaries = {}
        for name, bucket in sorted(self.dataset_metrics.items()):
            latency = _latency_summary(bucket["latencies"])
            mean_ms = latency["mean_ms"]
            summaries[name] = {
                "successful_tasks": bucket["success"],
                "failed_tasks": bucket["errors"],
                "e2e_latency": latency,
                "average_e2e_target_ms": E2E_TARGET_MS,
                "average_e2e_target_met": mean_ms is not None and mean_ms <= E2E_TARGET_MS,
                "cloud_request_ratio_percent": _round(
                    _safe_percent(bucket["cloud_attempts"], bucket["success"])
                ),
                "multi_edge_tasks": bucket["multi_edge_tasks"],
                "conflict_ratio_percent": _round(
                    _safe_percent(bucket["conflicts"], bucket["multi_edge_tasks"])
                ),
                "conflict_resolution_accuracy_percent": _round(
                    _safe_percent(bucket["resolution_correct"], bucket["resolution_gt"])
                ),
                "conflict_resolution_success_rate_percent": _round(
                    _safe_percent(bucket["resolution_success"], bucket["conflicts"])
                ),
            }
        return summaries

    def _build_summary(self) -> Dict[str, Any]:
        elapsed_seconds = (perf_counter_ns() - self.started_ns) / 1_000_000_000
        total_tasks = self.success_count + self.error_count
        functional_retention = _safe_percent(
            self.business_completed_count,
            total_tasks,
        )
        deadline_retention = _safe_percent(
            self.deadline_retained_count,
            total_tasks,
        )
        affected_deadline_retention = _safe_percent(
            self.cloud_required_deadline_retained_count,
            self.cloud_required_business_count,
        )
        offline_accuracy = _safe_percent(
            self.ground_truth_correct_count,
            self.ground_truth_task_count,
        )
        conflict_ratio = _safe_percent(self.conflict_count, self.multi_edge_task_count)
        resolution_accuracy = _safe_percent(
            self.resolution_correct_count, self.resolution_ground_truth_count
        )
        resolution_completion = _safe_percent(
            self.resolution_completion_count, self.conflict_count
        )
        cloud_arbitration_success = _safe_percent(
            self.cloud_arbitration_success_count, self.conflict_count
        )
        gateway_request_count = int(
            self.gateway_run_metrics.get("request_count", self.cloud_attempt_count)
        )
        gateway_success_count = int(
            self.gateway_run_metrics.get(
                "successful_request_count", self.cloud_success_count
            )
        )
        gateway_failed_count = int(
            self.gateway_run_metrics.get(
                "failed_request_count", gateway_request_count - gateway_success_count
            )
        )
        uploaded_image_bytes = int(
            self.gateway_run_metrics.get("image_bytes", self.cloud_uploaded_source_bytes)
        )
        uploaded_request_body_bytes = int(
            self.gateway_run_metrics.get(
                "request_body_bytes", self.cloud_request_body_bytes
            )
        )
        cloud_request_ratio = _safe_percent(gateway_request_count, total_tasks)
        cloud_request_reduction = (
            None if cloud_request_ratio is None else 100.0 - cloud_request_ratio
        )
        upload_reduction = (
            _safe_percent(
                self.total_source_bytes - uploaded_image_bytes,
                self.total_source_bytes,
            )
        )
        mean_request_overhead = (
            statistics.fmean(self.request_overhead_bytes)
            if self.request_overhead_bytes
            else None
        )
        estimated_all_cloud_request_bytes = (
            None
            if mean_request_overhead is None
            else self.total_source_bytes
            + round(mean_request_overhead * total_tasks)
        )
        request_body_reduction = (
            None
            if not estimated_all_cloud_request_bytes
            else (
                estimated_all_cloud_request_bytes - uploaded_request_body_bytes
            )
            / estimated_all_cloud_request_bytes
            * 100.0
        )
        communication_window_seconds = float(
            self.gateway_run_metrics.get("elapsed_seconds", elapsed_seconds)
        )
        wire_uplink_mbps = (
            8.0
            * uploaded_request_body_bytes
            / communication_window_seconds
            / 1_000_000
            if communication_window_seconds > 0
            else None
        )
        bandwidth_occupancy_percent = (
            None
            if wire_uplink_mbps is None or self.config.network_link_capacity_mbps is None
            else wire_uplink_mbps / self.config.network_link_capacity_mbps * 100.0
        )
        e2e = _latency_summary(self.e2e_latencies)
        latency_cv_percent = (
            None
            if not e2e["mean_ms"]
            else e2e["std_ms"] / e2e["mean_ms"] * 100.0
        )
        cloud_http = _latency_summary(self.cloud_round_trip_ms)
        estimated_all_cloud_ms = (
            None
            if cloud_http["mean_ms"] is None
            else cloud_http["mean_ms"] * total_tasks
        )
        observed_cloud_ms = sum(self.cloud_round_trip_ms)
        estimated_cloud_time_reduction = (
            None
            if not estimated_all_cloud_ms
            else (estimated_all_cloud_ms - observed_cloud_ms) / estimated_all_cloud_ms * 100.0
        )

        node_summary = {}
        for node_id in sorted(self.node_latencies):
            node_summary[node_id] = {
                "latency": _latency_summary(self.node_latencies[node_id]),
                "label_counts": dict(self.node_labels[node_id]),
            }

        dataset_summaries = self._dataset_summaries()
        all_dataset_latency_targets_met = bool(dataset_summaries) and all(
            dataset["average_e2e_target_met"] for dataset in dataset_summaries.values()
        )

        gateway_gpu = self.gateway_run_metrics.get("gpu") or {}
        gpu_sample_count = int(
            gateway_gpu.get("sample_count", len(self.gpu_utilization_samples))
        )
        gpu_unavailable_reason = None
        if self.config.cloud_gpu_index is None:
            gpu_unavailable_reason = "cloud.gpu_index 为 null，未启用本机 GPU 采样"
        elif not gpu_sample_count:
            gpu_unavailable_reason = (
                gateway_gpu.get("unavailable_reason")
                or "; ".join(gateway_gpu.get("errors", self.gpu_monitor_errors))
                or "云端完整评测窗口内没有取得有效 nvidia-smi 样本"
            )

        server_gpu = {
            "gpu_index": gateway_gpu.get("gpu_index", self.config.cloud_gpu_index),
            "scope": gateway_gpu.get("scope", "指定 GPU 的主机级指标"),
            "sample_count": gpu_sample_count,
            "average_utilization_percent": _round(
                gateway_gpu.get("average_utilization_percent")
                if gateway_gpu
                else statistics.fmean(self.gpu_utilization_samples)
                if self.gpu_utilization_samples
                else None
            ),
            "p95_utilization_percent": _round(
                gateway_gpu.get("p95_utilization_percent")
                if gateway_gpu
                else _percentile(self.gpu_utilization_samples, 95)
            ),
            "peak_utilization_percent": _round(
                gateway_gpu.get("peak_utilization_percent")
                if gateway_gpu
                else max(self.gpu_utilization_samples)
                if self.gpu_utilization_samples
                else None
            ),
            "gpu_seconds": _round(gateway_gpu.get("gpu_seconds")),
            "average_memory_used_mib": _round(
                gateway_gpu.get("average_memory_used_mib")
                if gateway_gpu
                else statistics.fmean(self.gpu_memory_used_samples_mib)
                if self.gpu_memory_used_samples_mib
                else None
            ),
            "peak_memory_used_mib": _round(
                gateway_gpu.get("peak_memory_used_mib")
                if gateway_gpu
                else max(self.gpu_memory_used_samples_mib)
                if self.gpu_memory_used_samples_mib
                else None
            ),
            "memory_total_mib": _round(
                gateway_gpu.get("memory_total_mib", self.gpu_memory_total_mib)
            ),
            "errors": gateway_gpu.get("errors", self.gpu_monitor_errors),
            "unavailable_reason": gpu_unavailable_reason,
            "attribution_note": (
                "nvidia-smi 提供整张 GPU 卡的主机级数据；同卡其他进程的负载也会计入"
            ),
        }
        communication_efficiency = {
            "primary_protocol": "edge_to_gateway_http_multipart",
            "upload_metrics_source": (
                "cloud_gateway_run_aggregate"
                if self.gateway_run_metrics
                else "edge_client_event_fallback"
            ),
            "gateway_run_metrics_available": bool(self.gateway_run_metrics),
            "all_cloud_baseline_original_image_bytes": self.total_source_bytes,
            "actual_uploaded_original_image_bytes": uploaded_image_bytes,
            "original_image_upload_reduction_percent": _round(upload_reduction),
            "estimated_all_cloud_request_body_bytes": estimated_all_cloud_request_bytes,
            "actual_request_body_bytes": uploaded_request_body_bytes,
            "request_body_upload_reduction_percent": _round(request_body_reduction),
            "client_observed_response_body_bytes": self.cloud_response_body_bytes,
            "actual_response_body_bytes": self.cloud_response_body_bytes,
            "request_body_size_sample_count": self.cloud_wire_size_samples,
            "gateway_recorded_request_count": gateway_request_count,
            "mean_multipart_overhead_bytes": _round(mean_request_overhead),
            "uploaded_original_bytes_per_task": _round(
                uploaded_image_bytes / total_tasks if total_tasks else None
            ),
            "gateway_round_trip": _latency_summary(self.gateway_round_trip_ms),
            "measurement_window_seconds": _round(communication_window_seconds),
            "average_http_body_uplink_mbps_over_run": _round(wire_uplink_mbps),
            "configured_link_capacity_mbps": self.config.network_link_capacity_mbps,
            "average_bandwidth_occupancy_percent": _round(
                bandwidth_occupancy_percent, 6
            ),
            "note": (
                "上传量以边缘到云端网关的图像检测 multipart HTTP 实体字节计算，"
                "不含控制请求、HTTP 头和 TCP/IP 开销；带宽占用率按完整评测窗口的平均"
                "上行吞吐除以配置链路容量计算，链路容量不是实测值。网关聚合覆盖已识别"
                "run_id 的完整请求；响应体字节来自边缘客户端观测"
            ),
        }

        return {
            "schema_version": "1.0",
            "run": {
                "started_at": self.started_at,
                "run_id": self.run_id,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": _round(elapsed_seconds),
                "evaluation_mode": "simulation/proxy"
                if self.config.edge_model_path is None
                else "model",
                "run_label": self.config.evaluation_run_label,
                "sample_description": self.config.evaluation_sample_description,
                "edge_model": self.config.edge_model_path or "deterministic-placeholder",
                "cloud_model": self.config.cloud_model,
                "multi_edge_enabled": self.config.multi_edge_enabled,
                "edge_nodes": [
                    {
                        "id": node.node_id,
                        "augmentation": node.augmentation,
                        "parameter": node.parameter,
                    }
                    for node in self.config.edge_nodes
                ],
            },
            "benchmark_basis": {
                "source_pdf": str(Path(__file__).resolve().parent.parent / PDF_NAME),
                "core_indicators_pages": [4, 5],
                "evaluation_indicator_1_3_page": 6,
                "targets": {
                    "average_e2e_latency_ms_max": E2E_TARGET_MS,
                    "decision_conflict_ratio_percent_max": CONFLICT_TARGET_PERCENT,
                    "conflict_resolution_success_percent_min": RESOLUTION_TARGET_PERCENT,
                    "network_outage_business_retention_percent_min": (
                        self.config.business_retention_target_percent
                    ),
                },
            },
            "tasks": {
                "total": total_tasks,
                "successful": self.success_count,
                "failed": self.error_count,
                "success_rate_percent": _round(
                    _safe_percent(self.success_count, total_tasks)
                ),
            },
            "end_to_end_latency": {
                "definition": "从读取单张图像前到最终 DetectionResult 就绪；不含可视化与结果落盘",
                "all_successful_tasks": e2e,
                "edge_only_tasks": _latency_summary(self.edge_only_latencies),
                "cloud_routed_tasks": _latency_summary(self.cloud_routed_latencies),
                "within_200ms_count": self.sla_success_count,
                "within_200ms_rate_percent": _round(
                    _safe_percent(self.sla_success_count, self.success_count)
                ),
                "average_target_met": e2e["mean_ms"] is not None
                and e2e["mean_ms"] <= E2E_TARGET_MS,
                "all_dataset_average_targets_met": all_dataset_latency_targets_met,
                "latency_coefficient_of_variation_percent": _round(latency_cv_percent),
            },
            "network_resilience": {
                "scenario": "simulated_full_disconnect"
                if self.config.network_outage_simulation_enabled
                else "not_injected",
                "fault_injection_active": self.config.network_outage_simulation_enabled,
                "definition": (
                    "限时业务保持率 = 在业务时限内产生合法 normal/anomaly 决策，且随后"
                    "同步写入本地事件文件成功的任务数 / 故障窗口内总任务数；文件写入"
                    "耗时不计入 200ms 决策时限"
                ),
                "business_deadline_ms": self.config.business_deadline_ms,
                "business_retention_target_percent_min": (
                    self.config.business_retention_target_percent
                ),
                "functional_completed_count": self.business_completed_count,
                "functional_retention_rate_percent": _round(functional_retention),
                "deadline_retained_count": self.deadline_retained_count,
                "deadline_missed_or_failed_count": (
                    total_tasks - self.deadline_retained_count
                ),
                "deadline_business_retention_rate_percent": _round(
                    deadline_retention
                ),
                "business_retention_target_met": (
                    deadline_retention is not None
                    and deadline_retention
                    >= self.config.business_retention_target_percent
                )
                if self.config.network_outage_simulation_enabled
                else None,
                "cloud_required_tasks": self.cloud_required_business_count,
                "cloud_required_deadline_retained_count": (
                    self.cloud_required_deadline_retained_count
                ),
                "cloud_required_deadline_retention_rate_percent": _round(
                    affected_deadline_retention
                ),
                "cloud_required_target_met": (
                    affected_deadline_retention is not None
                    and affected_deadline_retention
                    >= self.config.business_retention_target_percent
                )
                if self.config.network_outage_simulation_enabled
                else None,
                "fallback_count": self.fallback_count,
                "provisional_decision_count": self.provisional_count,
                "network_failure_or_open_circuit_count": self.network_failure_count,
                "circuit_short_circuit_count": self.circuit_short_circuit_count,
                "cloud_wait": _latency_summary(self.cloud_wait_ms),
                "ground_truth_available_tasks": self.ground_truth_task_count,
                "ground_truth_correct_tasks": self.ground_truth_correct_count,
                "offline_decision_accuracy_percent": _round(offline_accuracy),
                "accuracy_note": (
                    "业务保持率只衡量业务连续性，不等于检测准确率；当前占位边缘模型"
                    "不能用于证明工业检测质量"
                ),
            },
            "cloud_compute_load_proxy": {
                "note": "本节为补充诊断；评选指标 1.3 的计算资源主指标采用云端网关完整运行窗口的 GPU 使用率",
                "all_cloud_baseline_request_count": total_tasks,
                "actual_cloud_request_count": gateway_request_count,
                "successful_cloud_request_count": gateway_success_count,
                "failed_cloud_request_count": gateway_failed_count,
                "cloud_request_ratio_percent": _round(cloud_request_ratio),
                "cloud_request_reduction_vs_all_cloud_percent": _round(
                    cloud_request_reduction
                ),
                "http_round_trip": cloud_http,
                "observed_http_round_trip_total_ms": _round(observed_cloud_ms),
                "estimated_all_cloud_http_total_ms": _round(estimated_all_cloud_ms),
                "estimated_cloud_time_reduction_percent": _round(
                    estimated_cloud_time_reduction
                ),
                "cloud_total": _latency_summary(self.cloud_total_ms),
                "tokens": {
                    "prompt": self.prompt_tokens,
                    "completion": self.completion_tokens,
                    "total": self.total_tokens,
                },
                "server_cpu_percent": None,
                "server_cpu_unavailable_reason": "当前 OpenAI 兼容 API 未暴露服务端 CPU 监控接口",
                "server_gpu": server_gpu,
            },
            "communication_efficiency": communication_efficiency,
            "resource_and_communication_efficiency": {
                "definition": "评选指标 1.3 的三个主指标；其余请求、token 和时延仅作诊断信息",
                "data_upload": {
                    "http_request_body_bytes": uploaded_request_body_bytes,
                    "original_image_bytes": uploaded_image_bytes,
                    "request_body_reduction_vs_all_cloud_percent": _round(
                        request_body_reduction
                    ),
                },
                "bandwidth": {
                    "average_uplink_mbps_over_run": _round(wire_uplink_mbps),
                    "configured_link_capacity_mbps": self.config.network_link_capacity_mbps,
                    "average_occupancy_percent": _round(
                        bandwidth_occupancy_percent, 6
                    ),
                    "measurement_window_seconds": _round(
                        communication_window_seconds
                    ),
                },
                "cloud_gpu": server_gpu,
            },
            "stability_and_consistency": {
                "multi_edge_evaluated_tasks": self.multi_edge_task_count,
                "conflicted_tasks": self.conflict_count,
                "consistent_tasks": self.multi_edge_task_count - self.conflict_count,
                "conflict_ratio_percent": _round(conflict_ratio),
                "conflict_ratio_target_percent_max": CONFLICT_TARGET_PERCENT,
                "conflict_ratio_target_met": conflict_ratio is not None
                and conflict_ratio <= CONFLICT_TARGET_PERCENT,
                "resolution_completion_rate_percent": _round(resolution_completion),
                "cloud_arbitration_success_rate_percent": _round(
                    cloud_arbitration_success
                ),
                "conflict_resolution_success_rate_percent": _round(
                    cloud_arbitration_success
                ),
                "conflicts_with_ground_truth": self.resolution_ground_truth_count,
                "correctly_resolved_conflicts": self.resolution_correct_count,
                "ground_truth_resolution_accuracy_percent": _round(
                    resolution_accuracy
                ),
                # 兼容已有结果消费者；该字段是推理正确率，不作为“解决成功率”达标依据。
                "conflict_resolution_accuracy_percent": _round(resolution_accuracy),
                "conflict_resolution_target_percent_min": RESOLUTION_TARGET_PERCENT,
                "conflict_resolution_target_met": cloud_arbitration_success is not None
                and cloud_arbitration_success >= RESOLUTION_TARGET_PERCENT,
                "zero_conflict_note": "N/A：没有冲突，无法评价解决成功率"
                if self.conflict_count == 0
                else None,
                "per_node": node_summary,
            },
            "per_dataset": dataset_summaries,
            "scope_notes": {
                "measured": [
                    "端到端时延",
                    "云端请求比例与累计往返时间",
                    "原始图像上传量与 HTTP 请求/响应体字节",
                    "多节点标签冲突与基于数据集真值的解决正确率",
                    "同机部署且配置 gpu_index 时的云端 GPU 利用率与显存",
                ]
                + (
                    ["100% 模拟断网下的限时业务功能保持率"]
                    if self.config.network_outage_simulation_enabled
                    else []
                ),
                "not_measured": [
                    "服务端 CPU 真实利用率",
                    "配对 cloud-only 基准的实测端到端时延",
                    "TTFT、边缘模型内存与推理精度",
                ]
                + (
                    []
                    if self.config.network_outage_simulation_enabled
                    else ["网络波动注入下的业务功能保持率"]
                ),
            },
        }

    @staticmethod
    def _markdown(summary: Dict[str, Any]) -> str:
        e2e = summary["end_to_end_latency"]
        cloud = summary["cloud_compute_load_proxy"]
        communication = summary["communication_efficiency"]
        resource = summary["resource_and_communication_efficiency"]
        resilience = summary["network_resilience"]
        consistency = summary["stability_and_consistency"]
        gpu = resource["cloud_gpu"]

        def value(item: Any, suffix: str = "") -> str:
            return "N/A" if item is None else f"{item}{suffix}"

        lines = [
            "# 云边协同评测报告",
            "",
            f"- 开始时间：{summary['run']['started_at']}",
            f"- 评测模式：{summary['run']['evaluation_mode']}",
            f"- 报告范围：{summary['run']['run_label']}（{summary['run']['sample_description']}）",
            f"- 成功/总任务：{summary['tasks']['successful']}/{summary['tasks']['total']}",
            "",
            "## 核心指标",
            "",
            "| 指标 | 实测值 | PDF 目标 | 是否达标 |",
            "| --- | ---: | ---: | :---: |",
            f"| 平均端到端时延 | {value(e2e['all_successful_tasks']['mean_ms'], ' ms')} | 各数据集 ≤ 200 ms | {e2e['all_dataset_average_targets_met']} |",
            f"| 冲突比例 | {value(consistency['conflict_ratio_percent'], '%')} | ≤ 5% | {consistency['conflict_ratio_target_met']} |",
            f"| 冲突解决成功率 | {value(consistency['conflict_resolution_success_rate_percent'] if not resilience['fault_injection_active'] else None, '%')} | ≥ 90% | {value(consistency['conflict_resolution_target_met'] if not resilience['fault_injection_active'] else None)} |",
            f"| 断网限时业务保持率 | {value(resilience['deadline_business_retention_rate_percent'] if resilience['fault_injection_active'] else None, '%')} | ≥ {resilience['business_retention_target_percent_min']}% | {value(resilience['business_retention_target_met'])} |",
            "",
            "## 断网业务连续性",
            "",
            f"- 场景：{resilience['scenario']}；业务时限 {resilience['business_deadline_ms']} ms。",
            f"- 功能保持率：{value(resilience['functional_retention_rate_percent'], '%')}；限时业务保持率：{value(resilience['deadline_business_retention_rate_percent'], '%')}。",
            f"- 需云任务限时保持率：{value(resilience['cloud_required_deadline_retention_rate_percent'], '%')}（{resilience['cloud_required_deadline_retained_count']}/{resilience['cloud_required_tasks']}）。",
            f"- 本地降级 {resilience['fallback_count']} 次，断路器快速跳过 {resilience['circuit_short_circuit_count']} 次，临时隔离决策 {resilience['provisional_decision_count']} 次。",
            f"- 断网决策真值正确率：{value(resilience['offline_decision_accuracy_percent'], '%')}；该值与业务保持率分开评价。",
            "",
            "## 资源与通信效率（评选指标 1.3）",
            "",
            f"- 数据上传量：边缘到网关 HTTP 实体 {resource['data_upload']['http_request_body_bytes']} bytes，其中原始图像 {resource['data_upload']['original_image_bytes']} bytes；相对同协议全云估算削减 {value(resource['data_upload']['request_body_reduction_vs_all_cloud_percent'], '%')}。",
            f"- 带宽占用：完整评测窗口平均上行 {value(resource['bandwidth']['average_uplink_mbps_over_run'], ' Mbps')}，配置链路容量 {value(resource['bandwidth']['configured_link_capacity_mbps'], ' Mbps')}，平均占用率 {value(resource['bandwidth']['average_occupancy_percent'], '%')}。",
            (
                f"- 计算资源：GPU {gpu['gpu_index']} 完整评测窗口平均/P95/峰值利用率："
                f"{value(gpu['average_utilization_percent'], '%')}/"
                f"{value(gpu['p95_utilization_percent'], '%')}/"
                f"{value(gpu['peak_utilization_percent'], '%')}；峰值显存："
                f"{value(gpu['peak_memory_used_mib'], ' MiB')}。"
                if gpu["sample_count"]
                else f"- GPU 指标不可用：{gpu['unavailable_reason']}。"
            ),
            "- 口径：上传量不含 HTTP 头与 TCP/IP 开销；链路容量来自 config.yaml，并非实测；GPU 为整卡主机级数据，可能包含同卡其他进程负载。",
            "",
            "诊断信息："
            f" 云端请求 {cloud['actual_cloud_request_count']}/{cloud['all_cloud_baseline_request_count']}，"
            f"网关往返均值 {value(communication['gateway_round_trip']['mean_ms'], ' ms')}，"
            f"模型 API 累计往返 {value(cloud['observed_http_round_trip_total_ms'], ' ms')}。",
            "",
            "## 分数据集端到端时延",
            "",
            "| 数据集 | 样本数 | 平均时延 | P95 | ≤200ms |",
            "| --- | ---: | ---: | ---: | :---: |",
        ]
        for name, dataset in summary["per_dataset"].items():
            latency = dataset["e2e_latency"]
            lines.append(
                f"| {name} | {dataset['successful_tasks']} | {value(latency['mean_ms'], ' ms')} | "
                f"{value(latency['p95_ms'], ' ms')} | {dataset['average_e2e_target_met']} |"
            )
        lines.extend(
            [
                "",
                "## 口径说明",
                "",
                "端到端时延从读取单张图像前开始，到最终检测结果就绪结束，不包含可视化和文件落盘。断网限时业务保持率的分母包含失败与超时任务；决策需在业务时限内就绪，且随后的本地事件同步写入必须成功，写入耗时本身不计入 200ms。该指标是软件链路代理，尚不包含 PLC 动作确认。冲突解决成功率表示冲突发生后云端仲裁成功返回有效结果的比例；最终标签与数据集路径真值的一致率另列为推理正确率。当前边缘模型为确定性占位模拟器，因此只能验证业务连续性，不能证明工业检测精度。",
                "",
            ]
        )
        return "\n".join(lines)

    def finalize(self) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        if self._detail_file is not None and not self._detail_file.closed:
            self._detail_file.close()
        summary = self._build_summary()
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.report_path.write_text(self._markdown(summary), encoding="utf-8")
        return summary
