from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    dataset_type: str
    root: str
    split: str
    categories: List[str]
    subsets: List[str]
    max_images: Optional[int]


@dataclass(frozen=True)
class EdgeNodeConfig:
    node_id: str
    augmentation: str
    parameter: Optional[float]


@dataclass(frozen=True)
class Config:
    device_id: str
    cloud_gateway_url: str
    cloud_gateway_host: str
    cloud_gateway_port: int
    cloud_gateway_max_upload_bytes: int
    cloud_model_api_base_url: str
    # 保留旧字段名作为云端内部模型 API 地址的兼容别名。
    cloud_base_url: str
    cloud_model: str
    cloud_api_key: Optional[str]
    cloud_gateway_read_timeout: float
    cloud_timeout: float
    cloud_connect_timeout: float
    cloud_max_tokens: int
    cloud_temperature: float
    cloud_gpu_index: Optional[int]
    cloud_gpu_sample_interval: float
    circuit_breaker_enabled: bool
    circuit_breaker_failure_threshold: int
    circuit_breaker_recovery_timeout_seconds: float
    offline_fail_safe_enabled: bool
    quality_score_threshold: float
    quality_cloud_ratio: Optional[float]
    quality_calibration_max_images: Optional[int]
    quality_calibration_workers: int
    quality_score_size: int
    quality_jpeg_quality: int
    quality_weights: Dict[str, float]
    max_history: int
    edge_model_path: Optional[str]
    target_size: Optional[Tuple[int, int]]
    datasets: List[DatasetConfig]
    image_extensions: Tuple[str, ...]
    output_path: str
    output_append: bool
    visualization_output_dir: str
    evaluation_enabled: bool
    evaluation_run_label: str
    evaluation_sample_description: str
    network_outage_simulation_enabled: bool
    business_retention_target_percent: float
    business_deadline_ms: float
    network_link_capacity_mbps: Optional[float]
    metrics_summary_path: str
    metrics_detail_path: str
    metrics_report_path: str
    multi_edge_enabled: bool
    conflict_resolution: str
    edge_nodes: List[EdgeNodeConfig]

    @staticmethod
    def load(path: str = "config.yaml") -> "Config":
        config_path = Path(path)
        if not config_path.is_absolute():
            config_path = Path(__file__).resolve().parent.parent / config_path

        with config_path.open("r", encoding="utf-8") as file:
            data: Dict[str, Any] = yaml.safe_load(file) or {}

        device = data.get("device", {})
        edge = data.get("edge", {})
        cloud = data.get("cloud", {})
        policy = data.get("policy", {})
        task = data.get("task", {})
        app = data.get("app", {})
        evaluation = data.get("evaluation", {})
        resilience = data.get("resilience", {})
        outage_evaluation = evaluation.get("network_outage", {})
        multi_edge = evaluation.get("multi_edge", {})
        target_size = edge.get("target_size")
        cloud_gpu_index = cloud.get("gpu_index")
        model_api_base_url = str(
            cloud.get(
                "model_api_base_url",
                cloud.get("base_url", "http://127.0.0.1:7788/v1"),
            )
        ).rstrip("/")
        model_timeout = float(cloud.get("model_timeout", cloud.get("timeout", 60)))
        link_capacity_mbps = evaluation.get("link_capacity_mbps")
        datasets = []
        for item in data.get("datasets", []):
            max_images = item.get("max_images")
            datasets.append(
                DatasetConfig(
                    name=str(item.get("name", item.get("type", "dataset"))),
                    dataset_type=str(item.get("type", "generic")).lower(),
                    root=str(item["root"]),
                    split=str(item.get("split", "test")),
                    categories=[str(value) for value in item.get("categories", [])],
                    subsets=[str(value) for value in item.get("subsets", [])],
                    max_images=int(max_images) if max_images is not None else None,
                )
            )

        edge_nodes = []
        for index, item in enumerate(multi_edge.get("nodes", []), start=1):
            parameter = item.get("parameter")
            edge_nodes.append(
                EdgeNodeConfig(
                    node_id=str(item.get("id", f"edge-node-{index:03d}")),
                    augmentation=str(item.get("augmentation", "none")).lower(),
                    parameter=float(parameter) if parameter is not None else None,
                )
            )

        return Config(
            device_id=str(device.get("id", "edge-device-001")),
            cloud_gateway_url=str(
                cloud.get("gateway_url", "http://127.0.0.1:7790")
            ).rstrip("/"),
            cloud_gateway_host=str(cloud.get("gateway_host", "0.0.0.0")),
            cloud_gateway_port=int(cloud.get("gateway_port", 7790)),
            cloud_gateway_max_upload_bytes=int(
                cloud.get("gateway_max_upload_bytes", 100 * 1024 * 1024)
            ),
            cloud_model_api_base_url=model_api_base_url,
            cloud_base_url=model_api_base_url,
            cloud_model=str(cloud.get("model", "qwen25vl")),
            cloud_api_key=cloud.get("api_key") or None,
            cloud_gateway_read_timeout=float(
                cloud.get("gateway_read_timeout", model_timeout + 5.0)
            ),
            cloud_timeout=model_timeout,
            cloud_connect_timeout=float(cloud.get("connect_timeout", 0.5)),
            cloud_max_tokens=int(cloud.get("max_tokens", 512)),
            cloud_temperature=float(cloud.get("temperature", 0.1)),
            cloud_gpu_index=int(cloud_gpu_index) if cloud_gpu_index is not None else None,
            cloud_gpu_sample_interval=float(cloud.get("gpu_sample_interval", 0.25)),
            circuit_breaker_enabled=bool(
                resilience.get("circuit_breaker_enabled", True)
            ),
            circuit_breaker_failure_threshold=int(
                resilience.get("failure_threshold", 1)
            ),
            circuit_breaker_recovery_timeout_seconds=float(
                resilience.get("recovery_timeout_seconds", 10.0)
            ),
            offline_fail_safe_enabled=bool(
                resilience.get("fail_safe_uncertain_as_anomaly", True)
            ),
            quality_score_threshold=float(
                policy.get(
                    "quality_score_threshold",
                    policy.get("complexity_score_threshold", 0.5),
                )
            ),
            quality_cloud_ratio=(
                float(policy["quality_cloud_ratio"])
                if policy.get("quality_cloud_ratio") is not None
                else None
            ),
            quality_calibration_max_images=(
                int(policy["quality_calibration_max_images"])
                if policy.get("quality_calibration_max_images") is not None
                else None
            ),
            quality_calibration_workers=int(
                policy.get("quality_calibration_workers", 8)
            ),
            quality_score_size=int(policy.get("quality_score_size", 192)),
            quality_jpeg_quality=int(policy.get("quality_jpeg_quality", 30)),
            quality_weights={
                str(key): float(value)
                for key, value in (policy.get("quality_weights") or {}).items()
            },
            max_history=int(task.get("max_history", 100)),
            edge_model_path=edge.get("model_path") or None,
            target_size=(int(target_size[0]), int(target_size[1])) if target_size else None,
            datasets=datasets,
            image_extensions=tuple(
                str(extension).lower() for extension in app.get("image_extensions", [".png", ".jpg", ".jpeg"])
            ),
            output_path=str(app.get("output_path", "outputs/results.jsonl")),
            output_append=bool(app.get("output_append", False)),
            visualization_output_dir=str(
                app.get("visualization_output_dir", "outputs/visualizations")
            ),
            evaluation_enabled=bool(evaluation.get("enabled", True)),
            evaluation_run_label=str(evaluation.get("run_label", "full_dataset")),
            evaluation_sample_description=str(
                evaluation.get(
                    "sample_description",
                    "处理 datasets 配置选中的全部图像",
                )
            ),
            network_outage_simulation_enabled=bool(
                outage_evaluation.get("enabled", False)
            ),
            business_retention_target_percent=float(
                outage_evaluation.get("business_retention_target_percent", 90.0)
            ),
            business_deadline_ms=float(
                outage_evaluation.get("business_deadline_ms", 200.0)
            ),
            network_link_capacity_mbps=(
                float(link_capacity_mbps) if link_capacity_mbps is not None else None
            ),
            metrics_summary_path=str(
                evaluation.get("summary_output_path", "outputs/metrics/summary.json")
            ),
            metrics_detail_path=str(
                evaluation.get("detail_output_path", "outputs/metrics/events.jsonl")
            ),
            metrics_report_path=str(
                evaluation.get("report_output_path", "outputs/metrics/report.md")
            ),
            multi_edge_enabled=bool(multi_edge.get("enabled", False)),
            conflict_resolution=str(multi_edge.get("conflict_resolution", "cloud")).lower(),
            edge_nodes=edge_nodes,
        )
