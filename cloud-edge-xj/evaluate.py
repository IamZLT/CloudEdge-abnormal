import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter_ns
from typing import List
from uuid import uuid4

from cloud.gateway_client import CloudGatewayClient
from common.config import Config
from common.dataset_loader import discover_dataset_images
from common.metrics import MetricsCollector
from common.utils import log_info
from control.circuit_breaker import CloudCircuitBreaker
from control.decision_policy import DecisionPolicy
from control.sync_handler import SyncHandler
from edge.edge_service import EdgeService
from edge.inference_engine import InferenceEngine
from edge.multi_node_service import MultiNodeEdgeService


def _uniform_sample(paths: List[str], sample_count: int) -> List[str]:
    if sample_count >= len(paths):
        return paths
    if sample_count == 1:
        return [paths[len(paths) // 2]]
    return [
        paths[round(index * (len(paths) - 1) / (sample_count - 1))]
        for index in range(sample_count)
    ]


def run_evaluation(
    config_path: str,
    samples_per_dataset: int,
    output_dir: str,
    simulate_outage: bool = False,
) -> dict:
    if samples_per_dataset <= 0:
        raise ValueError("samples_per_dataset 必须大于 0")
    config = Config.load(config_path)
    output_root = Path(output_dir)
    config = replace(
        config,
        evaluation_run_label=(
            f"offline_smoke_{samples_per_dataset}_per_dataset"
            if simulate_outage
            else f"smoke_{samples_per_dataset}_per_dataset"
        ),
        evaluation_sample_description=(
            f"每个数据集按路径顺序均匀抽取 {samples_per_dataset} 张；"
            + (
                "100% 模拟断网并强制所有任务请求云复核；"
                if simulate_outage
                else ""
            )
            + "非全量正式结果"
        ),
        network_outage_simulation_enabled=simulate_outage,
        quality_cloud_ratio=(1.0 if simulate_outage else config.quality_cloud_ratio),
        metrics_summary_path=str(output_root / "summary.json"),
        metrics_detail_path=str(output_root / "events.jsonl"),
        metrics_report_path=str(output_root / "report.md"),
    )
    run_id = uuid4().hex
    gateway_client = CloudGatewayClient(
        config.cloud_gateway_url,
        config.device_id,
        run_id,
        config.cloud_gateway_read_timeout,
        config.cloud_connect_timeout,
    )
    circuit_breaker = CloudCircuitBreaker(
        enabled=config.circuit_breaker_enabled,
        failure_threshold=config.circuit_breaker_failure_threshold,
        recovery_timeout_seconds=(
            1_000_000_000.0
            if simulate_outage
            else config.circuit_breaker_recovery_timeout_seconds
        ),
    )
    sync_handler = SyncHandler(gateway_client, circuit_breaker)
    decision_policy = DecisionPolicy(config)
    inference_engine = InferenceEngine(config.edge_model_path)
    edge_service = EdgeService(
        config,
        inference_engine,
        decision_policy,
        sync_handler,
    )
    processor = (
        MultiNodeEdgeService(
            config,
            inference_engine,
            decision_policy,
            sync_handler,
        )
        if config.multi_edge_enabled
        else edge_service
    )
    metrics = MetricsCollector(config, run_id)
    sampled_dataset_images = []
    for dataset in config.datasets:
        paths = discover_dataset_images(dataset, config.image_extensions)
        selected_paths = _uniform_sample(paths, samples_per_dataset)
        sampled_dataset_images.append((dataset, paths, selected_paths))

    calibration_paths = [
        image_path
        for _, _, selected_paths in sampled_dataset_images
        for image_path in selected_paths
    ]
    calibration_sample_count = decision_policy.calibration_sample_count(
        calibration_paths
    )
    if config.quality_cloud_ratio is not None:
        log_info(
            "Calibrating image-quality routing threshold",
            {
                "quality_cloud_ratio": config.quality_cloud_ratio,
                "total_image_count": len(calibration_paths),
                "calibration_sample_count": calibration_sample_count,
                "calibration_workers": config.quality_calibration_workers,
            },
        )
    calibrated_threshold = decision_policy.calibrate_quality_threshold(
        calibration_paths
    )
    if calibrated_threshold is not None:
        log_info(
            "Image-quality routing threshold calibrated",
            {
                "quality_cloud_ratio": config.quality_cloud_ratio,
                "quality_score_threshold": calibrated_threshold,
                "calibration_sample_count": calibration_sample_count,
                "calibration_error_count": len(decision_policy.last_calibration_errors),
            },
        )

    if simulate_outage:
        sync_handler.mark_cloud_unavailable("simulated_full_disconnect")
        log_info(
            "Network outage simulation enabled; all cloud reviews use edge fallback",
            {"gateway_url": config.cloud_gateway_url},
        )
    else:
        try:
            gateway_client.start_run()
        except Exception as exc:
            sync_handler.mark_cloud_unavailable(exc)
            log_info(
                "Cloud gateway run registration failed; evaluation will use edge fallback",
                {"gateway_url": config.cloud_gateway_url, "error": str(exc)},
            )

    for dataset, paths, selected_paths in sampled_dataset_images:
        log_info(
            "Evaluation dataset sampled",
            {
                "dataset": dataset.name,
                "total_images": len(paths),
                "sampled_images": len(selected_paths),
            },
        )
        for image_path in selected_paths:
            started_ns = perf_counter_ns()
            try:
                result = processor.handle_task(image_path)
                e2e_latency_ms = (perf_counter_ns() - started_ns) / 1_000_000
                metrics.record_success(dataset, image_path, result, e2e_latency_ms)
            except Exception as exc:
                elapsed_ms = (perf_counter_ns() - started_ns) / 1_000_000
                metrics.record_error(dataset, image_path, str(exc), elapsed_ms)

    if gateway_client.run_registered:
        try:
            gateway_run = gateway_client.finish_run()
            metrics.set_gateway_run_metrics(gateway_run.get("metrics"))
        except Exception as exc:
            log_info("Cloud gateway metrics collection failed", {"error": str(exc)})
    gateway_client.close()
    summary = metrics.finalize()
    print(
        json.dumps(
            {
                "summary_path": str(metrics.summary_path),
                "detail_path": str(metrics.detail_path),
                "report_path": str(metrics.report_path),
                "tasks": summary["tasks"],
                "end_to_end_latency": summary["end_to_end_latency"],
                "network_resilience": summary["network_resilience"],
                "cloud_compute_load_proxy": summary["cloud_compute_load_proxy"],
                "stability_and_consistency": summary["stability_and_consistency"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="运行可复现的云边协同抽样评测")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--samples-per-dataset", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs/metrics/smoke")
    parser.add_argument(
        "--simulate-outage",
        action="store_true",
        help="不访问云端，模拟评测窗口内 100% 断网并验证边缘降级保持率",
    )
    args = parser.parse_args()
    run_evaluation(
        args.config,
        args.samples_per_dataset,
        args.output_dir,
        args.simulate_outage,
    )


if __name__ == "__main__":
    main()
