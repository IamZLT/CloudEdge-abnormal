from time import perf_counter_ns
from uuid import uuid4

from common.config import Config
from common.dataset_loader import discover_dataset_images
from common.metrics import MetricsCollector
from common.result_writer import JsonlResultWriter
from common.visualization import ResultVisualizer
from common.utils import log_info
from edge.inference_engine import InferenceEngine
from edge.edge_service import EdgeService
from edge.multi_node_service import MultiNodeEdgeService
from control.decision_policy import DecisionPolicy
from control.circuit_breaker import CloudCircuitBreaker
from control.sync_handler import SyncHandler
from cloud.gateway_client import CloudGatewayClient


def main():
    config = Config.load()
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
        recovery_timeout_seconds=config.circuit_breaker_recovery_timeout_seconds,
    )
    sync_handler = SyncHandler(
        cloud_service=gateway_client,
        circuit_breaker=circuit_breaker,
    )

    decision_policy = DecisionPolicy(config=config)
    inference_engine = InferenceEngine(model_path=config.edge_model_path)

    edge_service = EdgeService(
        config=config,
        inference_engine=inference_engine,
        decision_policy=decision_policy,
        sync_handler=sync_handler,
    )
    visualizer = ResultVisualizer(config.visualization_output_dir)
    processor = (
        MultiNodeEdgeService(
            config=config,
            inference_engine=inference_engine,
            decision_policy=decision_policy,
            sync_handler=sync_handler,
        )
        if config.multi_edge_enabled
        else edge_service
    )
    metrics = MetricsCollector(config, run_id)
    try:
        gateway_client.start_run()
    except Exception as exc:
        sync_handler.mark_cloud_unavailable(exc)
        log_info(
            "Cloud gateway run registration failed; edge fallback remains available",
            {"gateway_url": config.cloud_gateway_url, "error": str(exc)},
        )

    success_count = 0
    error_count = 0
    dataset_images = []
    for dataset in config.datasets:
        try:
            image_paths = discover_dataset_images(dataset, config.image_extensions)
            dataset_images.append((dataset, image_paths))
        except Exception as exc:
            error_count += 1
            dataset_images.append((dataset, exc))

    calibration_paths = [
        image_path
        for _, image_paths in dataset_images
        if isinstance(image_paths, list)
        for image_path in image_paths
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
                "calibration_image_count": len(calibration_paths),
                "calibration_sample_count": calibration_sample_count,
                "calibration_error_count": len(decision_policy.last_calibration_errors),
            },
        )

    with JsonlResultWriter(config.output_path, config.output_append) as result_writer:
        log_info("Result output initialized", {"output_path": str(result_writer.path)})
        for dataset, image_paths in dataset_images:
            if isinstance(image_paths, Exception):
                result_writer.write_error(dataset.name, None, str(image_paths))
                log_info("Dataset loading failed", {"dataset": dataset.name, "error": str(image_paths)})
                continue

            log_info("Dataset loaded", {"dataset": dataset.name, "image_count": len(image_paths)})
            for image_path in image_paths:
                task_started_ns = perf_counter_ns()
                try:
                    result = processor.handle_task(image_path)
                    e2e_latency_ms = (perf_counter_ns() - task_started_ns) / 1_000_000
                except Exception as exc:
                    error_count += 1
                    elapsed_ms = (perf_counter_ns() - task_started_ns) / 1_000_000
                    metrics.record_error(dataset, image_path, str(exc), elapsed_ms)
                    result_writer.write_error(dataset.name, image_path, str(exc))
                    log_info(
                        "Task failed",
                        {"dataset": dataset.name, "image_path": image_path, "error": str(exc)},
                    )
                    continue

                visualization_path = None
                visualization_error = None
                try:
                    visualization_path = visualizer.save(
                        dataset_name=dataset.name,
                        dataset_root=dataset.root,
                        image_path=image_path,
                        result=result,
                    )
                except Exception as exc:
                    visualization_error = str(exc)
                    log_info(
                        "Visualization failed",
                        {
                            "dataset": dataset.name,
                            "image_path": image_path,
                            "error": visualization_error,
                        },
                    )

                evaluation = metrics.record_success(
                    dataset,
                    image_path,
                    result,
                    e2e_latency_ms,
                )
                success_count += 1
                result_writer.write_success(
                    dataset.name,
                    image_path,
                    result,
                    visualization_path=visualization_path,
                    visualization_error=visualization_error,
                    evaluation=evaluation or None,
                )
                log_info(
                    "Final result",
                    {
                        "dataset": dataset.name,
                        "image_path": image_path,
                        "result": result.to_dict(),
                        "e2e_latency_ms": e2e_latency_ms,
                        "visualization_path": visualization_path,
                        "visualization_error": visualization_error,
                    },
                )

    if gateway_client.run_registered:
        try:
            gateway_run = gateway_client.finish_run()
            metrics.set_gateway_run_metrics(gateway_run.get("metrics"))
        except Exception as exc:
            log_info("Cloud gateway metrics collection failed", {"error": str(exc)})
    gateway_client.close()

    metrics_summary = metrics.finalize()
    if metrics_summary:
        log_info(
            "Evaluation metrics",
            {
                "summary_path": str(metrics.summary_path),
                "detail_path": str(metrics.detail_path),
                "report_path": str(metrics.report_path),
                "end_to_end_latency": metrics_summary["end_to_end_latency"],
                "cloud_compute_load_proxy": metrics_summary["cloud_compute_load_proxy"],
                "stability_and_consistency": metrics_summary["stability_and_consistency"],
            },
        )

    log_info(
        "Processing completed",
        {
            "success_count": success_count,
            "error_count": error_count,
            "output_path": config.output_path,
            "visualization_output_dir": config.visualization_output_dir,
            "metrics_summary_path": config.metrics_summary_path,
            "metrics_detail_path": config.metrics_detail_path,
            "metrics_report_path": config.metrics_report_path,
        },
    )


if __name__ == "__main__":
    main()
