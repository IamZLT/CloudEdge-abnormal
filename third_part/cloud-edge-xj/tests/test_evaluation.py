import os
import tempfile
import threading
import unittest
from dataclasses import replace
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from common.config import Config, DatasetConfig
from common.metrics import MetricsCollector
from common.schemas import DetectionResult
from common.visualization import ResultVisualizer
from cloud.gateway_client import CloudGatewayClient
from cloud.gateway_server import CloudGatewayApplication, create_handler
from cloud.model_api import LargeModelAPI
from edge.augmentations import apply_augmentation
from edge.inference_engine import InferenceEngine
from edge.edge_service import EdgeService
from edge.multi_node_service import MultiNodeEdgeService
from control.circuit_breaker import CloudCircuitBreaker
from control.decision_policy import DecisionPolicy
from control.sync_handler import SyncHandler


class EvaluationTest(unittest.TestCase):
    def test_single_edge_falls_back_and_circuit_breaker_skips_repeated_network_calls(self):
        class LowConfidenceEngine:
            def predict(self, image):
                return DetectionResult(
                    label="normal",
                    confidence=0.1,
                    source="edge",
                )

        class OfflineCloud:
            def __init__(self):
                self.calls = 0
                self.last_call_metrics = {}

            def review_result(self, result, image_path):
                self.calls += 1
                raise ConnectionError("simulated disconnect")

        with tempfile.TemporaryDirectory() as temporary_dir:
            image_path = Path(temporary_dir) / "image.png"
            image = np.indices((32, 32)).sum(axis=0) % 2 * 255
            image = np.repeat(image[:, :, None], 3, axis=2).astype(np.uint8)
            self.assertTrue(
                cv2.imwrite(
                    str(image_path),
                    image,
                )
            )
            config = replace(
                Config.load(),
                multi_edge_enabled=False,
                quality_score_threshold=-1.0,
                quality_cloud_ratio=None,
            )
            cloud = OfflineCloud()
            sync_handler = SyncHandler(
                cloud,
                CloudCircuitBreaker(
                    enabled=True,
                    failure_threshold=1,
                    recovery_timeout_seconds=60,
                ),
            )
            service = EdgeService(
                config,
                LowConfidenceEngine(),
                DecisionPolicy(config),
                sync_handler,
            )

            first = service.handle_task(str(image_path))
            second = service.handle_task(str(image_path))

            self.assertEqual(cloud.calls, 1)
            self.assertEqual(first.label, "anomaly")
            self.assertTrue(first.metadata["resilience"]["fallback_used"])
            self.assertEqual(first.metadata["resilience"]["business_action"], "divert_for_review")
            self.assertTrue(second.metadata["resilience"]["circuit_short_circuited"])
            self.assertTrue(second.metadata["resilience"]["business_completed"])

    def test_image_quality_router_replaces_confidence_routing(self):
        config = replace(
            Config.load(),
            quality_score_threshold=0.2,
            quality_cloud_ratio=None,
        )
        policy = DecisionPolicy(config)
        high_complexity = np.random.default_rng(0).integers(
            0,
            256,
            size=(64, 64, 3),
            dtype=np.uint8,
        )
        low_complexity = np.full((64, 64, 3), 127, dtype=np.uint8)

        complex_result = DetectionResult(label="normal", confidence=0.99)
        simple_result = DetectionResult(label="normal", confidence=0.01)
        policy.attach_quality_score(complex_result, high_complexity)
        policy.attach_quality_score(simple_result, low_complexity)

        complex_decision = policy.should_upload(complex_result)
        simple_decision = policy.should_upload(simple_result)

        self.assertTrue(complex_decision.should_upload)
        self.assertEqual(complex_decision.reason, "high_image_complexity")
        self.assertEqual(complex_decision.policy, "image_quality")
        self.assertFalse(simple_decision.should_upload)
        self.assertEqual(simple_decision.reason, "low_image_complexity")

    def test_quality_router_calibrates_threshold_from_dataset_ratio(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            simple_path = Path(temporary_dir) / "simple.png"
            complex_path = Path(temporary_dir) / "complex.png"
            self.assertTrue(
                cv2.imwrite(str(simple_path), np.full((32, 32, 3), 127, dtype=np.uint8))
            )
            noisy = np.random.default_rng(0).integers(
                0,
                256,
                size=(32, 32, 3),
                dtype=np.uint8,
            )
            self.assertTrue(cv2.imwrite(str(complex_path), noisy))
            config = replace(
                Config.load(),
                quality_cloud_ratio=0.5,
            )
            policy = DecisionPolicy(config)
            threshold = policy.calibrate_quality_threshold(
                [str(simple_path), str(complex_path)]
            )
            self.assertIsNotNone(threshold)
            self.assertEqual(len(policy.last_calibration_errors), 0)

            simple_result = DetectionResult(label="normal", confidence=0.99)
            complex_result = DetectionResult(label="normal", confidence=0.99)
            policy.attach_quality_score(simple_result, cv2.imread(str(simple_path)), str(simple_path))
            policy.attach_quality_score(complex_result, cv2.imread(str(complex_path)), str(complex_path))

            self.assertFalse(policy.should_upload(simple_result).should_upload)
            self.assertTrue(policy.should_upload(complex_result).should_upload)

    def test_edge_uses_http_gateway_and_uploads_original_image_bytes(self):
        class FakeCloudService:
            def __init__(self):
                self.calls = []

            def inspect_image_bytes(self, image_bytes, mime_type):
                self.calls.append((image_bytes, mime_type))
                return DetectionResult(
                    label="normal",
                    confidence=0.95,
                    source="cloud",
                    metadata={},
                )

        with tempfile.TemporaryDirectory() as temporary_dir:
            image_path = Path(temporary_dir) / "original.png"
            image = np.arange(16 * 20 * 3, dtype=np.uint8).reshape(16, 20, 3)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            original_bytes = image_path.read_bytes()

            fake_cloud = FakeCloudService()
            application = CloudGatewayApplication(fake_cloud, max_upload_bytes=1024 * 1024)
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                create_handler(application, max_request_body_bytes=2 * 1024 * 1024),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            client = CloudGatewayClient(
                f"http://127.0.0.1:{server.server_port}",
                "test-edge",
                "test-run",
                timeout=5,
            )
            try:
                client.start_run()
                result = client.review_result(
                    DetectionResult(label="anomaly", confidence=0.1, source="edge"),
                    str(image_path),
                )
                run_metrics = client.finish_run()["metrics"]
            finally:
                client.session.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

            self.assertEqual(len(fake_cloud.calls), 1)
            self.assertEqual(fake_cloud.calls[0], (original_bytes, "image/png"))
            self.assertEqual(result.label, "normal")
            self.assertEqual(result.metadata["gateway_metrics"]["context_keys"], [])
            communication = result.metadata["communication_metrics"]
            self.assertEqual(communication["protocol"], "http_multipart")
            self.assertEqual(communication["original_image_bytes"], len(original_bytes))
            self.assertGreater(communication["request_body_bytes"], len(original_bytes))
            self.assertEqual(run_metrics["request_count"], 1)
            self.assertEqual(
                run_metrics["request_body_bytes"],
                communication["request_body_bytes"],
            )

    def test_multi_node_conflict_uses_cloud_resolution(self):
        class ThresholdEngine:
            def predict(self, image):
                scale = 255.0 if np.issubdtype(image.dtype, np.integer) else 1.0
                label = "normal" if float(np.mean(image)) / scale >= 0.5 else "anomaly"
                return DetectionResult(label=label, confidence=0.9, source="edge")

        class FakeSyncHandler:
            def upload_result(self, result, image_path):
                return DetectionResult(
                    label="normal",
                    confidence=0.95,
                    source="cloud",
                    metadata={},
                )

        with tempfile.TemporaryDirectory() as temporary_dir:
            image_path = Path(temporary_dir) / "image.png"
            image = np.full((16, 16, 3), 125, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            config = Config.load()
            service = MultiNodeEdgeService(
                config,
                ThresholdEngine(),
                DecisionPolicy(config),
                FakeSyncHandler(),
            )
            result = service.handle_task(str(image_path))
            self.assertEqual(result.label, "normal")
            self.assertEqual(result.source, "cloud")
            self.assertTrue(result.metadata["multi_edge"]["conflict"])
            self.assertTrue(result.metadata["multi_edge"]["resolution_success"])

    def test_cloud_call_records_transport_and_token_metrics(self):
        class FakeCloudClient:
            last_call_metrics = {
                "request_body_bytes": 456,
                "response_body_bytes": 123,
                "http_round_trip_ms": 20.0,
                "http_status": 200,
            }

            def post(self, payload):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": '{"label":"normal","confidence":0.9,'
                                '"defect category":null,"bbox":null,"metadata":{}}'
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                }

        with tempfile.TemporaryDirectory() as temporary_dir:
            image_path = Path(temporary_dir) / "image.png"
            self.assertTrue(cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8)))
            api = LargeModelAPI(FakeCloudClient(), "test-model")
            result = api.call_large_model(
                DetectionResult(label="normal", confidence=0.9),
                str(image_path),
            )
            cloud_metrics = result.metadata["cloud_metrics"]
            self.assertEqual(cloud_metrics["request_body_bytes"], 456)
            self.assertEqual(cloud_metrics["total_tokens"], 15)

    def test_placeholder_is_deterministic_and_augmentations_keep_resolution(self):
        image = np.full((24, 32, 3), 0.5, dtype=np.float32)
        engine = InferenceEngine()
        first = engine.predict(image)
        second = engine.predict(image)
        self.assertEqual(first.label, second.label)
        self.assertEqual(first.confidence, second.confidence)
        for name, parameter in (("none", None), ("brightness", 1.05), ("contrast", 0.95)):
            augmented = apply_augmentation(image, name, parameter)
            self.assertEqual(augmented.shape, image.shape)

    def test_visualizer_can_replace_read_only_previous_output(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir) / "dataset"
            source = root / "category" / "test" / "good" / "000.png"
            source.parent.mkdir(parents=True)
            image = np.zeros((30, 40, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(source), image))
            visualizer = ResultVisualizer(str(Path(temporary_dir) / "visualizations"))
            result = DetectionResult(label="normal", confidence=0.9)
            destination = Path(visualizer.save("dataset", str(root), str(source), result))
            os.chmod(destination, 0o440)
            visualizer.save("dataset", str(root), str(source), result)
            self.assertEqual(source.read_bytes(), destination.read_bytes())
            self.assertTrue(destination.stat().st_mode & 0o200)

    def test_metrics_formulas_and_pdf_targets(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir) / "MVTec"
            source = root / "bottle" / "test" / "good" / "000.png"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"abc")
            dataset = DatasetConfig(
                name="MVTec",
                dataset_type="mvtec",
                root=str(root),
                split="test",
                categories=[],
                subsets=[],
                max_images=None,
            )
            config = replace(
                Config.load(),
                metrics_summary_path=str(Path(temporary_dir) / "summary.json"),
                metrics_detail_path=str(Path(temporary_dir) / "events.jsonl"),
                metrics_report_path=str(Path(temporary_dir) / "report.md"),
            )
            result = DetectionResult(
                label="normal",
                confidence=0.95,
                source="cloud",
                metadata={
                    "cloud_metrics": {
                        "request_body_bytes": 100,
                        "response_body_bytes": 20,
                        "http_round_trip_ms": 50.0,
                        "cloud_total_ms": 55.0,
                        "gpu": {
                            "utilization_samples_percent": [10.0, 20.0],
                            "memory_used_samples_mib": [1000.0, 1100.0],
                            "memory_total_mib": 24000.0,
                            "errors": [],
                        },
                    },
                    "communication_metrics": {
                        "protocol": "http_multipart",
                        "request_body_bytes": 100,
                        "response_body_bytes": 20,
                        "edge_gateway_round_trip_ms": 60.0,
                    },
                    "multi_edge": {
                        "cloud_attempted": True,
                        "cloud_success": True,
                        "conflict": True,
                        "resolution_success": True,
                        "nodes": [
                            {"node_id": "edge-1", "label": "normal", "latency_ms": 2.0},
                            {"node_id": "edge-2", "label": "anomaly", "latency_ms": 3.0},
                        ],
                    },
                },
            )
            collector = MetricsCollector(config)
            collector.record_success(dataset, str(source), result, 125.0)
            collector.set_gateway_run_metrics(
                {
                    "elapsed_seconds": 2.0,
                    "request_count": 1,
                    "successful_request_count": 1,
                    "failed_request_count": 0,
                    "request_body_bytes": 100,
                    "image_bytes": 3,
                    "gpu": {
                        "gpu_index": 1,
                        "scope": "test",
                        "sample_count": 2,
                        "average_utilization_percent": 40.0,
                        "p95_utilization_percent": 49.0,
                        "peak_utilization_percent": 50.0,
                        "gpu_seconds": 0.8,
                        "average_memory_used_mib": 1200.0,
                        "peak_memory_used_mib": 1300.0,
                        "memory_total_mib": 24000.0,
                        "errors": [],
                    },
                }
            )
            summary = collector.finalize()
            self.assertEqual(
                summary["end_to_end_latency"]["all_successful_tasks"]["mean_ms"],
                125.0,
            )
            self.assertTrue(summary["end_to_end_latency"]["average_target_met"])
            consistency = summary["stability_and_consistency"]
            self.assertEqual(consistency["conflict_ratio_percent"], 100.0)
            self.assertEqual(consistency["conflict_resolution_accuracy_percent"], 100.0)
            communication = summary["communication_efficiency"]
            self.assertEqual(communication["all_cloud_baseline_original_image_bytes"], 3)
            self.assertEqual(communication["actual_request_body_bytes"], 100)
            self.assertEqual(communication["estimated_all_cloud_request_body_bytes"], 100)
            self.assertEqual(communication["average_bandwidth_occupancy_percent"], 0.00004)
            gpu = summary["cloud_compute_load_proxy"]["server_gpu"]
            self.assertEqual(gpu["average_utilization_percent"], 40.0)
            self.assertEqual(gpu["peak_memory_used_mib"], 1300.0)
            primary = summary["resource_and_communication_efficiency"]
            self.assertEqual(primary["data_upload"]["http_request_body_bytes"], 100)
            self.assertEqual(primary["bandwidth"]["measurement_window_seconds"], 2.0)

    def test_offline_business_retention_uses_all_tasks_and_accepts_exactly_90_percent(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir) / "MVTec"
            source = root / "bottle" / "test" / "good" / "000.png"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"abc")
            dataset = DatasetConfig(
                name="MVTec",
                dataset_type="mvtec",
                root=str(root),
                split="test",
                categories=[],
                subsets=[],
                max_images=None,
            )
            config = replace(
                Config.load(),
                network_outage_simulation_enabled=True,
                business_deadline_ms=200.0,
                business_retention_target_percent=90.0,
                metrics_summary_path=str(Path(temporary_dir) / "summary.json"),
                metrics_detail_path=str(Path(temporary_dir) / "events.jsonl"),
                metrics_report_path=str(Path(temporary_dir) / "report.md"),
            )
            collector = MetricsCollector(config)
            result = DetectionResult(
                label="anomaly",
                confidence=0.2,
                source="edge",
                metadata={
                    "resilience": {
                        "cloud_required": True,
                        "cloud_attempted": False,
                        "cloud_success": False,
                        "fallback_used": True,
                        "provisional": True,
                        "business_completed": True,
                        "circuit_short_circuited": True,
                        "cloud_error": "offline",
                        "cloud_wait_ms": 0.1,
                    }
                },
            )
            for index in range(10):
                collector.record_success(
                    dataset,
                    str(source),
                    result,
                    100.0 if index < 9 else 250.0,
                )
            summary = collector.finalize()
            resilience = summary["network_resilience"]
            self.assertEqual(
                resilience["deadline_business_retention_rate_percent"],
                90.0,
            )
            self.assertTrue(resilience["business_retention_target_met"])
            self.assertEqual(
                resilience["cloud_required_deadline_retention_rate_percent"],
                90.0,
            )


if __name__ == "__main__":
    unittest.main()
