from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from time import perf_counter_ns
from typing import Dict, List, Tuple

import numpy as np

from common.config import Config, EdgeNodeConfig
from common.schemas import DetectionResult
from common.utils import current_timestamp, log_info
from control.decision_policy import DecisionPolicy
from control.sync_handler import SyncHandler
from edge.augmentations import apply_augmentation
from edge.image_reader import read_image
from edge.inference_engine import InferenceEngine
from edge.preprocessor import preprocess


class MultiNodeEdgeService:
    def __init__(
        self,
        config: Config,
        inference_engine: InferenceEngine,
        decision_policy: DecisionPolicy,
        sync_handler: SyncHandler,
    ):
        if len(config.edge_nodes) < 2:
            raise ValueError("启用多边缘节点评估时，evaluation.multi_edge.nodes 至少需要两个节点")
        if config.conflict_resolution != "cloud":
            raise ValueError("当前 conflict_resolution 仅支持 cloud")
        self.config = config
        self.inference_engine = inference_engine
        self.decision_policy = decision_policy
        self.sync_handler = sync_handler

    def _run_node(
        self,
        node: EdgeNodeConfig,
        image: np.ndarray,
    ) -> Tuple[DetectionResult, Dict[str, object]]:
        started_ns = perf_counter_ns()
        augmented = apply_augmentation(image, node.augmentation, node.parameter)
        result = self.inference_engine.predict(augmented)
        latency_ms = (perf_counter_ns() - started_ns) / 1_000_000
        result.timestamp = current_timestamp()
        result.source = f"edge:{node.node_id}"
        node_metric = {
            "node_id": node.node_id,
            "augmentation": node.augmentation,
            "augmentation_parameter": node.parameter,
            "label": result.label,
            "confidence": result.confidence,
            "latency_ms": latency_ms,
        }
        return result, node_metric

    @staticmethod
    def _select_edge_result(results: List[DetectionResult]) -> DetectionResult:
        counts = Counter(result.label for result in results)
        max_votes = max(counts.values())
        candidate_labels = {label for label, count in counts.items() if count == max_votes}
        return max(
            (result for result in results if result.label in candidate_labels),
            key=lambda result: result.confidence,
        )

    def handle_task(self, image_path: str) -> DetectionResult:
        read_started_ns = perf_counter_ns()
        original = read_image(image_path)
        read_ms = (perf_counter_ns() - read_started_ns) / 1_000_000
        preprocess_started_ns = perf_counter_ns()
        processed = preprocess(original)
        preprocess_ms = (perf_counter_ns() - preprocess_started_ns) / 1_000_000
        nodes_started_ns = perf_counter_ns()
        with ThreadPoolExecutor(max_workers=len(self.config.edge_nodes)) as executor:
            node_outputs = list(
                executor.map(
                    lambda node: self._run_node(node, processed),
                    self.config.edge_nodes,
                )
            )
        nodes_parallel_wall_ms = (perf_counter_ns() - nodes_started_ns) / 1_000_000

        aggregation_started_ns = perf_counter_ns()
        node_results = [output[0] for output in node_outputs]
        node_metrics = [output[1] for output in node_outputs]
        labels = sorted({result.label for result in node_results})
        conflict = len(labels) > 1
        selected = self._select_edge_result(node_results)
        self.decision_policy.attach_quality_score(selected, processed, image_path)
        original_edge_label = selected.label
        decision = self.decision_policy.should_upload(selected)
        cloud_attempted = conflict or decision.should_upload
        cloud_success = False
        cloud_error = None
        failed_communication_metrics = None
        cloud_attempt_latency_ms = None
        final_result = selected
        offload_reason = None

        if cloud_attempted:
            offload_reason = "node_conflict" if conflict else decision.reason
            circuit_state = dict(
                getattr(self.sync_handler, "circuit_state", {}) or {}
            ).get("state")
            if circuit_state != "open":
                log_info(
                    "Uploading image to cloud",
                    {"reason": offload_reason, "image_path": image_path},
                )
            cloud_started_ns = perf_counter_ns()
            try:
                final_result = self.sync_handler.upload_result(selected, image_path)
                cloud_success = True
            except Exception as exc:
                # 弱网或云端失败时保留边缘多数决策，业务仍可继续，并单独统计仲裁失败。
                cloud_error = str(exc)
                failed_communication_metrics = dict(
                    getattr(self.sync_handler, "last_call_metrics", {}) or {}
                )
                resilience_event = dict(
                    getattr(self.sync_handler, "last_resilience_event", {}) or {}
                )
                if not resilience_event.get("circuit_short_circuited"):
                    log_info(
                        "Cloud arbitration failed; using edge consensus",
                        {"image_path": image_path, "error": cloud_error},
                    )
            cloud_attempt_latency_ms = (perf_counter_ns() - cloud_started_ns) / 1_000_000

        aggregation_ms = (perf_counter_ns() - aggregation_started_ns) / 1_000_000

        if cloud_attempted and not cloud_success and self.config.offline_fail_safe_enabled:
            final_result.label = "anomaly"
            final_result.defect_category = (
                final_result.defect_category or "requires_cloud_review"
            )

        metadata = dict(final_result.metadata or {})
        if final_result is not selected:
            metadata["image_quality"] = selected.metadata.get("image_quality")
        metadata["routing"] = asdict(decision)
        resilience = (
            dict(getattr(self.sync_handler, "last_resilience_event", {}) or {})
            if cloud_attempted
            else {
                "cloud_attempted": False,
                "cloud_success": False,
                "circuit_state_after": dict(
                    getattr(self.sync_handler, "circuit_state", {}) or {}
                ).get("state", "unknown"),
            }
        )
        resilience.update(
            {
                "cloud_required": cloud_attempted,
                "fallback_used": cloud_attempted and not cloud_success,
                "fallback_source": "edge_consensus"
                if cloud_attempted and not cloud_success
                else None,
                "fallback_reason": "cloud_unavailable"
                if cloud_attempted and not cloud_success
                else None,
                "provisional": cloud_attempted and not cloud_success,
                "requires_cloud_review": cloud_attempted and not cloud_success,
                "original_edge_label": original_edge_label,
                "business_completed": True,
                "business_action": (
                    "pass" if final_result.label == "normal" else "divert_for_review"
                    if cloud_attempted and not cloud_success
                    else "divert"
                ),
            }
        )
        metadata["resilience"] = resilience
        metadata["multi_edge"] = {
            "simulation": True,
            "node_count": len(node_results),
            "nodes": node_metrics,
            "conflict": conflict,
            "unique_labels": labels,
            "cloud_attempted": cloud_attempted,
            "cloud_success": cloud_success,
            "cloud_error": cloud_error,
            "failed_communication_metrics": failed_communication_metrics,
            "offload_reason": offload_reason,
            "cloud_attempt_latency_ms": cloud_attempt_latency_ms,
            "resolution_attempted": conflict,
            "resolution_method": "cloud" if conflict else None,
            "resolution_success": cloud_success if conflict else None,
            "fallback_used": cloud_attempted and not cloud_success,
            "final_decision_method": (
                "cloud_review"
                if cloud_success
                else "edge_fallback"
                if cloud_attempted
                else "edge_consensus"
            ),
            "stage_timings_ms": {
                "image_read": read_ms,
                "preprocess": preprocess_ms,
                "nodes_parallel_wall": nodes_parallel_wall_ms,
                "node_max": max(metric["latency_ms"] for metric in node_metrics),
                "aggregation_and_cloud": aggregation_ms,
            },
        }
        final_result.metadata = metadata
        return final_result
