from dataclasses import asdict
from time import perf_counter_ns

from common.config import Config
from common.schemas import DetectionResult
from common.utils import current_timestamp, log_info
from edge.image_reader import read_image
from edge.preprocessor import preprocess
from edge.inference_engine import InferenceEngine
from control.decision_policy import DecisionPolicy
from control.sync_handler import SyncHandler


class EdgeService:
    def __init__(
        self,
        config: Config,
        inference_engine: InferenceEngine,
        decision_policy: DecisionPolicy,
        sync_handler: SyncHandler,
    ):
        self.config = config
        self.inference_engine = inference_engine
        self.decision_policy = decision_policy
        self.sync_handler = sync_handler

    def run_inference(self, image_path: str) -> DetectionResult:
        read_started_ns = perf_counter_ns()
        image = read_image(image_path)
        read_ms = (perf_counter_ns() - read_started_ns) / 1_000_000
        preprocess_started_ns = perf_counter_ns()
        processed = preprocess(image)
        preprocess_ms = (perf_counter_ns() - preprocess_started_ns) / 1_000_000
        inference_started_ns = perf_counter_ns()
        result = self.inference_engine.predict(processed)
        inference_ms = (perf_counter_ns() - inference_started_ns) / 1_000_000
        result.timestamp = current_timestamp()
        result.source = "edge"
        metadata = dict(result.metadata or {})
        metadata["edge_pipeline"] = {
            "image_read_ms": read_ms,
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
        }
        result.metadata = metadata
        self.decision_policy.attach_quality_score(result, processed, image_path)
        log_info("Edge inference completed", {"image_path": image_path, "result": result.to_dict()})
        return result

    def handle_task(self, image_path: str) -> DetectionResult:
        result = self.run_inference(image_path)
        decision = self.decision_policy.should_upload(result)
        metadata = dict(result.metadata or {})
        metadata["routing"] = asdict(decision)
        result.metadata = metadata
        if decision.should_upload:
            if self.sync_handler.circuit_state["state"] != "open":
                log_info("Uploading result to cloud", {"reason": decision.reason})
            original_edge_label = result.label
            try:
                cloud_result = self.sync_handler.upload_result(result, image_path)
            except Exception as exc:
                # 云端不可用不能中断现场业务；保留本地结果并对不确定样本执行保守隔离。
                if self.config.offline_fail_safe_enabled:
                    result.label = "anomaly"
                    result.defect_category = result.defect_category or "requires_cloud_review"
                metadata = dict(result.metadata or {})
                resilience = dict(self.sync_handler.last_resilience_event)
                resilience.update(
                    {
                        "cloud_required": True,
                        "fallback_used": True,
                        "fallback_source": "single_edge",
                        "fallback_reason": "cloud_unavailable",
                        "provisional": True,
                        "requires_cloud_review": True,
                        "original_edge_label": original_edge_label,
                        "business_completed": True,
                        "business_action": "divert_for_review",
                    }
                )
                metadata["resilience"] = resilience
                failed_transport = self.sync_handler.last_call_metrics
                if failed_transport:
                    metadata["communication_metrics"] = failed_transport
                metadata["offload_reason"] = decision.reason
                result.metadata = metadata
                if not resilience.get("circuit_short_circuited"):
                    log_info(
                        "Cloud review unavailable; using fail-safe edge decision",
                        {
                            "image_path": image_path,
                            "error": str(exc),
                            "label": result.label,
                        },
                    )
                return result
            metadata = dict(cloud_result.metadata or {})
            metadata["edge_pipeline"] = result.metadata["edge_pipeline"]
            metadata["image_quality"] = result.metadata.get("image_quality")
            metadata["routing"] = result.metadata.get("routing")
            metadata["offload_reason"] = decision.reason
            resilience = dict(self.sync_handler.last_resilience_event)
            resilience.update(
                {
                    "cloud_required": True,
                    "fallback_used": False,
                    "provisional": False,
                    "requires_cloud_review": False,
                    "business_completed": True,
                    "business_action": (
                        "pass" if cloud_result.label == "normal" else "divert"
                    ),
                }
            )
            metadata["resilience"] = resilience
            cloud_result.metadata = metadata
            return cloud_result
        metadata = dict(result.metadata or {})
        metadata["resilience"] = {
            "cloud_required": False,
            "cloud_attempted": False,
            "cloud_success": False,
            "fallback_used": False,
            "provisional": False,
            "requires_cloud_review": False,
            "business_completed": True,
            "business_action": "pass" if result.label == "normal" else "divert",
            "circuit_state_after": self.sync_handler.circuit_state["state"],
        }
        result.metadata = metadata
        return result
