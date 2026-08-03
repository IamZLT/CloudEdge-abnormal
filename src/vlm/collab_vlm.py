from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.vlm.qwen_client import QwenVLClient, VLMResult


@dataclass
class CollabVLMConfig:
    conf_low: float = 0.55
    conf_high: float = 0.85
    uncertain_band: bool = True
    cloud_extra_latency_ms: float = 50.0


def fuse_edge_cloud(
    edge_res: VLMResult,
    cloud_res: VLMResult | None,
    hard: bool,
    cfg: CollabVLMConfig,
) -> dict[str, Any]:
    """Sedna-style: edge default; hard → cloud review."""
    if hard and cloud_res is not None:
        final = cloud_res
        path = "CLOUD_REVIEW"
        latency = edge_res.latency_ms + cloud_res.latency_ms + cfg.cloud_extra_latency_ms
    else:
        final = edge_res
        path = "LOCAL"
        latency = edge_res.latency_ms

    return {
        "path": path,
        "hard": hard,
        "decision": final.decision,
        "confidence": final.confidence,
        "defect_type": final.defect_type,
        "reason": final.reason,
        "latency_ms": float(latency),
        "edge": edge_res.to_dict(),
        "cloud": cloud_res.to_dict() if cloud_res is not None else None,
    }


def decide_route(edge: QwenVLClient, edge_res: VLMResult, cfg: CollabVLMConfig) -> bool:
    return edge.is_hard(edge_res, cfg.conf_low, cfg.conf_high, use_band=cfg.uncertain_band)
