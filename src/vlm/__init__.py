"""Qwen-VL helpers for cloud-edge defect review + Qwen3.5 route agent."""

from .parse import parse_vlm_json
from .qwen_client import QwenVLClient, VLMResult
from .model_clients import (
    InternVLClient,
    MiniCPMVClient,
    TransformersVLMClient,
    create_vlm_client,
    detect_vlm_backend,
)
from .route_agent import (
    RouteAgent,
    RouteContext,
    RouteDecision,
    heuristic_upload,
    parse_route_json,
    resolve_include_image,
)

__all__ = [
    "QwenVLClient",
    "VLMResult",
    "TransformersVLMClient",
    "InternVLClient",
    "MiniCPMVClient",
    "create_vlm_client",
    "detect_vlm_backend",
    "parse_vlm_json",
    "RouteAgent",
    "RouteContext",
    "RouteDecision",
    "heuristic_upload",
    "parse_route_json",
    "resolve_include_image",
]
