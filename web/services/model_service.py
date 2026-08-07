"""Lazy cloud VLM + RouteAgent loaders for the web console."""
from __future__ import annotations

import os
import threading
from typing import Any

import yaml

from web.config import ROOT, load_default_collab

_cloud_client = None
_cloud_lock = threading.Lock()
_cloud_error: str | None = None

_route_agent = None
_route_agent_lock = threading.Lock()
_route_agent_error: str | None = None


def cloud_loaded() -> bool:
    return _cloud_client is not None


def cloud_error() -> str | None:
    return _cloud_error


def get_cloud_client():
    global _cloud_client, _cloud_error
    if _cloud_client is not None:
        return _cloud_client
    with _cloud_lock:
        if _cloud_client is not None:
            return _cloud_client
        try:
            from src.vlm import QwenVLClient

            cfg_path = ROOT / "configs" / "hybrid_lora.yaml"
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            cloud = cfg["cloud"]
            device = os.environ.get("WEB_VLM_DEVICE", cloud.get("device", "cuda:0"))
            _cloud_client = QwenVLClient(
                model_path=cloud["model_path"],
                adapter_path=cloud.get("adapter_path"),
                device=device,
                dtype=cloud.get("dtype", "bfloat16"),
                max_new_tokens=int(cloud.get("max_new_tokens", 128)),
                role="cloud",
                prompt=cfg.get("prompt"),
            )
            _cloud_error = None
            return _cloud_client
        except Exception as exc:  # noqa: BLE001
            _cloud_error = str(exc)
            raise


def get_route_agent():
    """Lazy-load Qwen3.5 RouteAgent (default: GGUF Q4 + mmproj)."""
    global _route_agent, _route_agent_error
    if _route_agent is not None:
        return _route_agent
    with _route_agent_lock:
        if _route_agent is not None:
            return _route_agent
        try:
            from src.vlm.route_agent import RouteAgent

            collab = load_default_collab()
            ra_cfg = dict(collab.get("route_agent") or {})
            ra_cfg["device"] = os.environ.get("WEB_ROUTE_DEVICE", ra_cfg.get("device", "cuda:0"))
            if os.environ.get("WEB_ROUTE_BACKEND"):
                ra_cfg["backend"] = os.environ["WEB_ROUTE_BACKEND"]
            if os.environ.get("WEB_ROUTE_GGUF_DIR"):
                ra_cfg["gguf_dir"] = os.environ["WEB_ROUTE_GGUF_DIR"]
            ra_cfg.setdefault("backend", "gguf")
            _route_agent = RouteAgent.from_config(ra_cfg)
            _route_agent_error = None
            return _route_agent
        except Exception as exc:  # noqa: BLE001
            _route_agent_error = str(exc)
            raise


def route_agent_info() -> dict[str, Any]:
    if _route_agent is None:
        return {
            "loaded": False,
            "error": _route_agent_error,
            "backend": (load_default_collab().get("route_agent") or {}).get("backend", "gguf"),
        }
    meta = dict(getattr(_route_agent, "meta", None) or {})
    return {
        "loaded": True,
        "error": None,
        "backend": getattr(_route_agent, "backend", meta.get("backend")),
        "weight_source": meta.get("weight_source"),
        "gpu_footprint_mb": meta.get("gpu_footprint_mb"),
        "package_disk_mb": meta.get("package_disk_mb"),
        "n_gpu_layers": meta.get("n_gpu_layers"),
        "gpu_offload_reported": meta.get("gpu_offload_reported"),
    }


def preload_models() -> None:
    try:
        get_route_agent()
        print("[web] RouteAgent preloaded", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[web] RouteAgent preload failed: {exc}", flush=True)
    preload_cloud = os.environ.get("WEB_PRELOAD_CLOUD", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if not preload_cloud:
        print("[web] Cloud preload skipped (WEB_PRELOAD_CLOUD=0)", flush=True)
        return
    try:
        get_cloud_client()
        print("[web] Cloud VLM preloaded", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[web] Cloud preload failed: {exc}", flush=True)
