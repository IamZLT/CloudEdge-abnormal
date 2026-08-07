"""Health + collab routing meta."""
from __future__ import annotations

import os

from fastapi import APIRouter

from web.config import DATA_ROOT, load_default_collab
from web.services import fleet_service, model_service

router = APIRouter(tags=["health"])


@router.get("/api/health")
def health():
    from src.collab_routing import get_router, list_policies

    snap = fleet_service.network_snapshot()
    ra = model_service.route_agent_info()
    fleet = fleet_service.get_fleet()
    try:
        router_name = get_router().name
    except Exception:  # noqa: BLE001
        router_name = None
    return {
        "ok": True,
        "cloud_loaded": model_service.cloud_loaded(),
        "cloud_error": model_service.cloud_error(),
        "route_agent_loaded": ra["loaded"],
        "route_agent_error": ra.get("error"),
        "route_agent": ra,
        "route_policy": router_name,
        "route_policies": list_policies(),
        "network_profile": snap.get("profile"),
        "edge_fleet": fleet.summary(),
        "active_edge_node": fleet.active_id,
        "data_root": str(DATA_ROOT),
        "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


@router.get("/api/collab_routing")
def api_collab_routing():
    from src.collab_routing import get_router, list_policies

    collab = load_default_collab()
    active = get_router(collab)
    return {
        "active": active.name,
        "policies": list_policies(),
        "config": {
            "route_policy": collab.get("route_policy"),
            "cost_risk": collab.get("cost_risk"),
            "cloud_admission": collab.get("cloud_admission"),
            "hard_margin": collab.get("hard_margin"),
        },
    }
