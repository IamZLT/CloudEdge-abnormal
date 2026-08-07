"""Edge fleet selection APIs."""
from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException

from web.services import fleet_service, live_service

router = APIRouter(tags=["fleet"])


@router.get("/api/edge_nodes")
def api_edge_nodes():
    fleet = fleet_service.get_fleet()
    return {"ok": True, **fleet.summary()}


@router.post("/api/edge_nodes/active")
async def api_edge_nodes_set_active(edge_node_id: str = Form(...)):
    fleet = fleet_service.get_fleet()
    try:
        with fleet_service.net_lock():
            node = fleet.set_active(str(edge_node_id))
            fleet_service.sync_active_network_unlocked()
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    snap = fleet_service.network_snapshot(node.id)
    return {
        "ok": True,
        "active_id": node.id,
        "edge_node": node.to_dict(),
        "network": snap,
        "disconnected": str(snap.get("profile") or "").lower() == "outage",
    }


@router.get("/api/fleet/live")
def api_fleet_live_status():
    return {"ok": True, **live_service.status()}


@router.post("/api/fleet/live/start")
async def api_fleet_live_start(
    interval_s: float = Form(2.0),
    use_route_agent: str = Form("false"),
    live_cloud: str = Form("false"),
):
    st = live_service.start(
        interval_s=float(interval_s),
        use_route_agent=str(use_route_agent).lower() in {"1", "true", "yes", "on"},
        live_cloud=str(live_cloud).lower() in {"1", "true", "yes", "on"},
    )
    return {"ok": True, **st}


@router.post("/api/fleet/live/stop")
def api_fleet_live_stop():
    return {"ok": True, **live_service.stop()}


@router.post("/api/fleet/live/tick")
def api_fleet_live_tick():
    """One synchronous round across all edge nodes."""
    return {"ok": True, **live_service.tick_now(), **{"status": live_service.status()}}
