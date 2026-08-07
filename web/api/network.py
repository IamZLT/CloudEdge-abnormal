"""Network / topology APIs."""
from __future__ import annotations

from collections import deque

from fastapi import APIRouter, Form, HTTPException, Query

from web.services import fleet_service

router = APIRouter(tags=["network"])


@router.get("/api/network")
def api_network(edge_node_id: str | None = Query(None)):
    fleet = fleet_service.get_fleet()
    with fleet_service.net_lock():
        node = fleet.get(edge_node_id)
        hist = fleet_service.get_node_histories().get(node.id) or deque()
        n = len(hist)
        last = dict(hist[-1]) if hist else None
        prev = fleet_service.get_node_prev_cfg().get(node.id)
        disconnected = str(node.network.get("profile") or "").lower() == "outage"
    return {
        "ok": True,
        "edge_node_id": node.id,
        "edge_node": node.to_dict(),
        "network": fleet_service.network_snapshot(node.id),
        "history_len": n,
        "last_sample": last,
        "disconnected": disconnected,
        "restore_profile": (prev or {}).get("profile") if prev else None,
        "fleet": {"num_nodes": fleet.num_nodes, "active_id": fleet.active_id},
    }


@router.get("/api/network/profiles")
def api_network_profiles():
    from src.network_sim import PROFILES

    fleet = fleet_service.get_fleet()
    return {
        "mode": "physical_geo_temporal" if fleet.env is not None else "legacy_profile",
        "profiles": {k: v.to_dict() for k, v in PROFILES.items()},
        "note": (
            "Physical mode: RTT comes from geo distance + live congestion; "
            "use outage/disconnect to force link down, restore to resume physics."
            if fleet.env is not None
            else "Legacy static profiles."
        ),
        "current": fleet_service.network_snapshot(),
        "edge_fleet": fleet.summary(),
    }


@router.get("/api/network/env")
def api_network_env():
    fleet = fleet_service.get_fleet()
    if fleet.env is None:
        return {"ok": True, "enabled": False, "mode": "legacy_profile"}
    return {"ok": True, "enabled": True, **fleet.env.summary()}


@router.post("/api/network/profile")
async def api_network_set_profile(
    profile: str = Form(...),
    edge_node_id: str | None = Form(None),
    rtt_ms: float | None = Form(None),
    bandwidth_mbps: float | None = Form(None),
    loss_prob: float | None = Form(None),
    timeout_ms: float | None = Form(None),
):
    from src.network_sim import PROFILES

    name = str(profile).lower().strip()
    allowed = set(PROFILES) | {"custom", "geo", "physical"}
    if name not in allowed:
        raise HTTPException(400, f"unknown profile: {name}; choose {sorted(allowed)}")
    fleet = fleet_service.get_fleet()
    try:
        with fleet_service.net_lock():
            node = fleet.get(edge_node_id)
            cfg = dict(node.network_snapshot())
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    if fleet.env is not None and name not in {"outage"}:
        name = "geo"
    cfg["profile"] = name
    if fleet.env is None:
        if rtt_ms is not None:
            cfg["rtt_ms"] = float(rtt_ms)
        if bandwidth_mbps is not None:
            cfg["bandwidth_mbps"] = float(bandwidth_mbps)
        if loss_prob is not None:
            cfg["loss_prob"] = float(loss_prob)
        if timeout_ms is not None:
            cfg["timeout_ms"] = float(timeout_ms)
    if name != "outage":
        with fleet_service.net_lock():
            fleet_service.get_node_prev_cfg()[node.id] = None
    sample = fleet_service.apply_network_cfg(cfg, node_id=node.id)
    snap = fleet_service.network_snapshot(node.id)
    return {
        "ok": True,
        "edge_node_id": node.id,
        "network": snap,
        "sample": sample,
        "disconnected": str(snap.get("profile") or "").lower() == "outage",
    }


@router.post("/api/network/disconnect")
def api_network_disconnect(edge_node_id: str | None = Query(None)):
    fleet = fleet_service.get_fleet()
    try:
        with fleet_service.net_lock():
            node = fleet.get(edge_node_id)
            cur = str(node.network.get("profile") or "fair").lower()
            if cur != "outage":
                fleet_service.get_node_prev_cfg()[node.id] = dict(node.network)
            cfg = dict(node.network)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    cfg["profile"] = "outage"
    sample = fleet_service.apply_network_cfg(cfg, node_id=node.id)
    with fleet_service.net_lock():
        prev = fleet_service.get_node_prev_cfg().get(node.id)
    return {
        "ok": True,
        "disconnected": True,
        "edge_node_id": node.id,
        "network": fleet_service.network_snapshot(node.id),
        "sample": sample,
        "restore_profile": (prev or {}).get("profile") or "fair",
    }


@router.post("/api/network/restore")
def api_network_restore(edge_node_id: str | None = Query(None)):
    fleet = fleet_service.get_fleet()
    try:
        with fleet_service.net_lock():
            node = fleet.get(edge_node_id)
            prev = dict(
                fleet_service.get_node_prev_cfg().get(node.id)
                or {"profile": "fair", "seed": 42 + node.index}
            )
            fleet_service.get_node_prev_cfg()[node.id] = None
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    if str(prev.get("profile") or "").lower() == "outage":
        prev["profile"] = "fair"
    sample = fleet_service.apply_network_cfg(prev, node_id=node.id)
    return {
        "ok": True,
        "disconnected": False,
        "edge_node_id": node.id,
        "network": fleet_service.network_snapshot(node.id),
        "sample": sample,
        "restore_profile": None,
    }


@router.get("/api/network/timeseries")
def api_network_timeseries(
    n: int = Query(120, ge=10, le=180),
    edge_node_id: str | None = Query(None),
):
    fleet = fleet_service.get_fleet()
    with fleet_service.net_lock():
        node = fleet.get(edge_node_id)
        hist = fleet_service.get_node_histories().get(node.id) or deque()
        pts = list(hist)[-int(n) :]
    return {
        "points": pts,
        "network": fleet_service.network_snapshot(node.id),
        "edge_node_id": node.id,
    }
