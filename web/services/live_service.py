"""Fleet live monitor: auto-run all edge nodes and stream events for the Topology UI."""
from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import quote

from web.services import catalog_service, demo_service, fleet_service

_lock = threading.Lock()
_stop = threading.Event()
_thread: threading.Thread | None = None
_running = False
_cursor: dict[str, int] = {}  # edge_id -> image index
_events: deque[dict[str, Any]] = deque(maxlen=80)
_last_by_node: dict[str, dict[str, Any]] = {}
_stats = {
    "ticks": 0,
    "n_local": 0,
    "n_upload_want": 0,
    "n_cloud_ok": 0,
    "n_fallback": 0,
    "started_at": None,
}
_interval_s = 2.0
_use_route_agent = False  # CRR rules only — keep live loop snappy
_live_cloud = False


def status() -> dict[str, Any]:
    with _lock:
        return {
            "running": _running,
            "interval_s": _interval_s,
            "use_route_agent": _use_route_agent,
            "live_cloud": _live_cloud,
            "stats": dict(_stats),
            "last_by_node": dict(_last_by_node),
            "events": list(_events)[-40:],
            "cloud": _cloud_view(),
        }


def _cloud_view() -> dict[str, Any]:
    fleet = fleet_service.get_fleet()
    cs = fleet.cloud_state
    return {
        "inflight": int(cs.inflight),
        "queue": int(cs.queue),
        "max_inflight": int(cs.max_inflight),
        "recent_cloud": dict(cs.recent_cloud),
    }


def _pick_image(node_id: str, category: str) -> dict[str, Any] | None:
    imgs = catalog_service.list_test_images(category, limit=48)
    if not imgs:
        return None
    idx = int(_cursor.get(node_id, 0)) % len(imgs)
    _cursor[node_id] = idx + 1
    return imgs[idx]


def _tick_once() -> dict[str, Any]:
    fleet = fleet_service.get_fleet()
    tick_results: list[dict[str, Any]] = []
    t_tick = time.time()

    for nid in list(fleet.order):
        node = fleet.get(nid)
        img = _pick_image(nid, node.category)
        if img is None:
            continue
        try:
            result = demo_service.run_demo(
                category=node.category,
                image_path=img["path"],
                live_cloud=_live_cloud,
                use_route_agent=_use_route_agent,
                edge_node_id=nid,
            )
        except Exception as exc:  # noqa: BLE001
            err_ev = {
                "t": time.time(),
                "edge_node_id": nid,
                "city": node.city,
                "category": node.category,
                "ok": False,
                "error": str(exc),
            }
            with _lock:
                _events.appendleft(err_ev)
                _last_by_node[nid] = err_ev
            tick_results.append(err_ev)
            continue

        path_type = str(result.get("route") or "LOCAL")
        upload_want = bool(result.get("upload_want"))
        net = result.get("network") or {}
        ra = result.get("route_agent") or {}
        cr = result.get("collab_routing") or {}
        final = result.get("final_decision")
        edge = result.get("edge") or {}
        resolved = str(Path(img["path"]).resolve())
        image_url = f"/api/image?path={quote(resolved, safe='')}"
        ev = {
            "t": time.time(),
            "edge_node_id": nid,
            "city": node.city or net.get("city"),
            "category": node.category,
            "image": Path(img["path"]).name,
            "image_rel": img.get("rel"),
            "image_url": image_url,
            "gt": img.get("gt"),
            "edge_pred": edge.get("edge_pred"),
            "edge_score": edge.get("edge_score"),
            "final": final,
            "path_type": path_type,
            "upload_want": upload_want,
            "route_reason": ra.get("reason") or cr.get("reason"),
            "algorithm": cr.get("algorithm") or ra.get("source"),
            "utility": cr.get("utility"),
            "features": cr.get("features") or {},
            "rtt_ms": net.get("rtt_ms"),
            "bandwidth_mbps": net.get("bandwidth_mbps"),
            "loss_prob": net.get("loss_prob"),
            "distance_geo_km": net.get("distance_geo_km"),
            "prop_rtt_ms": net.get("prop_rtt_ms"),
            "outage": bool(net.get("outage") or str(net.get("profile") or "").lower() == "outage"),
            "ok": True,
        }
        with _lock:
            _events.appendleft(ev)
            _last_by_node[nid] = ev
            if upload_want:
                _stats["n_upload_want"] = int(_stats["n_upload_want"]) + 1
            else:
                _stats["n_local"] = int(_stats["n_local"]) + 1
            if path_type == "CLOUD_REVIEW":
                _stats["n_cloud_ok"] = int(_stats["n_cloud_ok"]) + 1
            elif path_type == "LOCAL_NET_FALLBACK":
                _stats["n_fallback"] = int(_stats["n_fallback"]) + 1
        tick_results.append(ev)

    with _lock:
        _stats["ticks"] = int(_stats.get("ticks") or 0) + 1
        _stats["last_tick_at"] = t_tick
        _stats["last_tick_n"] = len(tick_results)

    # refresh fleet network snapshots for topology
    try:
        for nid in list(fleet.order):
            fleet_service.push_net_sample(node_id=nid)
    except Exception:
        pass

    return {
        "t": t_tick,
        "results": tick_results,
        "cloud": _cloud_view(),
        "fleet": fleet.summary(),
    }


def _loop() -> None:
    global _running
    while not _stop.is_set():
        try:
            _tick_once()
        except Exception as exc:  # noqa: BLE001
            with _lock:
                _events.appendleft(
                    {"t": time.time(), "ok": False, "error": f"tick_failed: {exc}"}
                )
        _stop.wait(_interval_s)
    with _lock:
        _running = False


def start(
    *,
    interval_s: float = 2.0,
    use_route_agent: bool = False,
    live_cloud: bool = False,
) -> dict[str, Any]:
    global _thread, _running, _interval_s, _use_route_agent, _live_cloud
    with _lock:
        if _running:
            return status()
        _interval_s = max(0.8, float(interval_s))
        _use_route_agent = bool(use_route_agent)
        _live_cloud = bool(live_cloud)
        _stats.update(
            {
                "ticks": 0,
                "n_local": 0,
                "n_upload_want": 0,
                "n_cloud_ok": 0,
                "n_fallback": 0,
                "started_at": time.time(),
            }
        )
        _events.clear()
        _last_by_node.clear()
        _stop.clear()
        _running = True
        _thread = threading.Thread(target=_loop, name="fleet-live", daemon=True)
        _thread.start()
    return status()


def stop() -> dict[str, Any]:
    global _running
    _stop.set()
    t = _thread
    if t and t.is_alive():
        t.join(timeout=3.0)
    with _lock:
        _running = False
    return status()


def tick_now() -> dict[str, Any]:
    """Synchronous single round (for debugging / manual refresh)."""
    return _tick_once()
