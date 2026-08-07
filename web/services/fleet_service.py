"""Multi-edge fleet + per-node network probe state."""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from web.config import DATA_ROOT, load_default_collab, load_default_yaml

_net_lock = threading.Lock()
_fleet = None  # EdgeFleet
_net_cfg: dict[str, Any] = {"profile": "fair", "seed": 42}
_net_prev_cfg: dict[str, Any] | None = None
_node_prev_cfg: dict[str, dict[str, Any] | None] = {}
_net_history: deque[dict[str, Any]] = deque(maxlen=180)
_node_histories: dict[str, deque[dict[str, Any]]] = {}
_net_sampler_stop = threading.Event()
_net_sim = None


def net_lock() -> threading.Lock:
    return _net_lock


def get_node_prev_cfg() -> dict[str, dict[str, Any] | None]:
    return _node_prev_cfg


def get_node_histories() -> dict[str, deque[dict[str, Any]]]:
    return _node_histories


def get_fleet():
    """Return the process-wide EdgeFleet (create if missing)."""
    global _fleet
    if _fleet is not None:
        return _fleet
    with _net_lock:
        if _fleet is None:
            from src.edge_fleet import EdgeFleet

            cfg = load_default_yaml()
            _fleet = EdgeFleet.from_config(cfg, data_root=DATA_ROOT)
            for nid in _fleet.order:
                _node_histories.setdefault(nid, deque(maxlen=180))
                _node_prev_cfg.setdefault(nid, None)
            sync_active_network_unlocked()
    return _fleet


def sync_active_network_unlocked() -> None:
    """Copy active edge node's network into legacy globals (waveform / APIs)."""
    global _net_cfg, _net_sim, _net_history
    if _fleet is None:
        return
    node = _fleet.get()
    _net_cfg = dict(node.network)
    _net_sim = node.sim
    _net_history = _node_histories.setdefault(node.id, deque(maxlen=180))


def init_network_state() -> None:
    get_fleet()


def network_snapshot(node_id: str | None = None) -> dict[str, Any]:
    fleet = get_fleet()
    with _net_lock:
        node = fleet.get(node_id)
        snap = node.network_snapshot()
        snap["edge_node_id"] = node.id
        snap["edge_node_name"] = node.name
        return snap


def push_net_sample(
    sample: dict[str, Any] | None = None,
    *,
    node_id: str | None = None,
) -> dict[str, Any]:
    """Append one waveform point for the given (or active) edge node."""
    fleet = get_fleet()
    with _net_lock:
        node = fleet.get(node_id)
        sim = node.sim
        hist = _node_histories.setdefault(node.id, deque(maxlen=180))
    snap = node.network_snapshot()
    if sample is None:
        out = sim.try_upload(int((load_default_collab().get("upload_bytes_hard") or 80000)))
        snap = node.network_snapshot()
        sample = {
            "t": time.time(),
            "profile": snap.get("profile") or snap.get("name") or "geo",
            "edge_node_id": node.id,
            "city": snap.get("city") or node.city,
            "distance_geo_km": snap.get("distance_geo_km"),
            "distance_fiber_km": snap.get("distance_fiber_km"),
            "prop_rtt_ms": snap.get("prop_rtt_ms"),
            "congestion": snap.get("congestion"),
            "rtt_ms": float(
                out.rtt_ms
                if (out.ok or out.failed_reason in {"timeout", "outage"}) and out.rtt_ms > 0
                else snap.get("rtt_ms") or 0.0
            ),
            "tx_ms": float(out.tx_ms),
            "bandwidth_mbps": float(snap.get("bandwidth_mbps") or 0.0),
            "loss_prob": float(snap.get("loss_prob") or 0.0),
            "timeout_ms": float(snap.get("timeout_ms") or 0.0),
            "upload_ok": bool(out.ok),
            "failed_reason": out.failed_reason,
            "source": "probe",
        }
        if out.failed_reason == "loss" and not sample["rtt_ms"]:
            sample["rtt_ms"] = float(snap.get("rtt_ms") or 0.0)
            bw = max(1e-6, float(snap.get("bandwidth_mbps") or 1.0))
            sample["tx_ms"] = float((80000 * 8) / (bw * 1e6) * 1000)
    else:
        sample = dict(sample)
        sample.setdefault("t", time.time())
        sample.setdefault("profile", snap.get("profile") or "geo")
        sample.setdefault("edge_node_id", node.id)
        sample.setdefault("city", snap.get("city") or node.city)
    with _net_lock:
        hist.append(sample)
    return sample


def net_sampler_loop() -> None:
    while not _net_sampler_stop.is_set():
        try:
            fleet = get_fleet()
            for nid in list(fleet.order):
                push_net_sample(node_id=nid)
        except Exception:
            pass
        _net_sampler_stop.wait(0.75)


def start_net_sampler() -> threading.Thread:
    t = threading.Thread(target=net_sampler_loop, name="net-sampler", daemon=True)
    t.start()
    return t


def apply_network_cfg(
    cfg: dict[str, Any],
    *,
    node_id: str | None = None,
) -> dict[str, Any]:
    fleet = get_fleet()
    with _net_lock:
        node = fleet.get(node_id)
        node.set_network(cfg)
        if node.id == fleet.active_id:
            sync_active_network_unlocked()
    return push_net_sample(node_id=node.id)
