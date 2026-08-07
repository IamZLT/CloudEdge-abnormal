#!/usr/bin/env python3
"""CloudEdge-abnormal Web console.

Env: conda activate base
Run:
  CUDA_VISIBLE_DEVICES=0 uvicorn web.app:app --host 0.0.0.0 --port 7860
  # or
  python -m web.app
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
DATA_ROOT = ROOT / "datasets" / "mvtec"
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_CFG = ROOT / "configs" / "default.yaml"

app = FastAPI(title="CloudEdge Defect Console", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_cloud_client = None
_cloud_lock = threading.Lock()
_cloud_error: str | None = None

_route_agent = None
_route_agent_lock = threading.Lock()
_route_agent_error: str | None = None

# ---- multi-edge fleet + per-node network simulation ----
_net_lock = threading.Lock()
_fleet = None  # EdgeFleet
_net_cfg: dict[str, Any] = {"profile": "fair", "seed": 42}  # mirrors active node
_net_prev_cfg: dict[str, Any] | None = None  # legacy; prefer _node_prev_cfg
_node_prev_cfg: dict[str, dict[str, Any] | None] = {}
_net_history: deque[dict[str, Any]] = deque(maxlen=180)  # active node view
_node_histories: dict[str, deque[dict[str, Any]]] = {}
_net_sampler_stop = threading.Event()
_net_sim = None


def _load_default_yaml() -> dict:
    if DEFAULT_CFG.exists():
        return yaml.safe_load(DEFAULT_CFG.read_text(encoding="utf-8")) or {}
    return {}


def _load_default_collab() -> dict:
    return dict(_load_default_yaml().get("collab") or {})


def _get_fleet():
    """Return the process-wide EdgeFleet (create if missing)."""
    global _fleet
    if _fleet is not None:
        return _fleet
    with _net_lock:
        if _fleet is None:
            from src.edge_fleet import EdgeFleet

            cfg = _load_default_yaml()
            _fleet = EdgeFleet.from_config(cfg, data_root=DATA_ROOT)
            for nid in _fleet.order:
                _node_histories.setdefault(nid, deque(maxlen=180))
                _node_prev_cfg.setdefault(nid, None)
            _sync_active_network_unlocked()
    return _fleet


def _sync_active_network_unlocked() -> None:
    """Copy active edge node's network into legacy globals (waveform / APIs)."""
    global _net_cfg, _net_sim, _net_history
    if _fleet is None:
        return
    node = _fleet.get()
    _net_cfg = dict(node.network)
    _net_sim = node.sim
    _net_history = _node_histories.setdefault(node.id, deque(maxlen=180))


def _init_network_state() -> None:
    """Boot multi-edge fleet (num_nodes from config, default 3)."""
    _get_fleet()


def _network_snapshot(node_id: str | None = None) -> dict[str, Any]:
    fleet = _get_fleet()
    with _net_lock:
        node = fleet.get(node_id)
        snap = node.network_snapshot()
        snap["edge_node_id"] = node.id
        snap["edge_node_name"] = node.name
        return snap


def _push_net_sample(
    sample: dict[str, Any] | None = None,
    *,
    node_id: str | None = None,
) -> dict[str, Any]:
    """Append one waveform point for the given (or active) edge node."""
    fleet = _get_fleet()
    with _net_lock:
        node = fleet.get(node_id)
        sim = node.sim
        hist = _node_histories.setdefault(node.id, deque(maxlen=180))
    # Live geo snapshot (advances OU / diurnal) or legacy profile
    snap = node.network_snapshot()
    if sample is None:
        out = sim.try_upload(int((_load_default_collab().get("upload_bytes_hard") or 80000)))
        snap = node.network_snapshot()  # refresh after upload attempt
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


def _net_sampler_loop() -> None:
    """Probe every edge node's link so the fleet stays live."""
    while not _net_sampler_stop.is_set():
        try:
            fleet = _get_fleet()
            for nid in list(fleet.order):
                _push_net_sample(node_id=nid)
        except Exception:
            pass
        _net_sampler_stop.wait(0.75)


def _preload_models() -> None:
    """Warm RouteAgent + optional cloud VLM so Demo is ready without Preload buttons."""
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


@app.on_event("startup")
def _on_startup() -> None:
    _init_network_state()
    t = threading.Thread(target=_net_sampler_loop, name="net-sampler", daemon=True)
    t.start()
    preload = os.environ.get("WEB_PRELOAD", "1").lower() not in {"0", "false", "no", "off"}
    if preload:
        threading.Thread(target=_preload_models, name="model-preload", daemon=True).start()
    else:
        print("[web] model preload skipped (WEB_PRELOAD=0)", flush=True)


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_summary() -> dict:
    all_cats = _read_json(ROOT / "outputs" / "hybrid_lora_8b" / "all_categories.json") or []
    anomalib = None
    mean_md = ROOT / "outputs" / "reports" / "mvtec_mean.md"
    # parse light stats from hybrid means
    h8 = _read_json(ROOT / "outputs" / "hybrid_lora_8b" / "hybrid_mean.json") or {}
    h4 = _read_json(ROOT / "outputs" / "hybrid_lora" / "hybrid_mean.json") or {}
    if all_cats:
        b1 = float(np.mean([r["B1_f1"] for r in all_cats]))
        zs = float(np.mean([r["ZS8B_f1"] for r in all_cats if r.get("ZS8B_f1") is not None]))
        l4 = float(np.mean([r["Lora4B_f1"] for r in all_cats if r.get("Lora4B_f1") is not None]))
        l8 = float(np.mean([r["Lora8B_f1"] for r in all_cats if r.get("Lora8B_f1") is not None]))
    else:
        b1 = zs = l4 = l8 = float("nan")
    return {
        "categories": all_cats,
        "means": {
            "B1_f1": b1,
            "ZS8B_f1": zs,
            "Lora4B_f1": l4,
            "Lora8B_f1": l8,
        },
        "anomalib_report": str(mean_md) if mean_md.exists() else None,
        "hybrid_lora_8b": h8.get("mean_S_f1"),
        "hybrid_lora_4b": h4.get("mean_S_f1"),
        "stack": {
            "edge": "Qwen3.5-0.8B vision multi-layer patch gallery (PaDiM optional)",
            "cloud": "Qwen3-VL-8B + LoRA (also 4B LoRA available)",
            "collab": "Multi-edge fleet + Qwen3.5 RouteAgent (GGUF Q4) + per-node network sim",
        },
    }


def list_categories() -> list[str]:
    if not DATA_ROOT.exists():
        return []
    return sorted(
        p.name
        for p in DATA_ROOT.iterdir()
        if p.is_dir() and (p / "test").exists()
    )


def _sample_key(path: str | Path) -> str:
    """Stable key: defect_folder/filename (avoids 000.png collisions across classes)."""
    p = Path(path)
    return f"{p.parent.name}/{p.name}"


def _cloud_reviewed_keys(category: str) -> set[str]:
    """Keys of images that already have cloud LLM outputs in benches."""
    keys: set[str] = set()
    for source in ("hybrid_lora_8b", "hybrid_lora", "hybrid"):
        bench = _read_json(ROOT / "outputs" / source / category / "bench.json")
        if not bench:
            continue
        for row in bench.get("rows", []):
            if row.get("cloud") and row.get("path"):
                keys.add(_sample_key(row["path"]))
    return keys


def list_test_images(category: str, limit: int = 40) -> list[dict]:
    test = DATA_ROOT / category / "test"
    if not test.exists():
        return []
    reviewed = _cloud_reviewed_keys(category)
    items = []
    for sub in sorted(test.iterdir()):
        if not sub.is_dir():
            continue
        label = 0 if sub.name == "good" else 1
        for p in sorted(sub.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                key = _sample_key(p)
                items.append(
                    {
                        "path": str(p.resolve()),
                        "rel": f"{category}/test/{sub.name}/{p.name}",
                        "name": p.name,
                        "defect_type": "none" if label == 0 else sub.name,
                        "label": label,
                        "gt": "OK" if label == 0 else "NG",
                        "has_llm": key in reviewed,
                        "sample_key": key,
                    }
                )
    # Prefer samples with cached cloud LLM output, then balance OK/NG
    with_llm = [x for x in items if x["has_llm"]]
    without = [x for x in items if not x["has_llm"]]
    picked: list[dict] = []
    for pool in (with_llm, without):
        oks = [x for x in pool if x["label"] == 0]
        ngs = [x for x in pool if x["label"] == 1]
        while (oks or ngs) and len(picked) < limit:
            if ngs:
                picked.append(ngs.pop(0))
            if len(picked) >= limit:
                break
            if oks:
                picked.append(oks.pop(0))
    return picked[:limit]


def edge_lookup(category: str, image_path: str) -> dict | None:
    pack = _read_json(ROOT / "outputs" / "hybrid_lora_8b" / category / "edge_scores.json")
    if not pack:
        pack = _read_json(ROOT / "outputs" / "hybrid" / category / "edge_scores.json")
    if not pack:
        return None
    target = str(Path(image_path).resolve())
    name = Path(image_path).name
    parent = Path(image_path).parent.name
    for it in pack.get("items", []):
        p = it.get("path") or ""
        if str(Path(p).resolve()) == target or (Path(p).name == name and Path(p).parent.name == parent):
            return {
                "edge_score": it["edge_score"],
                "edge_pred": "NG" if it.get("edge_pred") else "OK",
                "hard": bool(it.get("hard")),
                "threshold": pack.get("threshold"),
                "band_low": pack.get("band_low"),
                "band_high": pack.get("band_high"),
            }
    return None


def case_lookup(category: str, image_path: str, source: str = "hybrid_lora_8b") -> dict | None:
    bench = _read_json(ROOT / "outputs" / source / category / "bench.json")
    if not bench:
        return None
    key = _sample_key(image_path)
    for row in bench.get("rows", []):
        p = row.get("path") or ""
        if p and _sample_key(p) == key:
            return row
    return None


def case_lookup_with_cloud(category: str, image_path: str) -> dict | None:
    """Prefer a bench row that includes cloud LLM output across hybrid sources."""
    best: dict | None = None
    for source in ("hybrid_lora_8b", "hybrid_lora", "hybrid"):
        row = case_lookup(category, image_path, source)
        if not row:
            continue
        if row.get("cloud"):
            return row
        if best is None:
            best = row
    return best


def _url_for_file(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return f"/api/image?path={path.resolve()}"


def find_anomalib_viz(category: str, image_path: str, role: str) -> Path | None:
    """Locate Anomalib strip viz: Image | GT | Heatmap | Pred mask."""
    parent = Path(image_path).parent.name
    name = Path(image_path).name
    root = ROOT / "outputs" / "anomalib" / category / role
    if not root.exists():
        return None
    hits = sorted(root.glob(f"**/images/{parent}/{name}"))
    return hits[0] if hits else None


def find_gt_mask(category: str, image_path: str) -> Path | None:
    parent = Path(image_path).parent.name
    if parent == "good":
        return None
    stem = Path(image_path).stem
    mask = DATA_ROOT / category / "ground_truth" / parent / f"{stem}_mask.png"
    return mask if mask.exists() else None


def build_viz_payload(
    category: str,
    image_path: str,
    *,
    include_cloud: bool = True,
) -> dict[str, Any]:
    edge_viz = find_anomalib_viz(category, image_path, "edge")
    cloud_viz = find_anomalib_viz(category, image_path, "cloud") if include_cloud else None
    gt_mask = find_gt_mask(category, image_path)
    return {
        "edge_strip": _url_for_file(edge_viz),
        # heavy Anomalib cloud (PatchCore) - NOT Qwen-VL; only when routed to cloud
        "cloud_strip": _url_for_file(cloud_viz) if include_cloud else None,
        "gt_mask": _url_for_file(gt_mask),
        "include_cloud": bool(include_cloud),
        "legend": "Anomalib only (PaDiM/PatchCore). VLM outputs JSON, not heatmaps.",
    }


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

            collab = _load_default_collab()
            ra_cfg = dict(collab.get("route_agent") or {})
            # env overrides for web deploy
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


def _route_agent_info() -> dict[str, Any]:
    """Status snippet for health / preload responses."""
    if _route_agent is None:
        return {
            "loaded": False,
            "error": _route_agent_error,
            "backend": (_load_default_collab().get("route_agent") or {}).get("backend", "gguf"),
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


def _run_route_decision(
    *,
    image_path: Path,
    category: str,
    edge: dict | None,
    use_agent: bool,
    edge_node_id: str | None = None,
) -> dict[str, Any]:
    """Decide upload via RouteAgent (or heuristic fallback) + per-node network try_upload."""
    from src.vlm.route_agent import RouteContext, heuristic_upload, resolve_network_profile

    collab = _load_default_collab()
    fleet = _get_fleet()
    with _net_lock:
        node = fleet.get(edge_node_id)
        net_cfg = dict(node.network)
        sim = node.sim
    profile, net = resolve_network_profile({"network": net_cfg})
    score = float(edge["edge_score"]) if edge and edge.get("edge_score") is not None else 0.5
    thr = float(edge["threshold"]) if edge and edge.get("threshold") is not None else 0.5
    decision = str(edge.get("edge_pred") or ("NG" if score >= thr else "OK"))
    hard_margin = float(collab.get("thr_margin") or 0.05)
    n_gallery = int(collab.get("n_gallery_default") or 16)
    ctx = RouteContext(
        image=image_path,
        category=category,
        n_gallery=n_gallery,
        edge_score=score,
        edge_thr=thr,
        edge_decision=decision,
        network_profile=profile,
        network=net,
        hard_margin=hard_margin,
    )

    route_info: dict[str, Any]
    if use_agent:
        try:
            agent = get_route_agent()
            dec = agent.decide(ctx)
            route_info = dec.to_dict()
            meta = dict(getattr(agent, "meta", None) or {})
            route_info["backend"] = getattr(agent, "backend", meta.get("backend"))
            route_info["weight_source"] = meta.get("weight_source")
            route_info["gpu_footprint_mb"] = meta.get("gpu_footprint_mb")
        except Exception as exc:  # noqa: BLE001
            upload = heuristic_upload(ctx)
            route_info = {
                "upload": upload,
                "confidence": 0.0,
                "reason": f"route_agent_unavailable -> heuristic: {exc}",
                "source": "heuristic_fallback",
                "parse_ok": False,
                "latency_ms": 0.0,
                "raw": "",
                "network_profile": profile,
                "backend": (_load_default_collab().get("route_agent") or {}).get("backend", "gguf"),
            }
    else:
        upload = heuristic_upload(ctx)
        route_info = {
            "upload": upload,
            "confidence": 1.0 if profile == "outage" else 0.6,
            "reason": "heuristic (RouteAgent disabled)",
            "source": "heuristic",
            "parse_ok": True,
            "latency_ms": 0.0,
            "raw": "",
            "network_profile": profile,
        }

    upload_want = bool(route_info.get("upload"))
    net_outcome = None
    path_type = "LOCAL"
    if upload_want:
        up_hard = int(collab.get("upload_bytes_hard") or 80000)
        out = sim.try_upload(up_hard)
        net_outcome = out.to_dict()
        path_type = "CLOUD_REVIEW" if out.ok else "LOCAL_NET_FALLBACK"
        _push_net_sample(
            {
                "t": time.time(),
                "profile": profile,
                "edge_node_id": node.id,
                "rtt_ms": float(out.rtt_ms),
                "tx_ms": float(out.tx_ms),
                "bandwidth_mbps": float(net.get("bandwidth_mbps") or 0),
                "loss_prob": float(net.get("loss_prob") or 0),
                "timeout_ms": float(net.get("timeout_ms") or 0),
                "upload_ok": bool(out.ok),
                "failed_reason": out.failed_reason,
                "source": "demo_upload",
            },
            node_id=node.id,
        )
    else:
        path_type = "LOCAL"

    with _net_lock:
        node.stats.record_path(
            path_type=path_type, upload_want=upload_want, network_profile=profile
        )
        # Prefer live geo snapshot (distance / prop_rtt / congestion) for UI
        live_net = node.network_snapshot()

    return {
        "route_agent": route_info,
        "network": live_net,
        "network_outcome": net_outcome,
        "path_type": path_type,
        "upload_want": upload_want,
        "edge_node": node.to_dict(),
    }


@app.get("/", response_class=HTMLResponse)
def index():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(
        html,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )


@app.get("/api/health")
def health():
    snap = _network_snapshot()
    ra = _route_agent_info()
    fleet = _get_fleet()
    return {
        "ok": True,
        "cloud_loaded": _cloud_client is not None,
        "cloud_error": _cloud_error,
        "route_agent_loaded": ra["loaded"],
        "route_agent_error": ra.get("error"),
        "route_agent": ra,
        "network_profile": snap.get("profile"),
        "edge_fleet": fleet.summary(),
        "active_edge_node": fleet.active_id,
        "data_root": str(DATA_ROOT),
        "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _apply_network_cfg(
    cfg: dict[str, Any],
    *,
    node_id: str | None = None,
) -> dict[str, Any]:
    """Replace active (or given) edge node's network config and rebuild simulator."""
    fleet = _get_fleet()
    with _net_lock:
        node = fleet.get(node_id)
        node.set_network(cfg)
        if node.id == fleet.active_id:
            _sync_active_network_unlocked()
    sample = _push_net_sample(node_id=node.id)
    return sample


@app.get("/api/edge_nodes")
def api_edge_nodes():
    """List configured edge nodes (num_nodes from config, default 3)."""
    fleet = _get_fleet()
    return {"ok": True, **fleet.summary()}


@app.post("/api/edge_nodes/active")
async def api_edge_nodes_set_active(edge_node_id: str = Form(...)):
    """Select which edge node drives demo + network waveform."""
    fleet = _get_fleet()
    try:
        with _net_lock:
            node = fleet.set_active(str(edge_node_id))
            _sync_active_network_unlocked()
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    snap = _network_snapshot(node.id)
    return {
        "ok": True,
        "active_id": node.id,
        "edge_node": node.to_dict(),
        "network": snap,
        "disconnected": str(snap.get("profile") or "").lower() == "outage",
    }


@app.get("/api/network")
def api_network(edge_node_id: str | None = Query(None)):
    fleet = _get_fleet()
    with _net_lock:
        node = fleet.get(edge_node_id)
        hist = _node_histories.get(node.id) or deque()
        n = len(hist)
        last = dict(hist[-1]) if hist else None
        prev = _node_prev_cfg.get(node.id)
        disconnected = str(node.network.get("profile") or "").lower() == "outage"
    return {
        "ok": True,
        "edge_node_id": node.id,
        "edge_node": node.to_dict(),
        "network": _network_snapshot(node.id),
        "history_len": n,
        "last_sample": last,
        "disconnected": disconnected,
        "restore_profile": (prev or {}).get("profile") if prev else None,
        "fleet": {"num_nodes": fleet.num_nodes, "active_id": fleet.active_id},
    }


@app.get("/api/network/profiles")
def api_network_profiles():
    from src.network_sim import PROFILES

    fleet = _get_fleet()
    return {
        "mode": "physical_geo_temporal" if fleet.env is not None else "legacy_profile",
        "profiles": {k: v.to_dict() for k, v in PROFILES.items()},
        "note": (
            "Physical mode: RTT comes from geo distance + live congestion; "
            "use outage/disconnect to force link down, restore to resume physics."
            if fleet.env is not None
            else "Legacy static profiles."
        ),
        "current": _network_snapshot(),
        "edge_fleet": fleet.summary(),
    }


@app.get("/api/network/env")
def api_network_env():
    """Full physical environment snapshot (cloud + all edge links)."""
    fleet = _get_fleet()
    if fleet.env is None:
        return {"ok": True, "enabled": False, "mode": "legacy_profile"}
    return {"ok": True, "enabled": True, **fleet.env.summary()}


@app.post("/api/network/profile")
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
    fleet = _get_fleet()
    try:
        with _net_lock:
            node = fleet.get(edge_node_id)
            cfg = dict(node.network_snapshot())
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    # In physical mode, non-outage profiles mean "resume geo physics"
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
        with _net_lock:
            _node_prev_cfg[node.id] = None
    sample = _apply_network_cfg(cfg, node_id=node.id)
    snap = _network_snapshot(node.id)
    return {
        "ok": True,
        "edge_node_id": node.id,
        "network": snap,
        "sample": sample,
        "disconnected": str(snap.get("profile") or "").lower() == "outage",
    }


@app.post("/api/network/disconnect")
def api_network_disconnect(edge_node_id: str | None = Query(None)):
    """Simulate full outage on one edge node; remember previous profile for restore."""
    fleet = _get_fleet()
    try:
        with _net_lock:
            node = fleet.get(edge_node_id)
            cur = str(node.network.get("profile") or "fair").lower()
            if cur != "outage":
                _node_prev_cfg[node.id] = dict(node.network)
            cfg = dict(node.network)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    cfg["profile"] = "outage"
    sample = _apply_network_cfg(cfg, node_id=node.id)
    with _net_lock:
        prev = _node_prev_cfg.get(node.id)
    return {
        "ok": True,
        "disconnected": True,
        "edge_node_id": node.id,
        "network": _network_snapshot(node.id),
        "sample": sample,
        "restore_profile": (prev or {}).get("profile") or "fair",
    }


@app.post("/api/network/restore")
def api_network_restore(edge_node_id: str | None = Query(None)):
    """Restore one edge node's network to the profile used before disconnect."""
    fleet = _get_fleet()
    try:
        with _net_lock:
            node = fleet.get(edge_node_id)
            prev = dict(_node_prev_cfg.get(node.id) or {"profile": "fair", "seed": 42 + node.index})
            _node_prev_cfg[node.id] = None
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    if str(prev.get("profile") or "").lower() == "outage":
        prev["profile"] = "fair"
    sample = _apply_network_cfg(prev, node_id=node.id)
    return {
        "ok": True,
        "disconnected": False,
        "edge_node_id": node.id,
        "network": _network_snapshot(node.id),
        "sample": sample,
        "restore_profile": None,
    }


@app.get("/api/network/timeseries")
def api_network_timeseries(
    n: int = Query(120, ge=10, le=180),
    edge_node_id: str | None = Query(None),
):
    fleet = _get_fleet()
    with _net_lock:
        node = fleet.get(edge_node_id)
        hist = _node_histories.get(node.id) or deque()
        pts = list(hist)[-int(n) :]
    return {
        "points": pts,
        "network": _network_snapshot(node.id),
        "edge_node_id": node.id,
    }


@app.post("/api/route_agent/load")
def api_route_agent_load():
    try:
        get_route_agent()
        info = _route_agent_info()
        return {"ok": True, "loaded": True, **info}
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            {"ok": False, "error": str(exc), **_route_agent_info()},
            status_code=503,
        )


@app.get("/api/summary")
def api_summary():
    return load_summary()


@app.get("/api/categories")
def api_categories():
    return {"categories": list_categories()}


@app.get("/api/images")
def api_images(category: str = Query(...), limit: int = Query(36, ge=1, le=120)):
    if category not in list_categories():
        raise HTTPException(404, f"unknown category: {category}")
    return {"category": category, "images": list_test_images(category, limit=limit)}


@app.get("/api/image")
def api_image(path: str = Query(...)):
    p = Path(path).resolve()
    # allow dataset + outputs only
    allowed = [DATA_ROOT.resolve(), (ROOT / "outputs").resolve()]
    if not any(str(p).startswith(str(a)) for a in allowed):
        raise HTTPException(403, "path not allowed")
    if not p.exists() or p.suffix.lower() not in IMG_EXT:
        raise HTTPException(404, "image not found")
    return FileResponse(p)


@app.get("/api/cases")
def api_cases(category: str = Query(...), source: str = Query("hybrid_lora_8b"), limit: int = 24):
    bench = _read_json(ROOT / "outputs" / source / category / "bench.json")
    if not bench:
        raise HTTPException(404, f"no bench for {category} under {source}")
    rows = []
    for row in bench.get("rows", []):
        if not row.get("cloud") and source.startswith("hybrid"):
            # still include some local-only for context, but prefer reviewed
            continue
        viz = build_viz_payload(category, row["path"])
        rows.append(
            {
                "path": row["path"],
                "name": Path(row["path"]).name,
                "gt": "NG" if row.get("label") == 1 else "OK",
                "edge_pred": row.get("edge_pred"),
                "edge_score": row.get("edge_score"),
                "final": row.get("final_decision"),
                "path_type": row.get("path_type"),
                "viz": viz,
                "cloud": {
                    "decision": (row.get("cloud") or {}).get("decision"),
                    "confidence": (row.get("cloud") or {}).get("confidence"),
                    "defect_type": (row.get("cloud") or {}).get("defect_type"),
                    "reason": (row.get("cloud") or {}).get("reason"),
                    "raw": (row.get("cloud") or {}).get("raw"),
                    "latency_ms": (row.get("cloud") or {}).get("latency_ms"),
                }
                if row.get("cloud")
                else None,
            }
        )
        if len(rows) >= limit:
            break
    # if few cloud rows, fill with first rows
    if len(rows) < min(8, limit):
        for row in bench.get("rows", []):
            if any(r["path"] == row["path"] for r in rows):
                continue
            rows.append(
                {
                    "path": row["path"],
                    "name": Path(row["path"]).name,
                    "gt": "NG" if row.get("label") == 1 else "OK",
                    "edge_pred": row.get("edge_pred"),
                    "edge_score": row.get("edge_score"),
                    "final": row.get("final_decision"),
                    "path_type": row.get("path_type"),
                    "cloud": None,
                }
            )
            if len(rows) >= limit:
                break
    return {
        "category": category,
        "source": source,
        "detection": bench.get("detection"),
        "cloud_model": bench.get("cloud"),
        "cases": rows,
    }


@app.post("/api/demo")
async def api_demo(
    category: str = Form(...),
    image_path: str | None = Form(None),
    live_cloud: str = Form("false"),
    use_route_agent: str = Form("true"),
    edge_node_id: str | None = Form(None),
    file: UploadFile | None = File(None),
):
    """Demo: per-edge score -> RouteAgent + that node's network sim -> optional cloud."""
    live = str(live_cloud).lower() in {"1", "true", "yes", "on"}
    use_agent = str(use_route_agent).lower() in {"1", "true", "yes", "on"}
    tmp_path = None
    if file is not None and file.filename:
        name = Path(file.filename).name
        suffix = Path(name).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            raise HTTPException(400, "unsupported image type; use png/jpg/bmp/webp")
        data = await file.read()
        if not data:
            raise HTTPException(400, "empty upload")
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(400, "upload too large (max 20MB)")
        tmp_dir = ROOT / "outputs" / "web_uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"{int(time.time())}_{uuid.uuid4().hex[:8]}{suffix}"
        tmp_path.write_bytes(data)
        image_path = str(tmp_path.resolve())

    if not image_path:
        raise HTTPException(400, "image_path or file required")
    p = Path(image_path)
    if not p.exists():
        raise HTTPException(404, f"image not found: {image_path}")

    fleet = _get_fleet()
    try:
        with _net_lock:
            node = fleet.get(edge_node_id)
            # keep UI selection in sync when demo specifies a node
            if edge_node_id:
                fleet.set_active(node.id)
                _sync_active_network_unlocked()
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    edge = edge_lookup(category, image_path)
    cached = case_lookup_with_cloud(category, image_path)
    route_pack = _run_route_decision(
        image_path=p,
        category=category,
        edge=edge,
        use_agent=use_agent,
        edge_node_id=node.id,
    )

    path_type = route_pack["path_type"]
    went_cloud = path_type == "CLOUD_REVIEW"

    result: dict[str, Any] = {
        "category": category,
        "edge_node": route_pack.get("edge_node") or node.to_dict(),
        "image_path": str(p.resolve()),
        "image_url": f"/api/image?path={p.resolve()}",
        "edge": edge,
        # Cloud heatmap / PatchCore strip only when this run actually routes to cloud.
        "viz": build_viz_payload(category, image_path, include_cloud=went_cloud),
        "cached_case": None,
        "cloud_live": None,
        "route": path_type,
        "route_agent": route_pack["route_agent"],
        "network": route_pack["network"],
        "network_outcome": route_pack["network_outcome"],
        "upload_want": route_pack["upload_want"],
        "final_decision": None,
    }

    if cached:
        result["cached_case"] = {
            "gt": "NG" if cached.get("label") == 1 else "OK",
            "edge_pred": cached.get("edge_pred"),
            "edge_score": cached.get("edge_score"),
            "final": cached.get("final_decision"),
            "path_type": cached.get("path_type"),
            # Only expose cached cloud JSON when this run went to cloud.
            "cloud": (cached.get("cloud") if went_cloud else None),
        }

    # default final = edge; upgrade if cloud runs or cached cloud on same path
    result["final_decision"] = (edge or {}).get("edge_pred") or (
        cached.get("final_decision") if cached and went_cloud else None
    )

    if not went_cloud:
        result["cloud_live"] = {
            "skipped": True,
            "reason": (
                "network fallback - edge local"
                if path_type == "LOCAL_NET_FALLBACK"
                else route_pack["route_agent"].get("reason") or "RouteAgent: stay local"
            ),
        }
        return result

    if not live:
        # offline cloud path: reuse cached cloud JSON when available
        if cached and cached.get("cloud"):
            result["final_decision"] = cached.get("final_decision")
            result["cloud_live"] = {
                **(cached.get("cloud") or {}),
                "from_cache": True,
            }
        else:
            result["cloud_live"] = {
                "skipped": True,
                "reason": "CLOUD_REVIEW but no cached cloud JSON; enable Live cloud LoRA",
            }
        return result

    try:
        client = get_cloud_client()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"cloud model unavailable: {exc}") from exc

    vlm = client.infer(p)
    result["cloud_live"] = vlm.to_dict()
    result["final_decision"] = vlm.decision
    return result


@app.post("/api/cloud/load")
def api_cloud_load():
    try:
        get_cloud_client()
        return {"ok": True, "loaded": True}
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=503)


def main():
    import uvicorn

    host = os.environ.get("WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("WEB_PORT", "7860"))
    uvicorn.run("web.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
