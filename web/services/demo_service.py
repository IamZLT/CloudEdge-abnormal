"""Single-node demo: edge score → collab route → optional cloud VLM."""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from web.config import IMG_EXT, ROOT, load_default_collab
from web.services import catalog_service, fleet_service, model_service


def run_route_decision(
    *,
    image_path: Path,
    category: str,
    edge: dict | None,
    use_agent: bool,
    edge_node_id: str | None = None,
) -> dict[str, Any]:
    """Decide upload via RouteAgent (or collab router) + per-node network try_upload."""
    from src.collab_routing import (
        cloud_state_from_fleet,
        configure_routing,
        push_routing_context,
        reset_routing_context,
        rule_decide,
    )
    from src.vlm.route_agent import RouteContext

    collab = load_default_collab()
    configure_routing(collab)
    fleet = fleet_service.get_fleet()
    with fleet_service.net_lock():
        node = fleet.get(edge_node_id)
        net = dict(node.network_snapshot())
        sim = node.sim
        cloud = cloud_state_from_fleet(fleet, collab)
        recent = fleet.recent_cloud_for(node.id)
    profile = str(net.get("profile") or "geo").lower()
    score = float(edge["edge_score"]) if edge and edge.get("edge_score") is not None else 0.5
    thr = float(edge["threshold"]) if edge and edge.get("threshold") is not None else 0.5
    decision = str(edge.get("edge_pred") or ("NG" if score >= thr else "OK"))
    hard_margin = float(collab.get("hard_margin") or collab.get("thr_margin") or 0.05)
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
        cloud=cloud.to_dict() if hasattr(cloud, "to_dict") else dict(cloud or {}),
    )

    use_llm = False
    tokens = push_routing_context(cloud=cloud, edge_node_id=node.id, recent_cloud=recent)
    try:
        verdict = rule_decide(
            ctx,
            collab_cfg=collab,
            cloud=cloud,
            edge_node_id=node.id,
            recent_cloud=recent,
        )
        # Keep CRR for UI/logging/rules_snap; LLM CONTEXT does not include CRR priors.
        ctx.crr = verdict.to_dict()
        ra_cfg = dict(collab.get("route_agent") or {})
        cascade = bool(ra_cfg.get("cascade_skip_low_uncertainty", False))
        unc = ctx._edge_uncertainty()
        cold = int(n_gallery) <= 0
        # Default: always RouteAgent LLM when use_agent. Optional cascade skips LLM on low unc.
        use_llm = bool(use_agent) and (not cascade or unc in {"mid", "high"} or cold)

        route_info: dict[str, Any]
        if use_llm:
            try:
                agent = model_service.get_route_agent()
                dec = agent.decide(ctx)
                route_info = dec.to_dict()
                meta = dict(getattr(agent, "meta", None) or {})
                route_info["backend"] = getattr(agent, "backend", meta.get("backend"))
                route_info["weight_source"] = meta.get("weight_source")
                route_info["gpu_footprint_mb"] = meta.get("gpu_footprint_mb")
                route_info["llm_invoked"] = True
            except Exception as exc:  # noqa: BLE001
                route_info = {
                    "upload": False,
                    "decision": decision,
                    "confidence": 0.0,
                    "reason": f"route_agent_unavailable -> stay local: {exc}",
                    "source": "agent_unavailable_local",
                    "parse_ok": False,
                    "latency_ms": 0.0,
                    "raw": "",
                    "network_profile": profile,
                    "backend": ra_cfg.get("backend", "gguf"),
                    "llm_invoked": False,
                }
        else:
            why = (
                f"cascade: uncertainty={unc} → CRR without LLM"
                if use_agent and cascade
                else verdict.reason
            )
            route_info = {
                "upload": verdict.upload,
                "decision": decision,
                "confidence": 1.0 if profile == "outage" else 0.7,
                "reason": why,
                "source": (
                    f"collab_routing_cascade:{verdict.algorithm}"
                    if use_agent and cascade
                    else f"collab_routing:{verdict.algorithm}"
                ),
                "parse_ok": True,
                "latency_ms": 0.0,
                "raw": "",
                "network_profile": profile,
                "llm_invoked": False,
                "edge_uncertainty": unc,
            }
    finally:
        reset_routing_context(tokens)

    route_info["collab_routing"] = verdict.to_dict()
    route_info["crr_suggest_upload"] = bool(verdict.upload)
    # Detection label is always edge AD here; cloud may override later in run_demo.
    route_info["decision"] = decision
    route_info["analysis"] = route_info.get("reason")
    if not use_llm:
        route_info["upload"] = bool(verdict.upload)
        if not route_info.get("reason"):
            route_info["reason"] = verdict.reason
    elif not route_info.get("reason"):
        route_info["reason"] = verdict.reason

    upload_want = bool(route_info.get("upload"))
    net_outcome = None
    path_type = "LOCAL"
    if upload_want:
        up_hard = int(collab.get("upload_bytes_hard") or 80000)
        with fleet_service.net_lock():
            fleet.cloud_state.inflight += 1
        try:
            out = sim.try_upload(up_hard)
        finally:
            with fleet_service.net_lock():
                fleet.cloud_state.inflight = max(0, fleet.cloud_state.inflight - 1)
        net_outcome = out.to_dict()
        path_type = "CLOUD_REVIEW" if out.ok else "LOCAL_NET_FALLBACK"
        with fleet_service.net_lock():
            fleet.cloud_state.note_upload(node.id, ok=bool(out.ok))
            fleet.cloud_state.decay_recent()
        fleet_service.push_net_sample(
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

    with fleet_service.net_lock():
        node.stats.record_path(
            path_type=path_type, upload_want=upload_want, network_profile=profile
        )
        live_net = node.network_snapshot()

    return {
        "route_agent": route_info,
        "collab_routing": verdict.to_dict(),
        "network": live_net,
        "network_outcome": net_outcome,
        "path_type": path_type,
        "upload_want": upload_want,
        "edge_node": node.to_dict(),
    }


def save_upload_bytes(filename: str, data: bytes) -> Path:
    name = Path(filename).name
    suffix = Path(name).suffix.lower()
    if suffix not in IMG_EXT:
        raise ValueError("unsupported image type; use png/jpg/bmp/webp")
    if not data:
        raise ValueError("empty upload")
    if len(data) > 20 * 1024 * 1024:
        raise ValueError("upload too large (max 20MB)")
    tmp_dir = ROOT / "outputs" / "web_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"{int(time.time())}_{uuid.uuid4().hex[:8]}{suffix}"
    path.write_bytes(data)
    return path.resolve()


def run_demo(
    *,
    category: str,
    image_path: str,
    live_cloud: bool,
    use_route_agent: bool,
    edge_node_id: str | None = None,
) -> dict[str, Any]:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    fleet = fleet_service.get_fleet()
    with fleet_service.net_lock():
        node = fleet.get(edge_node_id)
        if edge_node_id:
            fleet.set_active(node.id)
            fleet_service.sync_active_network_unlocked()

    edge = catalog_service.edge_lookup(category, image_path)
    cached = catalog_service.case_lookup_with_cloud(category, image_path)
    route_pack = run_route_decision(
        image_path=p,
        category=category,
        edge=edge,
        use_agent=use_route_agent,
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
        "viz": catalog_service.build_viz_payload(category, image_path, include_cloud=went_cloud),
        "cached_case": None,
        "cloud_live": None,
        "route": path_type,
        "route_agent": route_pack["route_agent"],
        "collab_routing": route_pack.get("collab_routing"),
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
            "cloud": (cached.get("cloud") if went_cloud else None),
        }

    ra = route_pack.get("route_agent") or {}
    edge_pred = (edge or {}).get("edge_pred")
    # Final detection: edge AD locally; cloud overrides only after successful upload.
    result["final_decision"] = edge_pred or (
        cached.get("final_decision") if cached and went_cloud else None
    )
    result["route_analysis"] = ra.get("analysis") or ra.get("reason")

    if not went_cloud:
        result["cloud_live"] = {
            "skipped": True,
            "reason": (
                "network fallback - keep edge AD decision"
                if path_type == "LOCAL_NET_FALLBACK"
                else ra.get("reason") or "stay local (edge AD final)"
            ),
        }
        return result

    if not live_cloud:
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

    client = model_service.get_cloud_client()
    vlm = client.infer(p)
    result["cloud_live"] = vlm.to_dict()
    result["final_decision"] = vlm.decision
    return result
