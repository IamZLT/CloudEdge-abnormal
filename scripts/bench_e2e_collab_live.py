#!/usr/bin/env python3
"""Live end-to-end cloud-edge collab bench (real models, real latencies).

Pipeline per sample:
  1) Edge AD: Qwen3.5-0.8B multi-layer patch gallery (live vision)
  2) RouteAgent LLM (always; cascade off) + CRR CONTEXT + network sim
  3) Cloud VLM: Qwen3-VL-4B+LoRA on successful upload

Two-pass loading (fits one GPU): score all edges first, then unload edge and
load RouteAgent + cloud for routing/review.

Example:
  conda activate clip
  CUDA_VISIBLE_DEVICES=0 python scripts/bench_e2e_collab_live.py \\
    --categories bottle,cable --profiles fair,weak --limit 12
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics as st
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge.infer import DEFAULT_PATHS, _gallery_paths  # noqa: E402
from edge.methods.encoders import load_qwen35_vision_encoder  # noqa: E402
from edge.methods.gallery_ad import mvtec_test_split  # noqa: E402
from edge.methods.patch_gallery_ad import PatchGalleryAD  # noqa: E402
from src.collab_routing import CloudState, RouteSignal, build_router, configure_routing  # noqa: E402
from src.network_geo import live_network_dict, make_geo_simulator  # noqa: E402
from src.vlm import QwenVLClient  # noqa: E402
from src.vlm.route_agent import RouteAgent, RouteContext  # noqa: E402

OUT_DIR = ROOT / "outputs" / "reports" / "e2e_collab_live"


def _agg(xs: list[float]) -> dict[str, float] | None:
    if not xs:
        return None
    return {
        "n": len(xs),
        "mean": float(st.mean(xs)),
        "p50": float(st.median(xs)),
        "p95": float(np.percentile(xs, 95)),
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _label_from_path(p: Path) -> int:
    return 0 if p.parent.name == "good" else 1


def _subsample(items: list[dict], limit: int, seed: int) -> list[dict]:
    if limit <= 0 or len(items) <= limit:
        return items
    rng = np.random.default_rng(seed)
    # balance OK/NG a bit
    ok = [it for it in items if it["label"] == 0]
    ng = [it for it in items if it["label"] == 1]
    n_ok = max(1, limit // 3)
    n_ng = limit - n_ok
    pick_ok = ok if len(ok) <= n_ok else [ok[i] for i in sorted(rng.choice(len(ok), n_ok, replace=False))]
    pick_ng = ng if len(ng) <= n_ng else [ng[i] for i in sorted(rng.choice(len(ng), n_ng, replace=False))]
    out = pick_ok + pick_ng
    rng.shuffle(out)
    return out[:limit]


def _cuda_sync(device: str) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_edges(
    *,
    cfg: dict[str, Any],
    categories: list[str],
    limit: int,
    seed: int,
    device: str,
    warmup: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    edge_cfg = dict(cfg.get("edge") or {})
    data_root = Path(cfg.get("data_root") or ROOT / "datasets" / "mvtec")
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    image_size = int(cfg.get("image_size") or 224)
    max_gallery = int(edge_cfg.get("max_gallery") or 16)
    layers = edge_cfg.get("layers") or [6, 8, 10, 12]
    fusion_temp = float(edge_cfg.get("fusion_temp") or 0.5)
    model_path = edge_cfg.get("model_path") or DEFAULT_PATHS["qwen35"]

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    print(f"[edge] load Qwen3.5 vision @ {device} ...")
    t0 = time.perf_counter()
    _, encode_patches, meta = load_qwen35_vision_encoder(
        model_path,
        device=device,
        max_pixels=int(edge_cfg.get("max_pixels") or image_size * image_size),
        layers=layers,
    )
    load_s = time.perf_counter() - t0
    print(f"[edge] loaded in {load_s:.1f}s layers={meta.get('layers')}")

    rows: list[dict[str, Any]] = []
    load_meta = {
        "edge_load_s": load_s,
        "edge_method": "qwen35_0.8b_vision_mlpatch",
        "device": device,
        "edge_warmup": int(warmup),
    }

    for cat in categories:
        ad = PatchGalleryAD(
            encode_patches, device=device, name="qwen35_0.8b_vision_mlpatch", fusion_temperature=fusion_temp
        )
        gallery = _gallery_paths(data_root, cat, max_gallery, seed)
        thr = edge_cfg.get("threshold")
        if thr is None:
            thr = ad.calibrate_threshold_loo(
                gallery,
                seed=seed,
                quantile=float(edge_cfg.get("thr_quantile") or 0.95),
            )
            build_s = 0.0
        else:
            thr = float(thr)
            build_s = ad.build_gallery(gallery, seed=seed)

        test_items = mvtec_test_split(data_root, cat)
        items = [
            {"path": str(p), "label": int(y), "category": cat}
            for p, y in test_items
        ]
        items = _subsample(items, limit, seed + hash(cat) % 1000)
        print(f"[edge] {cat}: gallery={len(gallery)} thr={thr:.4f} n_test={len(items)} build={build_s:.1f}s")

        # Warm CUDA/cudnn kernels before timed loop (same pattern as PatchGalleryAD.evaluate).
        warm_n = min(int(warmup), len(items))
        if warm_n > 0:
            print(f"[edge] warmup x{warm_n} (excluded from latency stats) ...")
            for i in range(warm_n):
                _ = ad.score_image(Image.open(items[i]["path"]).convert("RGB"))
            _cuda_sync(device)

        for it in items:
            img = Image.open(it["path"]).convert("RGB")
            _cuda_sync(device)
            t1 = time.perf_counter()
            score, _amap = ad.score_image(img)
            _cuda_sync(device)
            edge_ms = (time.perf_counter() - t1) * 1000.0
            decision = "NG" if float(score) >= thr else "OK"
            rows.append(
                {
                    **it,
                    "edge_score": float(score),
                    "edge_thr": thr,
                    "edge_decision": decision,
                    "edge_pred": 1 if decision == "NG" else 0,
                    "edge_ms": float(edge_ms),
                    "n_gallery": len(gallery),
                }
            )
            print(
                f"  edge {cat}/{Path(it['path']).parent.name}/{Path(it['path']).name} "
                f"score={score:.3f} {decision} {edge_ms:.0f}ms"
            )

    del encode_patches, ad
    _free_cuda()
    print("[edge] unloaded")
    return rows, load_meta


def run_route_cloud(
    *,
    edge_rows: list[dict[str, Any]],
    collab: dict[str, Any],
    cloud_cfg: dict[str, Any],
    prompt: str | None,
    profiles: list[str],
    device: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    configure_routing(collab)
    ra_cfg = dict(collab.get("route_agent") or {})
    ra_cfg["device"] = device
    ra_cfg.setdefault("backend", "gguf")
    ra_cfg["enforce_context_rules"] = False
    ra_cfg["hard_block_on_outage"] = False
    ra_cfg["cascade_skip_low_uncertainty"] = False

    print(f"[route] load RouteAgent backend={ra_cfg.get('backend')} ...")
    t0 = time.perf_counter()
    agent = RouteAgent.from_config(ra_cfg)
    route_load_s = time.perf_counter() - t0
    print(f"[route] loaded in {route_load_s:.1f}s")

    print("[cloud] load Qwen3-VL-4B+LoRA ...")
    t1 = time.perf_counter()
    cloud = QwenVLClient(
        model_path=cloud_cfg["model_path"],
        adapter_path=cloud_cfg.get("adapter_path"),
        device=device,
        dtype=cloud_cfg.get("dtype", "bfloat16"),
        max_new_tokens=int(cloud_cfg.get("max_new_tokens", 160)),
        role="cloud",
        prompt=prompt,
    )
    cloud_load_s = time.perf_counter() - t1
    print(f"[cloud] loaded in {cloud_load_s:.1f}s")

    router = build_router("cost_risk", {**collab, "route_policy": "cost_risk"})
    adm = dict(collab.get("cloud_admission") or {})
    max_inflight = int(adm.get("max_inflight", 2))
    hard_margin = float(collab.get("hard_margin") or collab.get("thr_margin") or 0.05)
    up_bytes = int(collab.get("upload_bytes_hard") or 80000)

    # warmup route + cloud
    w = edge_rows[0]
    warm_sim = make_geo_simulator(collab, profiles[0] if profiles else "fair", seed=seed, edge_id="edge-warmup")
    warm_net = live_network_dict(warm_sim)
    wctx = RouteContext(
        image=w["path"],
        category=w["category"],
        n_gallery=int(w["n_gallery"]),
        edge_score=float(w["edge_score"]),
        edge_thr=float(w["edge_thr"]),
        edge_decision=str(w["edge_decision"]),
        network_profile=str(warm_net.get("profile") or "geo"),
        network=warm_net,
        hard_margin=hard_margin,
        cloud={"inflight": 0, "queue": 0, "max_inflight": max_inflight},
        include_image=False,
    )
    print("[warmup] route+cloud ...")
    _ = agent.decide(wctx)
    _ = cloud.infer(Path(w["path"]))

    all_rows: list[dict[str, Any]] = []
    for profile in profiles:
        print(f"\n=== geo-scenario={profile} n={len(edge_rows)} (full LLM) ===")
        sim = make_geo_simulator(collab, profile, seed=seed, edge_id=f"edge-{profile}")
        cloud_state = CloudState(inflight=0, queue=0, max_inflight=max_inflight)
        n_want = n_ok = n_fb = 0

        for i, er in enumerate(edge_rows):
            net = live_network_dict(sim)
            sig = RouteSignal(
                category=er["category"],
                n_gallery=int(er["n_gallery"]),
                edge_score=float(er["edge_score"]),
                edge_thr=float(er["edge_thr"]),
                edge_decision=str(er["edge_decision"]),
                network_profile=str(net.get("profile") or "geo"),
                network=net,
                hard_margin=hard_margin,
                edge_node_id=f"edge-{profile}",
            )
            cloud_state.inflight = min(max_inflight, max(0, n_want - n_ok - n_fb))
            verd = router.decide(sig, cloud_state)

            ctx = RouteContext(
                image=er["path"],
                category=er["category"],
                n_gallery=int(er["n_gallery"]),
                edge_score=float(er["edge_score"]),
                edge_thr=float(er["edge_thr"]),
                edge_decision=str(er["edge_decision"]),
                network_profile=str(net.get("profile") or "geo"),
                network=net,
                hard_margin=hard_margin,
                crr=verd.to_dict(),
                cloud=cloud_state.to_dict(),
                include_image=False,
            )

            t_route0 = time.perf_counter()
            dec = agent.decide(ctx)
            route_wall_ms = (time.perf_counter() - t_route0) * 1000.0
            ra_ms = float(dec.latency_ms or 0.0)
            upload = bool(dec.upload)

            net_ms = 0.0
            path_type = "LOCAL"
            net_ok = False
            if upload:
                n_want += 1
                out = sim.try_upload(up_bytes)
                net_ms = float(out.rtt_ms) + float(out.tx_ms)
                route_wall_ms += net_ms  # include sim hop in route wall for parity
                if out.ok:
                    n_ok += 1
                    path_type = "CLOUD_REVIEW"
                    net_ok = True
                else:
                    n_fb += 1
                    path_type = "LOCAL_NET_FALLBACK"

            cloud_ms = 0.0
            cloud_dec = None
            cloud_parse_ok = None
            final_pred = int(er["edge_pred"])
            if path_type == "CLOUD_REVIEW":
                t_c = time.perf_counter()
                vlm = cloud.infer(Path(er["path"]))
                cloud_ms = float(vlm.latency_ms or (time.perf_counter() - t_c) * 1000.0)
                cloud_dec = vlm.decision
                cloud_parse_ok = bool(vlm.parse_ok)
                if vlm.decision:
                    final_pred = 1 if str(vlm.decision).upper() == "NG" else 0

            total_ms = float(er["edge_ms"]) + route_wall_ms + cloud_ms
            row = {
                "profile": profile,
                "category": er["category"],
                "path": er["path"],
                "label": int(er["label"]),
                "edge_score": er["edge_score"],
                "edge_thr": er["edge_thr"],
                "edge_decision": er["edge_decision"],
                "edge_pred": int(er["edge_pred"]),
                "edge_ms": float(er["edge_ms"]),
                "llm_invoked": True,
                "upload": upload,
                "path_type": path_type,
                "net_ok": net_ok,
                "ra_model_ms": ra_ms,
                "route_wall_ms": float(route_wall_ms),
                "net_ms": float(net_ms),
                "cloud_ms": float(cloud_ms),
                "total_ms": float(total_ms),
                "cloud_dec": cloud_dec,
                "cloud_parse_ok": cloud_parse_ok,
                "final_pred": int(final_pred),
                "agree_crr": upload == bool(verd.upload),
                "parse_ok": bool(dec.parse_ok),
                "source": dec.source,
                "crr_suggest": bool(verd.upload),
                "reason": dec.reason,
            }
            all_rows.append(row)
            print(
                f"[{profile} {i}] {er['category']}/{Path(er['path']).name} "
                f"{path_type} up={upload} | edge={er['edge_ms']:.0f} ra={ra_ms:.0f} "
                f"net={net_ms:.0f} cloud={cloud_ms:.0f} | TOTAL={total_ms:.0f} "
                f"| edge={er['edge_decision']} final={'NG' if final_pred else 'OK'} gt={'NG' if er['label'] else 'OK'}"
            )

    meta = {
        "route_load_s": route_load_s,
        "cloud_load_s": cloud_load_s,
        "route_backend": getattr(agent, "backend", None),
        "cloud_model": cloud_cfg.get("model_path"),
        "cloud_adapter": cloud_cfg.get("adapter_path"),
    }
    return all_rows, meta


def _summarize_profile(rows: list[dict[str, Any]], profile: str) -> dict[str, Any]:
    rs = [r for r in rows if r["profile"] == profile]
    y_true = [r["label"] for r in rs]
    y_edge = [r["edge_pred"] for r in rs]
    y_final = [r["final_pred"] for r in rs]
    y_score = [r["edge_score"] for r in rs]
    local = [r for r in rs if r["path_type"] == "LOCAL"]
    cloud = [r for r in rs if r["path_type"] == "CLOUD_REVIEW"]
    fb = [r for r in rs if r["path_type"] == "LOCAL_NET_FALLBACK"]
    up = [r for r in rs if r["upload"]]

    def _f1(yt, yp):
        return float(f1_score(yt, yp, zero_division=0)) if yt else 0.0

    try:
        auroc = float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan")
    except ValueError:
        auroc = float("nan")

    return {
        "profile": profile,
        "n": len(rs),
        "upload_rate": sum(1 for r in rs if r["upload"]) / max(1, len(rs)),
        "cloud_review_rate": len(cloud) / max(1, len(rs)),
        "fallback_rate": len(fb) / max(1, len(rs)),
        "local_rate": len(local) / max(1, len(rs)),
        "llm_invoke_rate": 1.0,
        "agree_crr_rate": sum(1 for r in rs if r["agree_crr"]) / max(1, len(rs)),
        "parse_fail_rate": sum(1 for r in rs if not r["parse_ok"]) / max(1, len(rs)),
        "edge_f1": _f1(y_true, y_edge),
        "final_f1": _f1(y_true, y_final),
        "edge_auroc": auroc,
        "edge_ms": _agg([r["edge_ms"] for r in rs]),
        "ra_model_ms": _agg([r["ra_model_ms"] for r in rs]),
        "route_wall_ms": _agg([r["route_wall_ms"] for r in rs]),
        "net_ms_when_upload": _agg([r["net_ms"] for r in up]),
        "cloud_ms_when_review": _agg([r["cloud_ms"] for r in cloud]),
        "total_ms_all": _agg([r["total_ms"] for r in rs]),
        "total_ms_local": _agg([r["total_ms"] for r in local]),
        "total_ms_fallback": _agg([r["total_ms"] for r in fb]),
        "total_ms_cloud_review": _agg([r["total_ms"] for r in cloud]),
    }


def _fmt_agg(a: dict | None, digits: int = 0) -> str:
    if not a:
        return "—"
    return f"{a['mean']:.{digits}f} (p50={a['p50']:.{digits}f}, {a['min']:.{digits}f}–{a['max']:.{digits}f})"


def _md(summaries: list[dict], meta: dict, args: argparse.Namespace) -> str:
    lines = [
        "# Live e2e cloud-edge collab",
        "",
        f"Measured: {time.strftime('%Y-%m-%d %H:%M')}. "
        f"Categories={args.categories} limit/cat={args.limit} profiles={args.profiles}.",
        "",
        "Stack: **live** Qwen3.5-0.8B patch-gallery AD → RouteAgent GGUF (always) → net sim → "
        "**live** Qwen3-VL-4B+LoRA. Cascade off.",
        "",
        f"- Edge load: {meta.get('edge_load_s', 0):.1f}s",
        f"- RouteAgent load: {meta.get('route_load_s', 0):.1f}s",
        f"- Cloud load: {meta.get('cloud_load_s', 0):.1f}s",
        "",
        "## Detection + routing rates",
        "",
        "| profile | n | upload% | cloud% | fallback% | edge_F1 | final_F1 | agree_crr% | parse_fail% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            "| {profile} | {n} | {up:.1%} | {cloud:.1%} | {fb:.1%} | {ef:.3f} | {ff:.3f} | {ag:.1%} | {pf:.1%} |".format(
                profile=s["profile"],
                n=s["n"],
                up=s["upload_rate"],
                cloud=s["cloud_review_rate"],
                fb=s["fallback_rate"],
                ef=s["edge_f1"],
                ff=s["final_f1"],
                ag=s["agree_crr_rate"],
                pf=s["parse_fail_rate"],
            )
        )
    lines += [
        "",
        "## End-to-end latency (ms)",
        "",
        "| profile | path | n | mean total | detail |",
        "|---|---|---:|---:|---|",
    ]
    for s in summaries:
        for key, name in (
            ("total_ms_all", "ALL"),
            ("total_ms_local", "LOCAL"),
            ("total_ms_fallback", "LOCAL_NET_FALLBACK"),
            ("total_ms_cloud_review", "CLOUD_REVIEW"),
        ):
            a = s.get(key)
            if not a:
                continue
            lines.append(
                f"| {s['profile']} | {name} | {a['n']} | {a['mean']:.0f} | "
                f"p50={a['p50']:.0f}, {a['min']:.0f}–{a['max']:.0f} |"
            )
    lines += [
        "",
        "## Stage breakdown (when stage runs)",
        "",
        "| profile | stage | mean (ms) |",
        "|---|---|---|",
    ]
    for s in summaries:
        lines.append(f"| {s['profile']} | edge AD | {_fmt_agg(s['edge_ms'])} |")
        lines.append(f"| {s['profile']} | RouteAgent model | {_fmt_agg(s['ra_model_ms'])} |")
        lines.append(f"| {s['profile']} | route wall (+net) | {_fmt_agg(s['route_wall_ms'])} |")
        lines.append(f"| {s['profile']} | net RTT+TX (upload) | {_fmt_agg(s['net_ms_when_upload'])} |")
        lines.append(f"| {s['profile']} | cloud VLM | {_fmt_agg(s['cloud_ms_when_review'])} |")
    lines += [
        "",
        "Notes: Warmup excluded. Edge scored live then unloaded before RouteAgent/cloud load. "
        "Detection label = edge AD locally; cloud VLM overrides only after successful upload.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="bottle,cable")
    ap.add_argument("--profiles", default="fair,weak")
    ap.add_argument("--limit", type=int, default=12, help="max samples per category")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--edge-warmup", type=int, default=5, help="edge AD warmup forwards before timing")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--cloud-config", default=str(ROOT / "configs" / "hybrid_lora.yaml"))
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    cloud_yaml = _load_yaml(Path(args.cloud_config))
    collab = dict(cfg.get("collab") or {})
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]

    edge_rows, edge_meta = score_edges(
        cfg=cfg,
        categories=cats,
        limit=args.limit,
        seed=args.seed,
        device=args.device,
        warmup=args.edge_warmup,
    )
    if not edge_rows:
        print("no samples", file=sys.stderr)
        return 1

    route_rows, rc_meta = run_route_cloud(
        edge_rows=edge_rows,
        collab=collab,
        cloud_cfg=dict(cloud_yaml.get("cloud") or {}),
        prompt=cloud_yaml.get("prompt"),
        profiles=profiles,
        device=args.device,
        seed=args.seed,
    )

    meta = {**edge_meta, **rc_meta}
    summaries = [_summarize_profile(route_rows, p) for p in profiles]

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "args": {
            "categories": cats,
            "profiles": profiles,
            "limit": args.limit,
            "seed": args.seed,
            "device": args.device,
        },
        "summaries": summaries,
        "edge_rows": edge_rows,
        "rows": route_rows,
    }
    (args.out / "timing.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = _md(summaries, meta, args)
    (args.out / "timing.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"wrote {args.out / 'timing.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
