#!/usr/bin/env python3
"""Compare routing modes on cached hybrid scores:

  - baseline     : legacy margin rules (no LLM)
  - cost_risk    : CRR rules (no LLM)
  - llm_route    : RouteAgent LLM primary (no CRR in CONTEXT / no post guard)

Network: physical geo-temporal sim (NetworkEnvironment). ``--profiles`` only
selects edge city/access scenario (good/fair/weak), not static RTT tables.

Does not load cloud VLM; uses cached cloud JSON when upload succeeds.

Example:
  conda activate clip
  CUDA_VISIBLE_DEVICES=0 python scripts/bench_llm_crr_context_vs_baseline.py \\
    --categories bottle,cable --profiles fair,weak --limit 40
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.collab_routing import CloudState, RouteSignal, build_router, configure_routing  # noqa: E402
from src.network_geo import live_network_dict, make_geo_simulator  # noqa: E402
from src.vlm.route_agent import RouteAgent, RouteContext  # noqa: E402

OUT_DIR = ROOT / "outputs" / "reports" / "llm_crr_context_vs_baseline"
HYBRID = ROOT / "outputs" / "hybrid_lora_8b"


def _load_collab() -> dict[str, Any]:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")) or {}
    return dict(cfg.get("collab") or {})


def _load_category(cat: str) -> dict[str, Any] | None:
    edge_path = HYBRID / cat / "edge_scores.json"
    if not edge_path.exists():
        return None
    pack = json.loads(edge_path.read_text(encoding="utf-8"))
    cloud_by_key: dict[str, dict] = {}
    bench_path = HYBRID / cat / "bench.json"
    if bench_path.exists():
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        for row in bench.get("rows") or []:
            p = row.get("path") or ""
            if not p or not row.get("cloud"):
                continue
            key = f"{Path(p).parent.name}/{Path(p).name}"
            cloud_by_key[key] = row["cloud"]
    items = []
    for it in pack.get("items") or []:
        p = it.get("path") or ""
        key = f"{Path(p).parent.name}/{Path(p).name}"
        items.append(
            {
                "path": p,
                "label": int(it.get("label", 0)),
                "edge_score": float(it["edge_score"]),
                "edge_pred": 1 if it.get("edge_pred") else 0,
                "cloud": cloud_by_key.get(key),
            }
        )
    return {
        "category": cat,
        "threshold": float(pack.get("threshold") or 0.5),
        "items": items,
    }


def _subsample(items: list[dict], limit: int, seed: int) -> list[dict]:
    if limit <= 0 or len(items) <= limit:
        return items
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(items), size=limit, replace=False)
    idx.sort()
    return [items[i] for i in idx]


def _run_rules(
    *,
    policy: str,
    collab: dict[str, Any],
    packs: list[dict[str, Any]],
    profile: str,
    seed: int,
    cloud_extra_ms: float,
) -> dict[str, Any]:
    router = build_router(policy, {**collab, "route_policy": policy})
    adm = dict(collab.get("cloud_admission") or {})
    max_inflight = int(adm.get("max_inflight", 2))
    hard_margin = float(collab.get("hard_margin") or collab.get("thr_margin") or 0.05)
    n_gallery = int(collab.get("n_gallery_default") or 16)
    up_bytes = int(collab.get("upload_bytes_hard") or 80000)

    y_true: list[int] = []
    y_pred: list[int] = []
    n = n_want = n_ok = n_fb = n_local = 0
    agree = 0  # unused for rules
    lat_route: list[float] = []
    lat_e2e: list[float] = []
    t0 = time.perf_counter()

    for pi, pack in enumerate(packs):
        sim = make_geo_simulator(collab, profile, seed=seed + 17 * pi, edge_id=f"edge-{profile}-{pi}")
        cloud = CloudState(inflight=0, queue=0, max_inflight=max_inflight)
        thr = float(pack["threshold"])
        cat = pack["category"]
        for item in pack["items"]:
            n += 1
            net = live_network_dict(sim)
            score = float(item["edge_score"])
            edge_pred = int(item["edge_pred"])
            sig = RouteSignal(
                category=cat,
                n_gallery=n_gallery,
                edge_score=score,
                edge_thr=thr,
                edge_decision="NG" if score >= thr else "OK",
                network_profile=str(net.get("profile") or "geo"),
                network=net,
                hard_margin=hard_margin,
                edge_node_id=f"edge-{profile}-{pi % 3}",
            )
            cloud.inflight = min(max_inflight, max(0, n_want - n_ok - n_fb))
            t_r = time.perf_counter()
            verd = router.decide(sig, cloud)
            lat_route.append((time.perf_counter() - t_r) * 1000.0)

            final_pred = edge_pred
            sample_lat = 0.0
            if verd.upload:
                n_want += 1
                out = sim.try_upload(up_bytes)
                hop = float(out.rtt_ms) + float(out.tx_ms)
                if out.ok:
                    n_ok += 1
                    sample_lat = hop + cloud_extra_ms
                    cj = item.get("cloud") or {}
                    if cj.get("decision"):
                        final_pred = 1 if str(cj["decision"]).upper() == "NG" else 0
                else:
                    n_fb += 1
                    sample_lat = hop
            else:
                n_local += 1
            lat_e2e.append(sample_lat)
            y_true.append(int(item["label"]))
            y_pred.append(int(final_pred))

    return _summarize(
        mode=policy,
        profile=profile,
        n=n,
        n_want=n_want,
        n_ok=n_ok,
        n_fb=n_fb,
        n_local=n_local,
        agree=agree,
        y_true=y_true,
        y_pred=y_pred,
        lat_route=lat_route,
        lat_e2e=lat_e2e,
        wall_s=time.perf_counter() - t0,
        extra={"llm": False},
    )


def _run_llm(
    *,
    agent: RouteAgent,
    collab: dict[str, Any],
    packs: list[dict[str, Any]],
    profile: str,
    seed: int,
    cloud_extra_ms: float,
) -> dict[str, Any]:
    """LLM routes upload only; detection = edge AD / cloud cache. Cascade skips LLM on low unc."""
    configure_routing(collab)
    router = build_router("cost_risk", {**collab, "route_policy": "cost_risk"})
    adm = dict(collab.get("cloud_admission") or {})
    max_inflight = int(adm.get("max_inflight", 2))
    hard_margin = float(collab.get("hard_margin") or collab.get("thr_margin") or 0.05)
    n_gallery = int(collab.get("n_gallery_default") or 16)
    up_bytes = int(collab.get("upload_bytes_hard") or 80000)
    cascade = bool((collab.get("route_agent") or {}).get("cascade_skip_low_uncertainty", False))

    y_true: list[int] = []
    y_pred: list[int] = []
    n = n_want = n_ok = n_fb = n_local = 0
    n_agree = n_parse_fail = n_llm = 0
    lat_route: list[float] = []
    lat_e2e: list[float] = []
    t0 = time.perf_counter()

    for pi, pack in enumerate(packs):
        sim = make_geo_simulator(collab, profile, seed=seed + 17 * pi, edge_id=f"edge-{profile}-{pi}")
        cloud = CloudState(inflight=0, queue=0, max_inflight=max_inflight)
        thr = float(pack["threshold"])
        cat = pack["category"]
        for item in pack["items"]:
            n += 1
            net = live_network_dict(sim)
            score = float(item["edge_score"])
            edge_decision = "NG" if score >= thr else "OK"
            edge_pred = int(item["edge_pred"])
            sig = RouteSignal(
                category=cat,
                n_gallery=n_gallery,
                edge_score=score,
                edge_thr=thr,
                edge_decision=edge_decision,
                network_profile=str(net.get("profile") or "geo"),
                network=net,
                hard_margin=hard_margin,
                edge_node_id=f"edge-{profile}-{pi % 3}",
            )
            cloud.inflight = min(max_inflight, max(0, n_want - n_ok - n_fb))
            verd = router.decide(sig, cloud)

            img = item.get("path") or ""
            ctx = RouteContext(
                image=img if img and Path(img).exists() else ROOT / "README.md",
                category=cat,
                n_gallery=n_gallery,
                edge_score=score,
                edge_thr=thr,
                edge_decision=edge_decision,
                network_profile=str(net.get("profile") or "geo"),
                network=net,
                hard_margin=hard_margin,
                crr=verd.to_dict(),
                cloud=cloud.to_dict(),
                include_image=False,
            )
            unc = ctx._edge_uncertainty()
            cold = n_gallery <= 0
            invoke_llm = (not cascade) or unc in {"mid", "high"} or cold

            if invoke_llm:
                n_llm += 1
                dec = agent.decide(ctx)
                upload = bool(dec.upload)
                lat_route.append(float(dec.latency_ms or 0.0))
                sample_lat = float(dec.latency_ms or 0.0)
                if not dec.parse_ok or str(dec.source) == "parse_fail_local":
                    n_parse_fail += 1
            else:
                upload = bool(verd.upload)
                lat_route.append(0.0)
                sample_lat = 0.0

            if upload == bool(verd.upload):
                n_agree += 1

            # Detection always edge locally; cloud cache only after successful upload.
            final_pred = edge_pred
            if upload:
                n_want += 1
                out = sim.try_upload(up_bytes)
                hop = float(out.rtt_ms) + float(out.tx_ms)
                if out.ok:
                    n_ok += 1
                    sample_lat += hop + cloud_extra_ms
                    cj = item.get("cloud") or {}
                    if cj.get("decision"):
                        final_pred = 1 if str(cj["decision"]).upper() == "NG" else 0
                else:
                    n_fb += 1
                    sample_lat += hop
            else:
                n_local += 1

            lat_e2e.append(sample_lat)
            y_true.append(int(item["label"]))
            y_pred.append(int(final_pred))

    return _summarize(
        mode="llm_route" if not cascade else "llm_route_cascade",
        profile=profile,
        n=n,
        n_want=n_want,
        n_ok=n_ok,
        n_fb=n_fb,
        n_local=n_local,
        agree=n_agree,
        y_true=y_true,
        y_pred=y_pred,
        lat_route=lat_route,
        lat_e2e=lat_e2e,
        wall_s=time.perf_counter() - t0,
        extra={
            "llm": True,
            "agree_with_crr_rate": n_agree / max(1, n),
            "parse_fail_rate": n_parse_fail / max(1, n_llm),
            "llm_invoke_rate": n_llm / max(1, n),
            "cascade": cascade,
        },
    )


def _summarize(
    *,
    mode: str,
    profile: str,
    n: int,
    n_want: int,
    n_ok: int,
    n_fb: int,
    n_local: int,
    agree: int,
    y_true: list[int],
    y_pred: list[int],
    lat_route: list[float],
    lat_e2e: list[float],
    wall_s: float,
    extra: dict[str, Any],
) -> dict[str, Any]:
    f1 = float(f1_score(np.asarray(y_true), np.asarray(y_pred), zero_division=0))
    out = {
        "mode": mode,
        "network_profile": profile,
        "n_samples": n,
        "n_want_upload": n_want,
        "n_upload_ok": n_ok,
        "n_fallback": n_fb,
        "n_local": n_local,
        "upload_want_rate": n_want / max(1, n),
        "cloud_review_rate": n_ok / max(1, n),
        "fallback_rate": n_fb / max(1, n),
        "detection_f1": f1,
        "avg_route_latency_ms": float(np.mean(lat_route)) if lat_route else 0.0,
        "avg_e2e_extra_latency_ms": float(np.mean(lat_e2e)) if lat_e2e else 0.0,
        "p95_e2e_extra_latency_ms": float(np.percentile(lat_e2e, 95)) if lat_e2e else 0.0,
        "wall_s": wall_s,
        **extra,
    }
    if "agree_with_crr_rate" not in out and n:
        out["agree_with_crr_rate"] = agree / max(1, n)
    return out


def _md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# RouteAgent LLM vs baseline / CRR rules",
        "",
        "| mode | profile | n | upload% | cloud% | fallback% | route_F1 | llm_call% | route_ms | e2e_extra_ms | agree_crr% | parse_fail% | wall_s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {mode} | {profile} | {n} | {up:.1%} | {cloud:.1%} | {fb:.1%} | {f1:.3f} | {lc} | {rm:.0f} | {e2e:.0f} | {ag:.1%} | {pf:.1%} | {w:.1f} |".format(
                mode=r["mode"],
                profile=r["network_profile"],
                n=r["n_samples"],
                up=r["upload_want_rate"],
                cloud=r["cloud_review_rate"],
                fb=r["fallback_rate"],
                f1=r["detection_f1"],
                lc=(
                    f"{float(r['llm_invoke_rate']):.0%}"
                    if r.get("llm_invoke_rate") is not None
                    else "—"
                ),
                rm=r["avg_route_latency_ms"],
                e2e=r["avg_e2e_extra_latency_ms"],
                ag=float(r.get("agree_with_crr_rate") or 0.0),
                pf=float(r.get("parse_fail_rate") or 0.0),
                w=r["wall_s"],
            )
        )
    lines.append("")
    lines.append(
        "Notes: `llm_route` = every sample uses RouteAgent LLM; CRR is not in CONTEXT and "
        "no post-LLM guard overwrites valid output. Detection always edge AD / cached cloud."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="bottle,cable,capsule")
    ap.add_argument("--profiles", default="fair,weak")
    ap.add_argument("--limit", type=int, default=40, help="max samples per category (LLM cost)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cloud-extra-ms", type=float, default=800.0)
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    collab = _load_collab()
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    packs: list[dict[str, Any]] = []
    for c in cats:
        pack = _load_category(c)
        if not pack:
            print(f"[skip] missing hybrid scores for {c}")
            continue
        pack = dict(pack)
        cat_salt = int(hashlib.md5(c.encode()).hexdigest()[:8], 16) % 1000
        pack["items"] = _subsample(pack["items"], args.limit, args.seed + cat_salt)
        packs.append(pack)
    if not packs:
        print("no categories loaded", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    for profile in profiles:
        for policy in ("baseline", "cost_risk"):
            print(f"[run] {policy} @ {profile} ...")
            rows.append(
                _run_rules(
                    policy=policy,
                    collab=collab,
                    packs=packs,
                    profile=profile,
                    seed=args.seed,
                    cloud_extra_ms=args.cloud_extra_ms,
                )
            )

    if not args.skip_llm:
        ra_cfg = dict(collab.get("route_agent") or {})
        ra_cfg.setdefault("backend", "gguf")
        ra_cfg["enforce_context_rules"] = False
        ra_cfg["hard_block_on_outage"] = False
        ra_cfg["hard_route_guard"] = False
        print("[load] RouteAgent ...")
        agent = RouteAgent.from_config(ra_cfg)
        for profile in profiles:
            print(f"[run] llm_route @ {profile} ...")
            rows.append(
                _run_llm(
                    agent=agent,
                    collab=collab,
                    packs=packs,
                    profile=profile,
                    seed=args.seed,
                    cloud_extra_ms=args.cloud_extra_ms,
                )
            )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "compare.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md = _md(rows)
    (args.out / "compare.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {args.out / 'compare.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
