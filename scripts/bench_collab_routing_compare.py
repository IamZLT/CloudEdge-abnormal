#!/usr/bin/env python3
"""Compare collab route policies: baseline vs cost_risk (CRR).

Uses cached hybrid edge scores (+ optional cloud JSON). Does NOT load VLMs.
Sweeps network profiles and reports upload rate, upload success, fallback,
latency, and detection F1 under each policy.

Example:
  conda activate clip
  python scripts/bench_collab_routing_compare.py --categories bottle,cable,capsule
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.collab_routing import CloudState, RouteSignal, build_router  # noqa: E402
from src.network_sim import NetworkSimulator  # noqa: E402

OUT_DIR = ROOT / "outputs" / "reports" / "collab_routing_compare"
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
                "hard": bool(it.get("hard")),
                "cloud": cloud_by_key.get(key),
            }
        )
    return {
        "category": cat,
        "threshold": float(pack.get("threshold") or 0.5),
        "items": items,
    }


def _net_cfg(profile: str, seed: int) -> dict[str, Any]:
    return {"profile": profile, "seed": int(seed)}


def _signal_from_item(
    *,
    cat: str,
    item: dict,
    thr: float,
    n_gallery: int,
    hard_margin: float,
    snap: dict[str, Any],
    edge_node_id: str,
) -> RouteSignal:
    score = float(item["edge_score"])
    pred = "NG" if score >= thr else "OK"
    return RouteSignal(
        category=cat,
        n_gallery=n_gallery,
        edge_score=score,
        edge_thr=thr,
        edge_decision=pred,
        network_profile=str(snap.get("profile") or "fair"),
        network=dict(snap),
        hard_margin=hard_margin,
        edge_node_id=edge_node_id,
    )


def _run_policy(
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
    y_score: list[float] = []

    n = 0
    n_want = 0
    n_attempt = 0
    n_upload_ok = 0
    n_fallback = 0
    n_local = 0
    n_cloud_used_cache = 0
    lat_attempt_ms: list[float] = []
    lat_success_ms: list[float] = []
    lat_e2e_ms: list[float] = []  # all samples: edge path ~0 account + optional upload

    # one sim per category stream (reset seed for fair A/B)
    t0 = time.perf_counter()
    for pi, pack in enumerate(packs):
        sim = NetworkSimulator.from_config(_net_cfg(profile, seed + 17 * pi))
        cloud = CloudState(inflight=0, queue=0, max_inflight=max_inflight)
        thr = float(pack["threshold"])
        cat = pack["category"]
        for item in pack["items"]:
            n += 1
            # Link view for CRR (legacy profiles are static means + loss)
            net = dict(sim.profile.to_dict())
            net["profile"] = sim.profile.name
            net["outage"] = sim.profile.name == "outage"

            sig = _signal_from_item(
                cat=cat,
                item=item,
                thr=thr,
                n_gallery=n_gallery,
                hard_margin=hard_margin,
                snap=net,
                edge_node_id=f"edge-{pi % 3}",
            )
            # approximate concurrent cloud load from outstanding wants
            outstanding = max(0, n_want - n_upload_ok - n_fallback)
            cloud.inflight = min(max_inflight, outstanding)
            verd = router.decide(sig, cloud)

            edge_pred = int(item["edge_pred"])
            edge_score = float(item["edge_score"])
            final_pred = edge_pred
            final_score = edge_score
            sample_lat = 0.0  # accounting latency on top of edge infer

            if verd.upload:
                n_want += 1
                n_attempt += 1
                out = sim.try_upload(up_bytes)
                hop = float(out.rtt_ms) + float(out.tx_ms)
                lat_attempt_ms.append(hop)
                if out.ok:
                    n_upload_ok += 1
                    lat_success_ms.append(hop + cloud_extra_ms)
                    sample_lat = hop + cloud_extra_ms
                    cloud_json = item.get("cloud")
                    if cloud_json and cloud_json.get("decision"):
                        n_cloud_used_cache += 1
                        final_pred = 1 if str(cloud_json["decision"]).upper() == "NG" else 0
                        conf = cloud_json.get("confidence")
                        # score for AUROC: prefer cloud confidence oriented
                        if conf is not None:
                            final_score = float(conf) if final_pred == 1 else (1.0 - float(conf))
                    # else keep edge (no live VLM)
                else:
                    n_fallback += 1
                    sample_lat = hop  # failed attempt still paid partial cost
            else:
                n_local += 1

            lat_e2e_ms.append(sample_lat)
            y_true.append(int(item["label"]))
            y_pred.append(int(final_pred))
            y_score.append(float(final_score))

    elapsed = time.perf_counter() - t0
    y_true_a = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred)
    y_score_a = np.asarray(y_score)
    f1 = float(f1_score(y_true_a, y_pred_a, zero_division=0))
    try:
        auroc = float(roc_auc_score(y_true_a, y_score_a))
    except ValueError:
        auroc = float("nan")

    return {
        "policy": policy,
        "network_profile": profile,
        "n_samples": n,
        "n_want_upload": n_want,
        "n_upload_attempt": n_attempt,
        "n_upload_ok": n_upload_ok,
        "n_fallback": n_fallback,
        "n_local": n_local,
        "n_cloud_decision_from_cache": n_cloud_used_cache,
        "upload_want_rate": n_want / max(1, n),
        "upload_attempt_rate": n_attempt / max(1, n),
        "upload_success_rate": n_upload_ok / max(1, n_attempt),
        "fallback_rate": n_fallback / max(1, n),
        "cloud_review_rate": n_upload_ok / max(1, n),
        "avg_attempt_latency_ms": float(np.mean(lat_attempt_ms)) if lat_attempt_ms else 0.0,
        "avg_success_latency_ms": float(np.mean(lat_success_ms)) if lat_success_ms else 0.0,
        "avg_e2e_extra_latency_ms": float(np.mean(lat_e2e_ms)) if lat_e2e_ms else 0.0,
        "p95_e2e_extra_latency_ms": float(np.percentile(lat_e2e_ms, 95)) if lat_e2e_ms else 0.0,
        "detection_f1": f1,
        "detection_auroc": auroc,
        "wall_s": elapsed,
    }


def _delta(a: dict, b: dict, key: str) -> float | None:
    """b - a (cost_risk - baseline)."""
    if key not in a or key not in b:
        return None
    try:
        return float(b[key]) - float(a[key])
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--categories",
        default="bottle,cable,capsule",
        help="comma-separated MVTec categories with hybrid_lora_8b edge_scores",
    )
    ap.add_argument(
        "--profiles",
        default="good,fair,weak,outage",
        help="network profiles to sweep",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    collab = _load_collab()
    cats = [c.strip() for c in str(args.categories).split(",") if c.strip()]
    profiles = [p.strip() for p in str(args.profiles).split(",") if p.strip()]
    packs = []
    for c in cats:
        pack = _load_category(c)
        if pack is None:
            print(f"[skip] no edge_scores for {c}")
            continue
        packs.append(pack)
        print(f"[load] {c}: n={len(pack['items'])} thr={pack['threshold']:.4f}")
    if not packs:
        print("No categories loaded.", file=sys.stderr)
        return 1

    cloud_extra = float(collab.get("cloud_extra_latency_ms") or 80.0)
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        by_pol: dict[str, dict] = {}
        for policy in ("baseline", "cost_risk"):
            m = _run_policy(
                policy=policy,
                collab=collab,
                packs=packs,
                profile=profile,
                seed=args.seed,
                cloud_extra_ms=cloud_extra,
            )
            by_pol[policy] = m
            rows.append(m)
            print(
                f"[{profile:6}] {policy:10} "
                f"want={m['upload_want_rate']:.3f} ok_rate={m['upload_success_rate']:.3f} "
                f"fallback={m['fallback_rate']:.3f} "
                f"avg_e2e+={m['avg_e2e_extra_latency_ms']:.1f}ms "
                f"F1={m['detection_f1']:.4f}"
            )
        base, crr = by_pol["baseline"], by_pol["cost_risk"]
        print(
            f"[{profile:6}] delta(crr-base): "
            f"want={_delta(base, crr, 'upload_want_rate'):+.3f} "
            f"fallback={_delta(base, crr, 'fallback_rate'):+.3f} "
            f"e2e+={_delta(base, crr, 'avg_e2e_extra_latency_ms'):+.1f}ms "
            f"F1={_delta(base, crr, 'detection_f1'):+.4f}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    out_json = args.out / "compare.json"
    summary = {
        "categories": [p["category"] for p in packs],
        "n_samples_total": sum(len(p["items"]) for p in packs),
        "profiles": profiles,
        "policies": ["baseline", "cost_risk"],
        "rows": rows,
        "notes": [
            "No live VLM: CLOUD_REVIEW uses cached cloud JSON when present, else keeps edge decision.",
            "Latency is network accounting (RTT+TX) + cloud_extra_latency_ms on success.",
            "Same seeds per profile for fair A/B.",
        ],
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # markdown table
    md = [
        "# Collab routing compare: baseline vs cost_risk (CRR)",
        "",
        f"- Categories: {', '.join(summary['categories'])}",
        f"- Samples: {summary['n_samples_total']}",
        f"- Edge scores: `{HYBRID}/<cat>/edge_scores.json`",
        "",
        "| profile | policy | want_rate | upload_ok_rate | fallback_rate | cloud_review_rate | avg_e2e+ms | p95_e2e+ms | F1 | AUROC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        md.append(
            "| {network_profile} | {policy} | {upload_want_rate:.3f} | {upload_success_rate:.3f} | "
            "{fallback_rate:.3f} | {cloud_review_rate:.3f} | {avg_e2e_extra_latency_ms:.1f} | "
            "{p95_e2e_extra_latency_ms:.1f} | {detection_f1:.4f} | {detection_auroc:.4f} |".format(**r)
        )
    md.extend(
        [
            "",
            "## Deltas (cost_risk − baseline)",
            "",
            "| profile | Δwant | Δfallback | Δavg_e2e+ms | ΔF1 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for profile in profiles:
        base = next(r for r in rows if r["network_profile"] == profile and r["policy"] == "baseline")
        crr = next(r for r in rows if r["network_profile"] == profile and r["policy"] == "cost_risk")
        md.append(
            f"| {profile} | {_delta(base, crr, 'upload_want_rate'):+.3f} | "
            f"{_delta(base, crr, 'fallback_rate'):+.3f} | "
            f"{_delta(base, crr, 'avg_e2e_extra_latency_ms'):+.1f} | "
            f"{_delta(base, crr, 'detection_f1'):+.4f} |"
        )
    md.append("")
    out_md = args.out / "compare.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
