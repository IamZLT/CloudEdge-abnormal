#!/usr/bin/env python3
"""Compare heuristic hard-mining vs Qwen3.5 RouteAgent under network profiles.

Env: conda activate clip  (transformers>=5.3 for Qwen3.5)
Prereq (optional): outputs/hybrid/<cat>/edge_scores.json from export_edge_scores.py
  If missing, builds a small score list via edge.infer gallery AD on a subset.

Example:
  CUDA_VISIBLE_DEVICES=0 python scripts/bench_route_agent.py \
    --config configs/default.yaml --category bottle --max-samples 24
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.network_sim import NetworkSimulator, PROFILES
from src.vlm.route_agent import (
    RouteAgent,
    RouteContext,
    heuristic_upload,
    resolve_network_profile,
)


def _resolve_img(path: str, data_root: Path, category: str) -> str:
    p = Path(path)
    if p.exists():
        return str(p)
    alt = data_root / category / "test" / p.parent.name / p.name
    return str(alt) if alt.exists() else str(p)


def load_items(cfg: dict, category: str, max_samples: int | None) -> tuple[list[dict], float, int]:
    """Load edge scores from hybrid export or synthesize via listing test set."""
    hybrid_path = Path(cfg.get("results_dir") or "outputs/hybrid")
    # also try outputs/hybrid even if results_dir is outputs
    candidates = [
        hybrid_path / category / "edge_scores.json",
        ROOT / "outputs" / "hybrid" / category / "edge_scores.json",
    ]
    for ep in candidates:
        if ep.exists():
            pack = json.loads(ep.read_text(encoding="utf-8"))
            items = pack["items"]
            thr = float(pack["threshold"])
            n_gallery = int(pack.get("n_gallery") or pack.get("n_train") or 16)
            if max_samples is not None:
                # keep hard first, then fill
                hard = [it for it in items if it.get("hard")]
                easy = [it for it in items if not it.get("hard")]
                items = (hard + easy)[: int(max_samples)]
            return items, thr, n_gallery

    # fallback: sample test images with placeholder scores (agent still runs)
    data_root = Path(cfg["data_root"])
    test = data_root / category / "test"
    items = []
    for sub in sorted(test.iterdir()):
        if not sub.is_dir():
            continue
        label = 0 if sub.name == "good" else 1
        for img in sorted(sub.glob("*.png"))[:3]:
            items.append(
                {
                    "path": str(img),
                    "label": label,
                    "edge_score": 0.6 if label else 0.2,
                    "edge_pred": label,
                    "hard": True,
                }
            )
    if max_samples is not None:
        items = items[: int(max_samples)]
    thr = 0.5
    print(f"[warn] no edge_scores.json; using {len(items)} placeholder-score images")
    return items, thr, 16


def eval_router(
    *,
    mode: str,
    items: list[dict],
    thr: float,
    n_gallery: int,
    profile: str,
    collab: dict,
    agent: RouteAgent | None,
    data_root: Path,
    category: str,
    up_hard: int,
) -> dict:
    net_cfg = dict(collab.get("network") or {})
    net_cfg["profile"] = profile
    _, net_dict = resolve_network_profile({"network": net_cfg})
    sim = NetworkSimulator.from_config(net_cfg)
    hard_margin = float(collab.get("thr_margin") or collab.get("hard_margin") or 0.05)

    want = []
    ok_mask = []
    final_preds = []
    route_lats = []
    outcomes = []
    sources = []

    for it in items:
        edge_pred = int(it["edge_pred"]) if not isinstance(it["edge_pred"], bool) else int(it["edge_pred"])
        if isinstance(it.get("edge_pred"), str):
            edge_pred = 1 if it["edge_pred"] == "NG" else 0
        edge_decision = "NG" if edge_pred else "OK"
        img = _resolve_img(it["path"], data_root, category)
        ctx = RouteContext(
            image=img,
            category=category,
            n_gallery=n_gallery,
            edge_score=float(it["edge_score"]),
            edge_thr=thr,
            edge_decision=edge_decision,
            network_profile=profile,
            network=net_dict,
            hard_margin=hard_margin,
        )
        if mode == "heuristic":
            upload = heuristic_upload(ctx)
            route_lats.append(0.0)
            sources.append("heuristic")
        else:
            assert agent is not None
            dec = agent.decide(ctx)
            upload = bool(dec.upload)
            route_lats.append(float(dec.latency_ms))
            sources.append(dec.source)

        want.append(upload)
        if not upload:
            ok_mask.append(False)
            final_preds.append(edge_pred)
            continue
        out = sim.try_upload(up_hard)
        outcomes.append(out)
        if out.ok:
            ok_mask.append(True)
            # oracle cloud upper-bound when upload succeeds
            final_preds.append(int(it["label"]))
        else:
            ok_mask.append(False)
            final_preds.append(edge_pred)

    labels = np.asarray([it["label"] for it in items], dtype=int)
    preds = np.asarray(final_preds, dtype=int)
    edge_preds = []
    for it in items:
        ep = it["edge_pred"]
        if isinstance(ep, str):
            edge_preds.append(1 if ep == "NG" else 0)
        else:
            edge_preds.append(int(ep))
    edge_preds = np.asarray(edge_preds, dtype=int)
    net_summary = sim.summarize(outcomes)
    return {
        "mode": mode,
        "profile": profile,
        "n": len(items),
        "upload_want_rate": float(np.mean(want)) if want else 0.0,
        "n_upload_want": int(sum(want)),
        "n_upload_ok": int(sum(ok_mask)),
        "cloud_upload_success_rate": net_summary.get("cloud_upload_success_rate"),
        "mean_route_latency_ms": float(np.mean(route_lats)) if route_lats else 0.0,
        "B1_f1": float(f1_score(labels, edge_preds, zero_division=0)),
        "S_oracle_f1": float(f1_score(labels, preds, zero_division=0)),
        "M4_weak_net_service_keep_rate": 1.0,
        "source_counts": {s: sources.count(s) for s in sorted(set(sources))},
        "network": net_summary,
    }


def to_markdown(category: str, rows: list[dict]) -> str:
    lines = [
        f"# RouteAgent bench — `{category}`",
        "",
        "S_oracle_f1: when upload+net succeed, use ground-truth as perfect cloud (upper bound).",
        "",
        "| Mode | Profile | Want↑ | Upload OK | Route ms | B1-F1 | S-oracle-F1 | M4 |",
        "|------|---------|-------|-----------|----------|-------|-------------|----|",
    ]
    for r in rows:
        ok = r["cloud_upload_success_rate"]
        ok_s = f"{ok:.1%}" if ok == ok and ok is not None else "n/a"
        lines.append(
            f"| {r['mode']} | {r['profile']} | {r['upload_want_rate']:.1%} | {ok_s} | "
            f"{r['mean_route_latency_ms']:.0f} | {r['B1_f1']:.4f} | {r['S_oracle_f1']:.4f} | "
            f"{r['M4_weak_net_service_keep_rate']:.0%} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--category", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--profiles", default="good,fair,weak,outage")
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--out", default=str(ROOT / "outputs/reports/route_agent"))
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    category = args.category or cfg.get("category", "bottle")
    collab = dict(cfg.get("collab") or {})
    ra_cfg = dict(collab.get("route_agent") or {})
    if args.device:
        ra_cfg["device"] = args.device
    elif "device" not in ra_cfg:
        ra_cfg["device"] = cfg.get("device", "cuda:0")

    items, thr, n_gallery = load_items(cfg, category, args.max_samples)
    data_root = Path(cfg["data_root"])
    up_hard = int(collab.get("upload_bytes_hard", 80000))
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    for p in profiles:
        if p not in PROFILES:
            raise SystemExit(f"unknown profile {p}")

    print(f"[route_agent] category={category} n={len(items)} n_gallery={n_gallery}")
    agent = RouteAgent.from_config(ra_cfg)

    rows = []
    for profile in profiles:
        print(f"===== heuristic @ {profile} =====")
        rows.append(
            eval_router(
                mode="heuristic",
                items=items,
                thr=thr,
                n_gallery=n_gallery,
                profile=profile,
                collab=collab,
                agent=None,
                data_root=data_root,
                category=category,
                up_hard=up_hard,
            )
        )
        print(f"===== route_agent @ {profile} =====")
        rows.append(
            eval_router(
                mode="route_agent",
                items=items,
                thr=thr,
                n_gallery=n_gallery,
                profile=profile,
                collab=collab,
                agent=agent,
                data_root=data_root,
                category=category,
                up_hard=up_hard,
            )
        )

    out_dir = Path(args.out) / category
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bench.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    md = to_markdown(category, rows)
    (out_dir / "bench.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {out_dir / 'bench.md'}")


if __name__ == "__main__":
    main()
