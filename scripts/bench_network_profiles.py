#!/usr/bin/env python3
"""Sweep network profiles (good/fair/weak/outage) on fixed edge/cloud scores.

Reuses Anomalib scores once per category, then applies NetworkSimulator offline
(accounting only — no real sleep). Writes comparison tables under
outputs/reports/network_sim/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.network_sim import PROFILES, NetworkSimulator, apply_collab_uploads


def _best_thr(labels, scores):
    best_t, best_f1 = float(np.median(scores)), -1.0
    for t in np.quantile(scores, np.linspace(0.05, 0.95, 37)):
        f1 = f1_score(labels, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _collect_scores(cfg: dict, category: str, device: str, cache_path: Path):
    """Load or compute edge/cloud scores for one category."""
    if cache_path.exists():
        data = np.load(cache_path)
        return {
            "edge_scores": data["edge_scores"],
            "cloud_scores": data["cloud_scores"],
            "labels": data["labels"],
            "edge_lat": float(data["edge_lat"]),
            "cloud_lat": float(data["cloud_lat"]),
        }

    # reuse anomalib helpers (scripts/ on sys.path)
    from bench_anomalib import _latency_ms, _load, _scores, resolve_anomalib_root
    from src.offline_timm import enable as enable_offline_timm
    from anomalib.data import MVTecAD

    base = resolve_anomalib_root(cfg) / category
    edge_meta = base / "edge" / "train_meta.json"
    cloud_meta = base / "cloud" / "train_meta.json"
    if not edge_meta.exists() or not cloud_meta.exists():
        raise FileNotFoundError(
            f"[{category}] Need edge/cloud train_meta.json under {base}. "
            "Run train_anomalib.py first."
        )

    datamodule = MVTecAD(
        root=cfg["data_root"],
        category=category,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=4,
    )
    enable_offline_timm()
    edge_model, edge_engine, edge_ckpt, _ = _load(edge_meta, device)
    cloud_model, cloud_engine, cloud_ckpt, _ = _load(cloud_meta, device)
    edge_scores, labels = _scores(edge_engine, edge_model, edge_ckpt, datamodule)
    cloud_scores, labels2 = _scores(cloud_engine, cloud_model, cloud_ckpt, datamodule)
    if len(labels2) == len(labels):
        labels = labels2
    edge_lat, _ = _latency_ms(edge_engine, edge_model, edge_ckpt, datamodule, device)
    cloud_lat, _ = _latency_ms(cloud_engine, cloud_model, cloud_ckpt, datamodule, device)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        edge_scores=np.asarray(edge_scores, dtype=np.float64),
        cloud_scores=np.asarray(cloud_scores, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        edge_lat=np.asarray(edge_lat, dtype=np.float64),
        cloud_lat=np.asarray(cloud_lat, dtype=np.float64),
    )
    return {
        "edge_scores": np.asarray(edge_scores, dtype=np.float64),
        "cloud_scores": np.asarray(cloud_scores, dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.int64),
        "edge_lat": float(edge_lat),
        "cloud_lat": float(cloud_lat),
    }


def eval_profile(
    *,
    pack: dict,
    profile: str,
    collab: dict,
    seed: int,
) -> dict:
    edge_scores = pack["edge_scores"]
    cloud_scores = pack["cloud_scores"]
    labels = pack["labels"]
    edge_lat = pack["edge_lat"]
    cloud_lat = pack["cloud_lat"]
    n = len(labels)

    edge_thr = _best_thr(labels, edge_scores)
    cloud_thr = _best_thr(labels, cloud_scores)
    band_low = min(float(np.quantile(edge_scores, collab.get("low_quantile", 0.35))), edge_thr * 0.85)
    band_high = max(float(np.quantile(edge_scores, collab.get("high_quantile", 0.65))), edge_thr * 1.15)
    hard = (edge_scores >= band_low) & (edge_scores <= band_high)
    n_hard = int(hard.sum())
    up_hard = int(collab.get("upload_bytes_hard", 80000))

    sim = NetworkSimulator.from_config({"profile": profile, "seed": seed})
    s_scores, s_lat, cloud_ok, outcomes = apply_collab_uploads(
        hard_mask=hard,
        edge_scores=edge_scores,
        cloud_scores=cloud_scores,
        edge_lat_ms=[edge_lat] * n,
        cloud_lat_ms=[cloud_lat] * n,
        upload_bytes_hard=up_hard,
        sim=sim,
    )
    net = sim.summarize(outcomes)
    s_preds = np.where(cloud_ok, (s_scores >= cloud_thr).astype(int), (s_scores >= edge_thr).astype(int))
    s_f1 = float(f1_score(labels, s_preds, zero_division=0))
    s_auroc = float(roc_auc_score(labels, s_scores)) if len(np.unique(labels)) > 1 else float("nan")
    b1_preds = (edge_scores >= edge_thr).astype(int)
    b1_f1 = float(f1_score(labels, b1_preds, zero_division=0))

    return {
        "profile": profile,
        "n_test": n,
        "n_hard": n_hard,
        "hard_ratio": float(n_hard / max(1, n)),
        "n_cloud_upload_ok": int(cloud_ok.sum()),
        "cloud_upload_success_rate": net.get("cloud_upload_success_rate"),
        "mean_upload_rtt_ms": net.get("mean_upload_rtt_ms"),
        "mean_tx_ms": net.get("mean_tx_ms"),
        "mean_total_net_ms": net.get("mean_total_net_ms"),
        "S_all_mean_ms": float(np.mean(s_lat)),
        "S_f1": s_f1,
        "S_auroc": s_auroc,
        "B1_f1": b1_f1,
        "C5_f1_delta": float(s_f1 - b1_f1),
        "M4_weak_net_service_keep_rate": 1.0,
        "network": net,
    }


def to_markdown(category: str, rows: list[dict]) -> str:
    lines = [
        f"# Network profile sweep — `{category}`",
        "",
        "Hard-example uploads simulated offline (no real sleep). "
        "Upload failure → edge-local fallback; M4 service keep = 1.0.",
        "",
        "| Profile | Hard | Upload OK | RTT ms | Tx ms | S e2e ms | S-F1 | S-AUROC | M4 keep |",
        "|---------|------|-----------|--------|-------|----------|------|---------|---------|",
    ]
    for r in rows:
        ok = r["cloud_upload_success_rate"]
        ok_s = f"{ok:.1%}" if ok == ok else "n/a"
        rtt = r["mean_upload_rtt_ms"]
        tx = r["mean_tx_ms"]
        rtt_s = f"{rtt:.1f}" if rtt == rtt else "-"
        tx_s = f"{tx:.1f}" if tx == tx else "-"
        lines.append(
            f"| {r['profile']} | {r['n_hard']} ({r['hard_ratio']:.1%}) | {ok_s} | "
            f"{rtt_s} | {tx_s} | {r['S_all_mean_ms']:.1f} | {r['S_f1']:.4f} | "
            f"{r['S_auroc']:.4f} | {r['M4_weak_net_service_keep_rate']:.0%} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Sweep good/fair/weak/outage network profiles")
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--category", default=None)
    parser.add_argument("--categories", default=None, help="comma-separated; default: config category")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--profiles",
        default="good,fair,weak,outage",
        help="comma-separated profiles",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--out",
        default=str(ROOT / "outputs/reports/network_sim"),
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = args.device or cfg.get("device", "cuda:0")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    collab = cfg.get("collab", {})
    seed = int(args.seed if args.seed is not None else (collab.get("network") or {}).get("seed", 42))
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    for p in profiles:
        if p not in PROFILES and p != "custom":
            raise SystemExit(f"unknown profile {p}; choose {list(PROFILES)}")

    if args.categories:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    elif args.category:
        categories = [args.category]
    else:
        categories = [cfg["category"]]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    all_rows: dict[str, list[dict]] = {}

    for cat in categories:
        print(f"\n===== network profiles: {cat} =====")
        cache = out_root / cat / "scores.npz"
        pack = _collect_scores(cfg, cat, device, cache)
        rows = [eval_profile(pack=pack, profile=p, collab=collab, seed=seed) for p in profiles]
        all_rows[cat] = rows

        cat_dir = out_root / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        (cat_dir / "profiles.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        md = to_markdown(cat, rows)
        (cat_dir / "profiles.md").write_text(md, encoding="utf-8")
        print(md)

    # summary across categories (mean of numeric fields)
    if len(categories) >= 1:
        summary_lines = [
            "# Network profile sweep — summary",
            "",
            f"Categories: {', '.join(categories)}",
            "",
            "| Profile | mean Upload OK | mean S e2e ms | mean S-F1 | mean M4 keep |",
            "|---------|----------------|---------------|-----------|--------------|",
        ]
        summary = []
        for p in profiles:
            vals = [r for rows in all_rows.values() for r in rows if r["profile"] == p]
            def _m(key):
                arr = [v[key] for v in vals if v.get(key) == v.get(key)]
                return float(np.mean(arr)) if arr else float("nan")

            row = {
                "profile": p,
                "cloud_upload_success_rate": _m("cloud_upload_success_rate"),
                "S_all_mean_ms": _m("S_all_mean_ms"),
                "S_f1": _m("S_f1"),
                "M4_weak_net_service_keep_rate": _m("M4_weak_net_service_keep_rate"),
            }
            summary.append(row)
            ok = row["cloud_upload_success_rate"]
            ok_s = f"{ok:.1%}" if ok == ok else "n/a"
            summary_lines.append(
                f"| {p} | {ok_s} | {row['S_all_mean_ms']:.1f} | {row['S_f1']:.4f} | "
                f"{row['M4_weak_net_service_keep_rate']:.0%} |"
            )
        summary_lines.append("")
        (out_root / "summary.json").write_text(
            json.dumps({"categories": categories, "profiles": summary, "per_category": all_rows}, indent=2),
            encoding="utf-8",
        )
        (out_root / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
        print("\n".join(summary_lines))
        print(f"Wrote {out_root / 'summary.md'}")


if __name__ == "__main__":
    main()
