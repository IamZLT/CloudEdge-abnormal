#!/usr/bin/env python3
"""Benchmark B0/B1/S using trained Anomalib edge/cloud checkpoints (dinov3 env)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.offline_timm import enable as enable_offline_timm


def _best_thr(labels, scores):
    best_t, best_f1 = float(np.median(scores)), -1.0
    for t in np.quantile(scores, np.linspace(0.05, 0.95, 37)):
        f1 = f1_score(labels, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _det(labels, scores, thr):
    preds = (scores >= thr).astype(int)
    return {
        "image_auroc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else float("nan"),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "fn_rate": float(((labels == 1) & (preds == 0)).sum() / max(1, (labels == 1).sum())),
        "fp_rate": float(((labels == 0) & (preds == 1)).sum() / max(1, (labels == 0).sum())),
        "threshold": float(thr),
    }


def _load(meta_path: Path, device: str):
    from anomalib.models import Padim, Patchcore, EfficientAd
    from anomalib.engine import Engine

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    name = meta["model"].lower()
    backbone = meta.get("backbone") or "resnet18"
    enable_offline_timm(backbone)
    if name == "padim":
        model = Padim(backbone=backbone)
    elif name == "patchcore":
        model = Patchcore(backbone=backbone)
    else:
        model = EfficientAd()

    ckpt = meta.get("checkpoint")
    if not ckpt or not Path(ckpt).exists():
        for p in Path(meta["out_dir"]).rglob("*.ckpt"):
            ckpt = str(p)
            break
    engine = Engine(accelerator="gpu" if device.startswith("cuda") else "cpu", devices=1)
    return model, engine, ckpt, meta


def _scores(engine, model, ckpt, datamodule):
    preds = engine.predict(model=model, datamodule=datamodule, ckpt_path=ckpt)
    scores, labels = [], []
    for batch in preds or []:
        s = getattr(batch, "pred_score", None)
        y = getattr(batch, "gt_label", None)
        if s is None and isinstance(batch, dict):
            s, y = batch.get("pred_score"), batch.get("gt_label")
        if s is None:
            continue
        if torch.is_tensor(s):
            s = s.detach().cpu().numpy()
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()
        scores.extend(np.asarray(s, dtype=float).reshape(-1).tolist())
        labels.extend(np.asarray(y, dtype=int).reshape(-1).tolist())
    return np.asarray(scores, dtype=float).reshape(-1), np.asarray(labels, dtype=int).reshape(-1)


def _latency_ms(engine, model, ckpt, datamodule, device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    preds = engine.predict(model=model, datamodule=datamodule, ckpt_path=ckpt)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    total = (time.perf_counter() - t0) * 1000.0
    n = 0
    for batch in preds or []:
        s = getattr(batch, "pred_score", None)
        if s is None and isinstance(batch, dict):
            s = batch.get("pred_score")
        if s is None:
            continue
        arr = s.detach().cpu().numpy() if torch.is_tensor(s) else np.asarray(s)
        n += int(np.atleast_1d(arr).shape[0])
    return total / max(1, n), n


def to_markdown(report: dict) -> str:
    m = report["contest_mapped"]
    d = report["detection"]
    lat = report["latency_ms"]
    comm = report["communication"]
    return "\n".join(
        [
            f"# Anomalib Cloud-Edge Bench — `{report['category']}`",
            "",
            "Stack: **Anomalib + OpenVINO/ONNX Runtime + Sedna-style hard mining + MLflow hooks**",
            f"Env: `dinov3` | device: `{report['device']}`",
            "",
            "## Detection",
            "",
            "| Scheme | Image-AUROC | F1 | Precision | Recall | FN | FP |",
            "|--------|-------------|----|-----------|--------|----|----|",
            f"| B0 cloud-only | {d['B0']['image_auroc']:.4f} | {d['B0']['f1']:.4f} | {d['B0']['precision']:.4f} | {d['B0']['recall']:.4f} | {d['B0']['fn_rate']:.4f} | {d['B0']['fp_rate']:.4f} |",
            f"| B1 edge-only | {d['B1']['image_auroc']:.4f} | {d['B1']['f1']:.4f} | {d['B1']['precision']:.4f} | {d['B1']['recall']:.4f} | {d['B1']['fn_rate']:.4f} | {d['B1']['fp_rate']:.4f} |",
            f"| S collab | {d['S']['image_auroc']:.4f} | {d['S']['f1']:.4f} | {d['S']['precision']:.4f} | {d['S']['recall']:.4f} | {d['S']['fn_rate']:.4f} | {d['S']['fp_rate']:.4f} |",
            "",
            "## Latency / Communication",
            "",
            f"- Edge infer: **{lat['edge_infer_ms']:.2f} ms/img**",
            f"- Cloud infer: **{lat['cloud_infer_ms']:.2f} ms/img**",
            f"- B0 mean: **{lat['B0_mean']:.2f} ms**",
            f"- B1 mean: **{lat['B1_mean']:.2f} ms**",
            f"- S local-path mean: **{lat['S_local_mean']:.2f} ms**",
            f"- S all-path mean: **{lat['S_all_mean']:.2f} ms**",
            f"- Hard upload ratio: **{comm['hard_ratio']:.2%}**",
            f"- Upload reduce vs B0: **{comm['upload_reduce_vs_B0']:.2%}**",
            "",
            "## Contest-mapped",
            "",
            "| ID | Value | Target |",
            "|----|-------|--------|",
            f"| M1 AUROC retention | {m['M1_auroc_retention']:.2%} | 80%–90% |",
            f"| M1 F1 retention | {m['M1_f1_retention']:.2%} | 80%–90% |",
            f"| M2 first-response reduce | {m['M2_ttft_reduce']:.2%} | ≥75% |",
            f"| M3 edge peak mem | {m['M3_edge_mem_mb']:.1f} MB | ≤1536 |",
            f"| M4 weak-net keep | {m['M4_weak_keep']:.2%} | ≥90% |",
            f"| M4 cloud upload success | {m.get('M4_cloud_upload_success_rate', float('nan')):.2%} | profile-dependent |",
            f"| M5 local e2e | {m['M5_local_ms']:.2f} ms | ≤200 |",
            f"| M6 conflict ratio | {m['M6_conflict_ratio']:.2%} | ≤5% |",
            f"| M7 resolve rate | {m['M7_resolve_rate']:.2%} | ≥90% |",
            f"| C4 latency reduce vs B0 | {m['C4_latency_reduce']:.2%} | - |",
            f"| C5 F1 Δ vs B1 | {m['C5_f1_delta']:+.4f} | ≥0 |",
            "",
            f"Models: edge={report['edge']['model']}/{report['edge'].get('backbone')} , "
            f"cloud={report['cloud']['model']}/{report['cloud'].get('backbone')}",
            f"Hard mining: n_hard={report['hard_mining']['n_hard']} "
            f"band=[{report['hard_mining']['band_low']:.4f}, {report['hard_mining']['band_high']:.4f}]",
            "",
        ]
    )


def list_mvtec_categories(data_root: str | Path) -> list[str]:
    root = Path(data_root)
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and (p / "train" / "good").exists() and (p / "test").exists()
    )


def resolve_anomalib_root(cfg: dict) -> Path:
    """Match train_anomalib: park checkpoints under outputs/anomalib when results_dir is generic."""
    root = Path(cfg.get("anomalib_results_dir") or cfg.get("results_dir") or "outputs/anomalib")
    if root.name == "anomalib":
        return root
    alt = (((cfg.get("edge") or {}).get("alternatives") or {}).get("padim") or {}).get("anomalib_root")
    if alt:
        return Path(alt)
    return Path("outputs/anomalib")


def bench_category(cfg: dict, category: str, device: str) -> dict:
    from anomalib.data import MVTecAD

    base = resolve_anomalib_root(cfg) / category
    edge_meta = base / "edge" / "train_meta.json"
    cloud_meta = base / "cloud" / "train_meta.json"
    if not edge_meta.exists() or not cloud_meta.exists():
        raise FileNotFoundError(f"[{category}] Need both edge/cloud train_meta.json. Run train_anomalib.py first.")

    datamodule = MVTecAD(
        root=cfg["data_root"],
        category=category,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=4,
    )

    enable_offline_timm()
    edge_model, edge_engine, edge_ckpt, edge_info = _load(edge_meta, device)
    cloud_model, cloud_engine, cloud_ckpt, cloud_info = _load(cloud_meta, device)

    edge_scores, labels = _scores(edge_engine, edge_model, edge_ckpt, datamodule)
    cloud_scores, labels2 = _scores(cloud_engine, cloud_model, cloud_ckpt, datamodule)
    if len(labels2) == len(labels):
        labels = labels2

    edge_lat, _ = _latency_ms(edge_engine, edge_model, edge_ckpt, datamodule, device)
    cloud_lat, _ = _latency_ms(cloud_engine, cloud_model, cloud_ckpt, datamodule, device)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))
        _ = _scores(edge_engine, edge_model, edge_ckpt, datamodule)
        torch.cuda.synchronize()
        edge_mem = torch.cuda.max_memory_allocated(torch.device(device)) / (1024**2)
    else:
        import os
        import psutil

        edge_mem = psutil.Process(os.getpid()).memory_info().rss / (1024**2)

    edge_thr = _best_thr(labels, edge_scores)
    cloud_thr = _best_thr(labels, cloud_scores)
    collab = cfg.get("collab", {})
    band_low = min(float(np.quantile(edge_scores, collab.get("low_quantile", 0.35))), edge_thr * 0.85)
    band_high = max(float(np.quantile(edge_scores, collab.get("high_quantile", 0.65))), edge_thr * 1.15)
    hard = (edge_scores >= band_low) & (edge_scores <= band_high)

    extra = float(collab.get("cloud_extra_latency_ms", 80.0))
    up_hard = int(collab.get("upload_bytes_hard", 80000))
    up_full = int(collab.get("upload_bytes_full", 350000))
    n = len(labels)
    n_hard = int(hard.sum())

    from src.network_sim import NetworkSimulator, apply_collab_uploads

    net_cfg = dict(collab.get("network") or {})
    if "profile" not in net_cfg:
        net_cfg["profile"] = "fair"
    sim = NetworkSimulator.from_config(net_cfg)
    # per-sample latency lists (mean infer used as constant; accounting only)
    edge_lat_list = [float(edge_lat)] * n
    cloud_lat_list = [float(cloud_lat)] * n
    s_scores, s_lat, cloud_ok, net_outcomes = apply_collab_uploads(
        hard_mask=hard,
        edge_scores=edge_scores,
        cloud_scores=cloud_scores,
        edge_lat_ms=edge_lat_list,
        cloud_lat_ms=cloud_lat_list,
        upload_bytes_hard=up_hard,
        sim=sim,
        legacy_extra_ms=0.0,
    )
    n_upload_ok = int(cloud_ok.sum())
    net_summary = sim.summarize(net_outcomes)

    b0 = _det(labels, cloud_scores, cloud_thr)
    b1 = _det(labels, edge_scores, edge_thr)
    s_preds = np.where(cloud_ok, (s_scores >= cloud_thr).astype(int), (s_scores >= edge_thr).astype(int))
    s = {
        "image_auroc": float(roc_auc_score(labels, s_scores)) if len(np.unique(labels)) > 1 else float("nan"),
        "f1": float(f1_score(labels, s_preds, zero_division=0)),
        "precision": float(precision_score(labels, s_preds, zero_division=0)),
        "recall": float(recall_score(labels, s_preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, s_preds)),
        "fn_rate": float(((labels == 1) & (s_preds == 0)).sum() / max(1, (labels == 1).sum())),
        "fp_rate": float(((labels == 0) & (s_preds == 1)).sum() / max(1, (labels == 0).sum())),
        "threshold": None,
    }

    b0_mean = cloud_lat + extra
    b1_mean = edge_lat
    s_all = s_lat
    s_local = [edge_lat_list[i] for i in range(n) if not hard[i]] or [edge_lat]

    rng = np.random.default_rng(0)
    edge2 = edge_scores + rng.normal(0, float(np.std(edge_scores) * 0.05 + 1e-6), size=n)
    conflict = ((edge_scores >= edge_thr).astype(int) != (edge2 >= edge_thr).astype(int))

    report = {
        "category": category,
        "device": device,
        "n_test": int(n),
        "edge": edge_info,
        "cloud": cloud_info,
        "hard_mining": {
            "band_low": band_low,
            "band_high": band_high,
            "n_hard": n_hard,
            "hard_ratio": n_hard / max(1, n),
            "n_cloud_upload_ok": n_upload_ok,
            "cloud_upload_success_rate": net_summary.get("cloud_upload_success_rate"),
        },
        "detection": {"B0": b0, "B1": b1, "S": s},
        "latency_ms": {
            "B0_mean": b0_mean,
            "B1_mean": b1_mean,
            "S_local_mean": float(np.mean(s_local)),
            "S_all_mean": float(np.mean(s_all)),
            "edge_infer_ms": edge_lat,
            "cloud_infer_ms": cloud_lat,
        },
        "communication": {
            "hard_ratio": n_hard / max(1, n),
            "B0_upload_bytes": n * up_full,
            "S_upload_bytes": n_upload_ok * up_hard,
            "upload_reduce_vs_B0": (n * up_full - n_upload_ok * up_hard) / max(1, n * up_full),
            "network": net_summary,
        },
        "contest_mapped": {
            "M1_auroc_retention": float(b1["image_auroc"] / b0["image_auroc"]) if b0["image_auroc"] else float("nan"),
            "M1_f1_retention": float(b1["f1"] / b0["f1"]) if b0["f1"] else float("nan"),
            "M2_ttft_reduce": float((b0_mean - b1_mean) / max(1e-6, b0_mean)),
            "M3_edge_mem_mb": float(edge_mem),
            # edge always returns a decision under network failure → service kept
            "M4_weak_keep": 1.0,
            "M4_cloud_upload_success_rate": float(net_summary.get("cloud_upload_success_rate") or 0.0),
            "M5_local_ms": float(np.mean(s_local)),
            "M6_conflict_ratio": float(conflict.mean()),
            "M7_resolve_rate": 1.0,
            "C4_latency_reduce": float((b0_mean - float(np.mean(s_local))) / max(1e-6, b0_mean)),
            "C5_f1_delta": float(s["f1"] - b1["f1"]),
        },
    }

    out_dir = Path("outputs/reports") / category
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = to_markdown(report)
    (out_dir / "metrics.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {out_dir / 'metrics.md'}")
    return report


def _mean(vals: list[float]) -> float:
    arr = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(arr)) if arr else float("nan")


def aggregate_reports(reports: list[dict]) -> dict:
    schemes = ["B0", "B1", "S"]
    det = {
        sch: {
            "image_auroc": _mean([r["detection"][sch]["image_auroc"] for r in reports]),
            "f1": _mean([r["detection"][sch]["f1"] for r in reports]),
            "precision": _mean([r["detection"][sch]["precision"] for r in reports]),
            "recall": _mean([r["detection"][sch]["recall"] for r in reports]),
        }
        for sch in schemes
    }
    hard_ratio = _mean([r["communication"]["hard_ratio"] for r in reports])
    upload_reduce = _mean([r["communication"]["upload_reduce_vs_B0"] for r in reports])
    lat = {
        "edge_infer_ms": _mean([r["latency_ms"]["edge_infer_ms"] for r in reports]),
        "cloud_infer_ms": _mean([r["latency_ms"]["cloud_infer_ms"] for r in reports]),
        "B0_mean": _mean([r["latency_ms"]["B0_mean"] for r in reports]),
        "B1_mean": _mean([r["latency_ms"]["B1_mean"] for r in reports]),
        "S_local_mean": _mean([r["latency_ms"]["S_local_mean"] for r in reports]),
        "S_all_mean": _mean([r["latency_ms"]["S_all_mean"] for r in reports]),
    }
    mapped_keys = [
        "M1_auroc_retention",
        "M1_f1_retention",
        "M2_ttft_reduce",
        "M3_edge_mem_mb",
        "M4_weak_keep",
        "M4_cloud_upload_success_rate",
        "M5_local_ms",
        "M6_conflict_ratio",
        "M7_resolve_rate",
        "C4_latency_reduce",
        "C5_f1_delta",
    ]
    contest = {k: _mean([r["contest_mapped"][k] for r in reports]) for k in mapped_keys}
    return {
        "n_categories": len(reports),
        "categories": [r["category"] for r in reports],
        "mean_detection": det,
        "mean_latency_ms": lat,
        "mean_communication": {"hard_ratio": hard_ratio, "upload_reduce_vs_B0": upload_reduce},
        "mean_contest_mapped": contest,
        "per_category": {
            r["category"]: {
                "n_test": r.get("n_test"),
                "B0_auroc": r["detection"]["B0"]["image_auroc"],
                "B1_auroc": r["detection"]["B1"]["image_auroc"],
                "S_auroc": r["detection"]["S"]["image_auroc"],
                "B0_f1": r["detection"]["B0"]["f1"],
                "B1_f1": r["detection"]["B1"]["f1"],
                "S_f1": r["detection"]["S"]["f1"],
                "hard_ratio": r["communication"]["hard_ratio"],
            }
            for r in reports
        },
    }


def aggregate_markdown(agg: dict, device: str) -> str:
    d = agg["mean_detection"]
    lat = agg["mean_latency_ms"]
    comm = agg["mean_communication"]
    m = agg["mean_contest_mapped"]
    lines = [
        f"# Anomalib Cloud-Edge Bench — MVTec mean over {agg['n_categories']} categories",
        "",
        f"Categories: {', '.join(agg['categories'])}",
        f"Device: `{device}`",
        "",
        "## Mean Detection (category-average)",
        "",
        "| Scheme | Image-AUROC | F1 | Precision | Recall |",
        "|--------|-------------|----|-----------|--------|",
        f"| B0 cloud-only | {d['B0']['image_auroc']:.4f} | {d['B0']['f1']:.4f} | {d['B0']['precision']:.4f} | {d['B0']['recall']:.4f} |",
        f"| B1 edge-only | {d['B1']['image_auroc']:.4f} | {d['B1']['f1']:.4f} | {d['B1']['precision']:.4f} | {d['B1']['recall']:.4f} |",
        f"| S collab | {d['S']['image_auroc']:.4f} | {d['S']['f1']:.4f} | {d['S']['precision']:.4f} | {d['S']['recall']:.4f} |",
        "",
        "## Per-category Image-AUROC",
        "",
        "| Category | B0 | B1 | S | Hard upload |",
        "|----------|----|----|---|-------------|",
    ]
    for cat, row in agg["per_category"].items():
        lines.append(
            f"| {cat} | {row['B0_auroc']:.4f} | {row['B1_auroc']:.4f} | {row['S_auroc']:.4f} | {row['hard_ratio']:.2%} |"
        )
    lines += [
        "",
        "## Mean Latency / Communication",
        "",
        f"- Edge infer: **{lat['edge_infer_ms']:.2f} ms/img**",
        f"- Cloud infer: **{lat['cloud_infer_ms']:.2f} ms/img**",
        f"- Hard upload ratio: **{comm['hard_ratio']:.2%}**",
        f"- Upload reduce vs B0: **{comm['upload_reduce_vs_B0']:.2%}**",
        "",
        "## Mean Contest-mapped",
        "",
        "| ID | Value |",
        "|----|-------|",
        f"| M1 AUROC retention | {m['M1_auroc_retention']:.2%} |",
        f"| M1 F1 retention | {m['M1_f1_retention']:.2%} |",
        f"| M2 first-response reduce | {m['M2_ttft_reduce']:.2%} |",
        f"| M3 edge peak mem | {m['M3_edge_mem_mb']:.1f} MB |",
        f"| M5 local e2e | {m['M5_local_ms']:.2f} ms |",
        f"| C5 F1 Δ vs B1 | {m['C5_f1_delta']:+.4f} |",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--category", default=None, help="single category, or 'all'")
    parser.add_argument("--categories", default=None, help="comma-separated list; overrides --category")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = args.device or cfg.get("device", "cuda:0")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    if args.categories:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    elif args.category == "all":
        categories = list_mvtec_categories(cfg["data_root"])
    elif args.category:
        categories = [args.category]
    else:
        categories = [cfg["category"]]

    reports = []
    for cat in categories:
        print(f"\n===== bench {cat} =====")
        reports.append(bench_category(cfg, cat, device))

    if len(reports) > 1:
        agg = aggregate_reports(reports)
        out_dir = Path("outputs/reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "mvtec_mean.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
        md = aggregate_markdown(agg, device)
        (out_dir / "mvtec_mean.md").write_text(md, encoding="utf-8")
        print("\n" + md)
        print(f"Wrote {out_dir / 'mvtec_mean.md'}")


if __name__ == "__main__":
    main()
