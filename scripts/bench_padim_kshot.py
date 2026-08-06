#!/usr/bin/env python3
"""Train+eval PaDiM edge with k-shot normal samples only (fair vs feature-gallery).

Creates a temporary MVTec-like tree with:
  train/good = k sampled images (seed-controlled, train/good only)
  test/*     = symlinks to original test split

Env: conda activate dinov3
Example:
  CUDA_VISIBLE_DEVICES=3 python scripts/bench_padim_kshot.py --shots 16 --categories all
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge.methods.gallery_ad import list_images, mvtec_test_split, mvtec_train_good
from edge.methods.padim_ad import _estimate_resnet18_flops_g

CATS = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def _best_f1(labels, scores):
    best = (-1.0, 0.0, 0.0, float(np.median(scores)))
    for t in np.quantile(scores, np.linspace(0.02, 0.98, 49)):
        pred = (scores >= t).astype(int)
        f1 = float(f1_score(labels, pred, zero_division=0))
        if f1 > best[0]:
            best = (
                f1,
                float(precision_score(labels, pred, zero_division=0)),
                float(recall_score(labels, pred, zero_division=0)),
                float(t),
            )
    return best


def prepare_kshot_tree(
    data_root: Path,
    staging: Path,
    category: str,
    shots: int,
    seed: int,
) -> tuple[Path, list[str]]:
    """Build staging/mvtec/{cat} with k train/good + full test via symlinks."""
    rng = np.random.default_rng(seed)
    train_all = mvtec_train_good(data_root, category)
    if not train_all:
        raise FileNotFoundError(f"no train/good for {category}")
    k = min(shots, len(train_all))
    idx = rng.choice(len(train_all), size=k, replace=False)
    selected = [train_all[i] for i in sorted(idx)]

    cat_root = staging / category
    if cat_root.exists():
        shutil.rmtree(cat_root)
    good_dir = cat_root / "train" / "good"
    good_dir.mkdir(parents=True)
    for p in selected:
        (good_dir / p.name).symlink_to(p.resolve())

    # link entire test tree
    src_test = data_root / category / "test"
    dst_test = cat_root / "test"
    dst_test.mkdir(parents=True)
    for sub in sorted(src_test.iterdir()):
        if not sub.is_dir():
            continue
        dsub = dst_test / sub.name
        dsub.mkdir()
        for img in list_images(sub):
            (dsub / img.name).symlink_to(img.resolve())

    # ground_truth required by Anomalib MVTec parser (mask alignment)
    src_gt = data_root / category / "ground_truth"
    if src_gt.exists():
        dst_gt = cat_root / "ground_truth"
        dst_gt.mkdir(parents=True)
        for sub in sorted(src_gt.iterdir()):
            if not sub.is_dir():
                continue
            dsub = dst_gt / sub.name
            dsub.mkdir()
            for img in list_images(sub):
                (dsub / img.name).symlink_to(img.resolve())

    return cat_root, [str(p) for p in selected]


def train_and_eval_category(
    category: str,
    data_root: Path,
    staging: Path,
    out_root: Path,
    shots: int,
    seed: int,
    device: str,
    backbone: str = "resnet18",
) -> dict:
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import Padim

    from src.offline_timm import enable as enable_offline_timm

    enable_offline_timm(backbone)
    cat_root, selected = prepare_kshot_tree(data_root, staging, category, shots, seed)
    # MVTecAD expects root containing category folders
    dm = MVTecAD(
        root=str(staging),
        category=category,
        train_batch_size=min(16, shots),
        eval_batch_size=8,
        num_workers=2,
    )
    model = Padim(backbone=backbone)
    out_dir = out_root / category / "edge"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    engine = Engine(
        accelerator="gpu" if device.startswith("cuda") else "cpu",
        devices=1,
        default_root_dir=str(out_dir),
        max_epochs=1,
        enable_checkpointing=True,
    )

    print(f"[{category}] train PaDiM {backbone} on {len(selected)}-shot good ...")
    t0 = time.perf_counter()
    engine.fit(model=model, datamodule=dm)
    train_s = time.perf_counter() - t0

    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t1 = time.perf_counter()
    preds = engine.predict(model=model, datamodule=dm)
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.synchronize()
    predict_s = time.perf_counter() - t1

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

    labels_a = np.asarray(labels, dtype=int)
    scores_a = np.asarray(scores, dtype=float)
    n_test_files = len(mvtec_test_split(data_root, category))
    auroc = float(roc_auc_score(labels_a, scores_a)) if len(np.unique(labels_a)) > 1 else float("nan")
    f1, prec, rec, thr = _best_f1(labels_a, scores_a)
    peak = (
        float(torch.cuda.max_memory_allocated() / (1024**2))
        if torch.cuda.is_available() and device.startswith("cuda")
        else None
    )
    n = max(1, len(scores_a))

    # locate ckpt
    ckpt = next(out_dir.rglob("*.ckpt"), None)
    meta = {
        "method": "padim_resnet18_16shot",
        "category": category,
        "n_gallery": len(selected),
        "n_test": int(len(scores_a)),
        "n_test_files": n_test_files,
        "image_auroc": auroc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "threshold": thr,
        "gallery_build_s": float(train_s),
        "infer_latency_ms_mean": float(predict_s / n * 1000),
        "infer_latency_ms_std": 0.0,
        "flops_g": _estimate_resnet18_flops_g(224),
        "params_m": 11.7,
        "peak_mem_mb": peak,
        "notes": f"PaDiM trained on {len(selected)}-shot train/good only; eval on full test",
        "extra": {
            "shots": len(selected),
            "seed": seed,
            "selected_train_good": selected,
            "checkpoint": str(ckpt) if ckpt else None,
            "staging": str(cat_root),
            "train_s": train_s,
            "predict_s": predict_s,
        },
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"[{category}] AUROC={auroc:.4f} F1={f1:.4f} "
        f"train={train_s:.1f}s lat~{meta['infer_latency_ms_mean']:.1f}ms n_test={len(scores_a)}"
    )
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(ROOT / "datasets" / "mvtec"))
    ap.add_argument("--staging", default=str(ROOT / "outputs" / "mvtec_kshot_staging"))
    ap.add_argument("--out-root", default=str(ROOT / "outputs" / "anomalib_16shot"))
    ap.add_argument("--report-dir", default=str(ROOT / "outputs" / "reports" / "edge_methods"))
    ap.add_argument("--shots", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--categories", nargs="+", default=["all"])
    ap.add_argument("--tag", default="padim16shot")
    args = ap.parse_args()

    cats = CATS if args.categories == ["all"] else args.categories
    data_root = Path(args.data_root)
    staging = Path(args.staging)
    out_root = Path(args.out_root)
    report_dir = Path(args.report_dir)
    staging.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for cat in cats:
        rows.append(
            train_and_eval_category(
                cat,
                data_root,
                staging,
                out_root,
                args.shots,
                args.seed,
                args.device,
            )
        )
        # incremental save
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / f"edge_methods_{args.tag}.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )

    mean_auroc = float(np.mean([r["image_auroc"] for r in rows]))
    mean_f1 = float(np.mean([r["f1"] for r in rows]))
    mean_lat = float(np.mean([r["infer_latency_ms_mean"] for r in rows]))
    lines = [
        f"# PaDiM {args.shots}-shot (train/good subsample) · MVTec",
        "",
        f"Seed={args.seed}. Train bank = {args.shots} normals only; eval = full test.",
        "",
        f"- mean Image-AUROC: **{mean_auroc:.4f}**",
        f"- mean F1: **{mean_f1:.4f}**",
        f"- mean latency: **{mean_lat:.1f} ms/img**",
        "",
        "| Category | n_shot | n_test | Image-AUROC | F1 | Prec | Rec | Latency ms |",
        "|----------|--------|--------|-------------|----|------|-----|------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['category']} | {r['n_gallery']} | {r['n_test']} | {r['image_auroc']:.4f} | "
            f"{r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['infer_latency_ms_mean']:.1f} |"
        )
    # compare to full PaDiM if available
    full_path = report_dir / "edge_methods_padim16all.json"
    if full_path.exists():
        full = json.loads(full_path.read_text())
        full_mean = float(np.mean([r["image_auroc"] for r in full]))
        lines += [
            "",
            "## vs full-good PaDiM",
            f"- full-good mean Image-AUROC: **{full_mean:.4f}**",
            f"- {args.shots}-shot mean Image-AUROC: **{mean_auroc:.4f}** (Δ={mean_auroc - full_mean:+.4f})",
        ]
    md = report_dir / f"edge_methods_{args.tag}.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {md}")
    print(f"MEAN AUROC={mean_auroc:.4f} F1={mean_f1:.4f}")


if __name__ == "__main__":
    main()
