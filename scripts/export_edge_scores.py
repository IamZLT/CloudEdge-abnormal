#!/usr/bin/env python3
"""Export Anomalib edge scores + paths for hybrid cloud-VLM review.

Env: conda activate dinov3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.offline_timm import enable as enable_offline_timm


def _best_thr(labels, scores):
    from sklearn.metrics import f1_score

    best_t, best_f1 = float(np.median(scores)), -1.0
    for t in np.quantile(scores, np.linspace(0.05, 0.95, 37)):
        f1 = f1_score(labels, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/hybrid.yaml"))
    p.add_argument("--category", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    category = args.category or cfg.get("category", "bottle")
    device = args.device or cfg.get("edge", {}).get("device", "cuda:0")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    from anomalib.data import MVTecAD
    from anomalib.models import Padim, Patchcore
    from anomalib.engine import Engine

    meta_path = Path(cfg["edge"]["anomalib_root"]) / category / "edge" / "train_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing edge train_meta: {meta_path}. Train Anomalib edge first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    backbone = meta.get("backbone") or "resnet18"
    enable_offline_timm(backbone)
    name = (meta.get("model") or "Padim").lower()
    model = Padim(backbone=backbone) if name == "padim" else Patchcore(backbone=backbone)
    ckpt = meta.get("checkpoint")
    if not ckpt or not Path(ckpt).exists():
        ckpt = str(next(Path(meta["out_dir"]).rglob("*.ckpt")))

    dm = MVTecAD(
        root=cfg["data_root"],
        category=category,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=4,
    )
    engine = Engine(accelerator="gpu" if device.startswith("cuda") else "cpu", devices=1)
    preds = engine.predict(model=model, datamodule=dm, ckpt_path=ckpt)

    scores, labels, paths = [], [], []
    for batch in preds or []:
        s = getattr(batch, "pred_score", None)
        y = getattr(batch, "gt_label", None)
        path = getattr(batch, "image_path", None)
        if s is None and isinstance(batch, dict):
            s, y, path = batch.get("pred_score"), batch.get("gt_label"), batch.get("image_path")
        if s is None:
            continue
        if torch.is_tensor(s):
            s = s.detach().cpu().numpy()
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()
        s = np.asarray(s, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=int).reshape(-1)
        if path is None:
            path = [""] * len(s)
        elif isinstance(path, (str, Path)):
            path = [str(path)] * len(s)
        else:
            path = [str(x) for x in list(path)]
        scores.extend(s.tolist())
        labels.extend(y.tolist())
        # anomalib may return list longer than batch scores if nested
        if len(path) == 1 and len(s) > 1:
            path = path * len(s)
        paths.extend(path[: len(s)])

    scores_a = np.asarray(scores, dtype=float)
    labels_a = np.asarray(labels, dtype=int)
    thr = _best_thr(labels_a, scores_a)

    collab = cfg.get("collab", {})
    band_low = min(float(np.quantile(scores_a, collab.get("low_quantile", 0.35))), thr * 0.85)
    band_high = max(float(np.quantile(scores_a, collab.get("high_quantile", 0.65))), thr * 1.15)
    margin = float(collab.get("thr_margin", 0.08))
    near_thr = np.abs(scores_a - thr) <= (margin * max(1e-6, float(np.std(scores_a)) + abs(thr) * 0.1 + 1e-3))
    hard = ((scores_a >= band_low) & (scores_a <= band_high)) | near_thr

    out = {
        "category": category,
        "edge_model": meta.get("model"),
        "edge_backbone": backbone,
        "checkpoint": ckpt,
        "threshold": thr,
        "band_low": float(band_low),
        "band_high": float(band_high),
        "n": int(len(labels_a)),
        "n_hard": int(hard.sum()),
        "items": [
            {
                "path": paths[i],
                "label": int(labels_a[i]),
                "edge_score": float(scores_a[i]),
                "edge_pred": int(scores_a[i] >= thr),
                "hard": bool(hard[i]),
            }
            for i in range(len(labels_a))
        ],
    }
    out_dir = Path(cfg.get("results_dir", "outputs/hybrid")) / category
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "edge_scores.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{category}] n={out['n']} hard={out['n_hard']} thr={thr:.4f} -> {out_path}")


if __name__ == "__main__":
    main()
