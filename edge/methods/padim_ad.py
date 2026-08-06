"""PaDiM helpers that also return anomaly maps + pixel metrics."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .gallery_ad import MethodResult, best_f1_threshold, mvtec_test_split
from .pixel_metrics import best_pixel_f1, pixel_auroc, upsample_amap


def _estimate_resnet18_flops_g(image_size: int = 224) -> float:
    try:
        import torchvision
        from thop import profile

        m = torchvision.models.resnet18(weights=None).eval()
        macs, _ = profile(m, inputs=(torch.randn(1, 3, image_size, image_size),), verbose=False)
        return float(macs) / 1e9
    except Exception:
        return 1.8


def _enable_offline_timm(backbone: str) -> None:
    try:
        import sys

        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root))
        from src.offline_timm import enable as enable_offline_timm

        enable_offline_timm(backbone)
    except Exception:
        pass


def eval_padim_category(
    category: str,
    data_root: Path,
    anomalib_root: Path,
    device: str = "cuda:0",
    method_name: str = "padim_resnet18",
    map_cache: dict[str, np.ndarray] | None = None,
) -> MethodResult:
    """Load trained PaDiM edge ckpt and score the MVTec test split only."""
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import Padim

    meta_path = anomalib_root / category / "edge" / "train_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing {meta_path}; train PaDiM edge first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    backbone = meta.get("backbone") or "resnet18"
    ckpt = meta.get("checkpoint") or (meta.get("extra") or {}).get("checkpoint")
    candidates: list[Path] = []
    if ckpt:
        p = Path(ckpt)
        if not p.is_absolute():
            # relative to repo root
            candidates.append(Path(__file__).resolve().parents[2] / p)
            candidates.append(anomalib_root / p)
        else:
            candidates.append(p)
    edge_dir = anomalib_root / category / "edge"
    candidates.extend(sorted(edge_dir.rglob("*.ckpt")))
    ckpt_path = next((c for c in candidates if c.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(f"no PaDiM ckpt under {edge_dir} (meta={meta_path})")
    ckpt = str(ckpt_path)

    _enable_offline_timm(backbone)

    model = Padim(backbone=backbone)
    dm = MVTecAD(
        root=str(data_root),
        category=category,
        train_batch_size=32,
        eval_batch_size=1,
        num_workers=2,
    )
    engine = Engine(accelerator="gpu" if device.startswith("cuda") else "cpu", devices=1)

    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    preds = engine.predict(model=model, datamodule=dm, ckpt_path=ckpt)
    predict_s = time.perf_counter() - t0

    scores, labels = [], []
    gt_masks, amaps = [], []
    for batch in preds or []:
        s = getattr(batch, "pred_score", None)
        y = getattr(batch, "gt_label", None)
        amap = getattr(batch, "anomaly_map", None)
        gt = getattr(batch, "gt_mask", None)
        paths = getattr(batch, "image_path", None)
        if s is None and isinstance(batch, dict):
            s = batch.get("pred_score")
            y = batch.get("gt_label")
            amap = batch.get("anomaly_map")
            gt = batch.get("gt_mask")
            paths = batch.get("image_path")

        if s is None:
            continue
        if torch.is_tensor(s):
            s = s.detach().cpu().numpy()
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()
        s = np.asarray(s, dtype=float).reshape(-1)
        y = np.asarray(y).astype(int).reshape(-1)
        scores.extend(s.tolist())
        labels.extend(y.tolist())

        if amap is not None and gt is not None:
            if torch.is_tensor(amap):
                amap = amap.detach().cpu().numpy()
            if torch.is_tensor(gt):
                gt = gt.detach().cpu().numpy()
            amap = np.asarray(amap, dtype=np.float32)
            gt = np.asarray(gt).astype(np.uint8)
            # shapes: [B,H,W] or [B,1,H,W]
            if amap.ndim == 4:
                amap = amap[:, 0]
            if gt.ndim == 4:
                gt = gt[:, 0]
            if amap.ndim == 2:
                amap = amap[None]
            if gt.ndim == 2:
                gt = gt[None]
            if paths is None:
                paths = [None] * amap.shape[0]
            elif isinstance(paths, str):
                paths = [paths]
            for i in range(amap.shape[0]):
                a = amap[i]
                g = (gt[i] > 0).astype(np.uint8)
                if a.shape != g.shape:
                    a = upsample_amap(a, g.shape)
                amaps.append(a)
                gt_masks.append(g)
                if map_cache is not None and paths[i] is not None:
                    map_cache[str(Path(paths[i]).resolve())] = a

    labels_a = np.asarray(labels, dtype=int)
    scores_a = np.asarray(scores, dtype=float)
    n_test_expected = len(mvtec_test_split(data_root, category))
    auroc = float(roc_auc_score(labels_a, scores_a)) if len(np.unique(labels_a)) > 1 else float("nan")
    f1, prec, rec, thr = best_f1_threshold(labels_a, scores_a)

    p_auroc = p_f1 = p_prec = p_rec = p_thr = None
    if gt_masks and amaps:
        p_auroc = pixel_auroc(gt_masks, amaps)
        p_f1, p_prec, p_rec, p_thr = best_pixel_f1(gt_masks, amaps)

    peak = None
    if torch.cuda.is_available() and device.startswith("cuda"):
        peak = float(torch.cuda.max_memory_allocated() / (1024**2))

    n = max(1, len(scores_a))
    latency_ms = (predict_s / n) * 1000

    return MethodResult(
        method=method_name,
        category=category,
        n_gallery=int(meta.get("n_gallery") or 0),
        n_test=int(len(scores_a)),
        image_auroc=auroc,
        f1=f1,
        precision=prec,
        recall=rec,
        threshold=thr,
        gallery_build_s=0.0,
        infer_latency_ms_mean=float(latency_ms),
        infer_latency_ms_std=0.0,
        flops_g=_estimate_resnet18_flops_g(224),
        params_m=11.7,
        peak_mem_mb=peak,
        notes=(
            f"Anomalib PaDiM; test-only eval. "
            f"n_test_files={n_test_expected}; predict_wall_s={predict_s:.2f}; ckpt={ckpt}"
        ),
        extra={"backbone": backbone, "checkpoint": ckpt, "train_meta_metrics": meta.get("metrics")},
        pixel_auroc=p_auroc,
        pixel_f1=p_f1,
        pixel_precision=p_prec,
        pixel_recall=p_rec,
        pixel_threshold=p_thr,
    )


# keep old import path working
__all__ = ["eval_padim_category", "_estimate_resnet18_flops_g"]
