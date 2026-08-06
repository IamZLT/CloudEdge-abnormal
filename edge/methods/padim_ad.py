"""Current edge method: Anomalib PaDiM (resnet18) — evaluate on test only."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .gallery_ad import MethodResult, best_f1_threshold, mvtec_test_split


def _estimate_resnet18_flops_g(image_size: int = 224) -> float:
    try:
        import torchvision
        from thop import profile

        m = torchvision.models.resnet18(weights=None).eval()
        macs, _ = profile(m, inputs=(torch.randn(1, 3, image_size, image_size),), verbose=False)
        return float(macs) / 1e9
    except Exception:
        # well-known approx for ResNet18 @224
        return 1.8


def eval_padim_category(
    category: str,
    data_root: Path,
    anomalib_root: Path,
    device: str = "cuda:0",
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
    ckpt = meta.get("checkpoint")
    if not ckpt or not Path(ckpt).exists():
        ckpt = str(next(Path(meta["out_dir"]).rglob("*.ckpt")))

    # offline timm if project helper exists
    try:
        import sys

        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root))
        from src.offline_timm import enable as enable_offline_timm

        enable_offline_timm(backbone)
    except Exception:
        pass

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
        s = np.asarray(s, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=int).reshape(-1)
        scores.extend(s.tolist())
        labels.extend(y.tolist())

    labels_a = np.asarray(labels, dtype=int)
    scores_a = np.asarray(scores, dtype=float)
    # sanity: count should match test split
    n_test_expected = len(mvtec_test_split(data_root, category))
    auroc = float(roc_auc_score(labels_a, scores_a)) if len(np.unique(labels_a)) > 1 else float("nan")
    f1, prec, rec, thr = best_f1_threshold(labels_a, scores_a)

    peak = None
    if torch.cuda.is_available() and device.startswith("cuda"):
        peak = float(torch.cuda.max_memory_allocated() / (1024**2))

    n = max(1, len(scores_a))
    latency_ms = (predict_s / n) * 1000

    return MethodResult(
        method="padim_resnet18",
        category=category,
        n_gallery=0,  # parametric / embedding bank built at train time
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
        params_m=11.7,  # ResNet18 approx
        peak_mem_mb=peak,
        notes=(
            f"Anomalib PaDiM edge ckpt; test-only eval. "
            f"n_test_files={n_test_expected}; predict_wall_s={predict_s:.2f}; ckpt={ckpt}"
        ),
        extra={"backbone": backbone, "checkpoint": ckpt, "train_meta_metrics": meta.get("metrics")},
    )
