"""Shared feature-gallery anomaly detection utilities.

Protocol (strict MVTec split):
  - Gallery / memory bank: ONLY train/good
  - Evaluation: ONLY test/*  (good + defect)
  - Never mixes test images into the gallery
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class MethodResult:
    method: str
    category: str
    n_gallery: int
    n_test: int
    image_auroc: float
    f1: float
    precision: float
    recall: float
    threshold: float
    gallery_build_s: float
    infer_latency_ms_mean: float
    infer_latency_ms_std: float
    flops_g: float | None
    params_m: float | None
    peak_mem_mb: float | None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def mvtec_train_good(data_root: Path, category: str) -> list[Path]:
    return list_images(data_root / category / "train" / "good")


def mvtec_test_split(data_root: Path, category: str) -> list[tuple[Path, int]]:
    """Return (path, label) with label 0=good, 1=defect. Test set only."""
    test_root = data_root / category / "test"
    items: list[tuple[Path, int]] = []
    for sub in sorted(test_root.iterdir()):
        if not sub.is_dir():
            continue
        y = 0 if sub.name == "good" else 1
        for p in list_images(sub):
            items.append((p, y))
    return items


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, float, float]:
    """Return best_f1, precision, recall, threshold (calibrated on the same split — report as optimistic)."""
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


class FeatureGalleryAD:
    """Encode images → L2 feature → NN distance to normal gallery."""

    def __init__(
        self,
        encode_fn: Callable[[Image.Image], torch.Tensor],
        device: str = "cuda:0",
        name: str = "feature_gallery",
    ):
        self.encode_fn = encode_fn
        self.device = device
        self.name = name
        self.gallery: torch.Tensor | None = None

    @torch.inference_mode()
    def build_gallery(self, paths: list[Path]) -> float:
        import time

        feats = []
        t0 = time.perf_counter()
        for p in paths:
            img = Image.open(p).convert("RGB")
            f = self.encode_fn(img)
            if f.ndim != 1:
                f = f.reshape(-1)
            f = f.float().cpu()
            f = f / (f.norm() + 1e-8)
            feats.append(f)
        self.gallery = torch.stack(feats, dim=0) if feats else torch.empty(0)
        return time.perf_counter() - t0

    @torch.inference_mode()
    def score_image(self, image: Image.Image) -> float:
        assert self.gallery is not None and self.gallery.numel() > 0
        f = self.encode_fn(image).float().cpu().reshape(-1)
        f = f / (f.norm() + 1e-8)
        sims = self.gallery @ f
        return float(1.0 - sims.max().item())

    def evaluate(
        self,
        category: str,
        train_paths: list[Path],
        test_items: list[tuple[Path, int]],
        *,
        flops_g: float | None = None,
        params_m: float | None = None,
        warmup: int = 5,
        notes: str = "",
        extra: dict | None = None,
    ) -> MethodResult:
        import time

        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        gallery_s = self.build_gallery(train_paths)

        # warmup
        for i in range(min(warmup, len(test_items))):
            _ = self.score_image(Image.open(test_items[i][0]).convert("RGB"))

        labels, scores, lats = [], [], []
        for path, y in test_items:
            img = Image.open(path).convert("RGB")
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            s = self.score_image(img)
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000)
            labels.append(y)
            scores.append(s)

        labels_a = np.asarray(labels, dtype=int)
        scores_a = np.asarray(scores, dtype=float)
        auroc = float(roc_auc_score(labels_a, scores_a)) if len(np.unique(labels_a)) > 1 else float("nan")
        f1, prec, rec, thr = best_f1_threshold(labels_a, scores_a)

        peak = None
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            peak = float(torch.cuda.max_memory_allocated() / (1024**2))

        return MethodResult(
            method=self.name,
            category=category,
            n_gallery=len(train_paths),
            n_test=len(test_items),
            image_auroc=auroc,
            f1=f1,
            precision=prec,
            recall=rec,
            threshold=thr,
            gallery_build_s=float(gallery_s),
            infer_latency_ms_mean=float(np.mean(lats)),
            infer_latency_ms_std=float(np.std(lats)),
            flops_g=flops_g,
            params_m=params_m,
            peak_mem_mb=peak,
            notes=notes,
            extra=extra or {},
        )
