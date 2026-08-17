from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _trapezoid_area(y: np.ndarray, x: np.ndarray) -> float:
    """Version-independent trapezoidal integration for NumPy 1.x and 2.x."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if y.ndim != 1 or x.ndim != 1 or len(y) != len(x):
        raise ValueError("x and y must be one-dimensional arrays of equal length")
    if len(y) < 2:
        return 0.0
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) * 0.5))


def f1_max(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)[::-1]
    labels = labels[order].astype(np.int64)
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    positives = labels.sum()
    denominator = 2 * tp + fp + positives - tp
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)
    return float(f1.max(initial=0.0))


def binary_metrics(labels: list[int] | np.ndarray, scores: list[float] | np.ndarray) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.uint8)
    s = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "ap": float("nan"), "f1_max": float("nan")}
    return {
        "auroc": float(roc_auc_score(y, s)),
        "ap": float(average_precision_score(y, s)),
        "f1_max": f1_max(y, s),
    }


@dataclass
class HistogramMetrics:
    bins: int = 4096

    def __post_init__(self) -> None:
        self.positive = np.zeros(self.bins, dtype=np.int64)
        self.negative = np.zeros(self.bins, dtype=np.int64)

    def update(self, labels: np.ndarray, scores: np.ndarray) -> None:
        indices = np.minimum((np.clip(scores, 0, 1) * (self.bins - 1)).astype(np.int64), self.bins - 1)
        self.positive += np.bincount(indices[labels.astype(bool)].ravel(), minlength=self.bins)
        self.negative += np.bincount(indices[~labels.astype(bool)].ravel(), minlength=self.bins)

    def compute(self) -> dict[str, float]:
        pos_total, neg_total = self.positive.sum(), self.negative.sum()
        if pos_total == 0 or neg_total == 0:
            return {"auroc": float("nan"), "ap": float("nan"), "f1_max": float("nan")}
        tp = np.cumsum(self.positive[::-1])
        fp = np.cumsum(self.negative[::-1])
        tpr, fpr = tp / pos_total, fp / neg_total
        auroc = _trapezoid_area(np.r_[0, tpr], np.r_[0, fpr])
        precision = tp / np.maximum(tp + fp, 1)
        recall = tpr
        ap = float(np.sum(np.diff(np.r_[0, recall]) * precision))
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        return {"auroc": auroc, "ap": ap, "f1_max": float(f1.max(initial=0.0))}
