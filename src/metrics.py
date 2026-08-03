from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_detection_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    preds = (scores >= threshold).astype(int)

    out = {
        "image_auroc": float("nan"),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "fn_rate": float(((labels == 1) & (preds == 0)).sum() / max(1, (labels == 1).sum())),
        "fp_rate": float(((labels == 0) & (preds == 1)).sum() / max(1, (labels == 0).sum())),
        "threshold": float(threshold),
        "n": int(len(labels)),
        "n_anomaly": int((labels == 1).sum()),
        "n_normal": int((labels == 0).sum()),
    }
    if len(np.unique(labels)) > 1:
        out["image_auroc"] = float(roc_auc_score(labels, scores))
    return out


def latency_stats(latencies_ms: list[float]) -> Dict:
    arr = np.asarray(latencies_ms, dtype=float)
    if arr.size == 0:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def retention(edge_metric: float, cloud_metric: float) -> float:
    if cloud_metric is None or cloud_metric == 0 or np.isnan(cloud_metric):
        return float("nan")
    return float(edge_metric / cloud_metric)
