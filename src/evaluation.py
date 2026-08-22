"""Unified binary evaluation for cloud and edge anomaly detectors.

Scores always follow one convention: larger means more likely anomalous (NG).
Threshold fitting is deliberately separate from evaluation so callers can fit on
a calibration split and evaluate unchanged on a test split.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def qwen_anomaly_score(
    decision: str,
    confidence: float,
    parse_ok: bool = True,
) -> dict[str, Any]:
    """Convert a Qwen decision confidence into a higher-is-more-anomalous score.

    Invalid responses abstain instead of silently becoming OK.  The returned
    NaN score is excluded by :func:`evaluate_binary` when ``valid`` is used as
    its valid mask.
    """
    normalized = str(decision or "").strip().upper()
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        conf = float("nan")

    valid = bool(parse_ok) and normalized in {"OK", "NG"} and np.isfinite(conf)
    if not valid:
        return {
            "score": float("nan"),
            "prediction": None,
            "decision": "REVIEW",
            "valid": False,
        }

    conf = float(np.clip(conf, 0.0, 1.0))
    prediction = 1 if normalized == "NG" else 0
    score = conf if prediction == 1 else 1.0 - conf
    return {
        "score": float(score),
        "prediction": prediction,
        "decision": normalized,
        "valid": True,
    }


def _arrays(
    labels: Iterable[int],
    scores: Iterable[float],
    valid_mask: Iterable[bool] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(labels).astype(int).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if y.size != s.size:
        raise ValueError(f"labels/scores length mismatch: {y.size} != {s.size}")
    if np.any(~np.isin(y, [0, 1])):
        raise ValueError("labels must contain only 0 (OK) and 1 (NG)")

    if valid_mask is None:
        valid = np.ones(y.size, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if valid.size != y.size:
            raise ValueError(f"valid_mask length mismatch: {valid.size} != {y.size}")
    valid &= np.isfinite(s)
    return y, s, valid


def _target_recall_point(
    labels: np.ndarray,
    scores: np.ndarray,
    target_recall: float,
) -> tuple[float, float]:
    if not 0.0 < float(target_recall) <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return float("nan"), float("nan")

    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    eligible = np.flatnonzero(tpr >= float(target_recall))
    if eligible.size == 0:
        return float("nan"), float("nan")

    best_fpr = float(np.min(fpr[eligible]))
    tied = eligible[np.isclose(fpr[eligible], best_fpr)]
    # Prefer the strictest threshold when several points have the same FPR.
    idx = int(tied[np.argmax(thresholds[tied])])
    return best_fpr, float(thresholds[idx])


def fit_threshold(
    labels: Iterable[int],
    scores: Iterable[float],
    *,
    strategy: str = "max_f1",
    target_recall: float = 0.99,
    valid_mask: Iterable[bool] | None = None,
) -> float:
    """Fit a threshold on a calibration split.

    ``max_f1`` maximizes sample F1. ``target_recall`` selects the strictest
    threshold among ROC points with the lowest FPR that reach the target.
    """
    y_all, s_all, valid = _arrays(labels, scores, valid_mask)
    y, s = y_all[valid], s_all[valid]
    if y.size == 0:
        raise ValueError("cannot fit threshold: no valid samples")

    if strategy == "target_recall":
        _, threshold = _target_recall_point(y, s, target_recall)
        if not np.isfinite(threshold):
            raise ValueError("cannot fit target-recall threshold without both classes")
        return float(threshold)
    if strategy != "max_f1":
        raise ValueError(f"unknown threshold strategy: {strategy}")

    candidates = np.unique(s)
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        pred = (s >= threshold).astype(int)
        value = float(f1_score(y, pred, zero_division=0))
        # On ties prefer the stricter (higher) threshold.
        if value > best_f1 or (np.isclose(value, best_f1) and threshold > best_threshold):
            best_f1 = value
            best_threshold = float(threshold)
    return best_threshold


def evaluate_binary(
    labels: Iterable[int],
    anomaly_scores: Iterable[float],
    threshold: float,
    *,
    predictions: Iterable[int] | None = None,
    valid_mask: Iterable[bool] | None = None,
    target_recall: float = 0.99,
) -> dict[str, Any]:
    """Evaluate either detector branch with one stable, backward-compatible schema."""
    y_all, s_all, valid = _arrays(labels, anomaly_scores, valid_mask)
    n_total = int(y_all.size)
    y, s = y_all[valid], s_all[valid]

    if predictions is None:
        pred = (s >= float(threshold)).astype(int)
    else:
        pred_all = np.asarray(predictions).astype(int).reshape(-1)
        if pred_all.size != n_total:
            raise ValueError(f"predictions length mismatch: {pred_all.size} != {n_total}")
        if np.any(~np.isin(pred_all, [0, 1])):
            raise ValueError("predictions must contain only 0 (OK) and 1 (NG)")
        pred = pred_all[valid]

    n_valid = int(y.size)
    n_anomaly = int((y == 1).sum())
    n_normal = int((y == 0).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())

    if n_valid:
        f1 = float(f1_score(y, pred, zero_division=0))
        precision = float(precision_score(y, pred, zero_division=0))
        recall = float(recall_score(y, pred, zero_division=0))
        accuracy = float(accuracy_score(y, pred))
    else:
        f1 = precision = recall = accuracy = float("nan")
    fpr = float(fp / n_normal) if n_normal else float("nan")
    fnr = float(fn / n_anomaly) if n_anomaly else float("nan")

    if n_valid and n_anomaly and n_normal:
        auroc = float(roc_auc_score(y, s))
        auprc = float(average_precision_score(y, s))
    else:
        auroc = auprc = float("nan")
    fpr_target, threshold_target = _target_recall_point(y, s, target_recall)

    evaluation_v2 = {
        "ranking": {
            "image_auroc": auroc,
            "image_auprc": auprc,
        },
        "operating_point": {
            "threshold": float(threshold),
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "fpr": fpr,
            "fnr": fnr,
        },
        "industrial_point": {
            "target_recall": float(target_recall),
            "fpr_at_target_recall": fpr_target,
            "threshold_at_target_recall": threshold_target,
        },
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "quality": {
            "n_total": n_total,
            "n_valid": n_valid,
            "valid_rate": float(n_valid / n_total) if n_total else 0.0,
        },
    }

    # Flat aliases keep existing reports and Web consumers working during migration.
    return {
        "image_auroc": auroc,
        "image_auprc": auprc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "fn_rate": fnr,
        "fp_rate": fpr,
        "threshold": float(threshold),
        "target_recall": float(target_recall),
        "fpr_at_target_recall": fpr_target,
        "threshold_at_target_recall": threshold_target,
        "n": n_total,
        "n_total": n_total,
        "n_valid": n_valid,
        "valid_rate": float(n_valid / n_total) if n_total else 0.0,
        "n_anomaly": n_anomaly,
        "n_normal": n_normal,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "evaluation_v2": evaluation_v2,
    }


def summarize_latency(latencies_ms: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(latencies_ms), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def count_model_parameters(model: Any | None) -> dict[str, int | float | None]:
    """Count model parameters without assuming a specific ML framework.

    PyTorch modules, PEFT models, and Anomalib models all expose
    ``parameters()``. Keeping this helper framework-light lets the same result
    schema be reused for edge models later.
    """
    if model is None or not hasattr(model, "parameters"):
        return {
            "total": None,
            "trainable": None,
            "total_m": None,
            "trainable_m": None,
        }

    total = 0
    trainable = 0
    for parameter in model.parameters():
        n = int(parameter.numel())
        total += n
        if bool(getattr(parameter, "requires_grad", False)):
            trainable += n
    return {
        "total": total,
        "trainable": trainable,
        "total_m": float(total / 1_000_000),
        "trainable_m": float(trainable / 1_000_000),
    }


def summarize_inference(
    latencies_ms: Iterable[float],
    model: Any | None = None,
) -> dict[str, Any]:
    """Return one extensible runtime schema for the currently cloud-only metrics."""
    values = np.asarray(list(latencies_ms), dtype=float)
    n_valid = int(np.isfinite(values).sum())
    return {
        "n_inferences": int(values.size),
        "n_valid_latency": n_valid,
        "inference_latency_ms": summarize_latency(values.tolist()),
        "parameters": count_model_parameters(model),
    }
