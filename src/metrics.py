from __future__ import annotations

from typing import Dict

import numpy as np

from src.evaluation import evaluate_binary, summarize_latency


def binary_detection_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict:
    """Backward-compatible wrapper around the unified evaluation interface."""
    return evaluate_binary(labels, scores, threshold)


def latency_stats(latencies_ms: list[float]) -> Dict:
    """Backward-compatible alias for the unified latency summary."""
    return summarize_latency(latencies_ms)


def retention(edge_metric: float, cloud_metric: float) -> float:
    if cloud_metric is None or cloud_metric == 0 or np.isnan(cloud_metric):
        return float("nan")
    return float(edge_metric / cloud_metric)
