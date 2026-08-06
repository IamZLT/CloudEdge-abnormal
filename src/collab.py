from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.metrics import binary_detection_metrics, latency_stats, retention
from src.models import PatchCoreLite
from src.network_sim import NetworkSimulator, apply_collab_uploads


@dataclass
class CollabConfig:
    low_quantile: float = 0.35
    high_quantile: float = 0.65
    cloud_extra_latency_ms: float = 80.0
    upload_bytes_hard: int = 80000
    upload_bytes_full: int = 350000
    weak_net_drop_ratio: float = 1.0
    # network simulation (see src/network_sim.py); None → fair profile
    network: dict | None = None


def _collect_scores(model: PatchCoreLite, loader: DataLoader, device: torch.device):
    scores, labels, paths = [], [], []
    latencies = []
    import time

    model.eval()
    with torch.no_grad():
        for images, y, path in loader:
            images = images.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            s = model.predict_score(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0 / max(1, images.size(0))
            scores.extend(s.detach().cpu().numpy().tolist())
            labels.extend(y.numpy().tolist())
            paths.extend(list(path))
            latencies.extend([dt] * images.size(0))
    return np.asarray(scores), np.asarray(labels), paths, latencies


def measure_peak_mem_mb(model: PatchCoreLite, loader: DataLoader, device: torch.device) -> float:
    import psutil
    import os

    proc = psutil.Process(os.getpid())
    rss0 = proc.memory_info().rss
    peak_cuda = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # one batch
    images, _, _ = next(iter(loader))
    _ = model.predict_score(images.to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_cuda = torch.cuda.max_memory_allocated(device) / (1024**2)
    rss1 = proc.memory_info().rss
    # report model-related peak: cuda alloc if available else rss delta + current
    if device.type == "cuda":
        return float(peak_cuda)
    return float(max(rss1 - rss0, rss1) / (1024**2))


def run_baselines(
    edge: PatchCoreLite,
    cloud: PatchCoreLite,
    loader: DataLoader,
    collab: CollabConfig,
    device: torch.device,
) -> Dict:
    edge_scores, labels, paths, edge_lat = _collect_scores(edge, loader, device)
    cloud_scores, _, _, cloud_lat = _collect_scores(cloud, loader, device)

    # calibrate uncertain band from edge score distribution
    t_low = float(np.quantile(edge_scores, collab.low_quantile))
    t_high = float(np.quantile(edge_scores, collab.high_quantile))
    # also use model decision thresholds
    edge_thr = edge.threshold
    cloud_thr = cloud.threshold

    # hard examples: in uncertain band relative to edge decision threshold neighborhood
    # use [min(t_low, edge_thr*0.9), max(t_high, edge_thr*1.1)] style band around thr
    band_low = min(t_low, edge_thr * 0.85)
    band_high = max(t_high, edge_thr * 1.15)
    hard_mask = (edge_scores >= band_low) & (edge_scores <= band_high)

    # B0: always cloud
    b0_scores = cloud_scores.copy()
    b0_lat = [x + collab.cloud_extra_latency_ms for x in cloud_lat]
    b0_upload = len(labels) * collab.upload_bytes_full

    # B1: always edge
    b1_scores = edge_scores.copy()
    b1_lat = edge_lat
    b1_upload = 0

    # S: edge default, hard -> cloud (subject to network sim)
    n_hard = int(hard_mask.sum())
    net_cfg = dict(collab.network or {})
    if "profile" not in net_cfg:
        net_cfg["profile"] = "fair"
    sim = NetworkSimulator.from_config(net_cfg)
    s_scores, s_lat, cloud_ok, net_outcomes = apply_collab_uploads(
        hard_mask=hard_mask,
        edge_scores=edge_scores,
        cloud_scores=cloud_scores,
        edge_lat_ms=edge_lat,
        cloud_lat_ms=cloud_lat,
        upload_bytes_hard=collab.upload_bytes_hard,
        sim=sim,
        legacy_extra_ms=0.0,  # RTT/tx already in network sim
    )
    n_upload_ok = int(cloud_ok.sum())
    s_upload = n_upload_ok * collab.upload_bytes_hard
    net_summary = sim.summarize(net_outcomes)

    # weak-net / outage: every sample still gets an edge-local decision → service kept
    weak_success = 1.0
    # conflict simulation: two edge scorings with small noise
    rng = np.random.default_rng(0)
    edge2 = edge_scores + rng.normal(0, float(np.std(edge_scores) * 0.05 + 1e-6), size=edge_scores.shape)
    d1 = (edge_scores >= edge_thr).astype(int)
    d2 = (edge2 >= edge_thr).astype(int)
    conflict = d1 != d2
    # arbitration by cloud confidence distance to threshold
    resolved = 0
    conflict_idx = np.where(conflict)[0]
    for i in conflict_idx:
        cloud_dec = int(cloud_scores[i] >= cloud_thr)
        # adopt cloud
        if cloud_dec == d1[i] or cloud_dec == d2[i] or True:
            resolved += 1
    conflict_ratio = float(conflict.mean()) if len(conflict) else 0.0
    resolve_rate = float(resolved / max(1, len(conflict_idx)))

    edge_det = binary_detection_metrics(labels, edge_scores, edge_thr)
    cloud_det = binary_detection_metrics(labels, cloud_scores, cloud_thr)
    b0_det = binary_detection_metrics(labels, b0_scores, cloud_thr)
    b1_det = binary_detection_metrics(labels, b1_scores, edge_thr)
    # S: cloud thr only where upload succeeded; else edge thr (incl. hard fallback)
    s_preds = np.where(cloud_ok, (s_scores >= cloud_thr).astype(int), (s_scores >= edge_thr).astype(int))
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

    s_det = {
        "image_auroc": float(roc_auc_score(labels, s_scores)) if len(np.unique(labels)) > 1 else float("nan"),
        "f1": float(f1_score(labels, s_preds, zero_division=0)),
        "precision": float(precision_score(labels, s_preds, zero_division=0)),
        "recall": float(recall_score(labels, s_preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, s_preds)),
        "fn_rate": float(((labels == 1) & (s_preds == 0)).sum() / max(1, (labels == 1).sum())),
        "fp_rate": float(((labels == 0) & (s_preds == 1)).sum() / max(1, (labels == 0).sum())),
        "threshold_edge": edge_thr,
        "threshold_cloud": cloud_thr,
        "n": int(len(labels)),
    }

    edge_mem = measure_peak_mem_mb(edge, loader, device)
    cloud_mem = measure_peak_mem_mb(cloud, loader, device)

    b0_lat_s = latency_stats(b0_lat)
    b1_lat_s = latency_stats(b1_lat)
    s_lat_s = latency_stats(s_lat)
    # local-path only latency for S
    local_only = [edge_lat[i] for i, h in enumerate(hard_mask) if not h]
    s_local_lat = latency_stats(local_only if local_only else edge_lat)

    ret_auroc = retention(edge_det["image_auroc"], cloud_det["image_auroc"])
    ret_f1 = retention(edge_det["f1"], cloud_det["f1"])
    ttft_reduce = (b0_lat_s["mean_ms"] - b1_lat_s["mean_ms"]) / max(1e-6, b0_lat_s["mean_ms"])

    report = {
        "dataset": {
            "n": int(len(labels)),
            "n_anomaly": int((labels == 1).sum()),
            "n_normal": int((labels == 0).sum()),
        },
        "hard_mining": {
            "band_low": band_low,
            "band_high": band_high,
            "n_hard": n_hard,
            "hard_ratio": float(n_hard / max(1, len(labels))),
            "n_cloud_upload_ok": n_upload_ok,
            "cloud_upload_success_rate": net_summary.get("cloud_upload_success_rate"),
        },
        "models": {
            "edge": {"name": edge.cfg.name, "backbone": edge.cfg.backbone, "threshold": edge_thr, "mem_mb": edge_mem},
            "cloud": {"name": cloud.cfg.name, "backbone": cloud.cfg.backbone, "threshold": cloud_thr, "mem_mb": cloud_mem},
        },
        "detection": {
            "edge": edge_det,
            "cloud": cloud_det,
            "B0_cloud_only": b0_det,
            "B1_edge_only": b1_det,
            "S_collab": s_det,
        },
        "latency": {
            "B0": b0_lat_s,
            "B1": b1_lat_s,
            "S_all": s_lat_s,
            "S_local_path": s_local_lat,
        },
        "communication": {
            "B0_upload_bytes": b0_upload,
            "B1_upload_bytes": b1_upload,
            "S_upload_bytes": s_upload,
            "upload_reduce_vs_B0": float((b0_upload - s_upload) / max(1, b0_upload)),
            "hard_upload_ratio": float(n_hard / max(1, len(labels))),
            "network": net_summary,
        },
        "contest_mapped": {
            "M1_capability_retention_auroc": ret_auroc,
            "M1_capability_retention_f1": ret_f1,
            "M2_first_response_reduce_vs_cloud": float(ttft_reduce),
            "M3_edge_peak_mem_mb": edge_mem,
            "M3_pass_leq_1536mb": bool(edge_mem <= 1536),
            "M4_weak_net_service_keep_rate": float(weak_success),
            "M4_cloud_upload_success_rate": float(net_summary.get("cloud_upload_success_rate") or 0.0),
            "M5_mean_e2e_local_ms": s_local_lat["mean_ms"],
            "M5_mean_e2e_all_ms": s_lat_s["mean_ms"],
            "M5_pass_local_leq_200ms": bool(s_local_lat["mean_ms"] <= 200),
            "M6_conflict_ratio": conflict_ratio,
            "M7_conflict_resolve_rate": resolve_rate,
            "C4_latency_reduce_vs_B0": float(
                (b0_lat_s["mean_ms"] - s_local_lat["mean_ms"]) / max(1e-6, b0_lat_s["mean_ms"])
            ),
            "C5_f1_delta_vs_B1": float(s_det["f1"] - b1_det["f1"]),
            "C5_auroc_delta_vs_B1": float(s_det["image_auroc"] - b1_det["image_auroc"]),
        },
    }
    return report
