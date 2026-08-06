from __future__ import annotations

import torch
import torch.nn.functional as F


def robust_rescale(
    scores: torch.Tensor,
    labels: torch.Tensor | None = None,
    eps: float = 1e-6,
    clamp: float = 6.0,
) -> torch.Tensor:
    """Median/MAD rescale to [0, 1], optionally independently per region label."""
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
    b, p = scores.shape

    if labels is None:
        med = scores.median(dim=1, keepdim=True).values
        mad = (scores - med).abs().median(dim=1, keepdim=True).values.clamp_min(eps)
        z = ((scores - med) / mad).clamp(min=0.0, max=clamp)
        return z / clamp

    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
    out = torch.zeros_like(scores)
    for bi in range(b):
        for label in torch.unique(labels[bi]):
            mask = labels[bi] == label
            vals = scores[bi, mask]
            if vals.numel() == 0:
                continue
            med = vals.median()
            mad = (vals - med).abs().median().clamp_min(eps)
            z = ((vals - med) / mad).clamp(min=0.0, max=clamp)
            out[bi, mask] = z / clamp
    return out


def cluster_prototypes(
    patch_features: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Return each patch's same-cluster prototype. Supports [P, D] or [B, P, D]."""
    squeeze = False
    if patch_features.dim() == 2:
        patch_features = patch_features.unsqueeze(0)
        labels = labels.unsqueeze(0)
        squeeze = True

    if labels.dim() == 1:
        labels = labels.unsqueeze(0)

    feat = F.normalize(patch_features.float(), dim=-1)
    k = int(labels.max().item()) + 1
    one_hot = F.one_hot(labels.long(), num_classes=k).float()
    counts = one_hot.sum(dim=1, keepdim=True).clamp_min(1.0)
    centroids = torch.bmm(one_hot.transpose(1, 2), feat) / counts.transpose(1, 2)
    centroids = F.normalize(centroids, dim=-1)
    proto = centroids.gather(1, labels.unsqueeze(-1).expand(-1, -1, feat.shape[-1]))
    return proto[0] if squeeze else proto


def cluster_deviation_score(
    patch_features: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Patch-level deviation from its fine-grained cluster prototype."""
    if patch_features.dim() == 2:
        patch_features = patch_features.unsqueeze(0)
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
    feat = F.normalize(patch_features.float(), dim=-1)
    proto = cluster_prototypes(feat, labels)
    residual = 1.0 - (feat * proto).sum(dim=-1)
    return robust_rescale(residual.clamp_min(0.0), labels=labels, eps=eps)
