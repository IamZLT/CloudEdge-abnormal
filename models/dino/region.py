"""
INSID3-style DINOv3 region processing (3.1 + 3.2).

3.1  Debias: estimate positional subspace from noise images, project it out.
3.2  Cluster: self-similarity clustering of patch features within each image.
MoE routing uses cluster_id % num_experts (no seed selection).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class RegionBatchInfo:
    """Per-batch region metadata for MoE routing."""

    debiased_patch: torch.Tensor  # [B, P, D]
    cluster_labels: torch.Tensor  # [B, P] long
    cluster_centroids: torch.Tensor  # [B, K, D]
    routing_weights: torch.Tensor  # [B, P, E]
    num_clusters: int


def estimate_positional_subspace(
    noise_patch_features: torch.Tensor,
    n_components: int,
) -> torch.Tensor:
    """Estimate positional bias subspace from noise-image patch features (3.1)."""
    if noise_patch_features.dim() == 3:
        feats = noise_patch_features.reshape(-1, noise_patch_features.shape[-1])
    else:
        feats = noise_patch_features

    feats = feats.float()
    feats = feats - feats.mean(dim=0, keepdim=True)
    n_components = min(n_components, feats.shape[0] - 1, feats.shape[1])
    if n_components < 1:
        return torch.zeros(feats.shape[1], 1, device=feats.device, dtype=feats.dtype)

    _, _, vh = torch.linalg.svd(feats, full_matrices=False)
    return vh[:n_components].T.contiguous()


def debias_features(features: torch.Tensor, pos_basis: torch.Tensor) -> torch.Tensor:
    """F_tilde = F - P_pos F (3.1)."""
    if pos_basis is None or pos_basis.numel() == 0:
        return features
    proj = features @ pos_basis @ pos_basis.T
    return features - proj


def _deterministic_patch_indices(
    batch_size: int,
    num_patches: int,
    num_clusters: int,
    device: torch.device,
) -> torch.Tensor:
    """Evenly spaced patch indices [B, K] for reproducible k-means init."""
    k = min(num_clusters, num_patches)
    if num_patches >= k:
        idx = (torch.arange(k, device=device) * max(1, num_patches // k)).clamp(
            max=num_patches - 1
        )
    else:
        idx = torch.arange(k, device=device) % num_patches
    return idx.unsqueeze(0).expand(batch_size, -1)


def _kmeans_cosine_batched(
    features: torch.Tensor,
    n_clusters: int,
    n_iters: int = 15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched cosine k-means.

    Args:
        features: [B, P, D]
    Returns:
        labels [B, P], centroids [B, K, D]
    """
    b, p, d = features.shape
    k = min(n_clusters, p)
    device = features.device
    dtype = features.dtype

    if k <= 1:
        labels = torch.zeros(b, p, dtype=torch.long, device=device)
        centroids = features.mean(dim=1, keepdim=True)
        return labels, centroids

    init_idx = _deterministic_patch_indices(b, p, k, device)
    init_idx_exp = init_idx.unsqueeze(-1).expand(-1, -1, d)
    centroids = torch.gather(features, 1, init_idx_exp).clone()

    for _ in range(n_iters):
        sim = torch.bmm(features, centroids.transpose(1, 2))
        labels = sim.argmax(dim=-1)

        one_hot = F.one_hot(labels, num_classes=k).to(dtype=dtype)
        counts = one_hot.sum(dim=1)
        summed = torch.bmm(one_hot.transpose(1, 2), features)
        new_centroids = summed / counts.clamp_min(1).unsqueeze(-1)

        empty = counts == 0
        fallback_idx = init_idx[:, torch.arange(k, device=device) % k]
        fallback = torch.gather(
            features, 1, fallback_idx.unsqueeze(-1).expand(-1, -1, d)
        )
        new_centroids = torch.where(empty.unsqueeze(-1), fallback, new_centroids)
        centroids = F.normalize(new_centroids, dim=-1)

    return labels, centroids


def _kmeans_cosine(
    features: torch.Tensor,
    n_clusters: int,
    n_iters: int = 15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    labels, centroids = _kmeans_cosine_batched(
        features.unsqueeze(0), n_clusters, n_iters=n_iters
    )
    return labels[0], centroids[0]


def _augment_spatial_features(
    feat: torch.Tensor,
    grid_h: int,
    grid_w: int,
    spatial_weight: float,
) -> torch.Tensor:
    """Append normalized xy coords; feat is [B, P, D] or [P, D]."""
    if spatial_weight <= 0:
        return feat
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, grid_h, device=feat.device),
        torch.linspace(-1.0, 1.0, grid_w, device=feat.device),
        indexing="ij",
    )
    xy = torch.stack([yy, xx], dim=-1).reshape(1, -1, 2)
    if feat.dim() == 2:
        xy = xy.squeeze(0)
    else:
        xy = xy.expand(feat.shape[0], -1, -1)
    feat = torch.cat([feat, spatial_weight * xy], dim=-1)
    return F.normalize(feat, dim=-1)


def cluster_patches_self_similarity(
    patch_features: torch.Tensor,
    n_clusters: int,
    n_iters: int = 15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """3.2: partition patches into semantically coherent clusters."""
    feat = F.normalize(patch_features.float(), dim=-1)
    if feat.dim() == 2:
        return _kmeans_cosine(feat, n_clusters, n_iters=n_iters)
    labels, centroids = _kmeans_cosine_batched(feat, n_clusters, n_iters=n_iters)
    return labels, centroids


def cluster_patches_fine_grained(
    patch_features: torch.Tensor,
    grid_h: int,
    grid_w: int,
    n_clusters: int,
    n_iters: int = 25,
    spatial_weight: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fine-grained per-image clustering with optional spatial regularization.

    Supports single image [P, D] or batched [B, P, D].
    """
    feat = F.normalize(patch_features.float(), dim=-1)
    feat = _augment_spatial_features(feat, grid_h, grid_w, spatial_weight)
    if feat.dim() == 2:
        return _kmeans_cosine(feat, n_clusters, n_iters=n_iters)
    return _kmeans_cosine_batched(feat, n_clusters, n_iters=n_iters)


def build_cluster_routing_weights(
    cluster_labels: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Legacy hard assign: expert_id = cluster_id % num_experts (one-hot).
    Only used when cluster_expert_assign=True.
    """
    b, p = cluster_labels.shape
    e = num_experts
    weights = torch.zeros(b, p, e, device=cluster_labels.device, dtype=torch.float32)
    expert_ids = (cluster_labels % e).long()
    weights.scatter_(2, expert_ids.unsqueeze(-1), 1.0)
    return weights


class DinoRegionProcessor:
    """Orchestrates 3.1 debias + 3.2 cluster on DINOv3 patch features."""

    def __init__(
        self,
        pos_basis: torch.Tensor,
        n_clusters: int = 8,
        kmeans_iters: int = 15,
        num_experts: int = 4,
    ):
        self.pos_basis = pos_basis
        self.n_clusters = n_clusters
        self.kmeans_iters = kmeans_iters
        self.num_experts = num_experts

    @classmethod
    def from_noise_calibration(
        cls,
        dino_model,
        dino_processor,
        device: torch.device,
        image_size: int,
        n_noise_images: int = 32,
        n_pos_components: int = 16,
        **kwargs,
    ) -> "DinoRegionProcessor":
        from models.dino.encode import dinov3_encode_image

        noise_batch = torch.randn(n_noise_images, 3, image_size, image_size, device=device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        noise_batch = (noise_batch * 0.5 + 0.5 - mean) / std

        with torch.no_grad():
            out = dinov3_encode_image(
                noise_batch, dino_processor, dino_model, device=device, layer_indices=None
            )
            patches = out["patch_flat"]

        pos_basis = estimate_positional_subspace(patches, n_pos_components)
        return cls(pos_basis=pos_basis.to(device), **kwargs)

    def process_batch(
        self,
        patch_features: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> RegionBatchInfo:
        """
        Args:
            patch_features: [B, P, D] raw DINO patch tokens (last layer, no CLS)
            grid_h, grid_w: patch grid size (reserved for API compatibility)
        """
        del grid_h, grid_w
        b, _, d = patch_features.shape
        device = patch_features.device
        pos_basis = self.pos_basis.to(device)

        feat_norm = F.normalize(patch_features.float(), dim=-1)
        debiased_patch = F.normalize(debias_features(feat_norm, pos_basis), dim=-1)

        cluster_labels, cluster_centroids = cluster_patches_self_similarity(
            debiased_patch, self.n_clusters, n_iters=self.kmeans_iters
        )
        k = cluster_centroids.shape[1]

        routing_weights = build_cluster_routing_weights(cluster_labels, self.num_experts)

        return RegionBatchInfo(
            debiased_patch=debiased_patch,
            cluster_labels=cluster_labels,
            cluster_centroids=cluster_centroids,
            routing_weights=routing_weights,
            num_clusters=k,
        )
