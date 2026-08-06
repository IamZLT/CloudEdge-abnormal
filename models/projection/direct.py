"""Direct DINO->CLIP patch projection (no MoE, no clustering, no router).

Pure ablation baseline: LayerNorm + Linear per layer.
"""

from __future__ import annotations

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectPatchProjection(nn.Module):
    """Single-layer direct DINO -> CLIP projection: LayerNorm + Linear."""

    def __init__(self, in_dim: int = 1024, out_dim: int = 768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


class PerLayerDirectProjection(nn.Module):
    """Per-layer direct projection (one DirectPatchProjection per DINO layer)."""

    def __init__(self, num_layers: int, in_dim: int = 1024, out_dim: int = 768):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [DirectPatchProjection(in_dim, out_dim) for _ in range(num_layers)]
        )
        print(
            f"PerLayerDirectProjection: {num_layers} layers, {in_dim} -> {out_dim} (direct, no MoE)"
        )

    def project_layer_tokens(
        self,
        layer_idx: int,
        layer_feat: torch.Tensor,
        region_info=None,
        cpa_context=None,
        include_cls: bool = True,
    ) -> torch.Tensor:
        """Project one DINO layer's tokens (CLS + patches) to CLIP space.

        Same signature as PerLayerMoEVisualProjection.project_layer_tokens
        so the same pipeline functions work without modification.
        region_info and cpa_context are ignored.
        """
        return self.layers[layer_idx](layer_feat)

    def routing_regularization_loss(self, **kwargs):
        """No-op: direct projection has no router."""
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def configure_cluster_routing(self, **kwargs) -> None:
        """No-op: no routing to configure."""
        pass
