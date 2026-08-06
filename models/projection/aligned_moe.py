"""Category-independent DINO-to-CLIP attention and MoE projection."""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.projection.moe import LoRAExpert, _topk_route


class ResidualMapper(nn.Module):
    """Residual MLP mapping DINO tokens into CLIP embedding space."""

    def __init__(self, input_dim: int = 1024, output_dim: int = 768, hidden_dim: int = 1536):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.skip = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        return self.skip(x_norm) + self.mlp(x_norm)


class CrossLayerAttention(nn.Module):
    """Attend over feature levels independently at each spatial position."""

    def __init__(self, dim: int = 768, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, layers: torch.Tensor) -> torch.Tensor:
        # [B, L, N, C] -> [B*N, L, C]
        b, level_count, token_count, channels = layers.shape
        sequence = layers.permute(0, 2, 1, 3).reshape(b * token_count, level_count, channels)
        normed = self.norm(sequence)
        mixed, _ = self.attn(normed, normed, normed, need_weights=False)
        sequence = sequence + torch.tanh(self.scale) * mixed
        return sequence.view(b, token_count, level_count, channels).permute(0, 2, 1, 3)


class WindowTokenMixer(nn.Module):
    """Local non-overlapping window attention over patch tokens."""

    def __init__(self, dim: int = 768, num_heads: int = 8, window_size: int = 4):
        super().__init__()
        self.window_size = int(window_size)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, patches: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        b, token_count, channels = patches.shape
        if token_count != grid_h * grid_w:
            raise ValueError(f"Expected {grid_h * grid_w} patches, got {token_count}")
        window = self.window_size
        pad_h = (window - grid_h % window) % window
        pad_w = (window - grid_w % window) % window
        grid = patches.view(b, grid_h, grid_w, channels)
        grid = F.pad(grid, (0, 0, 0, pad_w, 0, pad_h))
        padded_h, padded_w = grid_h + pad_h, grid_w + pad_w
        windows = (
            grid.view(b, padded_h // window, window, padded_w // window, window, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, window * window, channels)
        )
        normed = self.norm(windows)
        mixed, _ = self.attn(normed, normed, normed, need_weights=False)
        windows = windows + torch.tanh(self.scale) * mixed
        grid = (
            windows.view(b, padded_h // window, padded_w // window, window, window, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(b, padded_h, padded_w, channels)
        )
        return grid[:, :grid_h, :grid_w].reshape(b, token_count, channels)


class CrossSpaceLoRAExpert(nn.Module):
    """Low-rank residual that directly maps DINO features into CLIP space."""

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 768,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.scaling = float(alpha) / max(int(rank), 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Linear(input_dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, output_dim, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


class CrossSpaceMapperMoE(nn.Module):
    """Category-independent Top-2 experts for the actual 1024→768 mapping."""

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 768,
        num_experts: int = 4,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
        top_k: int = 2,
        temperature: float = 1.0,
        residual_weight: float = 0.1,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.residual_weight = float(residual_weight)
        self.input_norm = nn.LayerNorm(input_dim)
        self.router = nn.Sequential(
            nn.LayerNorm(input_dim * 3),
            nn.Linear(input_dim * 3, num_experts),
        )
        self.experts = nn.ModuleList(
            [
                CrossSpaceLoRAExpert(
                    input_dim, output_dim, rank, alpha, dropout
                )
                for _ in range(num_experts)
            ]
        )
        self._last_router_probs = None
        self._last_expert_outputs = None

    def forward(
        self,
        patches: torch.Tensor,
        cls_token: Optional[torch.Tensor] = None,
        local_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        patches = self.input_norm(patches)
        if local_context is None:
            local_context = patches
        else:
            local_context = self.input_norm(local_context)
        if cls_token is None:
            cls_token = patches.mean(dim=1)
        else:
            cls_token = self.input_norm(cls_token)
        cls_context = cls_token.unsqueeze(1).expand(-1, patches.shape[1], -1)
        logits = self.router(torch.cat([patches, local_context, cls_context], dim=-1))
        dense, sparse, _ = _topk_route(logits, self.top_k, self.temperature)
        expert_outputs = torch.stack(
            [expert(patches) for expert in self.experts], dim=2
        )
        delta = (sparse.unsqueeze(-1) * expert_outputs).sum(dim=2)
        self._last_router_probs = dense
        self._last_expert_outputs = expert_outputs
        return self.residual_weight * delta


class ClipSpaceMoE(nn.Module):
    """Top-2 LoRA residual experts whose router is category independent."""

    def __init__(
        self,
        dim: int = 768,
        num_experts: int = 4,
        rank: int = 8,
        alpha: int = 16,
        top_k: int = 2,
        temperature: float = 1.0,
        residual_weight: float = 0.1,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.residual_weight = float(residual_weight)
        self.experts = nn.ModuleList(
            [LoRAExpert(dim=dim, rank=rank, alpha=alpha) for _ in range(num_experts)]
        )
        self.router = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, num_experts))
        self._last_router_probs = None
        self._last_expert_outputs = None

    def forward(
        self,
        patches: torch.Tensor,
        cls_token: Optional[torch.Tensor] = None,
        local_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if local_context is None:
            local_context = patches
        if cls_token is None:
            cls_token = patches.mean(dim=1)
        cls_context = cls_token.unsqueeze(1).expand(-1, patches.shape[1], -1)
        logits = self.router(torch.cat([patches, local_context, cls_context], dim=-1))
        dense, sparse, _ = _topk_route(logits, self.top_k, self.temperature)
        expert_outputs = torch.stack([expert(patches) for expert in self.experts], dim=2)
        delta = (sparse.unsqueeze(-1) * expert_outputs).sum(dim=2)
        self._last_router_probs = dense
        self._last_expert_outputs = expert_outputs
        return patches + self.residual_weight * delta


class _AlignedLayer(nn.Module):
    def __init__(
        self,
        mapper: ResidualMapper,
        moe: Optional[ClipSpaceMoE],
        cross_space_moe: Optional[CrossSpaceMapperMoE],
    ):
        super().__init__()
        self.mapper = mapper
        self.moe = moe
        self.cross_space_moe = cross_space_moe
        self._last_router_probs = None
        self._last_expert_outputs = None


class PerLayerAlignedProjection(nn.Module):
    """Drop-in per-layer projection with optional layer/local attention and CLIP MoE."""

    _is_per_layer_projection = True

    def __init__(
        self,
        num_layers: int = 4,
        vis_dim: int = 1024,
        output_dim: int = 768,
        mapper_hidden_dim: int = 1536,
        use_cross_layer_attention: bool = False,
        use_window_mixer: bool = False,
        window_size: int = 4,
        use_clip_space_moe: bool = False,
        use_cross_space_moe: bool = False,
        cross_space_moe_weight: Optional[float] = None,
        num_experts: int = 4,
        router_top_k: int = 2,
        router_temperature: float = 1.0,
        expert_rank: int = 8,
        expert_alpha: int = 16,
        expert_dropout: float = 0.05,
        adapt_weight: float = 0.1,
        **_: object,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.output_dim = int(output_dim)
        self.use_cross_layer_attention = bool(use_cross_layer_attention)
        self.use_window_mixer = bool(use_window_mixer)
        self.use_clip_space_moe = bool(use_clip_space_moe)
        self.use_cross_space_moe = bool(use_cross_space_moe)
        self.layers = nn.ModuleList()
        for _index in range(num_layers):
            mapper = ResidualMapper(vis_dim, output_dim, mapper_hidden_dim)
            moe = ClipSpaceMoE(
                output_dim, num_experts, expert_rank, expert_alpha,
                router_top_k, router_temperature, adapt_weight,
            ) if use_clip_space_moe else None
            cross_space_moe = CrossSpaceMapperMoE(
                vis_dim,
                output_dim,
                num_experts,
                expert_rank,
                expert_alpha,
                expert_dropout,
                router_top_k,
                router_temperature,
                (
                    adapt_weight
                    if cross_space_moe_weight is None
                    else float(cross_space_moe_weight)
                ),
            ) if use_cross_space_moe else None
            self.layers.append(_AlignedLayer(mapper, moe, cross_space_moe))
        self.cross_layer = CrossLayerAttention(output_dim) if use_cross_layer_attention else None
        self.window_mixer = (
            WindowTokenMixer(output_dim, window_size=window_size)
            if use_window_mixer else None
        )
        self.num_experts = (
            int(num_experts)
            if use_clip_space_moe or use_cross_space_moe
            else 0
        )
        self.num_abnormal = max(self.num_experts - 1, 0)

    @staticmethod
    def _source_local_context(patches: torch.Tensor) -> torch.Tensor:
        side = int(round(math.sqrt(patches.shape[1])))
        if side * side != patches.shape[1]:
            return patches
        grid = patches.view(patches.shape[0], side, side, patches.shape[-1])
        local = F.avg_pool2d(
            grid.permute(0, 3, 1, 2), kernel_size=3, stride=1, padding=1
        )
        return local.permute(0, 2, 3, 1).reshape_as(patches)

    def _map_layer(
        self,
        layer: _AlignedLayer,
        features: torch.Tensor,
    ) -> torch.Tensor:
        mapped = layer.mapper(features)
        if layer.cross_space_moe is None:
            return mapped
        source_cls, source_patches = features[:, 0], features[:, 1:]
        delta = layer.cross_space_moe(
            source_patches,
            cls_token=source_cls,
            local_context=self._source_local_context(source_patches),
        )
        return torch.cat(
            [mapped[:, :1], mapped[:, 1:] + delta],
            dim=1,
        )

    def project_layers(self, layer_features: List[torch.Tensor]) -> List[torch.Tensor]:
        mapped = torch.stack(
            [
                self._map_layer(layer, features)
                for layer, features in zip(self.layers, layer_features)
            ],
            dim=1,
        )
        if self.cross_layer is not None:
            mapped = self.cross_layer(mapped)
        outputs = []
        for index, layer in enumerate(self.layers):
            tokens = mapped[:, index]
            cls_token, patches = tokens[:, 0], tokens[:, 1:]
            side = int(round(math.sqrt(patches.shape[1])))
            local = patches
            if self.window_mixer is not None and side * side == patches.shape[1]:
                local = self.window_mixer(patches, side, side)
            if layer.moe is not None:
                patches = layer.moe(patches, cls_token=cls_token, local_context=local)
                layer._last_router_probs = layer.moe._last_router_probs
                layer._last_expert_outputs = layer.moe._last_expert_outputs
            else:
                patches = local
            outputs.append(torch.cat([cls_token.unsqueeze(1), patches], dim=1))
        return outputs

    def project_layer_tokens(self, layer_idx: int, layer_feat: torch.Tensor, **_: object) -> torch.Tensor:
        # Used only by legacy callers; joint project_layers is required for cross-layer attention.
        layer = self.layers[layer_idx]
        mapped = self._map_layer(layer, layer_feat)
        cls_token, patches = mapped[:, 0], mapped[:, 1:]
        if layer.moe is not None:
            patches = layer.moe(patches, cls_token=cls_token)
        return torch.cat([cls_token.unsqueeze(1), patches], dim=1)

    def routing_regularization_loss(
        self,
        balance_weight: float = 0.01,
        entropy_weight: float = 0.0,
        etf_weight: float = 0.0,
        **_: object,
    ) -> torch.Tensor:
        reference = next(self.parameters())
        losses = []
        for layer in self.layers:
            for module in (layer.cross_space_moe, layer.moe):
                if module is None or module._last_router_probs is None:
                    continue
                probabilities = module._last_router_probs
                target = torch.full_like(
                    probabilities.mean((0, 1)), 1.0 / module.num_experts
                )
                balance = F.mse_loss(probabilities.mean((0, 1)), target)
                entropy = -(
                    probabilities * probabilities.clamp_min(1e-8).log()
                ).sum(-1).mean()
                expert = module._last_expert_outputs
                diversity = reference.new_tensor(0.0)
                if expert is not None and module.num_experts > 1:
                    normalized = F.normalize(expert, dim=-1)
                    gram = torch.einsum(
                        "bnec,bnfc->bnef", normalized, normalized
                    )
                    eye = torch.eye(
                        module.num_experts,
                        device=gram.device,
                        dtype=gram.dtype,
                    )
                    diversity = ((gram - eye) ** 2).mean()
                losses.append(
                    float(balance_weight) * balance
                    - float(entropy_weight) * entropy
                    + float(etf_weight) * diversity
                )
        return sum(losses) / max(len(losses), 1) if losses else reference.new_tensor(0.0)
