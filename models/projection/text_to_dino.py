"""Talk2DINO-style CLIP-text → DINO-space projector."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextToDinoProjector(nn.Module):
    """
    Map frozen CLIP text embeddings into frozen DINO patch space.

    Coupled scoring:
      MoE-adapted DINO patches  ×  TextToDino(CLIP text banks)
    """

    def __init__(
        self,
        clip_dim: int = 768,
        dino_dim: int = 1024,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.clip_dim = int(clip_dim)
        self.dino_dim = int(dino_dim)
        self.norm = nn.LayerNorm(clip_dim)
        self.mlp = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dino_dim),
        )
        self.skip = nn.Linear(clip_dim, dino_dim, bias=False)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        if text_features.shape[-1] != self.clip_dim:
            raise ValueError(
                f"Expected CLIP dim {self.clip_dim}, got {text_features.shape[-1]}"
            )
        x = self.norm(text_features)
        return F.normalize(self.skip(x) + self.mlp(x), dim=-1)

    def project_banks(
        self,
        normal_bank: torch.Tensor,
        defect_bank: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(normal_bank), self.forward(defect_bank)


def collect_adapted_dino_features(
    patch_proj,
    patch_token_memory=None,
) -> list[torch.Tensor] | None:
    """
    Build [B, 1+P, DINO] tokens from MoE-adapted patch features.

    CLS = mean of adapted patches (Talk2DINO-style local readout).
    Falls back to raw DINO patches if `_last_x_adapt` is missing.
    """
    layers = getattr(patch_proj, "layers", None)
    if layers is None:
        return None
    outputs = []
    for index, layer in enumerate(layers):
        adapted = getattr(layer, "_last_x_adapt", None)
        if adapted is None:
            if patch_token_memory is None or index >= len(patch_token_memory):
                return None
            # Raw DINO tokens already include CLS.
            tokens = F.normalize(patch_token_memory[index], dim=-1)
            outputs.append(tokens)
            continue
        patches = F.normalize(adapted, dim=-1)
        cls = patches.mean(dim=1, keepdim=True)
        outputs.append(torch.cat([cls, patches], dim=1))
    return outputs
