from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src.evaluation import fit_threshold


BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
}


class FeatureExtractor(nn.Module):
    def __init__(self, backbone: str = "resnet18", layers: List[str] | None = None):
        super().__init__()
        layers = layers or ["layer2", "layer3"]
        ctor, weights = BACKBONES[backbone]
        net = ctor(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.layers = layers
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        feats = {}
        x = self.layer1(x)
        feats["layer1"] = x
        x = self.layer2(x)
        feats["layer2"] = x
        x = self.layer3(x)
        feats["layer3"] = x
        x = self.layer4(x)
        feats["layer4"] = x

        out = []
        ref = None
        for name in self.layers:
            f = feats[name]
            if ref is None:
                ref = f.shape[-2:]
            if f.shape[-2:] != ref:
                f = F.interpolate(f, size=ref, mode="bilinear", align_corners=False)
            out.append(f)
        fused = torch.cat(out, dim=1)
        b, c, h, w = fused.shape
        return fused.permute(0, 2, 3, 1).reshape(b, h * w, c)


class DINOv3FeatureExtractor(nn.Module):
    """Expose selected Hugging Face DINOv3 hidden states as patch embeddings."""

    def __init__(self, model_path: str, layers: List[int] | None = None):
        super().__init__()
        from transformers import AutoModel

        if not Path(model_path).is_dir():
            raise FileNotFoundError(f"DINOv3 model directory not found: {model_path}")
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.layers = layers or [int(self.model.config.num_hidden_layers)]
        n_layers = int(self.model.config.num_hidden_layers)
        for layer in self.layers:
            if layer < 1 or layer > n_layers:
                raise ValueError(f"DINOv3 layer {layer} out of range 1..{n_layers}")
        self.n_special_tokens = 1 + int(
            getattr(self.model.config, "num_register_tokens", 0) or 0
        )
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # UnifiedAnomalyDataset already applies the ImageNet normalization used
        # by the official DINOv3 processor, so pixel_values can be passed directly.
        outputs = self.model(pixel_values=x, output_hidden_states=True)
        patch_layers = [
            outputs.hidden_states[layer][:, self.n_special_tokens :, :]
            for layer in self.layers
        ]
        return torch.cat(patch_layers, dim=-1)


@dataclass
class PatchCoreConfig:
    name: str
    backbone: str = "resnet18"
    layers: List[str | int] = field(default_factory=lambda: ["layer2", "layer3"])
    model_path: str | None = None
    coreset_ratio: float = 0.1
    max_memory_bank: int = 10000
    device: str = "cuda:0"


class PatchCoreLite(nn.Module):
    """Embedding + nearest-neighbor memory bank (Anomalib PatchCore style)."""

    def __init__(self, cfg: PatchCoreConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        if cfg.backbone.startswith("dinov3"):
            if not cfg.model_path:
                raise ValueError("model_path is required for a DINOv3 backbone")
            layers = [int(layer) for layer in cfg.layers]
            self.extractor = DINOv3FeatureExtractor(cfg.model_path, layers).to(self.device)
        else:
            layers = [str(layer) for layer in cfg.layers]
            self.extractor = FeatureExtractor(cfg.backbone, layers).to(self.device)
        self.memory_bank: torch.Tensor | None = None
        self.reference_bank: torch.Tensor | None = None
        self.reference_paths: list[str] = []
        self.threshold: float = 0.5

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        self.extractor.eval()
        return self.extractor(x.to(self.device))

    @torch.no_grad()
    def fit(self, loader, max_batches: int | None = None):
        feats = []
        reference_feats = []
        reference_paths: list[str] = []
        for i, batch in enumerate(loader):
            images = batch[0]
            emb = self.embed(images)
            feats.append(emb.reshape(-1, emb.shape[-1]).cpu())
            pooled = F.normalize(F.normalize(emb.float(), dim=-1).mean(dim=1), dim=-1)
            reference_feats.append(pooled.cpu())
            if len(batch) >= 3:
                reference_paths.extend(str(path) for path in batch[2])
            else:
                reference_paths.extend([""] * int(images.shape[0]))
            if max_batches is not None and i + 1 >= max_batches:
                break
        bank = torch.cat(feats, dim=0)
        bank = self._coreset(bank, self.cfg.coreset_ratio, self.cfg.max_memory_bank)
        self.memory_bank = F.normalize(bank.float(), dim=1).to(self.device)
        self.reference_bank = torch.cat(reference_feats, dim=0).to(self.device)
        self.reference_paths = reference_paths

    def _coreset(self, bank: torch.Tensor, ratio: float, max_n: int) -> torch.Tensor:
        n = bank.shape[0]
        keep = min(max_n, max(1, int(n * ratio)))
        if keep >= n:
            return bank
        # random pre-sample to keep training practical on large banks
        pre = min(n, max(keep * 4, 20000))
        if pre < n:
            sel = torch.randperm(n)[:pre]
            bank = bank[sel]
            n = bank.shape[0]
            keep = min(keep, n)
        if keep >= n:
            return bank
        # lightweight random coreset (stable enough for demo benchmarks)
        idxs = torch.randperm(n)[:keep]
        return bank[idxs]

    @torch.no_grad()
    def predict_details(self, x: torch.Tensor) -> Dict[str, torch.Tensor | list[str] | tuple[int, int]]:
        """Return image scores, patch maps and nearest normal references.

        This is the expert-tool interface used by the detection Agent.  It
        preserves ``predict_score`` while exposing evidence without a second
        backbone forward pass.
        """
        assert self.memory_bank is not None, "Call fit() or load_bank() first"
        emb = F.normalize(self.embed(x), dim=-1)  # B,P,C
        b, p, _ = emb.shape
        image_scores = []
        patch_scores = []
        mem = self.memory_bank
        for i in range(b):
            e = emb[i]
            sim = e @ mem.T
            nn_sim, _ = sim.max(dim=1)
            patch_score = 1.0 - nn_sim
            k = max(1, int(0.01 * p))
            topk = torch.topk(patch_score, k=k).values.mean()
            image_scores.append(topk)
            patch_scores.append(patch_score)

        side = int(round(p**0.5))
        if side * side != p:
            raise ValueError(f"cannot infer square patch grid from {p} tokens")
        reference_indices = torch.full((b,), -1, dtype=torch.long, device=self.device)
        reference_similarity = torch.full((b,), float("nan"), device=self.device)
        nearest_paths = [""] * b
        if self.reference_bank is not None and len(self.reference_paths):
            query = F.normalize(emb.mean(dim=1), dim=-1)
            similarities = query @ self.reference_bank.T
            reference_similarity, reference_indices = similarities.max(dim=1)
            nearest_paths = [self.reference_paths[int(index)] for index in reference_indices]
        return {
            "image_scores": torch.stack(image_scores),
            "patch_scores": torch.stack(patch_scores).reshape(b, side, side),
            "grid_hw": (side, side),
            "reference_indices": reference_indices,
            "reference_similarity": reference_similarity,
            "reference_paths": nearest_paths,
        }

    @torch.no_grad()
    def predict_score(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_details(x)["image_scores"]  # type: ignore[return-value]

    def calibrate_threshold(self, scores: np.ndarray, labels: np.ndarray):
        from sklearn.metrics import f1_score

        best_t = fit_threshold(labels, scores, strategy="max_f1")
        pred = (np.asarray(scores) >= best_t).astype(int)
        best_f1 = float(f1_score(labels, pred, zero_division=0))
        self.threshold = best_t
        return best_t, best_f1

    def state_dict_bank(self) -> Dict:
        return {
            "cfg": self.cfg.__dict__,
            "memory_bank": None if self.memory_bank is None else self.memory_bank.detach().cpu(),
            "reference_bank": None if self.reference_bank is None else self.reference_bank.detach().cpu(),
            "reference_paths": self.reference_paths,
            "threshold": self.threshold,
        }

    def load_bank(self, state: Dict):
        self.threshold = float(state["threshold"])
        mb = state["memory_bank"]
        self.memory_bank = None if mb is None else mb.to(self.device)
        refs = state.get("reference_bank")
        self.reference_bank = None if refs is None else refs.to(self.device)
        self.reference_paths = [str(path) for path in state.get("reference_paths", [])]


def create_patchcore_expert(cfg: PatchCoreConfig) -> nn.Module:
    """Create either the lightweight experimental expert or standard WR50 PatchCore."""
    if cfg.backbone == "wide_resnet50_2":
        from src.wr50_patchcore import WR50PatchCoreExpert

        return WR50PatchCoreExpert(
            device=cfg.device,
            layers=[str(layer) for layer in cfg.layers],
            coreset_ratio=cfg.coreset_ratio,
        )
    return PatchCoreLite(cfg)
