from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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


@dataclass
class PatchCoreConfig:
    name: str
    backbone: str = "resnet18"
    layers: List[str] = field(default_factory=lambda: ["layer2", "layer3"])
    coreset_ratio: float = 0.1
    max_memory_bank: int = 10000
    device: str = "cuda:0"


class PatchCoreLite(nn.Module):
    """Embedding + nearest-neighbor memory bank (Anomalib PatchCore style)."""

    def __init__(self, cfg: PatchCoreConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.extractor = FeatureExtractor(cfg.backbone, cfg.layers).to(self.device)
        self.memory_bank: torch.Tensor | None = None
        self.threshold: float = 0.5

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        self.extractor.eval()
        return self.extractor(x.to(self.device))

    @torch.no_grad()
    def fit(self, loader, max_batches: int | None = None):
        feats = []
        for i, batch in enumerate(loader):
            images = batch[0]
            emb = self.embed(images)
            feats.append(emb.reshape(-1, emb.shape[-1]).cpu())
            if max_batches is not None and i + 1 >= max_batches:
                break
        bank = torch.cat(feats, dim=0)
        bank = self._coreset(bank, self.cfg.coreset_ratio, self.cfg.max_memory_bank)
        self.memory_bank = F.normalize(bank.float(), dim=1).to(self.device)

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
    def predict_score(self, x: torch.Tensor) -> torch.Tensor:
        assert self.memory_bank is not None, "Call fit() or load_bank() first"
        emb = F.normalize(self.embed(x), dim=-1)  # B,P,C
        b, p, _ = emb.shape
        scores = []
        mem = self.memory_bank
        for i in range(b):
            e = emb[i]
            sim = e @ mem.T
            nn_sim, _ = sim.max(dim=1)
            patch_score = 1.0 - nn_sim
            k = max(1, int(0.01 * p))
            topk = torch.topk(patch_score, k=k).values.mean()
            scores.append(topk)
        return torch.stack(scores)

    def calibrate_threshold(self, scores: np.ndarray, labels: np.ndarray):
        from sklearn.metrics import f1_score

        best_t, best_f1 = float(np.median(scores)), -1.0
        for t in np.quantile(scores, np.linspace(0.05, 0.95, 37)):
            pred = (scores >= t).astype(int)
            f1 = f1_score(labels, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        self.threshold = best_t
        return best_t, best_f1

    def state_dict_bank(self) -> Dict:
        return {
            "cfg": self.cfg.__dict__,
            "memory_bank": None if self.memory_bank is None else self.memory_bank.detach().cpu(),
            "threshold": self.threshold,
        }

    def load_bank(self, state: Dict):
        self.threshold = float(state["threshold"])
        mb = state["memory_bank"]
        self.memory_bank = None if mb is None else mb.to(self.device)
