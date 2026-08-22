"""Faithful Anomalib PatchCore WR50 expert with the project's evidence API."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class WR50PatchCoreExpert(nn.Module):
    """Anomalib PatchCore plus patch maps and nearest normal image retrieval.

    The memory bank uses Anomalib's k-center-greedy coreset.  A separate compact
    image-level reference bank is retained only to choose panel D for the Agent;
    it does not affect anomaly scores.
    """

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        layers: list[str] | tuple[str, ...] = ("layer2", "layer3"),
        coreset_ratio: float = 0.1,
        num_neighbors: int = 9,
    ) -> None:
        super().__init__()
        from src.offline_timm import enable as enable_offline_timm

        enable_offline_timm("wide_resnet50_2")
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel

        self.device = torch.device(device)
        self.coreset_ratio = float(coreset_ratio)
        self.model = PatchcoreModel(
            layers=list(layers),
            backbone="wide_resnet50_2",
            pre_trained=True,
            num_neighbors=int(num_neighbors),
        ).to(self.device)
        self.threshold = 0.5
        self.reference_bank: torch.Tensor | None = None
        self.reference_paths: list[str] = []

    @property
    def memory_bank(self) -> torch.Tensor:
        return self.model.memory_bank

    @torch.no_grad()
    def _embedding(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        images = images.to(self.device).type(self.model.memory_bank.dtype)
        features = self.model.feature_extractor(images)
        features = {
            layer: self.model.feature_pooler(feature)
            for layer, feature in features.items()
        }
        embedding_map = self.model.generate_embedding(features)
        batch, _, height, width = embedding_map.shape
        flat = self.model.reshape_embedding(embedding_map)
        return flat.reshape(batch, height * width, -1), (height, width)

    @torch.no_grad()
    def fit(self, loader, max_batches: int | None = None) -> None:
        self.model.train()
        reference_features: list[torch.Tensor] = []
        reference_paths: list[str] = []
        for index, batch in enumerate(loader):
            images = batch[0].to(self.device)
            flat = self.model(images)
            batch_size = int(images.shape[0])
            flat_by_image = flat.reshape(batch_size, -1, flat.shape[-1])
            pooled = F.normalize(flat_by_image.float().mean(dim=1), dim=-1)
            reference_features.append(pooled.cpu())
            if len(batch) >= 3:
                reference_paths.extend(str(path) for path in batch[2])
            else:
                reference_paths.extend([""] * batch_size)
            if max_batches is not None and index + 1 >= max_batches:
                break
        self.model.subsample_embedding(self.coreset_ratio)
        self.model.eval()
        self.reference_bank = torch.cat(reference_features, dim=0).to(self.device)
        self.reference_paths = reference_paths

    @torch.no_grad()
    def predict_details(self, images: torch.Tensor) -> dict[str, Any]:
        if self.memory_bank.numel() == 0:
            raise RuntimeError("Call fit() before predict_details()")
        self.model.eval()
        embeddings, grid_hw = self._embedding(images)
        batch, patches, channels = embeddings.shape
        flat = embeddings.reshape(batch * patches, channels)
        patch_scores, locations = self.model.nearest_neighbors(flat, n_neighbors=1)
        patch_scores_by_image = patch_scores.reshape(batch, patches)
        locations_by_image = locations.reshape(batch, patches)
        image_scores = self.model.compute_anomaly_score(
            patch_scores_by_image,
            locations_by_image,
            embeddings,
        )

        reference_indices = torch.full((batch,), -1, dtype=torch.long, device=self.device)
        reference_similarity = torch.full((batch,), float("nan"), device=self.device)
        reference_paths = [""] * batch
        if self.reference_bank is not None and self.reference_paths:
            query = F.normalize(embeddings.float().mean(dim=1), dim=-1)
            reference_similarity, reference_indices = (query @ self.reference_bank.T).max(dim=1)
            reference_paths = [self.reference_paths[int(index)] for index in reference_indices]
        height, width = grid_hw
        return {
            "image_scores": image_scores,
            "patch_scores": patch_scores_by_image.reshape(batch, height, width),
            "grid_hw": grid_hw,
            "reference_indices": reference_indices,
            "reference_similarity": reference_similarity,
            "reference_paths": reference_paths,
        }

    @torch.no_grad()
    def predict_score(self, images: torch.Tensor) -> torch.Tensor:
        return self.predict_details(images)["image_scores"]
