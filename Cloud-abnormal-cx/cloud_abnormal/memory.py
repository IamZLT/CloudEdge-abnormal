from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from .datasets import Sample
from .dino import DinoV3Encoder


@dataclass
class MemoryBank:
    features: torch.Tensor
    normal_median: float
    normal_scale: float
    reference_paths: list[str]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "features": self.features.half(),
                "normal_median": self.normal_median,
                "normal_scale": self.normal_scale,
                "reference_paths": self.reference_paths,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "MemoryBank":
        data = torch.load(path, map_location="cpu", weights_only=True)
        return cls(data["features"].float(), data["normal_median"], data["normal_scale"], data["reference_paths"])


def _coreset(features: torch.Tensor, limit: int, preselect: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    if len(features) > preselect:
        features = features[torch.randperm(len(features), generator=generator)[:preselect]]
    if len(features) <= limit:
        return features
    # Uniform patch sampling is intentionally used here: greedy k-center with
    # tens of thousands of centers is quadratic and impractical on full MVTec/VisA.
    return features[torch.randperm(len(features), generator=generator)[:limit]]


def patch_distances(query: torch.Tensor, bank: torch.Tensor, k: int, chunk: int = 4096) -> torch.Tensor:
    values: list[torch.Tensor] = []
    query = F.normalize(query.float(), dim=1)
    bank = F.normalize(bank.float(), dim=1)
    k = min(k, len(bank))
    for start in range(0, len(query), chunk):
        similarities = query[start : start + chunk] @ bank.T
        nearest = similarities.topk(k, dim=1).values
        values.append((1.0 - nearest).mean(dim=1))
    return torch.cat(values)


def fit_memory_bank(
    encoder: DinoV3Encoder,
    samples: list[Sample],
    max_patches: int,
    preselect_patches: int,
    knn: int,
    calibration_fraction: float,
    seed: int,
) -> MemoryBank:
    normal = [s for s in samples if s.split == "train" and s.label == 0]
    if not normal:
        raise RuntimeError("No normal training images found")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(normal))
    count = max(1, int(round(len(normal) * calibration_fraction))) if len(normal) >= 4 else 0
    calibration_ids = set(order[-count:].tolist()) if count else set()
    bank_features, calibration_features = [], []
    fit_count = max(1, len(normal) - len(calibration_ids))
    bank_quota = max(1, int(np.ceil(preselect_patches / fit_count)))
    calibration_limit = min(10000, preselect_patches)
    calibration_quota = max(1, int(np.ceil(calibration_limit / max(1, len(calibration_ids)))))
    sample_generator = torch.Generator().manual_seed(seed)
    for index, sample in enumerate(tqdm(normal, desc=f"DINO memory {normal[0].category}")):
        with Image.open(sample.image_path) as image:
            features, _ = encoder.encode(image)
        quota = calibration_quota if index in calibration_ids else bank_quota
        if len(features) > quota:
            ids = torch.randperm(len(features), generator=sample_generator)[:quota]
            features = features[ids]
        (calibration_features if index in calibration_ids else bank_features).append(features)
    if not bank_features:
        bank_features, calibration_features = calibration_features, []
    bank = _coreset(torch.cat(bank_features), max_patches, preselect_patches, seed)
    if calibration_features:
        device_bank = bank.to(encoder.device)
        calibration = torch.cat(calibration_features).to(encoder.device)
        distances = patch_distances(calibration, device_bank, knn, chunk=1024).cpu().numpy()
        median = float(np.median(distances))
        scale = float(max(np.quantile(distances, 0.995) - median, 1e-4))
    else:
        median, scale = 0.0, 0.25
    ref_count = min(8, len(normal))
    ref_ids = np.linspace(0, len(normal) - 1, ref_count, dtype=int)
    refs = [str(normal[index].image_path) for index in ref_ids]
    return MemoryBank(bank, median, scale, refs)


@torch.inference_mode()
def dino_anomaly_map(
    encoder: DinoV3Encoder,
    bank: MemoryBank,
    image: Image.Image,
    knn: int,
) -> tuple[np.ndarray, float]:
    features, grid = encoder.encode(image)
    features = features.to(bank.features.device)
    distances = patch_distances(features, bank.features, knn).cpu()
    normalized = ((distances - bank.normal_median) / bank.normal_scale).clamp(min=0)
    # Monotonic soft calibration preserves ranking above the normal range;
    # hard clipping at one would create ties among strong anomalies.
    calibrated = 1.0 - torch.exp(-normalized)
    patch_map = calibrated.reshape(1, 1, *grid)
    target_size = (image.height, image.width)
    anomaly = F.interpolate(patch_map, target_size, mode="bilinear", align_corners=False)[0, 0].numpy()
    anomaly = gaussian_filter(anomaly, sigma=max(target_size) * 0.008)
    top = max(1, int(anomaly.size * 0.01))
    score = float(np.partition(anomaly.reshape(-1), -top)[-top:].mean())
    return np.clip(anomaly, 0.0, 1.0), score
