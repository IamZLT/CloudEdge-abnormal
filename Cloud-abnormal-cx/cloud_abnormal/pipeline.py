from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .config import Config
from .datasets import Sample, load_dataset
from .dino import DinoV3Encoder, load_mask
from .memory import MemoryBank, dino_anomaly_map, fit_memory_bank
from .metrics import HistogramMetrics, binary_metrics
from .qwen import FrozenQwenInspector, QwenOpinion, opinion_map


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


class CloudAnomalyDetector:
    def __init__(self, cfg: Config, use_large: bool = False, disable_qwen: bool = False) -> None:
        self.cfg = cfg
        self.encoder = DinoV3Encoder(
            cfg.model.dino_path,
            cfg.model.dino_source,
            cfg.model.device,
            cfg.model.dtype,
            cfg.model.image_size,
            cfg.model.dino_layers,
        )
        enabled = cfg.qwen.enabled and not disable_qwen
        self.qwen = None
        if enabled:
            qwen_path = cfg.model.qwen_large_path if use_large else cfg.model.qwen_small_path
            self.qwen = FrozenQwenInspector(
                qwen_path,
                cfg.model.device,
                cfg.model.dtype,
                cfg.qwen.max_new_tokens,
                cfg.qwen.cache_dir,
            )

    def predict(self, sample: Sample, bank: MemoryBank) -> tuple[np.ndarray, float, QwenOpinion]:
        with Image.open(sample.image_path) as opened:
            image = opened.convert("RGB")
        dino_map, dino_score = dino_anomaly_map(self.encoder, bank, image, self.cfg.memory.knn)
        opinion = QwenOpinion()
        qmap = np.zeros_like(dino_map)
        if self.qwen is not None:
            refs = [Path(p) for p in bank.reference_paths[: self.cfg.qwen.normal_references]]
            opinion = self.qwen.inspect(sample.category, sample.image_path, refs)
            qmap = opinion_map(opinion, dino_map.shape, self.cfg.fusion.box_blur_fraction)
        wp = self.cfg.fusion.qwen_pixel_weight
        # Qwen adds semantic evidence but never suppresses DINO's fine anomaly response.
        fused_map = np.clip(dino_map + wp * qmap * (1.0 - dino_map), 0, 1)
        wi = self.cfg.fusion.qwen_image_weight if self.qwen is not None else 0.0
        semantic_score = max(opinion.anomaly_probability, float(qmap.max()))
        blended_score = (1.0 - wi) * dino_score + wi * semantic_score
        image_score = max(dino_score, blended_score)
        return fused_map, float(image_score), opinion


def group_categories(samples: list[Sample]) -> dict[str, list[Sample]]:
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.category].append(sample)
    return dict(sorted(grouped.items()))


def fit_dataset(
    cfg: Config, dataset: str, root: str, memory_dir: str, categories: list[str] | None = None
) -> None:
    samples = load_dataset(dataset, root)
    grouped = group_categories(samples)
    if categories is not None:
        requested = set(categories)
        grouped = {k: v for k, v in grouped.items() if k in requested}
    detector = CloudAnomalyDetector(cfg, disable_qwen=True)
    output = Path(memory_dir) / dataset
    for category, category_samples in grouped.items():
        bank = fit_memory_bank(
            detector.encoder,
            category_samples,
            cfg.memory.max_patches,
            cfg.memory.preselect_patches,
            cfg.memory.knn,
            cfg.memory.calibration_fraction,
            cfg.memory.seed,
        )
        bank.save(output / f"{category}.pt")


def evaluate_dataset(
    cfg: Config,
    dataset: str,
    root: str,
    memory_dir: str,
    use_large: bool = False,
    disable_qwen: bool = False,
    categories: list[str] | None = None,
) -> dict:
    samples = load_dataset(dataset, root)
    grouped = group_categories(samples)
    if categories is not None:
        requested = set(categories)
        grouped = {k: v for k, v in grouped.items() if k in requested}
    detector = CloudAnomalyDetector(cfg, use_large=use_large, disable_qwen=disable_qwen)
    per_category, all_labels, all_scores = {}, [], []
    overall_pixels = HistogramMetrics(cfg.evaluation.histogram_bins)
    all_times: list[float] = []
    for category, category_samples in grouped.items():
        bank_path = Path(memory_dir) / dataset / f"{category}.pt"
        if not bank_path.exists():
            raise FileNotFoundError(f"Run fit first; missing memory bank: {bank_path}")
        bank = MemoryBank.load(bank_path)
        bank.features = bank.features.to(cfg.model.device)
        test_samples = [s for s in category_samples if s.split == "test"]
        labels, scores, times = [], [], []
        pixels = HistogramMetrics(cfg.evaluation.histogram_bins)
        records = []
        for sample in tqdm(test_samples, desc=f"Evaluate {dataset}/{category}"):
            _sync(cfg.model.device)
            start = time.perf_counter()
            anomaly_map, score, opinion = detector.predict(sample, bank)
            _sync(cfg.model.device)
            elapsed = time.perf_counter() - start
            with Image.open(sample.image_path) as image:
                mask = load_mask(sample.mask_path, image.size)
            pixels.update(mask, anomaly_map)
            overall_pixels.update(mask, anomaly_map)
            labels.append(sample.label)
            scores.append(score)
            times.append(elapsed)
            records.append({
                "image": str(sample.image_path), "label": sample.label, "score": score,
                "time_seconds": elapsed, "qwen_probability": opinion.anomaly_probability,
                "qwen_defect_type": opinion.defect_type, "qwen_reason": opinion.reason,
                "qwen_regions": [region.__dict__ for region in opinion.regions],
            })
        per_category[category] = {
            "image_level": binary_metrics(labels, scores),
            "pixel_level": pixels.compute(),
            "mean_time_seconds": float(np.mean(times)) if times else float("nan"),
            "num_test_images": len(test_samples),
            "records": records,
        }
        all_labels.extend(labels)
        all_scores.extend(scores)
        all_times.extend(times)
    summary = {
        "dataset": dataset,
        "model": "Qwen3.5-9B" if use_large else "Qwen3.5-2B",
        "qwen_enabled": detector.qwen is not None,
        "overall": {
            "image_level": binary_metrics(all_labels, all_scores),
            "pixel_level": overall_pixels.compute(),
            "mean_time_seconds": float(np.mean(all_times)),
            "num_test_images": len(all_times),
            "macro_average": {
                "image_level": {
                    metric: float(np.nanmean([v["image_level"][metric] for v in per_category.values()]))
                    for metric in ("auroc", "ap", "f1_max")
                },
                "pixel_level": {
                    metric: float(np.nanmean([v["pixel_level"][metric] for v in per_category.values()]))
                    for metric in ("auroc", "ap", "f1_max")
                },
            },
        },
        "per_category": per_category,
    }
    output = Path(cfg.evaluation.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    suffix = "9b" if use_large else "2b"
    suffix += "_dino_only" if detector.qwen is None else "_fusion"
    (output / f"{dataset}_{suffix}_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8"
    )
    return summary
