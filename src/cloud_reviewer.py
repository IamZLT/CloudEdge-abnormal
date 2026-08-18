"""Unified cloud reviewer: DINOv3 (pixel kNN) + Qwen3.5 (semantic) fusion.

Single entry point for the cloud-side detector across the whole repo. Wraps the
frozen-foundation-model fusion detector from ``third_part/Cloud-abnormal-cx`` so
that benches, the CLI, and the web layer share one implementation instead of
maintaining parallel cloud code paths (PatchCore / Qwen-VL / fusion).

Detection:
  - DINO branch : DINOv3 ViT-L patch features -> kNN distance to a normal memory
                  bank -> pixel anomaly map + image-level score.
  - LLM branch  : Qwen3.5 (2B / 9B) compares the target against normal references
                  and returns anomaly probability / defect type / reason / boxes.
  - Fusion rule : hand-written "only enhance, never suppress" blend (see
                  ``CloudAnomalyDetector.predict``).

The reviewer outputs a continuous anomaly score (0..1) plus a fixed-threshold
OK/NG decision. Memory banks are per-category and loaded lazily (cached).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
CX_ROOT = ROOT / "third_part" / "Cloud-abnormal-cx"

if str(CX_ROOT) not in sys.path:
    sys.path.insert(0, str(CX_ROOT))


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


class CloudReviewer:
    """DINOv3 + Qwen3.5 fusion detector with a stable ``review()`` contract.

    Parameters
    ----------
    config_path : path to a ``Cloud-abnormal-cx`` config (default: default_224.yaml).
    memory_dir  : directory holding ``<dataset>/<category>.pt`` banks.
    dataset     : dataset key used to locate banks (default ``mvtec_llm``).
    use_large   : use Qwen3.5-9B instead of the 2B semantic verifier.
    disable_qwen: run the DINO branch only (no LLM fusion) — A/B of the fusion gain.
    device      : override ``cfg.model.device``.
    threshold   : score threshold for the OK/NG decision.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        memory_dir: str | Path | None = None,
        dataset: str = "mvtec_llm",
        use_large: bool = False,
        disable_qwen: bool = False,
        device: str | None = None,
        threshold: float = 0.5,
    ) -> None:
        from cloud_abnormal.config import load_config
        from cloud_abnormal.pipeline import CloudAnomalyDetector

        self.config_path = Path(config_path) if config_path else CX_ROOT / "configs" / "default_224.yaml"
        self.memory_dir = Path(memory_dir) if memory_dir else ROOT / "outputs" / "cloud_abnormal_cx_224" / "memory"
        self.dataset = dataset
        self.threshold = float(threshold)
        self.use_large = bool(use_large)

        cfg = load_config(self.config_path)
        if device:
            cfg.model.device = device
        # dino_source is configured relative to Cloud-abnormal-cx; resolve it so
        # the reviewer works regardless of the caller's cwd.
        if not Path(cfg.model.dino_source).is_absolute():
            cfg.model.dino_source = str((CX_ROOT / cfg.model.dino_source).resolve())

        self.detector = CloudAnomalyDetector(cfg, use_large=use_large, disable_qwen=disable_qwen)
        self.cfg = cfg
        self._banks: dict[str, Any] = {}

    def _bank(self, category: str):
        from cloud_abnormal.memory import MemoryBank

        if category in self._banks:
            return self._banks[category]
        bank_path = self.memory_dir / self.dataset / f"{category}.pt"
        if not bank_path.exists():
            raise FileNotFoundError(f"Memory bank missing: {bank_path} (run the fit step first)")
        bank = MemoryBank.load(bank_path)
        bank.features = bank.features.to(self.cfg.model.device)
        self._banks[category] = bank
        return bank

    def review(self, image: str | Path, category: str) -> dict[str, Any]:
        """Return a single detection dict for one image.

        ``image`` may be a path; ``category`` selects the per-category memory bank.
        """
        from cloud_abnormal.datasets import Sample

        sample = Sample(
            image_path=Path(image).resolve(),
            mask_path=None,
            category=category,
            defect_type="unknown",
            label=0,
            split="test",
        )
        bank = self._bank(category)

        _sync(self.cfg.model.device)
        t0 = time.perf_counter()
        _fused_map, image_score, opinion = self.detector.predict(sample, bank)
        _sync(self.cfg.model.device)
        latency_ms = (time.perf_counter() - t0) * 1000

        decision = "NG" if image_score >= self.threshold else "OK"
        return {
            "decision": decision,
            "score": float(image_score),
            "threshold": self.threshold,
            "qwen_enabled": self.detector.qwen is not None,
            "qwen_probability": float(opinion.anomaly_probability),
            "qwen_defect_type": opinion.defect_type,
            "qwen_reason": opinion.reason,
            "qwen_regions": [region.__dict__ for region in opinion.regions],
            "latency_ms": float(latency_ms),
            "model_version": "dino+qwen3.5-9b-fusion" if self.use_large else "dino+qwen3.5-2b-fusion",
        }
