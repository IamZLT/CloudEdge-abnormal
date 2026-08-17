from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    qwen_small_path: str = "/data2/zlt/anomaly_detection_llm/model_card/Qwen3.5-2B"
    qwen_large_path: str = "/data2/zlt/anomaly_detection_llm/model_card/Qwen3.5-9B"
    dino_path: str = "/data2/zlt/anomaly_detection_llm/model_card/dinov3-vitl16-pretrain-lvd1689m"
    dino_source: str = "./dinov3"
    device: str = "cuda"
    dtype: str = "bfloat16"
    image_size: int = 448
    dino_layers: list[int] = field(default_factory=lambda: [6, 12, 18, 24])


@dataclass
class DataConfig:
    mvtec_root: str = "/data2/zlt/code/CloudEdge-abnormal/datasets/mvtec"
    mvtec_llm_root: str = "/data2/zlt/code/CloudEdge-abnormal/datasets/mvtec_anomaly_llm"
    visa_root: str = "/data2/zlt/code/CloudEdge-abnormal/datasets/VisA"
    num_workers: int = 4


@dataclass
class MemoryConfig:
    max_patches: int = 15000
    preselect_patches: int = 80000
    knn: int = 3
    calibration_fraction: float = 0.2
    seed: int = 42


@dataclass
class QwenConfig:
    enabled: bool = True
    normal_references: int = 3
    max_new_tokens: int = 384
    cache_dir: str = "outputs/qwen_cache"


@dataclass
class FusionConfig:
    qwen_pixel_weight: float = 0.18
    qwen_image_weight: float = 0.25
    box_blur_fraction: float = 0.035


@dataclass
class EvaluationConfig:
    histogram_bins: int = 4096
    output_dir: str = "outputs"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    qwen: QwenConfig = field(default_factory=QwenConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _update_dataclass(instance: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise KeyError(f"Unknown configuration key: {key}")
        setattr(instance, key, value)


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path is None:
        return cfg
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    for section, values in raw.items():
        if not hasattr(cfg, section) or not isinstance(values, dict):
            raise KeyError(f"Unknown configuration section: {section}")
        _update_dataclass(getattr(cfg, section), values)
    return cfg
