from pathlib import Path
from typing import Optional

from common.config import DatasetConfig


def infer_ground_truth(dataset: DatasetConfig, image_path: str) -> Optional[str]:
    path = Path(image_path)
    if dataset.dataset_type == "mvtec":
        return "normal" if path.parent.name.lower() == "good" else "anomaly"
    if dataset.dataset_type == "visa":
        subset = path.parent.name.lower()
        if subset == "normal":
            return "normal"
        if subset == "anomaly":
            return "anomaly"
    return None
