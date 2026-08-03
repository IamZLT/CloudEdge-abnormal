from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".JPG", ".PNG", ".BMP"}


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class MVTecCategory(Dataset):
    """Minimal MVTec-style loader: train/good + test/{good,defect*}."""

    def __init__(self, root: str | Path, category: str, split: str, image_size: int = 224):
        self.root = Path(root) / category
        self.split = split
        self.transform = build_transform(image_size)
        self.samples: List[Tuple[Path, int]] = []

        if split == "train":
            good_dir = self.root / "train" / "good"
            self.samples = [(p, 0) for p in sorted(good_dir.iterdir()) if p.suffix in IMG_EXTS]
        elif split == "test":
            test_root = self.root / "test"
            for sub in sorted(test_root.iterdir()):
                if not sub.is_dir():
                    continue
                label = 0 if sub.name == "good" else 1
                for p in sorted(sub.iterdir()):
                    if p.suffix in IMG_EXTS:
                        self.samples.append((p, label))
        else:
            raise ValueError(f"Unknown split: {split}")

        if not self.samples:
            raise FileNotFoundError(f"No images found under {self.root} split={split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, str(path)


def summarize_split(ds: MVTecCategory) -> dict:
    n_pos = sum(1 for _, y in ds.samples if y == 1)
    n_neg = len(ds) - n_pos
    return {"n": len(ds), "anomaly": n_pos, "normal": n_neg}
