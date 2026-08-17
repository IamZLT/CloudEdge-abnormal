from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    mask_path: Path | None
    category: str
    defect_type: str
    label: int
    split: str


def _images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def _find_mask(folder: Path, image: Path) -> Path | None:
    candidates = [
        folder / f"{image.stem}_mask.png",
        folder / f"{image.stem}.png",
        folder / image.name,
    ]
    return next((p for p in candidates if p.exists()), None)


def load_mvtec(root: str | Path) -> list[Sample]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"MVTec root does not exist: {root}")
    samples: list[Sample] = []
    for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        category = category_dir.name
        for split in ("train", "test"):
            split_dir = category_dir / split
            if not split_dir.exists():
                continue
            for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                defect = defect_dir.name
                label = int(defect.lower() not in {"good", "normal"})
                for image in _images(defect_dir):
                    mask = None
                    if label:
                        mask = _find_mask(category_dir / "ground_truth" / defect, image)
                        if split == "test" and mask is None:
                            raise FileNotFoundError(f"Mask not found for anomalous image: {image}")
                    samples.append(Sample(image, mask, category, defect, label, split))
    if not samples:
        raise RuntimeError(f"No MVTec images found under {root}")
    return samples


def load_mvtec_llm(root: str | Path) -> list[Sample]:
    """Loads the LLM-oriented MVTec split used by `mvtec_anomaly_llm`.

    Layout is inverted relative to the standard MVTec tree: images live under
    `<root>/<split>/<category>/<defect>/...` (with `good` as the nominal class),
    and defect images are symlinks into the original MVTec root. Ground-truth
    masks are resolved through the symlink target's original `ground_truth/`.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"MVTec-LLM root does not exist: {root}")
    samples: list[Sample] = []
    for split in ("train", "test"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for category_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            category = category_dir.name
            for defect_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
                defect = defect_dir.name
                label = int(defect.lower() not in {"good", "normal"})
                for image in _images(defect_dir):
                    mask = None
                    if label and split == "test":
                        real = image.resolve()
                        mask = _find_mask(real.parents[2] / "ground_truth" / defect, real)
                        if mask is None:
                            raise FileNotFoundError(f"Mask not found for anomalous image: {image}")
                    samples.append(Sample(image, mask, category, defect, label, split))
    if not samples:
        raise RuntimeError(f"No MVTec-LLM images found under {root}")
    return samples


def _load_visa_pytorch(root: Path) -> list[Sample]:
    for candidate in (root / "visa_pytorch" / "1cls", root / "1cls", root):
        if any((candidate / name / "train").exists() for name in ("candle", "cashew", "capsules")):
            return load_mvtec(candidate)
    return []


def _first(row: dict[str, str], names: Iterable[str]) -> str | None:
    lowered = {k.lower().strip(): v for k, v in row.items()}
    for name in names:
        value = lowered.get(name)
        if value not in (None, ""):
            return value.strip()
    return None


def _resolve_visa_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    value = value.replace("\\", "/").lstrip("./")
    candidates = [root / value, root / "Data" / value]
    return next((p for p in candidates if p.exists()), candidates[0])


def load_visa(root: str | Path) -> list[Sample]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"VisA root does not exist: {root}")
    pytorch_samples = _load_visa_pytorch(root)
    if pytorch_samples:
        return pytorch_samples

    csv_candidates = [root / "split_csv" / "1cls.csv", root / "1cls.csv"]
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if csv_path is None:
        raise RuntimeError(
            "Unsupported VisA layout. Expected visa_pytorch/1cls or split_csv/1cls.csv."
        )
    samples: list[Sample] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            image_value = _first(row, ("image", "image_path", "img_path", "path"))
            category = _first(row, ("object", "category", "class", "class_name"))
            split = (_first(row, ("split", "phase", "set")) or "test").lower()
            label_text = (_first(row, ("label", "anomaly", "is_anomaly")) or "0").lower()
            label = int(label_text in {"1", "true", "anomaly", "anomalous", "bad"})
            if image_value is None or category is None:
                raise ValueError(f"VisA CSV lacks image/category columns: {row}")
            image = _resolve_visa_path(root, image_value)
            mask = _resolve_visa_path(root, _first(row, ("mask", "mask_path", "gt_path")))
            if image is None or not image.exists():
                raise FileNotFoundError(f"VisA image not found: {image}")
            if label and (mask is None or not mask.exists()):
                raise FileNotFoundError(f"VisA mask not found for: {image}")
            defect = "anomaly" if label else "good"
            samples.append(Sample(image, mask, category, defect, label, split))
    return samples


def load_dataset(name: str, root: str | Path) -> list[Sample]:
    if name.lower() == "mvtec":
        return load_mvtec(root)
    if name.lower() == "mvtec_llm":
        return load_mvtec_llm(root)
    if name.lower() == "visa":
        return load_visa(root)
    raise ValueError(f"Unknown dataset: {name}")

