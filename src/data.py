from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # Allow metadata validation before the ML env is installed.
    class Dataset:  # type: ignore[no-redef]
        pass


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DATASET_ALIASES = {
    "mvtec": "mvtec",
    "mvtec-ad": "mvtec",
    "mvtecad": "mvtec",
    "realiad": "realiad",
    "real-iad": "realiad",
    "real_iad": "realiad",
    "visa": "visa",
}
DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "configs" / "datasets.yaml"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    layout: str
    subdir: str | None = None
    image_root: str | None = None
    annotation_root: str | None = None

    @property
    def data_root(self) -> Path:
        return self.root / self.subdir if self.subdir else self.root


@dataclass(frozen=True)
class AnomalySample:
    dataset: str
    category: str
    split: str
    image_path: Path
    label: int
    defect_type: str
    mask_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["image_path"] = str(self.image_path)
        result["mask_path"] = str(self.mask_path) if self.mask_path else None
        return result


def _canonical_name(name: str) -> str:
    key = name.strip().lower()
    if key not in DATASET_ALIASES:
        choices = ", ".join(sorted(set(DATASET_ALIASES.values())))
        raise KeyError(f"Unknown dataset {name!r}; choose one of: {choices}")
    return DATASET_ALIASES[key]


def load_dataset_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> dict[str, DatasetSpec]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset registry not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("datasets", raw)
    if not isinstance(entries, Mapping):
        raise ValueError(f"Expected a 'datasets' mapping in {path}")

    registry: dict[str, DatasetSpec] = {}
    for raw_name, value in entries.items():
        if not isinstance(value, Mapping) or "root" not in value:
            raise ValueError(f"Dataset {raw_name!r} must define at least 'root'")
        name = _canonical_name(str(raw_name))
        registry[name] = DatasetSpec(
            name=name,
            root=Path(str(value["root"])).expanduser(),
            layout=str(value.get("layout", name)).lower(),
            subdir=value.get("subdir"),
            image_root=value.get("image_root"),
            annotation_root=value.get("annotation_root"),
        )
    return registry


def get_dataset_spec(
    name: str,
    root: str | Path | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> DatasetSpec:
    canonical = _canonical_name(name)
    registry = load_dataset_registry(registry_path)
    if canonical not in registry:
        raise KeyError(f"Dataset {canonical!r} is not configured in {registry_path}")
    spec = registry[canonical]
    if root is None:
        return spec
    return DatasetSpec(
        name=spec.name,
        root=Path(root).expanduser(),
        layout=spec.layout,
        subdir=spec.subdir,
        image_root=spec.image_root,
        annotation_root=spec.annotation_root,
    )


def build_transform(image_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _image_files(directory: Path, recursive: bool = False) -> list[Path]:
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMG_EXTS)


def _mask_for_mvtec_style(category_root: Path, defect_type: str, image_path: Path) -> Path | None:
    if defect_type == "good":
        return None
    mask_dir = category_root / "ground_truth" / defect_type
    candidates = [
        mask_dir / f"{image_path.stem}_mask.png",
        mask_dir / f"{image_path.stem}.png",
    ]
    return next((p for p in candidates if p.is_file()), None)


def _scan_mvtec_style(spec: DatasetSpec, category: str, split: str) -> list[AnomalySample]:
    category_root = spec.data_root / category
    split_root = category_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_root}")

    records: list[AnomalySample] = []
    for defect_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        defect_type = defect_dir.name
        label = 0 if defect_type.lower() in {"good", "ok", "normal"} else 1
        for image_path in _image_files(defect_dir):
            records.append(
                AnomalySample(
                    dataset=spec.name,
                    category=category,
                    split=split,
                    image_path=image_path,
                    label=label,
                    defect_type=defect_type,
                    mask_path=_mask_for_mvtec_style(category_root, defect_type, image_path),
                )
            )
    return records


def _scan_realiad(spec: DatasetSpec, category: str, split: str) -> list[AnomalySample]:
    annotation_root = spec.root / (spec.annotation_root or "realiad_jsons")
    image_root = spec.root / (spec.image_root or "realiad_extracted")
    annotation_path = annotation_root / f"{category}.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing Real-IAD annotation: {annotation_path}")

    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    if split not in payload:
        raise ValueError(f"Split {split!r} is not present in {annotation_path}")
    normal_class = str(payload.get("meta", {}).get("normal_class", "OK"))
    category_root = image_root / category

    records: list[AnomalySample] = []
    for item in payload[split]:
        defect_type = str(item.get("anomaly_class", normal_class))
        image_path = category_root / str(item["image_path"])
        raw_mask = item.get("mask_path")
        mask_path = category_root / str(raw_mask) if raw_mask else None
        records.append(
            AnomalySample(
                dataset=spec.name,
                category=category,
                split=split,
                image_path=image_path,
                label=0 if defect_type == normal_class else 1,
                defect_type=defect_type,
                mask_path=mask_path,
            )
        )
    return records


def list_categories(
    name: str,
    root: str | Path | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> list[str]:
    spec = get_dataset_spec(name, root=root, registry_path=registry_path)
    if spec.layout == "realiad":
        annotation_root = spec.root / (spec.annotation_root or "realiad_jsons")
        if not annotation_root.is_dir():
            raise FileNotFoundError(f"Missing Real-IAD annotation directory: {annotation_root}")
        return sorted(p.stem for p in annotation_root.glob("*.json"))
    if not spec.data_root.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {spec.data_root}")
    return sorted(
        p.name
        for p in spec.data_root.iterdir()
        if p.is_dir() and (p / "train").is_dir() and (p / "test").is_dir()
    )


class UnifiedAnomalyDataset(Dataset):
    """Unified image-level anomaly dataset for MVTec, Real-IAD and ViSA.

    ``output='tuple'`` keeps the repository's historical ``(image, label, path)``
    contract. ``output='dict'`` additionally exposes dataset/category/defect/mask
    metadata without loading a mask into memory.
    """

    def __init__(
        self,
        name: str,
        category: str,
        split: Literal["train", "test"] = "test",
        image_size: int = 224,
        root: str | Path | None = None,
        registry_path: str | Path = DEFAULT_REGISTRY_PATH,
        output: Literal["tuple", "dict"] = "tuple",
        validate_files: bool = False,
    ):
        if split not in {"train", "test"}:
            raise ValueError(f"Unknown split: {split!r}; expected 'train' or 'test'")
        if output not in {"tuple", "dict"}:
            raise ValueError(f"Unknown output mode: {output!r}")

        self.spec = get_dataset_spec(name, root=root, registry_path=registry_path)
        self.root = self.spec.root
        self.category = category
        self.split = split
        self.output = output
        self.image_size = image_size
        self.transform = None

        if self.spec.layout in {"mvtec", "mvtec_style", "visa"}:
            self.records = _scan_mvtec_style(self.spec, category, split)
        elif self.spec.layout == "realiad":
            self.records = _scan_realiad(self.spec, category, split)
        else:
            raise ValueError(f"Unsupported dataset layout: {self.spec.layout!r}")

        if not self.records:
            raise FileNotFoundError(
                f"No images found for dataset={self.spec.name} category={category} split={split}"
            )
        if validate_files:
            missing = [r.image_path for r in self.records if not r.image_path.is_file()]
            if missing:
                preview = ", ".join(str(p) for p in missing[:3])
                raise FileNotFoundError(f"{len(missing)} image files are missing; first: {preview}")

        # Compatibility with code that reads ``dataset.samples`` directly.
        self.samples = [(r.image_path, r.label) for r in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        from PIL import Image

        record = self.records[idx]
        image = Image.open(record.image_path).convert("RGB")
        if self.transform is None:
            self.transform = build_transform(self.image_size)
        image = self.transform(image)
        if self.output == "tuple":
            return image, record.label, str(record.image_path)
        item = record.to_dict()
        item["image"] = image
        return item


class MVTecCategory(UnifiedAnomalyDataset):
    """Backward-compatible MVTec loader used by existing scripts."""

    def __init__(self, root: str | Path, category: str, split: str, image_size: int = 224):
        super().__init__(
            name="mvtec",
            root=root,
            category=category,
            split=split,
            image_size=image_size,
            output="tuple",
        )
        # Preserve the historical attribute value (category directory).
        self.root = Path(root) / category


def build_dataset(
    name: str,
    category: str,
    split: Literal["train", "test"] = "test",
    image_size: int = 224,
    root: str | Path | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    output: Literal["tuple", "dict"] = "tuple",
    validate_files: bool = False,
) -> UnifiedAnomalyDataset:
    return UnifiedAnomalyDataset(
        name=name,
        root=root,
        category=category,
        split=split,
        image_size=image_size,
        registry_path=registry_path,
        output=output,
        validate_files=validate_files,
    )


def summarize_split(dataset: UnifiedAnomalyDataset) -> dict[str, Any]:
    n_anomaly = sum(r.label == 1 for r in dataset.records)
    return {
        "dataset": dataset.spec.name,
        "category": dataset.category,
        "split": dataset.split,
        "n": len(dataset),
        "anomaly": n_anomaly,
        "normal": len(dataset) - n_anomaly,
    }
