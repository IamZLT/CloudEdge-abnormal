from pathlib import Path
from typing import Iterable, List

from common.config import DatasetConfig


def _is_image(path: Path, extensions: Iterable[str]) -> bool:
    return path.is_file() and path.suffix.lower() in extensions


def _selected_categories(root: Path, configured: List[str]) -> List[Path]:
    if configured:
        return [root / category for category in configured]
    return sorted(path for path in root.iterdir() if path.is_dir())


def _discover_mvtec(dataset: DatasetConfig, extensions: Iterable[str]) -> List[Path]:
    root = Path(dataset.root)
    images = []
    for category in _selected_categories(root, dataset.categories):
        split_dir = category / dataset.split
        if split_dir.is_dir():
            images.extend(path for path in split_dir.rglob("*") if _is_image(path, extensions))
    return images


def _discover_visa(dataset: DatasetConfig, extensions: Iterable[str]) -> List[Path]:
    root = Path(dataset.root)
    data_root = root / "data" if (root / "data").is_dir() else root
    selected_subsets = {value.lower() for value in dataset.subsets}
    images = []
    for category in _selected_categories(data_root, dataset.categories):
        image_root = category / "Data" / "Images"
        if not image_root.is_dir():
            continue
        for path in image_root.rglob("*"):
            if not _is_image(path, extensions):
                continue
            if selected_subsets and path.parent.name.lower() not in selected_subsets:
                continue
            images.append(path)
    return images


def discover_dataset_images(dataset: DatasetConfig, extensions: Iterable[str]) -> List[str]:
    root = Path(dataset.root)
    if not root.is_dir():
        raise FileNotFoundError(f"数据集目录不存在：{root}")

    if dataset.dataset_type == "mvtec":
        paths = _discover_mvtec(dataset, extensions)
    elif dataset.dataset_type == "visa":
        paths = _discover_visa(dataset, extensions)
    else:
        paths = [path for path in root.rglob("*") if _is_image(path, extensions)]

    paths = sorted(set(paths))
    if dataset.max_images is not None:
        paths = paths[: dataset.max_images]
    if not paths:
        raise ValueError(f"数据集 {dataset.name} 中没有找到待检测图像")
    return [str(path) for path in paths]
