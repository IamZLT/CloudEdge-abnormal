from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.data import build_dataset, list_categories, summarize_split


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(20, 30, 40)).save(path)


def test_unified_registry_supports_all_layouts(tmp_path: Path):
    mvtec = tmp_path / "mvtec"
    _image(mvtec / "bottle" / "train" / "good" / "000.png")
    _image(mvtec / "bottle" / "test" / "good" / "001.png")
    _image(mvtec / "bottle" / "test" / "broken" / "002.png")
    _image(mvtec / "bottle" / "ground_truth" / "broken" / "002_mask.png")

    visa = tmp_path / "visa"
    _image(visa / "1cls" / "pcb1" / "train" / "good" / "000.JPG")
    _image(visa / "1cls" / "pcb1" / "test" / "good" / "001.JPG")
    _image(visa / "1cls" / "pcb1" / "test" / "bad" / "002.JPG")
    _image(visa / "1cls" / "pcb1" / "ground_truth" / "bad" / "002.png")

    realiad = tmp_path / "realiad"
    _image(realiad / "realiad_extracted" / "audiojack" / "OK" / "S1" / "ok.jpg")
    _image(realiad / "realiad_extracted" / "audiojack" / "NG" / "BX" / "ng.jpg")
    annotation = {
        "meta": {"normal_class": "OK"},
        "train": [],
        "test": [
            {"anomaly_class": "OK", "image_path": "OK/S1/ok.jpg", "mask_path": None},
            {"anomaly_class": "BX", "image_path": "NG/BX/ng.jpg", "mask_path": None},
        ],
    }
    annotation_path = realiad / "realiad_jsons" / "audiojack.json"
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")

    registry = tmp_path / "datasets.yaml"
    registry.write_text(
        "datasets:\n"
        f"  mvtec: {{root: {mvtec}, layout: mvtec}}\n"
        f"  visa: {{root: {visa}, layout: visa, subdir: 1cls}}\n"
        f"  realiad: {{root: {realiad}, layout: realiad, image_root: realiad_extracted, annotation_root: realiad_jsons}}\n",
        encoding="utf-8",
    )

    expected = {"mvtec": "bottle", "visa": "pcb1", "realiad": "audiojack"}
    for name, category in expected.items():
        assert list_categories(name, registry_path=registry) == [category]
        dataset = build_dataset(name, category, registry_path=registry, validate_files=True)
        assert sorted(record.label for record in dataset.records) == [0, 1]
        assert summarize_split(dataset)["anomaly"] == 1

    mvtec_ds = build_dataset("mvtec", "bottle", registry_path=registry, output="dict")
    mvtec_bad = next(mvtec_ds[i] for i, record in enumerate(mvtec_ds.records) if record.label == 1)
    assert mvtec_bad["mask_path"].endswith("002_mask.png")
    visa_ds = build_dataset("visa", "pcb1", registry_path=registry, output="dict")
    visa_bad = next(visa_ds[i] for i, record in enumerate(visa_ds.records) if record.label == 1)
    assert visa_bad["mask_path"].endswith("002.png")
