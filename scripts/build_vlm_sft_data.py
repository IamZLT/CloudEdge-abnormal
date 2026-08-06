#!/usr/bin/env python3
"""Build MVTec SFT JSONL for Qwen-VL OK/NG JSON (train + holdout split).

Train sources:
  - train/good → OK
  - (1 - holdout_ratio) of each test/* subfolder → OK/NG by folder name

Holdout:
  - holdout_ratio of each test/* subfolder (for fair eval, no train leak)

Example:
  python scripts/build_vlm_sft_data.py --config configs/qwen_vl_lora.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted([p for p in d.iterdir() if p.suffix.lower() in IMG_EXT])


def split_holdout(paths: list[Path], ratio: float, rng: random.Random) -> tuple[list[Path], list[Path]]:
    paths = list(paths)
    rng.shuffle(paths)
    n_hold = int(round(len(paths) * ratio))
    n_hold = min(max(n_hold, 0), len(paths))
    # keep at least 1 in train when possible
    if len(paths) >= 2 and n_hold >= len(paths):
        n_hold = len(paths) - 1
    hold = paths[:n_hold]
    train = paths[n_hold:]
    return train, hold


def make_target(decision: str, defect_type: str, category: str) -> str:
    if decision == "OK":
        obj = {
            "decision": "OK",
            "confidence": 0.95,
            "defect_type": "none",
            "reason": f"No visible defect on the {category}; product appears normal.",
        }
    else:
        nice = defect_type.replace("_", " ")
        obj = {
            "decision": "NG",
            "confidence": 0.95,
            "defect_type": defect_type,
            "reason": f"Visible {nice} defect on the {category}.",
        }
    return json.dumps(obj, ensure_ascii=False)


def add_sample(bucket: list, path: Path, decision: str, defect_type: str, category: str, split: str, prompt: str):
    bucket.append(
        {
            "image": str(path.resolve()),
            "category": category,
            "split": split,
            "label": 0 if decision == "OK" else 1,
            "defect_type": defect_type,
            "prompt": prompt,
            "response": make_target(decision, defect_type, category),
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/qwen_vl_lora.yaml"))
    parser.add_argument("--categories", default=None, help="comma list override")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_root = Path(cfg["data_root"])
    rng = random.Random(int(cfg.get("seed", 42)))
    holdout_ratio = float(cfg.get("holdout_ratio", 0.4))
    max_ok = cfg.get("max_ok_per_category", None)
    prompt = cfg.get("prompt", "Inspect the image. Reply JSON with decision OK/NG.")

    if args.categories:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    else:
        cats = cfg.get("categories")
        if cats in (None, "all"):
            categories = sorted(
                p.name
                for p in data_root.iterdir()
                if p.is_dir() and (p / "train" / "good").exists() and (p / "test").exists()
            )
        else:
            categories = list(cats)

    train_rows: list[dict] = []
    hold_rows: list[dict] = []

    for cat in categories:
        # train/good → OK (train only)
        ok_train = list_images(data_root / cat / "train" / "good")
        if max_ok is not None and len(ok_train) > int(max_ok):
            ok_train = rng.sample(ok_train, int(max_ok))
        for p in ok_train:
            add_sample(train_rows, p, "OK", "none", cat, "train_good", prompt)

        test_root = data_root / cat / "test"
        for sub in sorted(test_root.iterdir()):
            if not sub.is_dir():
                continue
            decision = "OK" if sub.name == "good" else "NG"
            defect_type = "none" if decision == "OK" else sub.name
            imgs = list_images(sub)
            tr, ho = split_holdout(imgs, holdout_ratio, rng)
            for p in tr:
                add_sample(train_rows, p, decision, defect_type, cat, f"test_{sub.name}_train", prompt)
            for p in ho:
                add_sample(hold_rows, p, decision, defect_type, cat, f"test_{sub.name}_holdout", prompt)

    out_dir = Path(cfg.get("results_dir", "outputs/qwen_vl_lora"))
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "sft_train.jsonl"
    hold_path = out_dir / "sft_holdout.jsonl"
    meta_path = out_dir / "sft_meta.json"

    with train_path.open("w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with hold_path.open("w", encoding="utf-8") as f:
        for r in hold_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ok_tr = sum(1 for r in train_rows if r["label"] == 0)
    n_ng_tr = sum(1 for r in train_rows if r["label"] == 1)
    n_ok_ho = sum(1 for r in hold_rows if r["label"] == 0)
    n_ng_ho = sum(1 for r in hold_rows if r["label"] == 1)
    meta = {
        "categories": categories,
        "holdout_ratio": holdout_ratio,
        "train": {"n": len(train_rows), "ok": n_ok_tr, "ng": n_ng_tr},
        "holdout": {"n": len(hold_rows), "ok": n_ok_ho, "ng": n_ng_ho},
        "train_path": str(train_path),
        "holdout_path": str(hold_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Wrote {train_path}")
    print(f"Wrote {hold_path}")


if __name__ == "__main__":
    main()
