#!/usr/bin/env python3
"""Split MVTec AD into a supervised train/test folder layout for LLM training.

Output layout (under datasets/mvtec_anomaly_llm):
    train/{category}/good/          # normal images (MVTec train/good)
    train/{category}/{defect}/      # 60% of each MVTec test/{defect}
    test/{category}/good/           # normal images (MVTec test/good)
    test/{category}/{defect}/       # 40% of each MVTec test/{defect}

Normal images follow the MVTec train/good vs test/good convention. Defect images
only exist under MVTec test/, so each defect folder is split by `holdout_ratio`.

Files are created as symlinks to the source images (no data duplication).

Also writes train.jsonl / test.jsonl manifests and split_meta.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted([p for p in d.iterdir() if p.suffix.lower() in IMG_EXT])


def symlink_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(str(src.resolve()), str(dst))


def discover_categories(data_root: Path) -> list[str]:
    return sorted(
        p.name
        for p in data_root.iterdir()
        if p.is_dir() and (p / "train" / "good").is_dir() and (p / "test").is_dir()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(ROOT / "datasets/mvtec_anomaly_detection"))
    parser.add_argument("--out-root", default=str(ROOT / "datasets/mvtec_anomaly_llm"))
    parser.add_argument("--holdout-ratio", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--categories", default=None, help="comma list override (default: all)")
    parser.add_argument("--force", action="store_true", help="clear existing output first")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve()
    rng = random.Random(args.seed)

    if args.categories:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    else:
        categories = discover_categories(data_root)

    if args.force and out_root.exists():
        for child in out_root.iterdir():
            if child.is_dir() and not child.is_symlink():
                import shutil
                shutil.rmtree(child)
            else:
                child.unlink()
    out_root.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict] = []
    test_rows: list[dict] = []
    stats: dict[str, dict] = {}

    for cat in categories:
        stats[cat] = {"train": {"good": 0, "defect": 0}, "test": {"good": 0, "defect": 0}}

        # --- Normal: MVTec train/good -> train, test/good -> test ---
        for p in list_images(data_root / cat / "train" / "good"):
            dst = out_root / "train" / cat / "good" / p.name
            symlink_image(p, dst)
            train_rows.append({"image": str(dst), "source": str(p), "category": cat,
                               "label": 0, "defect_type": "none", "split": "train"})
            stats[cat]["train"]["good"] += 1

        for p in list_images(data_root / cat / "test" / "good"):
            dst = out_root / "test" / cat / "good" / p.name
            symlink_image(p, dst)
            test_rows.append({"image": str(dst), "source": str(p), "category": cat,
                              "label": 0, "defect_type": "none", "split": "test"})
            stats[cat]["test"]["good"] += 1

        # --- Defect: split MVTec test/{defect} by holdout_ratio ---
        test_root = data_root / cat / "test"
        for sub in sorted(test_root.iterdir()):
            if not sub.is_dir() or sub.name == "good":
                continue
            defect_type = sub.name
            imgs = list_images(sub)
            rng.shuffle(imgs)
            n_hold = int(round(len(imgs) * args.holdout_ratio))
            n_hold = min(max(n_hold, 0), len(imgs))
            if len(imgs) >= 2 and n_hold >= len(imgs):
                n_hold = len(imgs) - 1
            hold = imgs[:n_hold]
            train = imgs[n_hold:]
            for p in train:
                dst = out_root / "train" / cat / defect_type / p.name
                symlink_image(p, dst)
                train_rows.append({"image": str(dst), "source": str(p), "category": cat,
                                   "label": 1, "defect_type": defect_type, "split": "train"})
                stats[cat]["train"]["defect"] += 1
            for p in hold:
                dst = out_root / "test" / cat / defect_type / p.name
                symlink_image(p, dst)
                test_rows.append({"image": str(dst), "source": str(p), "category": cat,
                                  "label": 1, "defect_type": defect_type, "split": "test"})
                stats[cat]["test"]["defect"] += 1

    def write_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(out_root / "train.jsonl", train_rows)
    write_jsonl(out_root / "test.jsonl", test_rows)

    meta = {
        "data_root": str(data_root),
        "out_root": str(out_root),
        "holdout_ratio": args.holdout_ratio,
        "seed": args.seed,
        "categories": categories,
        "per_category": stats,
        "totals": {
            "train": {"n": len(train_rows), "ok": sum(1 for r in train_rows if r["label"] == 0),
                      "ng": sum(1 for r in train_rows if r["label"] == 1)},
            "test": {"n": len(test_rows), "ok": sum(1 for r in test_rows if r["label"] == 0),
                     "ng": sum(1 for r in test_rows if r["label"] == 1)},
        },
    }
    (out_root / "split_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    print(json.dumps(meta["totals"], indent=2))
    for cat in categories:
        s = stats[cat]
        print(f"{cat:12} train good={s['train']['good']:4} defect={s['train']['defect']:4} | "
              f"test good={s['test']['good']:4} defect={s['test']['defect']:4}")
    print(f"Wrote {out_root}")


if __name__ == "__main__":
    main()
