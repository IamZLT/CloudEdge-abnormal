#!/usr/bin/env python3
"""Validate configured datasets and print category/split statistics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DEFAULT_REGISTRY_PATH, build_dataset, list_categories, summarize_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--dataset", choices=["mvtec", "realiad", "visa"], default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--all-categories", action="store_true")
    parser.add_argument("--validate-files", action="store_true")
    args = parser.parse_args()

    names = [args.dataset] if args.dataset else ["mvtec", "realiad", "visa"]
    reports = []
    for name in names:
        categories = list_categories(name, registry_path=args.registry)
        selected = categories if args.all_categories else [args.category or categories[0]]
        for category in selected:
            if category not in categories:
                raise ValueError(f"Unknown {name} category {category!r}; choices: {categories}")
            dataset = build_dataset(
                name,
                category,
                split=args.split,
                registry_path=args.registry,
                validate_files=args.validate_files,
            )
            report = summarize_split(dataset)
            report["first_image"] = str(dataset.records[0].image_path)
            report["categories_available"] = len(categories)
            reports.append(report)
            print(json.dumps(report, ensure_ascii=False))

    print(f"validated={len(reports)} dataset/category split(s)")


if __name__ == "__main__":
    main()
