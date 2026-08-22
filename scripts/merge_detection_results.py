#!/usr/bin/env python3
"""Merge compatible benchmark summaries and compute method-level macro means."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


METRICS = [
    "auroc", "auprc", "f1", "precision", "recall", "accuracy",
    "fpr_at_recall_99", "valid_rate", "latency_mean_ms",
    "latency_p95_ms", "peak_mem_mb",
]


def read_rows(root: Path) -> list[dict[str, str]]:
    path = root / "summary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = [row for root in args.roots for row in read_rows(root)]
    rows.sort(key=lambda row: (row["dataset"], row["method"], row["category"]))
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "summary.csv", rows)
    (args.out / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["branch"], row["method"])].append(row)
    model_rows = []
    for (dataset, branch, method), group in sorted(grouped.items()):
        result = {
            "dataset": dataset,
            "branch": branch,
            "method": method,
            "categories": len(group),
            "n": sum(int(row["n"]) for row in group),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group if row.get(metric) not in {"", "None"}]
            result[f"macro_{metric}"] = sum(values) / len(values) if values else None
        model_rows.append(result)
    write_csv(args.out / "model_summary.csv", model_rows)
    (args.out / "model_summary.json").write_text(
        json.dumps(model_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# MVTec PatchCore backbone comparison", "",
        "| Method | Categories | Images | AUROC | AUPRC | F1 | Precision | Recall | Accuracy | Mean latency (ms) | Peak memory (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in model_rows:
        lines.append(
            f"| {row['method']} | {row['categories']} | {row['n']} | "
            f"{row['macro_auroc']:.4f} | {row['macro_auprc']:.4f} | "
            f"{row['macro_f1']:.4f} | {row['macro_precision']:.4f} | "
            f"{row['macro_recall']:.4f} | {row['macro_accuracy']:.4f} | "
            f"{row['macro_latency_mean_ms']:.2f} | {row['macro_peak_mem_mb']:.1f} |"
        )
    (args.out / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
