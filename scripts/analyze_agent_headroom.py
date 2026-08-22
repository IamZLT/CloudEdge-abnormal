#!/usr/bin/env python3
"""Analyze PatchCore/VLM complementarity from already-recorded MVTec results.

The deployable replay policies never use test labels.  The explicitly named
``oracle`` section does use labels and is reported only as a diagnostic ceiling.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import evaluate_binary


def canonical_image_key(value: str) -> str:
    """Make Linux and server157 absolute MVTec paths comparable."""
    parts = Path(value).parts
    candidates = [i for i, part in enumerate(parts) if part.lower() == "mvtec"]
    if candidates:
        return "/".join(parts[candidates[-1] + 1 :])
    if "test" in parts:
        return "/".join(parts[max(0, parts.index("test") - 1) :])
    return Path(value).name


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(float), 1e-5, 1.0 - 1e-5)
    return np.log(values / (1.0 - values))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def calibrate_patchcore(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Map PatchCore distances to a 0.5-threshold probability without labels."""
    scale = max(abs(float(threshold)) * 0.25, 1e-6)
    return _sigmoid((scores - float(threshold)) / scale)


def conservative_replay(
    patch_probability: np.ndarray,
    vlm_scores: np.ndarray,
    vlm_confidence: np.ndarray,
    vlm_valid: np.ndarray,
    budget: float,
) -> tuple[np.ndarray, np.ndarray]:
    """A fixed, label-free preview of the planned conservative fusion policy."""
    uncertainty = np.abs(_logit(patch_probability))
    n_review = min(len(uncertainty), max(0, int(round(len(uncertainty) * budget))))
    review = np.zeros(len(uncertainty), dtype=bool)
    if n_review:
        review[np.argsort(uncertainty)[:n_review]] = True

    patch_logit = _logit(patch_probability)
    vlm_logit = _logit(vlm_scores)
    final_logit = patch_logit.copy()
    usable = review & vlm_valid & (vlm_confidence >= 0.85)
    # Qwen3.5-9B is high-precision but lower-recall: confident NG evidence may
    # raise a borderline score more strongly than an OK response may lower it.
    raise_mask = usable & (vlm_scores >= 0.5)
    lower_mask = usable & (vlm_scores < 0.5) & (np.abs(patch_logit) <= 1.25)
    final_logit[raise_mask] += 0.35 * np.maximum(vlm_logit[raise_mask], 0.0)
    final_logit[lower_mask] += 0.15 * np.minimum(vlm_logit[lower_mask], 0.0)
    return _sigmoid(final_logit), review


def _align(pc_report: dict[str, Any], vlm_report: dict[str, Any]) -> dict[str, np.ndarray]:
    vlm_by_key = {canonical_image_key(row["path"]): row for row in vlm_report["rows"]}
    aligned = []
    for pc_row in pc_report["rows"]:
        key = canonical_image_key(pc_row["path"])
        if key not in vlm_by_key:
            raise KeyError(f"VLM result missing image: {key}")
        aligned.append((pc_row, vlm_by_key[key], key))
    if len(aligned) != len(vlm_report["rows"]):
        raise ValueError("PatchCore/VLM row counts differ after path alignment")
    return {
        "keys": np.asarray([item[2] for item in aligned], dtype=object),
        "labels": np.asarray([int(item[0]["label"]) for item in aligned]),
        "pc_scores": np.asarray([float(item[0]["score"]) for item in aligned]),
        "pc_predictions": np.asarray([int(item[0]["prediction"]) for item in aligned]),
        "pc_latency": np.asarray([float(item[0].get("latency_ms", 0.0)) for item in aligned]),
        "vlm_scores": np.asarray([float(item[1]["score"]) for item in aligned]),
        "vlm_predictions": np.asarray([int(item[1]["prediction"]) for item in aligned]),
        "vlm_confidence": np.asarray(
            [float(item[1].get("result", {}).get("confidence", 0.0)) for item in aligned]
        ),
        "vlm_valid": np.asarray(
            [bool(item[1].get("result", {}).get("parse_ok", False)) for item in aligned]
        ),
        "vlm_latency": np.asarray(
            [float(item[1].get("result", {}).get("latency_ms", 0.0)) for item in aligned]
        ),
    }


def analyze_category(
    pc_path: Path, vlm_path: Path, budgets: list[float]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pc_report, vlm_report = _load(pc_path), _load(vlm_path)
    data = _align(pc_report, vlm_report)
    y = data["labels"]
    threshold = float(pc_report["threshold_protocol"]["threshold"])
    pc_probability = calibrate_patchcore(data["pc_scores"], threshold)
    pc_correct = data["pc_predictions"] == y
    vlm_correct = data["vlm_predictions"] == y
    disagreement = data["pc_predictions"] != data["vlm_predictions"]
    oracle_predictions = np.where(pc_correct, data["pc_predictions"], data["vlm_predictions"])
    oracle_metrics = evaluate_binary(
        y, oracle_predictions.astype(float), 0.5, predictions=oracle_predictions
    )

    summary = {
        "category": str(pc_report["category"]),
        "n": int(len(y)),
        "patchcore_errors": int((~pc_correct).sum()),
        "vlm_errors": int((~vlm_correct).sum()),
        "disagreements": int(disagreement.sum()),
        "patchcore_error_vlm_correct": int(((~pc_correct) & vlm_correct).sum()),
        "patchcore_correct_vlm_wrong": int((pc_correct & (~vlm_correct)).sum()),
        "both_wrong": int(((~pc_correct) & (~vlm_correct)).sum()),
        "oracle_f1": float(oracle_metrics["f1"]),
        "oracle_accuracy": float(oracle_metrics["accuracy"]),
        "patchcore_auroc": float(pc_report["metrics"]["image_auroc"]),
        "patchcore_f1": float(pc_report["metrics"]["f1"]),
        "vlm_auroc": float(vlm_report["metrics"]["image_auroc"]),
        "vlm_f1": float(vlm_report["metrics"]["f1"]),
    }
    replay_rows = []
    for budget in budgets:
        final_scores, review = conservative_replay(
            pc_probability,
            data["vlm_scores"],
            data["vlm_confidence"],
            data["vlm_valid"],
            budget,
        )
        metrics = evaluate_binary(y, final_scores, 0.5)
        reviewed_pc_wrong = review & (~pc_correct)
        replay_rows.append({
            "category": summary["category"],
            "budget": budget,
            "n_reviewed": int(review.sum()),
            "review_rate": float(review.mean()),
            "reviewed_patchcore_errors": int(reviewed_pc_wrong.sum()),
            "reviewed_correctable_errors": int((reviewed_pc_wrong & vlm_correct).sum()),
            "auroc": float(metrics["image_auroc"]),
            "auprc": float(metrics["image_auprc"]),
            "f1": float(metrics["f1"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "accuracy": float(metrics["accuracy"]),
            "estimated_latency_ms": float(
                data["pc_latency"].mean() + data["vlm_latency"][review].sum() / max(1, len(y))
            ),
        })
    return summary, replay_rows


def _macro(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patchcore-root", type=Path, default=ROOT / "outputs/mvtec_patchcore_dinov3/mvtec")
    parser.add_argument("--patchcore-file", default="traditional_patchcore_dinov3_vitl16.json")
    parser.add_argument("--vlm-root", type=Path, default=ROOT / "outputs/mvtec_full_vlm/mvtec")
    parser.add_argument("--vlm-file", default="vlm_qwen3_5_9b.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/agent_headroom")
    parser.add_argument("--budgets", default="0.1,0.2,0.3,1.0")
    args = parser.parse_args()
    budgets = [float(value) for value in args.budgets.split(",")]

    category_rows, replay_rows = [], []
    pc_files = sorted(args.patchcore_root.glob(f"*/{args.patchcore_file}"))
    if not pc_files:
        raise FileNotFoundError(f"no PatchCore reports under {args.patchcore_root}")
    for pc_path in pc_files:
        category = pc_path.parent.name
        vlm_path = args.vlm_root / category / args.vlm_file
        summary, replay = analyze_category(pc_path, vlm_path, budgets)
        category_rows.append(summary)
        replay_rows.extend(replay)

    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out / "category_complementarity.csv", category_rows)
    _write_csv(args.out / "replay_by_category.csv", replay_rows)
    replay_macro = []
    metric_keys = ["auroc", "auprc", "f1", "precision", "recall", "accuracy", "estimated_latency_ms"]
    for budget in budgets:
        group = [row for row in replay_rows if math.isclose(float(row["budget"]), budget)]
        replay_macro.append({"budget": budget, **_macro(group, metric_keys)})
    _write_csv(args.out / "replay_macro.csv", replay_macro)

    totals = {
        "n_categories": len(category_rows),
        "n_images": sum(row["n"] for row in category_rows),
        "patchcore_errors": sum(row["patchcore_errors"] for row in category_rows),
        "vlm_errors": sum(row["vlm_errors"] for row in category_rows),
        "disagreements": sum(row["disagreements"] for row in category_rows),
        "patchcore_error_vlm_correct": sum(row["patchcore_error_vlm_correct"] for row in category_rows),
        "patchcore_correct_vlm_wrong": sum(row["patchcore_correct_vlm_wrong"] for row in category_rows),
        "both_wrong": sum(row["both_wrong"] for row in category_rows),
        "macro": _macro(
            category_rows,
            ["patchcore_auroc", "patchcore_f1", "vlm_auroc", "vlm_f1", "oracle_f1", "oracle_accuracy"],
        ),
        "replay_macro": replay_macro,
        "warning": "oracle uses MVTec test labels and is diagnostic only; replay policies are label-free",
    }
    (args.out / "summary.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
    lines = [
        "# PatchCore + Qwen3.5-9B agent headroom", "",
        f"- Images: {totals['n_images']} across {totals['n_categories']} categories",
        f"- PatchCore errors: {totals['patchcore_errors']}",
        f"- PatchCore errors corrected by raw VLM: {totals['patchcore_error_vlm_correct']}",
        f"- PatchCore-correct samples harmed by raw VLM: {totals['patchcore_correct_vlm_wrong']}",
        f"- Both wrong: {totals['both_wrong']}", "",
        "| Review budget | AUROC | AUPRC | F1 | Precision | Recall | Accuracy | Estimated ms/image |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in replay_macro:
        lines.append(
            f"| {row['budget']:.0%} | {row['auroc']:.4f} | {row['auprc']:.4f} | {row['f1']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['accuracy']:.4f} | "
            f"{row['estimated_latency_ms']:.1f} |"
        )
    lines.extend(["", "> Oracle metrics use test labels and must not be reported as a deployable method."])
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
