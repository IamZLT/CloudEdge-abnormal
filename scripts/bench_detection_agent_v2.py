#!/usr/bin/env python3
"""Conflict-only second-pass verifier for completed Detection Agent v1 runs.

The script reuses v1 PatchCore evidence, evidence boards, and first VLM reviews.
It invokes the VLM only when v1 would reverse PatchCore, then either accepts that
override or safely restores the expert result. Runs are resumable per sample.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.bench_detection_agent import _expert_from_dict, _review_from_dict
from src.detection_agent.conflict_verifier import ConflictVerifier, VerificationGate
from src.detection_agent.schemas import VerificationEvidence
from src.evaluation import evaluate_binary, summarize_latency


def _verification_from_dict(value: dict[str, Any]) -> VerificationEvidence:
    return VerificationEvidence(
        confirm_override=bool(value.get("confirm_override", False)),
        decision=str(value.get("decision", "REVIEW")).upper(),
        confidence=float(value.get("confidence", 0.0)),
        region_agreement=bool(value.get("region_agreement", False)),
        reason=str(value.get("reason", "")),
        parse_ok=bool(value.get("parse_ok", False)),
        latency_ms=float(value.get("latency_ms", 0.0)),
        peak_mem_mb=value.get("peak_mem_mb"),
        raw=str(value.get("raw", "")),
    )


def _load_category(v1_root: Path, category: str) -> tuple[dict[str, Any], dict[str, Any]]:
    category_root = v1_root / "mvtec" / category
    manifest = json.loads((category_root / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((category_root / "agent_result.json").read_text(encoding="utf-8"))
    if len(manifest["rows"]) != len(report["rows"]):
        raise ValueError(f"row count mismatch for {category}")
    for manifest_row, result_row in zip(manifest["rows"], report["rows"]):
        if manifest_row["path"] != result_row["path"]:
            raise ValueError(f"row alignment mismatch for {category}: {manifest_row['path']}")
    return manifest, report


def _is_conflict(result_row: dict[str, Any]) -> bool:
    expert_prediction = int(str(result_row["agent"]["expert"]["decision"]).upper() == "NG")
    return expert_prediction != int(result_row["prediction"])


def verify_categories(
    categories: list[str],
    config: dict[str, Any],
    v1_root: Path,
    out_root: Path,
    device: str,
    resume: bool,
) -> list[dict[str, str]]:
    cfg = config["verifier"]
    verifier = ConflictVerifier.from_model(
        cfg["model_path"],
        backend=str(cfg.get("backend", "transformers")),
        device=device,
        dtype=str(cfg.get("dtype", "bfloat16")),
        max_new_tokens=int(cfg.get("max_new_tokens", 160)),
    )
    failures: list[dict[str, str]] = []
    for category in categories:
        manifest, report = _load_category(v1_root, category)
        verification_dir = out_root / "mvtec" / category / "verifications"
        verification_dir.mkdir(parents=True, exist_ok=True)
        candidates = [
            (manifest_row, result_row)
            for manifest_row, result_row in zip(manifest["rows"], report["rows"])
            if _is_conflict(result_row)
        ]
        for position, (manifest_row, result_row) in enumerate(candidates, start=1):
            index = int(manifest_row["index"])
            destination = verification_dir / f"{index:04d}.json"
            if resume and destination.is_file():
                continue
            try:
                expert = _expert_from_dict(result_row["agent"]["expert"])
                review = _review_from_dict(result_row["agent"]["review"])
                board = v1_root / str(manifest_row["board"])
                result = verifier.verify(board, expert, review)
                destination.write_text(
                    json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
                )
                print(
                    f"[verify {category} {position}/{len(candidates)}] "
                    f"confirm={result.confirm_override} decision={result.decision} "
                    f"conf={result.confidence:.2f} agree={result.region_agreement}"
                )
            except Exception as exc:  # noqa: BLE001
                failures.append({"category": category, "index": str(index), "error": str(exc)})
                print(f"[verify] FAILED {category}/{index}: {exc}")
    del verifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return failures


def evaluate_categories(
    categories: list[str], config: dict[str, Any], v1_root: Path, out_root: Path
) -> list[dict[str, Any]]:
    cfg = config["verifier"]
    gate = VerificationGate(
        min_confidence=float(cfg.get("min_confidence", 0.90)),
        require_region_agreement=bool(cfg.get("require_region_agreement", True)),
        allow_overrides=bool(cfg.get("allow_overrides", True)),
    )
    reports: list[dict[str, Any]] = []
    for category in categories:
        manifest, v1_report = _load_category(v1_root, category)
        labels: list[int] = []
        scores: list[float] = []
        predictions: list[int] = []
        latencies: list[float] = []
        output_rows: list[dict[str, Any]] = []
        conflict_count = accepted = corrected = harmed = parse_failures = 0
        gate_reasons: dict[str, int] = {}
        peak_memories: list[float] = []
        for manifest_row, v1_row in zip(manifest["rows"], v1_report["rows"]):
            label = int(v1_row["label"])
            expert = _expert_from_dict(v1_row["agent"]["expert"])
            expert_prediction = int(expert.decision == "NG")
            conflict = _is_conflict(v1_row)
            verification = None
            accept = False
            gate_reason = "no_v1_decision_conflict"
            latency = float(v1_row["latency_ms"])
            if conflict:
                conflict_count += 1
                verification_path = (
                    out_root / "mvtec" / category / "verifications"
                    / f"{int(manifest_row['index']):04d}.json"
                )
                if verification_path.is_file():
                    verification = _verification_from_dict(
                        json.loads(verification_path.read_text(encoding="utf-8"))
                    )
                    latency += verification.latency_ms
                    parse_failures += int(not verification.parse_ok)
                    if verification.peak_mem_mb is not None:
                        peak_memories.append(float(verification.peak_mem_mb))
                review = _review_from_dict(v1_row["agent"]["review"])
                accept, gate_reason = gate.accept(expert, review, verification)
            gate_reasons[gate_reason] = gate_reasons.get(gate_reason, 0) + 1
            if accept:
                score = float(v1_row["score"])
                prediction = int(v1_row["prediction"])
                accepted += 1
                corrected += int(expert_prediction != label and prediction == label)
                harmed += int(expert_prediction == label and prediction != label)
            else:
                # Preserve the primary expert's score and ordering unless an
                # actual decision override passes the independent verifier.
                score = float(expert.probability)
                prediction = expert_prediction
            labels.append(label)
            scores.append(score)
            predictions.append(prediction)
            latencies.append(latency)
            output_rows.append({
                "index": int(manifest_row["index"]),
                "path": v1_row["path"],
                "label": label,
                "score": score,
                "prediction": prediction,
                "latency_ms": latency,
                "v1_conflict": conflict,
                "override_accepted": accept,
                "gate_reason": gate_reason,
                "expert": expert.to_dict(),
                "v1_review": v1_row["agent"].get("review"),
                "verification": None if verification is None else verification.to_dict(),
            })
        metrics = evaluate_binary(labels, scores, 0.5, predictions=predictions)
        report = {
            "schema_version": 2,
            "branch": "agent_v2",
            "method": (
                f"{manifest['expert_method']}_{cfg.get('model_id', 'vlm')}_conflict_verifier"
                + ("_advisory" if not bool(cfg.get("allow_overrides", True)) else "")
            ),
            "dataset": "mvtec",
            "category": category,
            "sampling": {"n_test_evaluated": len(labels)},
            "threshold_protocol": {
                "source": "PatchCore held-out normal calibration + fixed conflict verification",
                "threshold": 0.5,
                "test_labels_used_for_threshold": False,
                "test_labels_used_for_inference": False,
                "validation_status": "exploratory; policy requires external held-out validation",
            },
            "metrics": metrics,
            "expert_metrics": manifest["expert_metrics"],
            "latency_ms": summarize_latency(latencies),
            "agent_runtime": {
                "first_pass_reviewed": int(v1_report["agent_runtime"]["reviewed"]),
                "v1_conflicts": conflict_count,
                "verified": conflict_count - gate_reasons.get("verification_missing", 0),
                "verification_parse_failures": parse_failures,
                "accepted_overrides": accepted,
                "corrected_errors": corrected,
                "harmful_overrides": harmed,
                "override_precision": corrected / max(1, corrected + harmed),
                "gate_reasons": gate_reasons,
                "override_policy": (
                    "automatic" if bool(cfg.get("allow_overrides", True)) else "advisory_only"
                ),
                "peak_mem_mb": max(
                    peak_memories + [float(v1_report["agent_runtime"].get("peak_mem_mb", 0.0))]
                ),
            },
            "rows": output_rows,
        }
        category_root = out_root / "mvtec" / category
        category_root.mkdir(parents=True, exist_ok=True)
        (category_root / "agent_v2_result.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        reports.append(report)
    return reports


def write_summary(reports: list[dict[str, Any]], out_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for report in reports:
        metrics, expert, runtime = report["metrics"], report["expert_metrics"], report["agent_runtime"]
        rows.append({
            "category": report["category"], "n": metrics["n"],
            "expert_auroc": expert["image_auroc"], "agent_v2_auroc": metrics["image_auroc"],
            "delta_auroc": metrics["image_auroc"] - expert["image_auroc"],
            "expert_auprc": expert["image_auprc"], "agent_v2_auprc": metrics["image_auprc"],
            "delta_auprc": metrics["image_auprc"] - expert["image_auprc"],
            "expert_f1": expert["f1"], "agent_v2_f1": metrics["f1"],
            "delta_f1": metrics["f1"] - expert["f1"],
            "precision": metrics["precision"], "recall": metrics["recall"],
            "accuracy": metrics["accuracy"], "v1_conflicts": runtime["v1_conflicts"],
            "accepted_overrides": runtime["accepted_overrides"],
            "corrected_errors": runtime["corrected_errors"],
            "harmful_overrides": runtime["harmful_overrides"],
            "override_precision": runtime["override_precision"],
            "latency_mean_ms": report["latency_ms"]["mean_ms"],
            "latency_p95_ms": report["latency_ms"]["p95_ms"],
            "peak_mem_mb": runtime["peak_mem_mb"],
        })
    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mean_fields = [
        "expert_auroc", "agent_v2_auroc", "delta_auroc", "expert_auprc", "agent_v2_auprc",
        "delta_auprc", "expert_f1", "agent_v2_f1", "delta_f1", "precision", "recall",
        "accuracy", "latency_mean_ms", "latency_p95_ms",
    ]
    macro = {key: float(np.mean([float(row[key]) for row in rows])) for key in mean_fields}
    corrected = sum(int(row["corrected_errors"]) for row in rows)
    harmed = sum(int(row["harmful_overrides"]) for row in rows)
    macro.update({
        "categories": len(rows), "n": sum(int(row["n"]) for row in rows),
        "v1_conflicts": sum(int(row["v1_conflicts"]) for row in rows),
        "accepted_overrides": sum(int(row["accepted_overrides"]) for row in rows),
        "corrected_errors": corrected, "harmful_overrides": harmed,
        "override_precision": corrected / max(1, corrected + harmed),
        "peak_mem_mb": max(float(row["peak_mem_mb"]) for row in rows),
        "validation_status": "exploratory; validate unchanged policy on an external held-out dataset",
    })
    (out_root / "model_summary.json").write_text(json.dumps(macro, indent=2), encoding="utf-8")
    lines = [
        "# Detection Agent v2 benchmark", "",
        "> Exploratory MVTec follow-up. The policy must be validated unchanged on external held-out data.", "",
        "| Category | Expert AUROC | Agent v2 AUROC | ΔAUROC | Expert F1 | Agent v2 F1 | ΔF1 | Accepted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['category']} | {row['expert_auroc']:.4f} | {row['agent_v2_auroc']:.4f} | "
            f"{row['delta_auroc']:+.4f} | {row['expert_f1']:.4f} | {row['agent_v2_f1']:.4f} | "
            f"{row['delta_f1']:+.4f} | {row['accepted_overrides']} |"
        )
    lines.append(
        f"| **macro** | **{macro['expert_auroc']:.4f}** | **{macro['agent_v2_auroc']:.4f}** | "
        f"**{macro['delta_auroc']:+.4f}** | **{macro['expert_f1']:.4f}** | "
        f"**{macro['agent_v2_f1']:.4f}** | **{macro['delta_f1']:+.4f}** | "
        f"**{macro['accepted_overrides']}** |"
    )
    (out_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(macro, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/agent_patchcore_qwen9b_v2.yaml")
    parser.add_argument("--v1-root", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/detection_agent_v2")
    parser.add_argument("--categories", default="all")
    parser.add_argument("--phase", choices=["verify", "evaluate", "all"], default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    v1_root = args.v1_root or ROOT / str(config.get("v1_root", "outputs/detection_agent_mvtec"))
    categories = (
        sorted(path.parent.name for path in (v1_root / "mvtec").glob("*/agent_result.json"))
        if args.categories == "all"
        else [value.strip() for value in args.categories.split(",") if value.strip()]
    )
    if not categories:
        raise FileNotFoundError(f"no completed v1 categories under {v1_root}")
    args.out.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    if args.phase in {"verify", "all"}:
        failures = verify_categories(
            categories, config, v1_root, args.out, args.device, args.resume
        )
    if args.phase in {"evaluate", "all"}:
        reports = evaluate_categories(categories, config, v1_root, args.out)
        write_summary(reports, args.out)
    (args.out / "failures.json").write_text(
        json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"DONE phase={args.phase} categories={len(categories)} failures={len(failures)} out={args.out}")


if __name__ == "__main__":
    main()
