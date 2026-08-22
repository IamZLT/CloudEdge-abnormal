#!/usr/bin/env python3
"""Resumable PatchCore-expert + VLM-review Agent benchmark."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.bench_detection_branches import split_normal_indices
from src.data import DEFAULT_REGISTRY_PATH, build_dataset, list_categories
from src.detection_agent.evidence_builder import (
    build_evidence_board,
    patch_concentration,
    top_regions,
)
from src.detection_agent.fusion import ConservativeFusion, calibrate_patchcore_score
from src.detection_agent.pipeline import DetectionAgent
from src.detection_agent.router import BudgetRouter
from src.detection_agent.schemas import ExpertEvidence, RegionEvidence, ReviewEvidence
from src.evaluation import evaluate_binary, summarize_latency
from src.models import PatchCoreConfig, PatchCoreLite, create_patchcore_expert


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(dataset, indices: list[int], batch_size: int, workers: int, device: str):
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.startswith("cuda"),
    )


def _score_loader(model: PatchCoreLite, loader: DataLoader, device: str) -> np.ndarray:
    scores = []
    with torch.inference_mode():
        for images, _, _ in loader:
            scores.extend(model.predict_score(images).detach().cpu().numpy().tolist())
    return np.asarray(scores, dtype=float)


def _region_to_dict(region: RegionEvidence) -> dict[str, Any]:
    return {
        "bbox_xyxy": list(region.bbox_xyxy),
        "score": region.score,
        "grid_rc": list(region.grid_rc),
    }


def _expert_from_dict(value: dict[str, Any]) -> ExpertEvidence:
    return ExpertEvidence(
        image_path=str(value["image_path"]),
        category=str(value["category"]),
        score=float(value["score"]),
        threshold=float(value["threshold"]),
        probability=float(value["probability"]),
        decision=str(value["decision"]),
        patch_scores=np.zeros((1, 1), dtype=float),
        reference_path=str(value.get("reference_path") or ""),
        reference_similarity=value.get("reference_similarity"),
        regions=[
            RegionEvidence(
                bbox_xyxy=tuple(int(v) for v in region["bbox_xyxy"]),
                score=float(region["score"]),
                grid_rc=tuple(int(v) for v in region["grid_rc"]),
            )
            for region in value.get("regions", [])
        ],
        concentration=float(value.get("concentration", 0.0)),
        latency_ms=float(value.get("latency_ms", 0.0)),
    )


def _review_from_dict(value: dict[str, Any]) -> ReviewEvidence:
    return ReviewEvidence(
        decision=str(value["decision"]),
        confidence=float(value["confidence"]),
        defect_type=str(value.get("defect_type", "none")),
        reason=str(value.get("reason", "")),
        region_agreement=bool(value.get("region_agreement", False)),
        parse_ok=bool(value.get("parse_ok", False)),
        latency_ms=float(value.get("latency_ms", 0.0)),
        peak_mem_mb=value.get("peak_mem_mb"),
        raw=str(value.get("raw", "")),
    )


def prepare_category(
    dataset_name: str,
    category: str,
    suite: dict[str, Any],
    registry: str,
    device: str,
    out_root: Path,
    resume: bool,
) -> Path:
    category_root = out_root / dataset_name / category
    manifest_path = category_root / "manifest.json"
    if resume and manifest_path.is_file():
        print(f"[prepare] resume {category}")
        return manifest_path

    protocol = suite["protocol"]
    expert_cfg = suite["expert"]
    image_size = int(protocol.get("image_size", 224))
    train_ds = build_dataset(
        dataset_name, category, split="train", image_size=image_size,
        registry_path=registry, output="tuple", validate_files=True,
    )
    test_ds = build_dataset(
        dataset_name, category, split="test", image_size=image_size,
        registry_path=registry, output="tuple", validate_files=True,
    )
    bank_idx, calibration_idx = split_normal_indices(
        [record.label for record in train_ds.records],
        None,
        float(protocol.get("calibration_fraction", 0.2)),
        int(protocol.get("seed", 42)),
    )
    batch_size = int(protocol.get("batch_size", 8))
    workers = int(protocol.get("num_workers", 2))
    bank_loader = _loader(train_ds, bank_idx, batch_size, workers, device)
    calibration_loader = _loader(train_ds, calibration_idx, batch_size, workers, device)
    test_loader = _loader(test_ds, list(range(len(test_ds))), batch_size, workers, device)

    model = create_patchcore_expert(PatchCoreConfig(
        name=str(expert_cfg.get("method", "patchcore_dinov3_vitl16")),
        backbone=str(expert_cfg["backbone"]),
        model_path=expert_cfg.get("model_path"),
        layers=list(expert_cfg.get("layers", [24])),
        coreset_ratio=float(expert_cfg.get("coreset_ratio", 0.1)),
        max_memory_bank=int(expert_cfg.get("max_memory_bank", 10000)),
        device=device,
    ))
    model.fit(bank_loader)
    calibration_scores = _score_loader(model, calibration_loader, device)
    quantile = float(protocol.get("normal_threshold_quantile", 0.99))
    model.threshold = float(np.quantile(calibration_scores, quantile))

    evidence_items: list[ExpertEvidence] = []
    labels: list[int] = []
    panel_cfg = suite.get("evidence", {})
    top_k = int(panel_cfg.get("top_k_regions", 2))
    with torch.inference_mode():
        for images, batch_labels, paths in test_loader:
            if device.startswith("cuda"):
                torch.cuda.synchronize(torch.device(device))
            started = time.perf_counter()
            details = model.predict_details(images)
            if device.startswith("cuda"):
                torch.cuda.synchronize(torch.device(device))
            per_image_ms = (time.perf_counter() - started) * 1000.0 / max(1, len(paths))
            image_scores = details["image_scores"].detach().cpu().numpy()
            maps = details["patch_scores"].detach().cpu().numpy()
            similarities = details["reference_similarity"].detach().cpu().numpy()
            references = list(details["reference_paths"])
            for index, path in enumerate(paths):
                with Image.open(path) as source:
                    image_wh = source.size
                score = float(image_scores[index])
                probability = calibrate_patchcore_score(score, model.threshold)
                regions = top_regions(maps[index], image_wh, top_k=top_k)
                evidence_items.append(ExpertEvidence(
                    image_path=str(path),
                    category=category,
                    score=score,
                    threshold=model.threshold,
                    probability=probability,
                    decision="NG" if score >= model.threshold else "OK",
                    patch_scores=maps[index],
                    reference_path=str(references[index]),
                    reference_similarity=float(similarities[index]),
                    regions=regions,
                    concentration=patch_concentration(maps[index]),
                    latency_ms=per_image_ms,
                ))
                labels.append(int(batch_labels[index]))

    router_cfg = suite.get("router", {})
    router = BudgetRouter(
        review_budget=float(router_cfg.get("review_budget", 0.30)),
        concentration_weight=float(router_cfg.get("concentration_weight", 0.15)),
    )
    selected = router.select(evidence_items)
    evidence_dir = category_root / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (evidence, label) in enumerate(zip(evidence_items, labels)):
        reviewed = index in selected
        board_rel = None
        if reviewed:
            board = build_evidence_board(evidence, panel_size=int(panel_cfg.get("panel_size", 336)))
            board_path = evidence_dir / f"{index:04d}.jpg"
            board.image.save(board_path, quality=92)
            board_rel = str(board_path.relative_to(out_root))
        expert_value = evidence.to_dict(include_patch_scores=False)
        expert_value["regions"] = [_region_to_dict(region) for region in evidence.regions]
        rows.append({
            "index": index,
            "path": evidence.image_path,
            "label": label,
            "reviewed": reviewed,
            "board": board_rel,
            "expert": expert_value,
        })

    pc_metrics = evaluate_binary(
        labels, [item.score for item in evidence_items], model.threshold,
    )
    manifest = {
        "schema_version": 1,
        "dataset": dataset_name,
        "category": category,
        "expert_method": expert_cfg.get("method"),
        "threshold_protocol": {
            "source": "held_out_train_normal",
            "quantile": quantile,
            "threshold": model.threshold,
            "test_labels_used_for_threshold": False,
        },
        "router": {
            "review_budget": router.review_budget,
            "n_reviewed": len(selected),
            "review_rate": len(selected) / len(rows),
            "test_labels_used": False,
        },
        "expert_metrics": pc_metrics,
        "rows": rows,
    }
    category_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[prepare] {category}: n={len(rows)} review={len(selected)} threshold={model.threshold:.6f}")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest_path


def review_manifests(
    manifests: list[Path], suite: dict[str, Any], device: str, out_root: Path, resume: bool
) -> list[dict[str, str]]:
    from src.detection_agent.vlm_reviewer import DetectionReviewer

    reviewer_cfg = suite["reviewer"]
    reviewer = DetectionReviewer.from_model(
        reviewer_cfg["model_path"],
        backend=str(reviewer_cfg.get("backend", "transformers")),
        device=device,
        dtype=str(reviewer_cfg.get("dtype", "bfloat16")),
        max_new_tokens=int(reviewer_cfg.get("max_new_tokens", 160)),
    )
    failures = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        category = manifest["category"]
        review_dir = manifest_path.parent / "reviews"
        review_dir.mkdir(parents=True, exist_ok=True)
        pending = [row for row in manifest["rows"] if row["reviewed"]]
        for position, row in enumerate(pending, start=1):
            review_path = review_dir / f"{int(row['index']):04d}.json"
            if resume and review_path.is_file():
                continue
            try:
                board_path = out_root / row["board"]
                result = reviewer.review(board_path)
                review_path.write_text(
                    json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
                )
                print(
                    f"[review {category} {position}/{len(pending)}] "
                    f"{result.decision} conf={result.confidence:.2f} "
                    f"agree={result.region_agreement}"
                )
            except Exception as exc:  # noqa: BLE001
                failures.append({"category": category, "index": str(row["index"]), "error": str(exc)})
                print(f"[review] FAILED {category}/{row['index']}: {exc}")
    del reviewer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return failures


def evaluate_manifests(
    manifests: list[Path], suite: dict[str, Any], out_root: Path
) -> list[dict[str, Any]]:
    fusion_cfg = suite.get("fusion", {})
    agent = DetectionAgent(ConservativeFusion(
        min_confidence=float(fusion_cfg.get("min_confidence", 0.85)),
        raise_weight=float(fusion_cfg.get("raise_weight", 0.35)),
        lower_weight=float(fusion_cfg.get("lower_weight", 0.15)),
        max_lower_abs_logit=float(fusion_cfg.get("max_lower_abs_logit", 1.25)),
        require_region_agreement=bool(fusion_cfg.get("require_region_agreement", True)),
    ))
    reports = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        labels, scores, predictions, latencies, rows = [], [], [], [], []
        applied = parse_failures = region_rejections = 0
        changed_decisions = corrected_errors = harmful_overrides = 0
        peak_memories: list[float] = []
        for row in manifest["rows"]:
            expert = _expert_from_dict(row["expert"])
            review = None
            if row["reviewed"]:
                review_path = manifest_path.parent / "reviews" / f"{int(row['index']):04d}.json"
                if review_path.is_file():
                    review = _review_from_dict(json.loads(review_path.read_text(encoding="utf-8")))
                else:
                    review = ReviewEvidence("REVIEW", 0.0, "none", "missing review", False, False, 0.0)
            decision = agent.decide(expert, reviewed=bool(row["reviewed"]), review=review)
            expert_prediction = int(expert.decision == "NG")
            if decision.prediction != expert_prediction:
                changed_decisions += 1
                expert_correct = expert_prediction == int(row["label"])
                agent_correct = decision.prediction == int(row["label"])
                corrected_errors += int((not expert_correct) and agent_correct)
                harmful_overrides += int(expert_correct and (not agent_correct))
            labels.append(int(row["label"]))
            scores.append(decision.final_score)
            predictions.append(decision.prediction)
            latency = expert.latency_ms + (review.latency_ms if review is not None else 0.0)
            latencies.append(latency)
            applied += int(decision.review_applied)
            parse_failures += int(review is not None and not review.parse_ok)
            region_rejections += int(decision.fallback_reason == "region_disagreement")
            if review is not None and review.peak_mem_mb is not None:
                peak_memories.append(float(review.peak_mem_mb))
            rows.append({
                "path": row["path"],
                "label": int(row["label"]),
                "score": decision.final_score,
                "prediction": decision.prediction,
                "latency_ms": latency,
                "agent": decision.to_dict(),
            })
        metrics = evaluate_binary(labels, scores, 0.5, predictions=predictions)
        reviewed_count = sum(bool(row["reviewed"]) for row in manifest["rows"])
        report = {
            "schema_version": 1,
            "branch": "agent",
            "method": f"{manifest['expert_method']}_qwen3_5_9b_agent",
            "dataset": manifest["dataset"],
            "category": manifest["category"],
            "sampling": {"n_test_evaluated": len(rows)},
            "threshold_protocol": {
                "source": "PatchCore held-out normal calibration + fixed label-free fusion",
                "threshold": 0.5,
                "test_labels_used_for_threshold": False,
                "test_labels_used_for_fusion": False,
            },
            "metrics": metrics,
            "expert_metrics": manifest["expert_metrics"],
            "latency_ms": summarize_latency(latencies),
            "agent_runtime": {
                "reviewed": reviewed_count,
                "review_rate": reviewed_count / max(1, len(rows)),
                "review_applied": applied,
                "parse_failures": parse_failures,
                "region_rejections": region_rejections,
                "changed_decisions": changed_decisions,
                "corrected_errors": corrected_errors,
                "harmful_overrides": harmful_overrides,
                "override_precision": corrected_errors / max(1, corrected_errors + harmful_overrides),
                "peak_mem_mb": max(peak_memories, default=0.0),
            },
            "rows": rows,
        }
        result_path = manifest_path.parent / "agent_result.json"
        result_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        reports.append(report)
    return reports


def write_summary(reports: list[dict[str, Any]], out_root: Path) -> None:
    rows = []
    for report in reports:
        metrics, expert = report["metrics"], report["expert_metrics"]
        rows.append({
            "dataset": report["dataset"],
            "category": report["category"],
            "method": report["method"],
            "n": metrics["n"],
            "expert_auroc": expert["image_auroc"],
            "agent_auroc": metrics["image_auroc"],
            "delta_auroc": metrics["image_auroc"] - expert["image_auroc"],
            "expert_auprc": expert["image_auprc"],
            "agent_auprc": metrics["image_auprc"],
            "delta_auprc": metrics["image_auprc"] - expert["image_auprc"],
            "expert_f1": expert["f1"],
            "agent_f1": metrics["f1"],
            "delta_f1": metrics["f1"] - expert["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "accuracy": metrics["accuracy"],
            "review_rate": report["agent_runtime"]["review_rate"],
            "review_applied": report["agent_runtime"]["review_applied"],
            "changed_decisions": report["agent_runtime"]["changed_decisions"],
            "corrected_errors": report["agent_runtime"]["corrected_errors"],
            "harmful_overrides": report["agent_runtime"]["harmful_overrides"],
            "override_precision": report["agent_runtime"]["override_precision"],
            "peak_mem_mb": report["agent_runtime"]["peak_mem_mb"],
            "latency_mean_ms": report["latency_ms"]["mean_ms"],
            "latency_p95_ms": report["latency_ms"]["p95_ms"],
        })
    fields = list(rows[0])
    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    macro_fields = [
        "expert_auroc", "agent_auroc", "delta_auroc", "expert_auprc", "agent_auprc", "delta_auprc",
        "expert_f1", "agent_f1", "delta_f1",
        "precision", "recall", "accuracy", "review_rate", "latency_mean_ms", "latency_p95_ms",
    ]
    macro = {key: float(np.mean([float(row[key]) for row in rows])) for key in macro_fields}
    corrected = sum(int(row["corrected_errors"]) for row in rows)
    harmed = sum(int(row["harmful_overrides"]) for row in rows)
    macro.update({
        "categories": len(rows),
        "n": sum(int(row["n"]) for row in rows),
        "reviewed": sum(int(round(float(row["review_rate"]) * int(row["n"]))) for row in rows),
        "review_applied": sum(int(row["review_applied"]) for row in rows),
        "changed_decisions": sum(int(row["changed_decisions"]) for row in rows),
        "corrected_errors": corrected,
        "harmful_overrides": harmed,
        "override_precision": corrected / max(1, corrected + harmed),
        "peak_mem_mb": max(float(row["peak_mem_mb"]) for row in rows),
    })
    (out_root / "model_summary.json").write_text(json.dumps(macro, indent=2), encoding="utf-8")
    lines = [
        "# Detection Agent benchmark", "",
        "| Category | Expert AUROC | Agent AUROC | ΔAUROC | Expert F1 | Agent F1 | ΔF1 | Review | Mean ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['category']} | {row['expert_auroc']:.4f} | {row['agent_auroc']:.4f} | "
            f"{row['delta_auroc']:+.4f} | {row['expert_f1']:.4f} | {row['agent_f1']:.4f} | "
            f"{row['delta_f1']:+.4f} | {row['review_rate']:.1%} | {row['latency_mean_ms']:.1f} |"
        )
    lines.append(
        f"| **macro** | **{macro['expert_auroc']:.4f}** | **{macro['agent_auroc']:.4f}** | "
        f"**{macro['delta_auroc']:+.4f}** | **{macro['expert_f1']:.4f}** | "
        f"**{macro['agent_f1']:.4f}** | **{macro['delta_f1']:+.4f}** | "
        f"**{macro['review_rate']:.1%}** | **{macro['latency_mean_ms']:.1f}** |"
    )
    (out_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(macro, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/agent_patchcore_qwen9b.yaml"))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--dataset", choices=["mvtec", "visa", "realiad"], default="mvtec")
    parser.add_argument("--categories", default="all")
    parser.add_argument("--phase", choices=["prepare", "review", "evaluate", "all"], default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=str(ROOT / "outputs/detection_agent"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    suite = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(suite.get("protocol", {}).get("seed", 42))
    set_seed(seed)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    categories = (
        list_categories(args.dataset, registry_path=args.registry)
        if args.categories == "all"
        else [value.strip() for value in args.categories.split(",") if value.strip()]
    )
    manifests = [out_root / args.dataset / category / "manifest.json" for category in categories]
    failures: list[dict[str, str]] = []
    if args.phase in {"prepare", "all"}:
        manifests = [
            prepare_category(
                args.dataset, category, suite, args.registry, args.device, out_root, args.resume
            )
            for category in categories
        ]
    missing = [path for path in manifests if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing manifests; run --phase prepare first: {missing[:3]}")
    if args.phase in {"review", "all"}:
        failures.extend(review_manifests(manifests, suite, args.device, out_root, args.resume))
    if args.phase in {"evaluate", "all"}:
        reports = evaluate_manifests(manifests, suite, out_root)
        write_summary(reports, out_root)
    (out_root / "failures.json").write_text(
        json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"DONE phase={args.phase} categories={len(categories)} failures={len(failures)} out={out_root}")


if __name__ == "__main__":
    main()
