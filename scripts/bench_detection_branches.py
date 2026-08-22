#!/usr/bin/env python3
"""Unified image-level benchmark for VLM and traditional AD branches.

Both branches consume ``src.data.UnifiedAnomalyDataset`` and report the same
``src.evaluation.evaluate_binary`` schema. Traditional AD uses only normal
training images: one subset builds the PatchCoreLite bank and a disjoint normal
calibration subset fixes the operating threshold before test evaluation.
"""
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
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DEFAULT_REGISTRY_PATH, build_dataset, list_categories
from src.evaluation import (
    count_model_parameters,
    evaluate_binary,
    qwen_anomaly_score,
    summarize_latency,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_indices(labels: list[int], maximum: int | None, seed: int) -> list[int]:
    """Deterministically cap a test split while keeping OK/NG near balanced."""
    if maximum is None or maximum <= 0 or maximum >= len(labels):
        return list(range(len(labels)))
    rng = np.random.default_rng(seed)
    groups = {value: np.flatnonzero(np.asarray(labels) == value) for value in (0, 1)}
    selected: list[int] = []
    per_class = maximum // 2
    for value in (0, 1):
        take = min(per_class, len(groups[value]))
        if take:
            selected.extend(rng.choice(groups[value], size=take, replace=False).tolist())
    remaining = maximum - len(selected)
    if remaining:
        pool = np.asarray(sorted(set(range(len(labels))) - set(selected)), dtype=int)
        selected.extend(rng.choice(pool, size=min(remaining, len(pool)), replace=False).tolist())
    return sorted(selected)


def split_normal_indices(
    labels: list[int], maximum: int | None, calibration_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    normal = np.flatnonzero(np.asarray(labels) == 0)
    if len(normal) < 2:
        raise ValueError("traditional branch needs at least two normal training images")
    rng = np.random.default_rng(seed)
    normal = rng.permutation(normal)
    if maximum is not None and maximum > 0:
        normal = normal[:maximum]
    if len(normal) < 2:
        raise ValueError("max_train_normal must leave at least two normal images")
    n_calibration = max(1, int(round(len(normal) * calibration_fraction)))
    n_calibration = min(n_calibration, len(normal) - 1)
    return sorted(normal[n_calibration:].tolist()), sorted(normal[:n_calibration].tolist())


def _output_path(out_root: Path, dataset: str, category: str, branch: str, method: str) -> Path:
    safe_method = method.replace("/", "_").replace(".", "_")
    return out_root / dataset / category / f"{branch}_{safe_method}.json"


def _score_traditional(model, loader: DataLoader, device: str):
    labels: list[int] = []
    scores: list[float] = []
    paths: list[str] = []
    latencies: list[float] = []
    with torch.inference_mode():
        for images, batch_labels, batch_paths in loader:
            if device.startswith("cuda"):
                torch.cuda.synchronize(torch.device(device))
            started = time.perf_counter()
            batch_scores = model.predict_score(images)
            if device.startswith("cuda"):
                torch.cuda.synchronize(torch.device(device))
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            count = int(len(batch_labels))
            latencies.extend([elapsed_ms / max(1, count)] * count)
            scores.extend(batch_scores.detach().cpu().numpy().astype(float).tolist())
            labels.extend(batch_labels.detach().cpu().numpy().astype(int).tolist())
            paths.extend(str(path) for path in batch_paths)
    return labels, scores, paths, latencies


def run_traditional(
    dataset_name: str,
    category: str,
    protocol: dict[str, Any],
    method_cfg: dict[str, Any],
    registry: str,
    device: str,
    out_root: Path,
) -> dict[str, Any]:
    from src.models import PatchCoreConfig, create_patchcore_expert

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
        protocol.get("max_train_normal"),
        float(protocol.get("calibration_fraction", 0.2)),
        int(protocol.get("seed", 42)),
    )
    test_idx = stratified_indices(
        [record.label for record in test_ds.records],
        protocol.get("max_test_per_category"),
        int(protocol.get("seed", 42)),
    )
    workers = int(protocol.get("num_workers", 2))
    batch_size = int(protocol.get("traditional_batch_size", 8))
    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": device.startswith("cuda"),
    }
    bank_loader = DataLoader(Subset(train_ds, bank_idx), batch_size=batch_size, shuffle=False, **loader_kwargs)
    calibration_loader = DataLoader(
        Subset(train_ds, calibration_idx), batch_size=batch_size, shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(Subset(test_ds, test_idx), batch_size=batch_size, shuffle=False, **loader_kwargs)

    method_id = str(method_cfg.get("method", "patchcore_lite"))
    model = create_patchcore_expert(PatchCoreConfig(
        name=method_id,
        backbone=str(method_cfg.get("backbone", "resnet18")),
        layers=list(method_cfg.get("layers", ["layer2", "layer3"])),
        model_path=method_cfg.get("model_path"),
        coreset_ratio=float(method_cfg.get("coreset_ratio", 0.1)),
        max_memory_bank=int(method_cfg.get("max_memory_bank", 10000)),
        device=device,
    ))
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    fit_started = time.perf_counter()
    model.fit(bank_loader)
    gallery_build_s = time.perf_counter() - fit_started

    _, calibration_scores, _, _ = _score_traditional(model, calibration_loader, device)
    quantile = float(protocol.get("normal_threshold_quantile", 0.99))
    threshold = float(np.quantile(np.asarray(calibration_scores, dtype=float), quantile))
    labels, scores, paths, latencies = _score_traditional(model, test_loader, device)
    metrics = evaluate_binary(labels, scores, threshold)
    peak = None
    if device.startswith("cuda"):
        peak = float(torch.cuda.max_memory_allocated(torch.device(device)) / (1024**2))

    report = {
        "schema_version": 1,
        "branch": "traditional",
        "method": method_id,
        "dataset": dataset_name,
        "category": category,
        "sampling": {
            "n_test_full": len(test_ds),
            "n_test_evaluated": len(test_idx),
            "stratified_cap": protocol.get("max_test_per_category"),
            "n_train_normal_full": sum(record.label == 0 for record in train_ds.records),
            "n_gallery": len(bank_idx),
            "n_calibration_normal": len(calibration_idx),
        },
        "threshold_protocol": {
            "source": "held_out_train_normal",
            "quantile": quantile,
            "threshold": threshold,
            "test_labels_used_for_threshold": False,
        },
        "metrics": metrics,
        "latency_ms": summarize_latency(latencies),
        "runtime": {
            "device": device,
            "gallery_build_s": gallery_build_s,
            "peak_mem_mb": peak,
            "parameters": count_model_parameters(model),
            "memory_bank_size": int(model.memory_bank.shape[0]),
            "backbone": method_cfg.get("backbone", "resnet18"),
            "model_path": method_cfg.get("model_path"),
            "layers": list(method_cfg.get("layers", ["layer2", "layer3"])),
        },
        "rows": [
            {"path": path, "label": int(label), "score": float(score),
             "prediction": int(score >= threshold), "latency_ms": float(latency)}
            for path, label, score, latency in zip(paths, labels, scores, latencies)
        ],
    }
    path = _output_path(out_root, dataset_name, category, "traditional", method_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def run_vlm_category(
    client,
    model_id: str,
    dataset_name: str,
    category: str,
    protocol: dict[str, Any],
    registry: str,
    out_root: Path,
) -> dict[str, Any]:
    test_ds = build_dataset(
        dataset_name, category, split="test", registry_path=registry,
        output="tuple", validate_files=True,
    )
    indices = stratified_indices(
        [record.label for record in test_ds.records],
        protocol.get("max_test_per_category"),
        int(protocol.get("seed", 42)),
    )
    labels: list[int] = []
    scores: list[float] = []
    predictions: list[int] = []
    valid: list[bool] = []
    latencies: list[float] = []
    rows: list[dict[str, Any]] = []
    for position, index in enumerate(indices, start=1):
        record = test_ds.records[index]
        result = client.infer(record.image_path)
        converted = qwen_anomaly_score(result.decision, result.confidence, result.parse_ok)
        labels.append(int(record.label))
        scores.append(float(converted["score"]))
        predictions.append(int(converted["prediction"]) if converted["valid"] else 0)
        valid.append(bool(converted["valid"]))
        latencies.append(float(result.latency_ms))
        rows.append({
            "path": str(record.image_path), "label": int(record.label),
            "defect_type_gt": record.defect_type, "score": converted["score"],
            "prediction": converted["prediction"], "result": result.to_dict(),
        })
        print(
            f"[vlm {model_id} {dataset_name}/{category} {position}/{len(indices)}] "
            f"GT={'NG' if record.label else 'OK'} pred={result.decision} "
            f"conf={result.confidence:.2f} valid={result.parse_ok}"
        )
    metrics = evaluate_binary(
        labels, scores, 0.5, predictions=predictions, valid_mask=valid,
    )
    report = {
        "schema_version": 1,
        "branch": "vlm",
        "method": model_id,
        "dataset": dataset_name,
        "category": category,
        "sampling": {
            "n_test_full": len(test_ds),
            "n_test_evaluated": len(indices),
            "stratified_cap": protocol.get("max_test_per_category"),
        },
        "threshold_protocol": {
            "source": "model_OK_NG_decision",
            "score_mapping": "NG:confidence; OK:1-confidence",
            "threshold": 0.5,
            "test_labels_used_for_threshold": False,
        },
        "metrics": metrics,
        "latency_ms": summarize_latency(latencies),
        "runtime": {
            "model_path": client.model_path,
            "peak_mem_mb": max(
                (row["result"].get("peak_mem_mb") or 0.0 for row in rows), default=0.0
            ),
            "parameters": count_model_parameters(client.model),
        },
        "rows": rows,
    }
    path = _output_path(out_root, dataset_name, category, "vlm", model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def summary_row(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report["metrics"]
    latency = report["latency_ms"]
    runtime = report.get("runtime", {})
    return {
        "dataset": report["dataset"], "category": report["category"],
        "branch": report["branch"], "method": report["method"],
        "n": metrics["n_total"], "n_valid": metrics["n_valid"],
        "auroc": metrics["image_auroc"], "auprc": metrics["image_auprc"],
        "f1": metrics["f1"], "precision": metrics["precision"],
        "recall": metrics["recall"], "accuracy": metrics["accuracy"],
        "fpr_at_recall_99": metrics["fpr_at_target_recall"],
        "valid_rate": metrics["valid_rate"],
        "latency_mean_ms": latency["mean_ms"], "latency_p95_ms": latency["p95_ms"],
        "peak_mem_mb": runtime.get("peak_mem_mb"),
    }


def write_summary(reports: list[dict[str, Any]], out_root: Path) -> None:
    rows = [summary_row(report) for report in reports]
    out_root.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (out_root / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Detection branch benchmark", "",
        "| Dataset | Category | Branch | Method | N | AUROC | AUPRC | F1 | Recall | Valid | Mean ms | P95 ms |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['category']} | {row['branch']} | {row['method']} | "
            f"{row['n']} | {row['auroc']:.4f} | {row['auprc']:.4f} | {row['f1']:.4f} | "
            f"{row['recall']:.4f} | {row['valid_rate']:.2%} | "
            f"{row['latency_mean_ms']:.1f} | {row['latency_p95_ms']:.1f} |"
        )
    (out_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_plan(args) -> tuple[dict[str, Any], dict[str, list[str]], list[str], dict[str, Any]]:
    suite = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    protocol = dict(suite.get("protocol") or {})
    if args.max_test is not None:
        protocol["max_test_per_category"] = None if args.max_test <= 0 else args.max_test
    if args.max_train is not None:
        protocol["max_train_normal"] = None if args.max_train <= 0 else args.max_train
    protocol["seed"] = args.seed

    planned = {str(k): list(v) for k, v in (suite.get("datasets") or {}).items()}
    if args.datasets:
        names = [name.strip() for name in args.datasets.split(",") if name.strip()]
        planned = {name: planned.get(name, ["all"]) for name in names}
    if args.categories:
        if len(planned) != 1:
            raise ValueError("--categories override requires exactly one selected dataset")
        only = next(iter(planned))
        planned[only] = [value.strip() for value in args.categories.split(",") if value.strip()]
    for dataset_name, categories in list(planned.items()):
        if categories == ["all"]:
            planned[dataset_name] = list_categories(dataset_name, registry_path=args.registry)

    vlm_models = list(suite.get("vlm_models") or [])
    if args.vlm_models:
        vlm_models = [value.strip() for value in args.vlm_models.split(",") if value.strip()]
    traditional = dict(suite.get("traditional") or {})
    return protocol, planned, vlm_models, traditional


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/detection_experiments.yaml"))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--model-registry", default=str(ROOT / "configs/vlm_models.yaml"))
    parser.add_argument("--qwen-config", default=str(ROOT / "configs/qwen_vl.yaml"))
    parser.add_argument("--branch", choices=["vlm", "traditional", "all"], default="all")
    parser.add_argument("--datasets", default=None, help="comma list overriding suite")
    parser.add_argument("--categories", default=None, help="comma list; only with one dataset")
    parser.add_argument("--vlm-models", default=None, help="comma list of registry IDs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-test", type=int, default=None, help="0 means full test split")
    parser.add_argument("--max-train", type=int, default=None, help="0 means all train normals")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(ROOT / "outputs/detection_experiments"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    protocol, planned, vlm_models, traditional_cfg = load_plan(args)
    out_root = Path(args.out)
    print(json.dumps({
        "branch": args.branch, "datasets": planned, "vlm_models": vlm_models,
        "protocol": protocol, "device": args.device, "out": str(out_root),
    }, indent=2, ensure_ascii=False))
    if args.dry_run:
        return

    reports: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    run_traditional_branch = args.branch in {"traditional", "all"}
    run_vlm_branch = args.branch in {"vlm", "all"}

    if run_traditional_branch:
        method_id = str(traditional_cfg.get("method", "patchcore_lite"))
        for dataset_name, categories in planned.items():
            for category in categories:
                path = _output_path(out_root, dataset_name, category, "traditional", method_id)
                if args.resume and path.is_file():
                    reports.append(json.loads(path.read_text(encoding="utf-8")))
                    continue
                try:
                    print(f"[traditional] {dataset_name}/{category}")
                    reports.append(run_traditional(
                        dataset_name, category, protocol, traditional_cfg,
                        args.registry, args.device, out_root,
                    ))
                except Exception as exc:  # noqa: BLE001
                    print(f"[traditional] FAILED {dataset_name}/{category}: {exc}")
                    failures.append({"branch": "traditional", "dataset": dataset_name,
                                     "category": category, "error": str(exc)})
                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    if run_vlm_branch:
        from src.vlm import create_vlm_client

        model_registry = yaml.safe_load(Path(args.model_registry).read_text(encoding="utf-8"))["models"]
        prompt = (yaml.safe_load(Path(args.qwen_config).read_text(encoding="utf-8")) or {}).get("prompt")
        for model_id in vlm_models:
            if model_id not in model_registry:
                raise KeyError(f"unknown VLM model ID {model_id!r}")
            model_cfg = dict(model_registry[model_id])
            client = None
            try:
                print(f"[vlm] loading {model_id}: {model_cfg['model_path']}")
                client = create_vlm_client(
                    model_path=model_cfg["model_path"], backend=model_cfg.get("backend", "auto"),
                    device=model_cfg.get("device", args.device),
                    dtype=model_cfg.get("dtype", "bfloat16"),
                    max_new_tokens=int(model_cfg.get("max_new_tokens", 128)),
                    max_tiles=int(model_cfg.get("max_tiles", 6)),
                    role="cloud", prompt=prompt,
                )
                for dataset_name, categories in planned.items():
                    for category in categories:
                        path = _output_path(out_root, dataset_name, category, "vlm", model_id)
                        if args.resume and path.is_file():
                            reports.append(json.loads(path.read_text(encoding="utf-8")))
                            continue
                        try:
                            reports.append(run_vlm_category(
                                client, model_id, dataset_name, category,
                                protocol, args.registry, out_root,
                            ))
                        except Exception as exc:  # noqa: BLE001
                            print(f"[vlm] FAILED {model_id} {dataset_name}/{category}: {exc}")
                            failures.append({"branch": "vlm", "method": model_id,
                                             "dataset": dataset_name, "category": category,
                                             "error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                print(f"[vlm] FAILED to load {model_id}: {exc}")
                failures.append({"branch": "vlm", "method": model_id, "error": str(exc)})
            finally:
                del client
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    write_summary(reports, out_root)
    (out_root / "failures.json").write_text(
        json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"DONE reports={len(reports)} failures={len(failures)} out={out_root}")


if __name__ == "__main__":
    main()
