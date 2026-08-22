#!/usr/bin/env python3
"""Benchmark edge(small Qwen-VL) / cloud(large Qwen-VL) / collab on MVTec test images.

Env: conda activate base
Example:
  CUDA_VISIBLE_DEVICES=0,1 python scripts/bench_qwen_vl.py --config configs/qwen_vl.yaml
  CUDA_VISIBLE_DEVICES=0,1 python scripts/bench_qwen_vl.py --max-images 8 --category bottle
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DEFAULT_REGISTRY_PATH, build_dataset
from src.evaluation import (
    evaluate_binary,
    qwen_anomaly_score,
    summarize_inference,
    summarize_latency,
)
from src.vlm import create_vlm_client
from src.vlm.collab_vlm import CollabVLMConfig, decide_route, fuse_edge_cloud


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/qwen_vl.yaml"))
    parser.add_argument("--dataset", choices=["mvtec", "realiad", "visa"], default=None)
    parser.add_argument("--data-root", default=None, help="override root from configs/datasets.yaml")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--category", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--edge-device", default=None)
    parser.add_argument("--cloud-device", default=None)
    parser.add_argument("--edge-model", default=None, help="override edge model_path")
    parser.add_argument("--cloud-model", default=None, help="override cloud model_path")
    parser.add_argument("--edge-backend", default=None, help="auto|qwen3_vl|transformers|internvl|minicpm")
    parser.add_argument("--cloud-backend", default=None, help="auto|qwen3_vl|transformers|internvl|minicpm")
    parser.add_argument("--skip-cloud", action="store_true", help="only run edge")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset_name = args.dataset or cfg.get("dataset", "mvtec")
    category = args.category or cfg.get("category", "bottle")
    max_images = args.max_images
    if max_images is None:
        max_images = cfg.get("bench", {}).get("max_images", 20)

    data_root = args.data_root or cfg.get("data_root")
    dataset = build_dataset(
        dataset_name,
        category,
        split="test",
        root=data_root,
        registry_path=args.registry,
        output="tuple",
        validate_files=True,
    )
    items = [(record.image_path, record.label) for record in dataset.records]
    if not items:
        raise FileNotFoundError(f"no test images for {dataset_name}/{category}")
    if max_images and max_images > 0:
        # balanced-ish: take from both ends (good last in sorted dirs often)
        items = items[: max_images // 2] + items[-(max_images - max_images // 2) :]
        # dedupe keep order
        seen = set()
        uniq = []
        for it in items:
            if it[0] in seen:
                continue
            seen.add(it[0])
            uniq.append(it)
        items = uniq

    prompt = cfg.get("prompt")
    edge_cfg = cfg["edge"]
    cloud_cfg = cfg["cloud"]
    edge_model_path = args.edge_model or edge_cfg["model_path"]
    cloud_model_path = args.cloud_model or cloud_cfg["model_path"]
    collab_cfg = CollabVLMConfig(
        conf_low=float(edge_cfg.get("conf_low", 0.55)),
        conf_high=float(edge_cfg.get("conf_high", 0.85)),
        uncertain_band=bool(cfg.get("collab", {}).get("uncertain_band", True)),
        cloud_extra_latency_ms=float(cfg.get("collab", {}).get("cloud_extra_latency_ms", 50.0)),
    )

    print(f"[bench] dataset={dataset_name} category={category} n={len(items)} edge={edge_model_path}")
    edge = create_vlm_client(
        model_path=edge_model_path,
        backend=args.edge_backend or edge_cfg.get("backend", "auto"),
        device=args.edge_device or edge_cfg.get("device", "cuda:0"),
        dtype=edge_cfg.get("dtype", "bfloat16"),
        max_new_tokens=int(edge_cfg.get("max_new_tokens", 128)),
        role="edge",
        prompt=prompt,
    )

    cloud = None
    if not args.skip_cloud:
        print(f"[bench] cloud={cloud_model_path}")
        cloud = create_vlm_client(
            model_path=cloud_model_path,
            backend=args.cloud_backend or cloud_cfg.get("backend", "auto"),
            device=args.cloud_device or cloud_cfg.get("device", "cuda:1"),
            dtype=cloud_cfg.get("dtype", "bfloat16"),
            max_new_tokens=int(cloud_cfg.get("max_new_tokens", 160)),
            role="cloud",
            prompt=prompt,
        )

    rows = []
    labels, e_preds, e_scores, e_valid = [], [], [], []
    c_preds, c_scores, c_valid = [], [], []
    s_preds, s_scores, s_valid = [], [], [], []
    n_hard = 0
    edge_lats, cloud_lats, s_lats = [], [], []

    for i, (path, y) in enumerate(items):
        edge_res = edge.infer(path)
        hard = decide_route(edge, edge_res, collab_cfg)
        cloud_res = None
        if hard and cloud is not None:
            cloud_res = cloud.infer(path)
            n_hard += 1
        elif cloud is not None:
            # still score cloud for B0 baseline on this smoke set
            cloud_res = cloud.infer(path)

        fused = fuse_edge_cloud(edge_res, cloud_res if hard else None, hard, collab_cfg)
        # for B0 we want cloud always — if cloud_res computed above use it
        if cloud_res is None and cloud is not None:
            cloud_res = cloud.infer(path)

        labels.append(y)
        edge_score = qwen_anomaly_score(edge_res.decision, edge_res.confidence, edge_res.parse_ok)
        e_preds.append(int(edge_score["prediction"]) if edge_score["valid"] else 0)
        e_scores.append(float(edge_score["score"]))
        e_valid.append(bool(edge_score["valid"]))
        edge_lats.append(edge_res.latency_ms)

        if cloud_res is not None:
            cloud_score = qwen_anomaly_score(cloud_res.decision, cloud_res.confidence, cloud_res.parse_ok)
            c_preds.append(int(cloud_score["prediction"]) if cloud_score["valid"] else 0)
            c_scores.append(float(cloud_score["score"]))
            c_valid.append(bool(cloud_score["valid"]))
            cloud_lats.append(cloud_res.latency_ms)
        else:
            c_preds.append(e_preds[-1])
            c_scores.append(e_scores[-1])
            c_valid.append(e_valid[-1])

        final_res = cloud_res if hard and cloud_res is not None else edge_res
        final_score = qwen_anomaly_score(final_res.decision, final_res.confidence, final_res.parse_ok)
        s_preds.append(int(final_score["prediction"]) if final_score["valid"] else 0)
        s_scores.append(float(final_score["score"]))
        s_valid.append(bool(final_score["valid"]))
        if not final_score["valid"]:
            fused["decision"] = "REVIEW"
        s_lats.append(fused["latency_ms"])

        row = {
            "path": str(path),
            "label": int(y),
            "edge": edge_res.to_dict(),
            "cloud": cloud_res.to_dict() if cloud_res is not None else None,
            "fused": fused,
        }
        rows.append(row)
        print(
            f"[{i+1}/{len(items)}] y={y} edge={edge_res.decision}/{edge_res.confidence:.2f} "
            f"hard={hard} final={fused['decision']} path={fused['path']} "
            f"lat={fused['latency_ms']:.0f}ms :: {path.name}"
        )

    labels_a = np.asarray(labels, dtype=int)
    report = {
        "dataset": dataset_name,
        "category": category,
        "n": len(items),
        "edge_model": edge_model_path,
        "cloud_model": None if args.skip_cloud else cloud_model_path,
        "hard_ratio": n_hard / max(1, len(items)),
        "n_hard": n_hard,
        "detection": {
            "B1_edge_only": evaluate_binary(labels_a, e_scores, 0.5, predictions=e_preds, valid_mask=e_valid),
            "B0_cloud_only": evaluate_binary(labels_a, c_scores, 0.5, predictions=c_preds, valid_mask=c_valid),
            "S_collab": evaluate_binary(labels_a, s_scores, 0.5, predictions=s_preds, valid_mask=s_valid),
        },
        "latency_ms": {
            "edge_mean": float(np.mean(edge_lats)) if edge_lats else None,
            "cloud_mean": float(np.mean(cloud_lats)) if cloud_lats else None,
            "S_mean": float(np.mean(s_lats)) if s_lats else None,
            "edge": summarize_latency(edge_lats),
            "cloud": summarize_latency(cloud_lats),
            "S": summarize_latency(s_lats),
        },
        "rows": rows,
    }
    if cloud is not None:
        report["cloud_runtime"] = summarize_inference(cloud_lats, cloud.model)

    out_dir = Path(cfg.get("results_dir", "outputs/qwen_vl")) / dataset_name / category
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bench.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    d = report["detection"]
    lines = [
        f"# Qwen-VL Cloud-Edge Bench — `{category}` (n={len(items)})",
        "",
        f"- Edge: `{report['edge_model']}`",
        f"- Cloud: `{report['cloud_model']}`",
        f"- Hard upload ratio: **{report['hard_ratio']:.2%}** ({n_hard}/{len(items)})",
        "",
        "## Detection metrics",
        "",
        "| Scheme | AUROC | AUPRC | F1 | P | R | FPR@R99 | Valid |",
        "|--------|-------|-------|----|---|---|---------|-------|",
        f"| B0 cloud | {d['B0_cloud_only']['image_auroc']:.4f} | {d['B0_cloud_only']['image_auprc']:.4f} | {d['B0_cloud_only']['f1']:.4f} | {d['B0_cloud_only']['precision']:.4f} | {d['B0_cloud_only']['recall']:.4f} | {d['B0_cloud_only']['fpr_at_target_recall']:.4f} | {d['B0_cloud_only']['valid_rate']:.2%} |",
        f"| B1 edge | {d['B1_edge_only']['image_auroc']:.4f} | {d['B1_edge_only']['image_auprc']:.4f} | {d['B1_edge_only']['f1']:.4f} | {d['B1_edge_only']['precision']:.4f} | {d['B1_edge_only']['recall']:.4f} | {d['B1_edge_only']['fpr_at_target_recall']:.4f} | {d['B1_edge_only']['valid_rate']:.2%} |",
        f"| S collab | {d['S_collab']['image_auroc']:.4f} | {d['S_collab']['image_auprc']:.4f} | {d['S_collab']['f1']:.4f} | {d['S_collab']['precision']:.4f} | {d['S_collab']['recall']:.4f} | {d['S_collab']['fpr_at_target_recall']:.4f} | {d['S_collab']['valid_rate']:.2%} |",
        "",
        f"- Edge mean latency: **{report['latency_ms']['edge_mean']:.1f} ms**",
    ]
    if report["latency_ms"]["cloud_mean"] is not None:
        lines.append(f"- Cloud mean latency: **{report['latency_ms']['cloud_mean']:.1f} ms**")
    if report.get("cloud_runtime"):
        runtime = report["cloud_runtime"]
        lines.append(
            f"- Cloud P95 latency: **{runtime['inference_latency_ms']['p95_ms']:.1f} ms**"
        )
        lines.append(f"- Cloud parameters: **{runtime['parameters']['total_m']:.2f} M**")
    lines += [
        f"- S mean latency: **{report['latency_ms']['S_mean']:.1f} ms**",
        "",
        "## LLM outputs (per image)",
        "",
        "Each row is the **raw structured reply** from Qwen-VL (decision / confidence / defect_type / reason).",
        "",
    ]
    for i, row in enumerate(rows):
        gt = "NG" if row["label"] == 1 else "OK"
        e = row["edge"]
        c = row.get("cloud") or {}
        f = row["fused"]
        lines += [
            f"### [{i+1}] `{Path(row['path']).name}`  (GT={gt})",
            "",
            f"- Route: **{f['path']}** | final={f['decision']} | hard={f['hard']}",
            f"- Edge ({e.get('latency_ms', 0):.0f} ms): `{e['decision']}` conf={e['confidence']:.2f} "
            f"| type={e.get('defect_type')} | reason={e.get('reason')}",
        ]
        if c:
            lines.append(
                f"- Cloud ({c.get('latency_ms', 0):.0f} ms): `{c.get('decision')}` conf={c.get('confidence', 0):.2f} "
                f"| type={c.get('defect_type')} | reason={c.get('reason')}"
            )
        lines += [
            "",
            "<details><summary>Edge raw LLM JSON</summary>",
            "",
            "```json",
            (e.get("raw") or "").strip() or "{}",
            "```",
            "",
            "</details>",
            "",
        ]
        if c.get("raw"):
            lines += [
                "<details><summary>Cloud raw LLM JSON</summary>",
                "",
                "```json",
                str(c.get("raw") or "").strip(),
                "```",
                "",
                "</details>",
                "",
            ]

    md = "\n".join(lines)
    (out_dir / "bench.md").write_text(md, encoding="utf-8")

    # compact LLM-only dump for demo /答辩
    llm_lines = [
        f"# LLM 文本输出 — `{category}`",
        "",
        "格式：图像 → 边侧/云端 Qwen-VL 生成的 JSON（decision / confidence / defect_type / reason）",
        "",
    ]
    for i, row in enumerate(rows):
        gt = "NG" if row["label"] == 1 else "OK"
        e = row["edge"]
        c = row.get("cloud") or {}
        llm_lines += [
            f"## {i+1}. {Path(row['path']).name} (GT={gt})",
            "",
            "### Edge LLM",
            "```json",
            (e.get("raw") or "").strip() or json.dumps(
                {
                    "decision": e["decision"],
                    "confidence": e["confidence"],
                    "defect_type": e.get("defect_type"),
                    "reason": e.get("reason"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
        ]
        if c:
            llm_lines += [
                "### Cloud LLM",
                "```json",
                (c.get("raw") or "").strip()
                or json.dumps(
                    {
                        "decision": c.get("decision"),
                        "confidence": c.get("confidence"),
                        "defect_type": c.get("defect_type"),
                        "reason": c.get("reason"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
                "",
                f"Final route: **{row['fused']['path']}** → `{row['fused']['decision']}`",
                "",
            ]
    (out_dir / "llm_outputs.md").write_text("\n".join(llm_lines), encoding="utf-8")

    print("\n" + md)
    print(f"Wrote {out_dir / 'bench.md'}")
    print(f"Wrote {out_dir / 'llm_outputs.md'}")


if __name__ == "__main__":
    main()
