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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vlm import QwenVLClient
from src.vlm.collab_vlm import CollabVLMConfig, decide_route, fuse_edge_cloud


def list_test_images(data_root: Path, category: str) -> list[tuple[Path, int]]:
    """Return (path, label) with label: 0=good, 1=defect."""
    test_root = data_root / category / "test"
    items: list[tuple[Path, int]] = []
    for sub in sorted(test_root.iterdir()):
        if not sub.is_dir():
            continue
        label = 0 if sub.name == "good" else 1
        for p in sorted(sub.glob("*")):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                items.append((p, label))
    return items


def _det(labels: np.ndarray, preds: np.ndarray, scores: np.ndarray) -> dict:
    out = {
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "n": int(len(labels)),
    }
    if len(np.unique(labels)) > 1 and len(np.unique(scores)) > 1:
        out["image_auroc"] = float(roc_auc_score(labels, scores))
    else:
        out["image_auroc"] = float("nan")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/qwen_vl.yaml"))
    parser.add_argument("--category", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--edge-device", default=None)
    parser.add_argument("--cloud-device", default=None)
    parser.add_argument("--skip-cloud", action="store_true", help="only run edge")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    category = args.category or cfg.get("category", "bottle")
    max_images = args.max_images
    if max_images is None:
        max_images = cfg.get("bench", {}).get("max_images", 20)

    data_root = Path(cfg["data_root"])
    items = list_test_images(data_root, category)
    if not items:
        raise FileNotFoundError(f"no test images under {data_root / category / 'test'}")
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
    collab_cfg = CollabVLMConfig(
        conf_low=float(edge_cfg.get("conf_low", 0.55)),
        conf_high=float(edge_cfg.get("conf_high", 0.85)),
        uncertain_band=bool(cfg.get("collab", {}).get("uncertain_band", True)),
        cloud_extra_latency_ms=float(cfg.get("collab", {}).get("cloud_extra_latency_ms", 50.0)),
    )

    print(f"[bench] category={category} n={len(items)} edge={edge_cfg['model_path']}")
    edge = QwenVLClient(
        model_path=edge_cfg["model_path"],
        device=args.edge_device or edge_cfg.get("device", "cuda:0"),
        dtype=edge_cfg.get("dtype", "bfloat16"),
        max_new_tokens=int(edge_cfg.get("max_new_tokens", 128)),
        role="edge",
        prompt=prompt,
    )

    cloud = None
    if not args.skip_cloud:
        print(f"[bench] cloud={cloud_cfg['model_path']}")
        cloud = QwenVLClient(
            model_path=cloud_cfg["model_path"],
            device=args.cloud_device or cloud_cfg.get("device", "cuda:1"),
            dtype=cloud_cfg.get("dtype", "bfloat16"),
            max_new_tokens=int(cloud_cfg.get("max_new_tokens", 160)),
            role="cloud",
            prompt=prompt,
        )

    rows = []
    labels, e_preds, e_scores = [], [], []
    c_preds, c_scores = [], []
    s_preds, s_scores = [], []
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
        e_preds.append(1 if edge_res.decision == "NG" else 0)
        # score: NG confidence or 1-OK confidence
        e_scores.append(edge_res.confidence if edge_res.decision == "NG" else 1.0 - edge_res.confidence)
        edge_lats.append(edge_res.latency_ms)

        if cloud_res is not None:
            c_preds.append(1 if cloud_res.decision == "NG" else 0)
            c_scores.append(cloud_res.confidence if cloud_res.decision == "NG" else 1.0 - cloud_res.confidence)
            cloud_lats.append(cloud_res.latency_ms)
        else:
            c_preds.append(e_preds[-1])
            c_scores.append(e_scores[-1])

        s_preds.append(1 if fused["decision"] == "NG" else 0)
        s_scores.append(fused["confidence"] if fused["decision"] == "NG" else 1.0 - fused["confidence"])
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
        "category": category,
        "n": len(items),
        "edge_model": edge_cfg["model_path"],
        "cloud_model": None if args.skip_cloud else cloud_cfg["model_path"],
        "hard_ratio": n_hard / max(1, len(items)),
        "n_hard": n_hard,
        "detection": {
            "B1_edge_only": _det(labels_a, np.asarray(e_preds), np.asarray(e_scores)),
            "B0_cloud_only": _det(labels_a, np.asarray(c_preds), np.asarray(c_scores)),
            "S_collab": _det(labels_a, np.asarray(s_preds), np.asarray(s_scores)),
        },
        "latency_ms": {
            "edge_mean": float(np.mean(edge_lats)) if edge_lats else None,
            "cloud_mean": float(np.mean(cloud_lats)) if cloud_lats else None,
            "S_mean": float(np.mean(s_lats)) if s_lats else None,
        },
        "rows": rows,
    }

    out_dir = Path(cfg.get("results_dir", "outputs/qwen_vl")) / category
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
        "| Scheme | AUROC | F1 | P | R | Acc |",
        "|--------|-------|----|---|---|-----|",
        f"| B0 cloud | {d['B0_cloud_only']['image_auroc']:.4f} | {d['B0_cloud_only']['f1']:.4f} | {d['B0_cloud_only']['precision']:.4f} | {d['B0_cloud_only']['recall']:.4f} | {d['B0_cloud_only']['accuracy']:.4f} |",
        f"| B1 edge | {d['B1_edge_only']['image_auroc']:.4f} | {d['B1_edge_only']['f1']:.4f} | {d['B1_edge_only']['precision']:.4f} | {d['B1_edge_only']['recall']:.4f} | {d['B1_edge_only']['accuracy']:.4f} |",
        f"| S collab | {d['S_collab']['image_auroc']:.4f} | {d['S_collab']['f1']:.4f} | {d['S_collab']['precision']:.4f} | {d['S_collab']['recall']:.4f} | {d['S_collab']['accuracy']:.4f} |",
        "",
        f"- Edge mean latency: **{report['latency_ms']['edge_mean']:.1f} ms**",
    ]
    if report["latency_ms"]["cloud_mean"] is not None:
        lines.append(f"- Cloud mean latency: **{report['latency_ms']['cloud_mean']:.1f} ms**")
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
