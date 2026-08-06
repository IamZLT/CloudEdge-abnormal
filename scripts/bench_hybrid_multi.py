#!/usr/bin/env python3
"""Batch hybrid cloud-VLM review across MVTec categories (load Qwen once).

Env: conda activate base
Prereq: edge_scores.json for each category (export_edge_scores.py)

Example:
  CUDA_VISIBLE_DEVICES=2 python scripts/bench_hybrid_multi.py \
    --categories cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,tile,toothbrush,transistor,wood,zipper \
    --max-cloud-reviews 16
"""
from __future__ import annotations

import argparse
import json
import subprocess
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


def _det_binary(labels, preds, scores=None):
    out = {
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
    }
    if scores is not None and len(np.unique(labels)) > 1 and len(np.unique(scores)) > 1:
        out["image_auroc"] = float(roc_auc_score(labels, scores))
    else:
        out["image_auroc"] = float("nan")
    return out


def run_one(cfg, category: str, client: QwenVLClient, max_reviews: int | None) -> dict:
    out_dir = Path(cfg.get("results_dir", "outputs/hybrid")) / category
    edge_path = out_dir / "edge_scores.json"
    if not edge_path.exists():
        raise FileNotFoundError(f"Missing {edge_path}")

    edge_pack = json.loads(edge_path.read_text(encoding="utf-8"))
    items = edge_pack["items"]
    thr = float(edge_pack["threshold"])
    collab = cfg.get("collab", {})

    hard_idx = [i for i, it in enumerate(items) if it.get("hard")]
    if max_reviews is not None and int(max_reviews) >= 0:
        hard_idx = hard_idx[: int(max_reviews)]
    hard_set = set(hard_idx)

    print(f"\n===== {category} n={len(items)} hard_marked={edge_pack['n_hard']} reviews={len(hard_set)} =====")

    labels = np.asarray([it["label"] for it in items], dtype=int)
    edge_scores = np.asarray([it["edge_score"] for it in items], dtype=float)
    edge_preds = (edge_scores >= thr).astype(int)
    b1 = _det_binary(labels, edge_preds, edge_scores)

    s_preds = edge_preds.copy()
    s_scores = edge_scores.copy()
    rows = []
    cloud_lats = []
    n_upload = 0
    cloud_fixed = 0
    cloud_wrong = 0

    for i, it in enumerate(items):
        row = {
            "path": it["path"],
            "label": it["label"],
            "edge_score": it["edge_score"],
            "edge_pred": "NG" if it["edge_pred"] else "OK",
            "hard_marked": bool(it.get("hard")),
            "cloud": None,
            "final_decision": "NG" if it["edge_pred"] else "OK",
            "path_type": "LOCAL",
        }
        if i in hard_set:
            n_upload += 1
            img = it["path"]
            if not Path(img).exists():
                alt = Path(cfg["data_root"]) / category / "test" / Path(img).parent.name / Path(img).name
                if alt.exists():
                    img = str(alt)
            res = client.infer(img)
            cloud_lats.append(res.latency_ms)
            cloud_pred = 1 if res.decision == "NG" else 0
            s_preds[i] = cloud_pred
            s_scores[i] = res.confidence if res.decision == "NG" else (1.0 - res.confidence)
            row["cloud"] = res.to_dict()
            row["final_decision"] = res.decision
            row["path_type"] = "CLOUD_REVIEW"
            gt = int(it["label"])
            edge_wrong = int(it["edge_pred"]) != gt
            cloud_ok = cloud_pred == gt
            if edge_wrong and cloud_ok:
                cloud_fixed += 1
            if not cloud_ok:
                cloud_wrong += 1
            print(
                f"  [{n_upload}/{len(hard_set)}] {Path(img).name} "
                f"GT={'NG' if gt else 'OK'} edge={row['edge_pred']} "
                f"cloud={res.decision}/{res.confidence:.2f} | {res.reason[:60]}"
            )
        rows.append(row)

    s = _det_binary(labels, s_preds, s_scores)
    report = {
        "mode": "hybrid_edge_anomalib_cloud_qwen_vl",
        "category": category,
        "edge": {
            "model": edge_pack.get("edge_model"),
            "backbone": edge_pack.get("edge_backbone"),
            "threshold": thr,
        },
        "cloud": {
            "model_path": cfg["cloud"]["model_path"],
            "adapter_path": cfg["cloud"].get("adapter_path"),
        },
        "n": len(items),
        "n_cloud_reviews": n_upload,
        "hard_upload_ratio": n_upload / max(1, len(items)),
        "cloud_fixed_edge": cloud_fixed,
        "cloud_wrong_on_reviewed": cloud_wrong,
        "detection": {"B1_edge_only": b1, "S_collab": s},
        "latency_ms": {
            "cloud_mean": float(np.mean(cloud_lats)) if cloud_lats else None,
            "cloud_extra": float(collab.get("cloud_extra_latency_ms", 50.0)),
        },
        "rows": rows,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bench.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# Hybrid Bench — edge Anomalib + cloud Qwen-VL — `{category}`",
        "",
        f"- Edge: {report['edge']['model']}/{report['edge']['backbone']}",
        f"- Cloud: `{report['cloud']['model_path']}`",
        f"- LoRA: `{report['cloud'].get('adapter_path')}`" if report["cloud"].get("adapter_path") else "",
        f"- Cloud reviews: **{n_upload}/{len(items)}** ({report['hard_upload_ratio']:.2%})",
        f"- Cloud fixed edge errors: **{cloud_fixed}** | cloud wrong on reviewed: **{cloud_wrong}**",
        "",
        "## Detection",
        "",
        "| Scheme | AUROC | F1 | P | R | Acc |",
        "|--------|-------|----|---|---|-----|",
        f"| B1 edge-only | {b1['image_auroc']:.4f} | {b1['f1']:.4f} | {b1['precision']:.4f} | {b1['recall']:.4f} | {b1['accuracy']:.4f} |",
        f"| S collab (edge+VLM) | {s['image_auroc']:.4f} | {s['f1']:.4f} | {s['precision']:.4f} | {s['recall']:.4f} | {s['accuracy']:.4f} |",
        "",
    ]
    if report["latency_ms"]["cloud_mean"] is not None:
        lines.append(f"- Cloud VLM mean latency: **{report['latency_ms']['cloud_mean']:.1f} ms**")
    lines += ["", "## Cloud LLM outputs (reviewed hard samples)", ""]

    llm_lines = [f"# Cloud LLM outputs — `{category}`", ""]
    for row in rows:
        if not row.get("cloud"):
            continue
        c = row["cloud"]
        gt = "NG" if row["label"] == 1 else "OK"
        name = Path(row["path"]).name
        lines += [
            f"### `{name}` (GT={gt}, edge={row['edge_pred']}, score={row['edge_score']:.4f})",
            "",
            f"- Final: **{row['final_decision']}** via `{row['path_type']}`",
            f"- LLM: `{c['decision']}` conf={c['confidence']:.2f} | {c.get('defect_type')} | {c.get('reason')}",
            "",
            "```json",
            (c.get("raw") or "").strip(),
            "```",
            "",
        ]
        llm_lines += [
            f"## {name} (GT={gt})",
            "",
            f"- Edge: {row['edge_pred']} (score={row['edge_score']:.4f})",
            f"- Cloud LLM → final `{row['final_decision']}`",
            "",
            "```json",
            (c.get("raw") or "").strip(),
            "```",
            "",
        ]

    (out_dir / "bench.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "llm_outputs.md").write_text("\n".join(llm_lines), encoding="utf-8")
    print(
        f"[{category}] B1_f1={b1['f1']:.4f} S_f1={s['f1']:.4f} "
        f"Δf1={s['f1']-b1['f1']:+.4f} fixed={cloud_fixed} wrong={cloud_wrong}"
    )
    return {
        "category": category,
        "n": len(items),
        "n_reviews": n_upload,
        "B1_auroc": b1["image_auroc"],
        "B1_f1": b1["f1"],
        "S_auroc": s["image_auroc"],
        "S_f1": s["f1"],
        "delta_f1": s["f1"] - b1["f1"],
        "cloud_fixed": cloud_fixed,
        "cloud_wrong": cloud_wrong,
        "cloud_lat_ms": report["latency_ms"]["cloud_mean"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/hybrid.yaml"))
    parser.add_argument("--categories", default=None, help="comma list; default=all with edge_scores.json")
    parser.add_argument("--max-cloud-reviews", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(cfg.get("results_dir", "outputs/hybrid"))
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    else:
        categories = sorted(p.parent.name for p in root.glob("*/edge_scores.json"))

    if args.skip_existing:
        categories = [c for c in categories if not (root / c / "bench.json").exists()]

    print(f"[multi] categories={categories} max_reviews={args.max_cloud_reviews}")
    cloud_cfg = cfg["cloud"]
    adapter = cloud_cfg.get("adapter_path")
    if adapter:
        print(f"[multi] LoRA adapter = {adapter}")
    client = QwenVLClient(
        model_path=cloud_cfg["model_path"],
        device=args.device or cloud_cfg.get("device", "cuda:0"),
        dtype=cloud_cfg.get("dtype", "bfloat16"),
        max_new_tokens=int(cloud_cfg.get("max_new_tokens", 160)),
        role="cloud",
        prompt=cfg.get("prompt"),
        adapter_path=adapter,
    )

    summaries = []
    for cat in categories:
        try:
            summaries.append(run_one(cfg, cat, client, args.max_cloud_reviews))
        except Exception as exc:  # noqa: BLE001
            print(f"[{cat}] FAILED: {exc}")
            summaries.append({"category": cat, "error": str(exc)})

    # include already-finished bottle/screw if present and not in this run
    for p in sorted(root.glob("*/bench.json")):
        cat = p.parent.name
        if cat in {s.get("category") for s in summaries}:
            continue
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
            d = r["detection"]
            summaries.append(
                {
                    "category": cat,
                    "n": r["n"],
                    "n_reviews": r["n_cloud_reviews"],
                    "B1_auroc": d["B1_edge_only"]["image_auroc"],
                    "B1_f1": d["B1_edge_only"]["f1"],
                    "S_auroc": d["S_collab"]["image_auroc"],
                    "S_f1": d["S_collab"]["f1"],
                    "delta_f1": d["S_collab"]["f1"] - d["B1_edge_only"]["f1"],
                    "cloud_fixed": r.get("cloud_fixed_edge"),
                    "cloud_wrong": r.get("cloud_wrong_on_reviewed"),
                    "cloud_lat_ms": (r.get("latency_ms") or {}).get("cloud_mean"),
                    "from_cache": True,
                }
            )
        except Exception:  # noqa: BLE001
            pass

    ok = [s for s in summaries if "error" not in s and "B1_f1" in s]
    mean_b1 = float(np.mean([s["B1_f1"] for s in ok])) if ok else float("nan")
    mean_s = float(np.mean([s["S_f1"] for s in ok])) if ok else float("nan")

    md = [
        "# Hybrid mean — edge Anomalib + cloud Qwen-VL",
        "",
        f"Categories with results: {len(ok)}",
        f"- Mean B1 F1: **{mean_b1:.4f}**",
        f"- Mean S F1: **{mean_s:.4f}**",
        f"- Mean ΔF1 (S-B1): **{mean_s-mean_b1:+.4f}**",
        "",
        "| Category | n | reviews | B1 AUROC | B1 F1 | S AUROC | S F1 | ΔF1 | fixed | wrong |",
        "|----------|---|---------|----------|-------|---------|------|-----|-------|-------|",
    ]
    for s in sorted(ok, key=lambda x: x["category"]):
        md.append(
            f"| {s['category']} | {s.get('n','')} | {s.get('n_reviews','')} | "
            f"{s['B1_auroc']:.4f} | {s['B1_f1']:.4f} | {s['S_auroc']:.4f} | {s['S_f1']:.4f} | "
            f"{s['delta_f1']:+.4f} | {s.get('cloud_fixed','')} | {s.get('cloud_wrong','')} |"
        )
    out_md = root / "hybrid_mean.md"
    out_json = root / "hybrid_mean.json"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps({"summaries": summaries, "mean_B1_f1": mean_b1, "mean_S_f1": mean_s}, indent=2), encoding="utf-8")
    print("\n" + "\n".join(md))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
