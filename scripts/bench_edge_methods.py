#!/usr/bin/env python3
"""Benchmark 4 edge AD methods with strict MVTec train/test split.

Methods:
  1. clip      — CLIP ViT-L/14 feature gallery (train/good only)
  2. dinov3    — DINOv3 ViT-L/16 feature gallery
  3. qwen35    — Qwen3.5-0.8B vision-encoder feature gallery
  4. padim     — current Anomalib PaDiM/resnet18 (existing edge ckpt)

Env tips:
  - CLIP / Qwen3.5: conda activate base
  - DINOv3 / PaDiM: conda activate dinov3  (or any env with transformers+anomalib)

Example:
  CUDA_VISIBLE_DEVICES=1 python scripts/bench_edge_methods.py --methods clip --categories bottle screw
  CUDA_VISIBLE_DEVICES=1 python scripts/bench_edge_methods.py --methods all --categories all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge.methods.gallery_ad import FeatureGalleryAD, mvtec_test_split, mvtec_train_good

CATS = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

DEFAULT_PATHS = {
    "clip": "/data2/zlt/anomaly_detection_llm/model_card/clip-vit-large-patch14",
    "dinov3": "/data2/zlt/anomaly_detection_llm/model_card/dinov3-vitl16-pretrain-lvd1689m",
    "qwen35": "/data2/zlt/anomaly_detection_llm/model_card/Qwen3.5-0.8B",
}


def _write_reports(rows: list[dict], out_dir: Path, tag: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"edge_methods_{tag}.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    # pivot by method
    methods = sorted({r["method"] for r in rows})
    cats = sorted({r["category"] for r in rows})
    lines = [
        f"# Edge methods bench ({tag})",
        "",
        "Split protocol: gallery/train bank = `train/good` only; metrics on `test/*` only.",
        "",
        "## Mean over categories",
        "",
        "| Method | Image-AUROC | Pixel-AUROC | Pixel-F1 | F1 | Prec | Rec | Latency ms | GFLOPs | Params M | Peak MB |",
        "|--------|-------------|-------------|----------|----|------|-----|------------|--------|----------|---------|",
    ]
    for m in methods:
        sub = [r for r in rows if r["method"] == m]
        def avg(k):
            vals = [r[k] for r in sub if r.get(k) is not None]
            return float(np.mean(vals)) if vals else float("nan")

        lines.append(
            f"| {m} | {avg('image_auroc'):.4f} | {avg('pixel_auroc'):.4f} | {avg('pixel_f1'):.4f} | "
            f"{avg('f1'):.4f} | {avg('precision'):.4f} | "
            f"{avg('recall'):.4f} | {avg('infer_latency_ms_mean'):.2f} | "
            f"{avg('flops_g'):.3f} | {avg('params_m'):.1f} | {avg('peak_mem_mb'):.0f} |"
        )

    lines += ["", "## Per-category Image-AUROC", ""]
    header = "| Category | " + " | ".join(methods) + " |"
    sep = "|----------|" + "|".join(["------"] * len(methods)) + "|"
    lines += [header, sep]
    for c in cats:
        vals = []
        for m in methods:
            hit = next((r for r in rows if r["method"] == m and r["category"] == c), None)
            vals.append(f"{hit['image_auroc']:.4f}" if hit else "—")
        lines.append(f"| {c} | " + " | ".join(vals) + " |")

    lines += ["", "## Per-category Latency (ms/img)", ""]
    lines += [header.replace("Image-AUROC", "Latency"), sep]
    for c in cats:
        vals = []
        for m in methods:
            hit = next((r for r in rows if r["method"] == m and r["category"] == c), None)
            vals.append(f"{hit['infer_latency_ms_mean']:.2f}" if hit else "—")
        lines.append(f"| {c} | " + " | ".join(vals) + " |")

    lines += ["", "## Notes", ""]
    for m in methods:
        note = next((r.get("notes", "") for r in rows if r["method"] == m), "")
        if note:
            lines.append(f"- **{m}**: {note}")

    md_path = out_dir / f"edge_methods_{tag}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {json_path} and {md_path}")


def run_feature_method(
    method: str,
    categories: list[str],
    data_root: Path,
    device: str,
    image_size: int,
    max_gallery: int | None,
    seed: int,
) -> list[dict]:
    from edge.methods.encoders import load_clip_encoder, load_dinov3_encoder, load_qwen35_vision_encoder

    if method == "clip":
        encode, _encode_patches, meta = load_clip_encoder(
            DEFAULT_PATHS["clip"], device=device, image_size=image_size
        )
        name = "clip_vitl14_gallery"
    elif method == "dinov3":
        encode, _encode_patches, meta = load_dinov3_encoder(
            DEFAULT_PATHS["dinov3"], device=device, image_size=image_size
        )
        name = "dinov3_vitl16_gallery"
    elif method == "qwen35":
        encode, _encode_patches, meta = load_qwen35_vision_encoder(
            DEFAULT_PATHS["qwen35"], device=device, max_pixels=image_size * image_size
        )
        name = "qwen35_0.8b_vision_gallery"
    else:
        raise ValueError(method)

    ad = FeatureGalleryAD(encode, device=device, name=name)
    rows = []
    rng = np.random.default_rng(seed)
    for cat in categories:
        train = mvtec_train_good(data_root, cat)
        if max_gallery and len(train) > max_gallery:
            idx = rng.choice(len(train), size=max_gallery, replace=False)
            train = [train[i] for i in sorted(idx)]
        test = mvtec_test_split(data_root, cat)
        # assert no leakage
        train_set = {str(p.resolve()) for p in train}
        for p, _ in test:
            if str(p.resolve()) in train_set:
                raise RuntimeError(f"train/test leakage: {p}")
        print(f"[{name}] {cat}: gallery={len(train)} test={len(test)}")
        res = ad.evaluate(
            cat,
            train,
            test,
            flops_g=meta.get("flops_g"),
            params_m=meta.get("params_m"),
            notes=json.dumps(meta, ensure_ascii=False),
            extra={"split": "train/good -> test/*", "encoder_meta": meta},
        )
        print(
            f"  AUROC={res.image_auroc:.4f} F1={res.f1:.4f} "
            f"lat={res.infer_latency_ms_mean:.2f}ms FLOPs={res.flops_g}G"
        )
        rows.append(res.to_dict())
    return rows


def run_padim(categories: list[str], data_root: Path, anomalib_root: Path, device: str) -> list[dict]:
    from edge.methods.padim_ad import eval_padim_category

    rows = []
    for cat in categories:
        print(f"[padim] {cat}")
        res = eval_padim_category(cat, data_root, anomalib_root, device=device)
        print(f"  AUROC={res.image_auroc:.4f} F1={res.f1:.4f} lat~{res.infer_latency_ms_mean:.2f}ms")
        rows.append(res.to_dict())
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["all"], help="clip dinov3 qwen35 padim all")
    ap.add_argument("--categories", nargs="+", default=["bottle", "screw"])
    ap.add_argument("--data-root", default=str(ROOT / "datasets" / "mvtec"))
    ap.add_argument("--anomalib-root", default=str(ROOT / "outputs" / "anomalib"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--max-gallery", type=int, default=None, help="subsample train/good (still train-only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "reports" / "edge_methods"))
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    methods = args.methods
    if methods == ["all"]:
        methods = ["clip", "dinov3", "qwen35", "padim"]
    cats = CATS if args.categories == ["all"] else args.categories
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)

    all_rows: list[dict] = []
    for m in methods:
        if m == "padim":
            all_rows.extend(run_padim(cats, data_root, Path(args.anomalib_root), args.device))
        else:
            all_rows.extend(
                run_feature_method(
                    m,
                    cats,
                    data_root,
                    args.device,
                    args.image_size,
                    args.max_gallery,
                    args.seed,
                )
            )
        # incremental save
        _write_reports(all_rows, out_dir, args.tag)

    _write_reports(all_rows, out_dir, args.tag)
    print("DONE")


if __name__ == "__main__":
    main()
