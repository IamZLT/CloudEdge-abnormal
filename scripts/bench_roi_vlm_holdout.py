#!/usr/bin/env python3
"""Holdout bench: edge anomaly-map ROI crops → Qwen3.5 VLM (new script).

Does NOT modify existing train/eval LoRA scripts or overwrite their outputs.
Writes to results_dir from config (default: outputs/qwen35_roi_vlm/).

Env: conda activate clip
Example:
  CUDA_VISIBLE_DEVICES=3 python scripts/bench_roi_vlm_holdout.py \
    --config configs/qwen35_roi_vlm.yaml --max-images 60
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge.methods.gallery_ad import mvtec_train_good
from edge.methods.patch_gallery_ad import PatchGalleryAD
from edge.methods.roi_crops import amap_to_rois, draw_roi_boxes, make_roi_collage
from src.vlm import QwenVLClient


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "n": int(len(y_true)),
        "pred_ok": int(sum(1 for p in y_pred if p == 0)),
        "pred_ng": int(sum(1 for p in y_pred if p == 1)),
    }


def subsample_balanced(rows: list[dict], max_images: int | None) -> list[dict]:
    if max_images is None or max_images <= 0:
        return rows
    ok = [r for r in rows if int(r["label"]) == 0]
    ng = [r for r in rows if int(r["label"]) == 1]
    n_ok = min(len(ok), max_images // 2)
    n_ng = min(len(ng), max_images - n_ok)
    return ok[:n_ok] + ng[:n_ng]


def build_edge_ad(cfg: dict, device: str) -> PatchGalleryAD:
    from edge.methods.encoders import load_qwen35_vision_encoder

    edge = cfg.get("edge") or {}
    model_path = edge.get("model_path")
    _, encode_patches, _meta = load_qwen35_vision_encoder(
        model_path,
        device=device,
        max_pixels=int(edge.get("max_pixels") or 50176),
        layers=edge.get("layers"),
    )
    return PatchGalleryAD(
        encode_patches,
        device=device,
        name="qwen35_roi_bench",
        fusion_temperature=float(edge.get("fusion_temp") or 0.5),
    )


def phase_extract_rois(
    rows: list[dict],
    cfg: dict,
    device: str,
    out_dir: Path,
) -> list[dict]:
    """Build per-category galleries, score maps, save ROI metadata (+ optional previews)."""
    data_root = Path(cfg.get("data_root") or "datasets/mvtec_anomaly_detection")
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    edge = cfg.get("edge") or {}
    roi_cfg = cfg.get("roi") or {}
    seed = int(cfg.get("seed") or 42)
    max_gallery = int(edge.get("max_gallery") or 16)
    top_k = int(roi_cfg.get("top_k") or 2)
    pad_ratio = float(roi_cfg.get("pad_ratio") or 0.35)
    min_side = int(roi_cfg.get("min_side") or 96)
    save_prev = bool(roi_cfg.get("save_previews", True))
    preview_dir = out_dir / "previews"
    if save_prev:
        preview_dir.mkdir(parents=True, exist_ok=True)

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[str(r.get("category") or "unknown")].append(r)

    print(f"[roi] loading edge vision encoder on {device}")
    ad = build_edge_ad(cfg, device)
    caches: list[dict] = []

    for cat, cat_rows in by_cat.items():
        train = mvtec_train_good(data_root, cat)
        if not train:
            raise FileNotFoundError(f"no train/good for {cat} under {data_root}")
        if len(train) > max_gallery:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(train), size=max_gallery, replace=False)
            train = [train[i] for i in sorted(idx)]
        print(f"[roi] gallery {cat}: {len(train)} good → {len(cat_rows)} queries")
        ad.build_gallery(train, seed=seed)

        for i, row in enumerate(cat_rows):
            img_path = Path(row["image"])
            pil = Image.open(img_path).convert("RGB")
            score, amap = ad.score_image(pil)
            rois = amap_to_rois(
                pil,
                amap,
                top_k=top_k,
                pad_ratio=pad_ratio,
                min_side=min_side,
            )
            entry = {
                "image": str(img_path),
                "category": cat,
                "label": int(row["label"]),
                "edge_score": float(score),
                "map_hw": list(amap.shape),
                "rois": [
                    {
                        "box": list(r.box),
                        "score": float(r.score),
                        "patch_ij": list(r.patch_ij),
                    }
                    for r in rois
                ],
            }
            if save_prev and rois:
                stem = f"{cat}_{img_path.stem}_gt{'NG' if row['label'] else 'OK'}"
                draw_roi_boxes(pil, rois).save(preview_dir / f"{stem}_boxes.jpg", quality=90)
                collage = make_roi_collage(rois)
                if collage is not None:
                    collage.save(preview_dir / f"{stem}_collage.jpg", quality=90)
                rois[0].crop.save(preview_dir / f"{stem}_top1.jpg", quality=90)
            caches.append(entry)
            print(
                f"[roi {len(caches)}/{len(rows)}] {cat} "
                f"gt={'NG' if row['label'] else 'OK'} score={score:.3f} n_roi={len(rois)}"
            )

    # free vision tower before loading full VLM
    del ad
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cache_path = out_dir / "roi_cache.jsonl"
    with cache_path.open("w", encoding="utf-8") as f:
        for e in caches:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"[roi] wrote {cache_path}")
    return caches


def pick_image_for_mode(pil: Image.Image, rois_meta: list[dict], mode: str) -> Image.Image | None:
    from edge.methods.roi_crops import RoiCrop

    if mode == "full":
        return pil
    rois = [
        RoiCrop(
            box=tuple(r["box"]),
            score=float(r["score"]),
            patch_ij=tuple(r["patch_ij"]),
            crop=pil.crop(tuple(r["box"])),
        )
        for r in rois_meta
    ]
    if mode == "roi_top1":
        return rois[0].crop if rois else None
    if mode == "roi_collage":
        return make_roi_collage(rois) if rois else None
    raise ValueError(f"unknown mode: {mode}")


def run_vlm_mode(
    client: QwenVLClient,
    caches: list[dict],
    mode: str,
    prompt: str,
) -> dict:
    client.prompt = prompt
    y_true, y_pred = [], []
    details = []
    for i, e in enumerate(caches):
        pil = Image.open(e["image"]).convert("RGB")
        img = pick_image_for_mode(pil, e.get("rois") or [], mode)
        if img is None:
            # no ROI → conservative OK (contrast found nothing salient)
            decision = "OK"
            conf = 0.4
            defect_type = "none"
            reason = "no salient ROI from edge map"
            latency = 0.0
            raw = ""
        else:
            res = client.infer(img)
            decision = res.decision
            conf = res.confidence
            defect_type = res.defect_type
            reason = res.reason
            latency = res.latency_ms
            raw = res.raw
        pred = 1 if decision == "NG" else 0
        y_true.append(int(e["label"]))
        y_pred.append(pred)
        details.append(
            {
                "image": e["image"],
                "category": e["category"],
                "gt": int(e["label"]),
                "pred": pred,
                "decision": decision,
                "confidence": conf,
                "defect_type": defect_type,
                "reason": reason,
                "edge_score": e.get("edge_score"),
                "n_roi": len(e.get("rois") or []),
                "latency_ms": latency,
                "raw": raw,
                "correct": pred == int(e["label"]),
            }
        )
        print(
            f"[{mode} {i+1}/{len(caches)}] {e['category']} "
            f"gt={'NG' if e['label'] else 'OK'} pred={decision} "
            f"ok={pred == int(e['label'])} | {(reason or '')[:48]}"
        )
    return {"metrics": metrics(y_true, y_pred), "details": details}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/qwen35_roi_vlm.yaml"))
    parser.add_argument("--max-images", type=int, default=60)
    parser.add_argument("--device", default=None)
    parser.add_argument("--adapter", default=None, help="override vlm.adapter_path")
    parser.add_argument("--skip-extract", action="store_true", help="reuse roi_cache.jsonl")
    parser.add_argument(
        "--modes",
        default=None,
        help="comma list override, e.g. full,roi_top1,roi_collage",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(cfg.get("results_dir") or "outputs/qwen35_roi_vlm")
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    holdout = Path(cfg.get("holdout_jsonl") or "outputs/qwen35_lora/sft_holdout.jsonl")
    if not holdout.is_absolute():
        holdout = ROOT / holdout
    rows = subsample_balanced(load_jsonl(holdout), args.max_images)
    device = args.device or cfg.get("device") or "cuda:0"
    roi_cfg = cfg.get("roi") or {}
    modes = (
        [m.strip() for m in args.modes.split(",") if m.strip()]
        if args.modes
        else list(roi_cfg.get("modes") or ["full", "roi_top1", "roi_collage"])
    )

    cache_path = out_dir / "roi_cache.jsonl"
    if args.skip_extract and cache_path.exists():
        caches = load_jsonl(cache_path)
        print(f"[roi] reused cache n={len(caches)}")
    else:
        caches = phase_extract_rois(rows, cfg, device, out_dir)

    vlm_cfg = cfg.get("vlm") or {}
    adapter = args.adapter if args.adapter is not None else vlm_cfg.get("adapter_path")
    if adapter in {"", "null", "None"}:
        adapter = None
    prompt_full = cfg.get("prompt_full") or ""
    prompt_roi = cfg.get("prompt_roi") or prompt_full

    print(f"[vlm] loading {vlm_cfg.get('model_path')} adapter={adapter}")
    client = QwenVLClient(
        model_path=vlm_cfg["model_path"],
        device=device,
        dtype=vlm_cfg.get("dtype", "bfloat16"),
        max_new_tokens=int(vlm_cfg.get("max_new_tokens") or 128),
        role="roi_bench",
        model_family=vlm_cfg.get("model_family"),
        adapter_path=adapter,
    )

    report = {
        "holdout": str(holdout),
        "n_eval": len(caches),
        "device": device,
        "adapter": adapter,
        "modes": modes,
        "edge": cfg.get("edge"),
        "roi": roi_cfg,
        "results": {},
    }
    t0 = time.perf_counter()
    for mode in modes:
        prompt = prompt_full if mode == "full" else prompt_roi
        report["results"][mode] = run_vlm_mode(client, caches, mode, prompt)
    report["elapsed_s"] = time.perf_counter() - t0

    out_json = out_dir / "bench_roi_vlm.json"
    out_md = out_dir / "bench_roi_vlm.md"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# ROI-conditioned Qwen3.5-0.8B holdout bench",
        "",
        f"- Holdout: `{holdout}`",
        f"- N: {len(caches)}",
        f"- Adapter: `{adapter or 'none (zero-shot)'}`",
        f"- Edge: Qwen3.5 vision patch gallery → top-{roi_cfg.get('top_k', 2)} ROI",
        "",
        "| Mode | Acc | F1 | P | R | pred OK/NG |",
        "|------|-----|----|---|---|------------|",
    ]
    for mode in modes:
        m = report["results"][mode]["metrics"]
        lines.append(
            f"| {mode} | {m['accuracy']:.4f} | {m['f1']:.4f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['pred_ok']}/{m['pred_ng']} |"
        )
    if "full" in report["results"] and "roi_top1" in report["results"]:
        df = (
            report["results"]["roi_top1"]["metrics"]["f1"]
            - report["results"]["full"]["metrics"]["f1"]
        )
        lines += ["", f"- ΔF1 (roi_top1 − full): **{df:+.4f}**", ""]
    if "full" in report["results"] and "roi_collage" in report["results"]:
        df = (
            report["results"]["roi_collage"]["metrics"]["f1"]
            - report["results"]["full"]["metrics"]["f1"]
        )
        lines += [f"- ΔF1 (roi_collage − full): **{df:+.4f}**", ""]
    lines += [
        "## Notes",
        "",
        "- `full`: whole image (same protocol as prior 0.8B LoRA holdout).",
        "- `roi_top1` / `roi_collage`: crops from edge contrast anomaly map.",
        "- Previews (if enabled): `previews/*_boxes.jpg`, `*_top1.jpg`, `*_collage.jpg`.",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
