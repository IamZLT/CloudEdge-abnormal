#!/usr/bin/env python3
"""Build ROI short-label SFT JSONL from edge anomaly maps (new script).

For each source SFT image:
  - NG → top-1 anomalous crop → decision NG
  - OK → top-1 crop (hard OK) + optional low-score crop (easy OK)

Crops are written as JPEG files; JSONL points to those paths.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge.methods.gallery_ad import mvtec_train_good
from edge.methods.patch_gallery_ad import PatchGalleryAD
from edge.methods.roi_crops import amap_to_rois


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def short_response(decision: str, defect_type: str = "none") -> str:
    if decision == "OK":
        obj = {
            "decision": "OK",
            "confidence": 0.9,
            "defect_type": "none",
            "reason": "normal region",
        }
    else:
        nice = (defect_type or "defect").replace("_", " ")
        obj = {
            "decision": "NG",
            "confidence": 0.9,
            "defect_type": defect_type or "defect",
            "reason": f"visible {nice}",
        }
    return json.dumps(obj, ensure_ascii=False)


def low_score_roi(image: Image.Image, amap: np.ndarray, *, pad_ratio: float, min_side: int):
    """Crop around the lowest-scoring grid cell (easy OK)."""
    gh, gw = amap.shape
    i, j = np.unravel_index(int(np.argmin(amap)), amap.shape)
    # reuse amap_to_rois machinery by fabricating a peak map
    fake = np.zeros_like(amap)
    fake[i, j] = 1.0
    rois = amap_to_rois(image, fake, top_k=1, pad_ratio=pad_ratio, min_side=min_side)
    return rois[0] if rois else None


def build_edge_ad(cfg: dict, device: str) -> PatchGalleryAD:
    from edge.methods.encoders import load_qwen35_vision_encoder

    edge = cfg.get("edge") or {}
    _, encode_patches, _ = load_qwen35_vision_encoder(
        edge["model_path"],
        device=device,
        max_pixels=int(edge.get("max_pixels") or 50176),
        layers=edge.get("layers"),
    )
    return PatchGalleryAD(
        encode_patches,
        device=device,
        name="roi_short_sft",
        fusion_temperature=float(edge.get("fusion_temp") or 0.5),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/qwen35_roi_gated.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", choices=["train", "holdout", "both"], default="both")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_root = Path(cfg.get("results_dir") or "outputs/qwen35_roi_gated")
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    crop_root = out_root / "crops"
    crop_root.mkdir(parents=True, exist_ok=True)

    data_root = Path(cfg.get("data_root") or "datasets/mvtec_anomaly_detection")
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    edge = cfg.get("edge") or {}
    roi_cfg = cfg.get("roi") or {}
    prompt = cfg.get("prompt_roi") or ""
    device = args.device or cfg.get("device") or "cuda:0"
    seed = int(cfg.get("seed") or 42)
    max_gallery = int(edge.get("max_gallery") or 16)
    top_k = int(roi_cfg.get("top_k") or 1)
    pad_ratio = float(roi_cfg.get("pad_ratio") or 0.4)
    min_side = int(roi_cfg.get("min_side") or 96)
    add_ok_low = bool(roi_cfg.get("add_ok_low_crop", True))

    splits = []
    if args.split in {"train", "both"}:
        splits.append(("train", Path(cfg["source_train_jsonl"])))
    if args.split in {"holdout", "both"}:
        splits.append(("holdout", Path(cfg["source_holdout_jsonl"])))

    ad = build_edge_ad(cfg, device)
    meta = {"splits": {}}

    for split_name, src in splits:
        if not src.is_absolute():
            src = ROOT / src
        rows = load_jsonl(src)
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_cat[str(r.get("category") or "unknown")].append(r)

        out_rows = []
        for cat, cat_rows in by_cat.items():
            train = mvtec_train_good(data_root, cat)
            if not train:
                raise FileNotFoundError(f"no train/good for {cat}")
            if len(train) > max_gallery:
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(train), size=max_gallery, replace=False)
                train = [train[i] for i in sorted(idx)]
            print(f"[{split_name}] gallery {cat}: {len(train)} → {len(cat_rows)} images")
            ad.build_gallery(train, seed=seed)

            # calibrate thr on gallery for cache (eval reuse)
            g_scores = []
            for gp in train:
                s, _ = ad.score_image(Image.open(gp).convert("RGB"))
                g_scores.append(float(s))
            thr = float(np.quantile(g_scores, float(edge.get("thr_quantile") or 0.95)))

            for r in cat_rows:
                img_path = Path(r["image"])
                pil = Image.open(img_path).convert("RGB")
                score, amap = ad.score_image(pil)
                label = int(r["label"])
                decision = "NG" if label == 1 else "OK"
                defect = r.get("defect_type") or ("defect" if label == 1 else "none")
                rois = amap_to_rois(
                    pil, amap, top_k=top_k, pad_ratio=pad_ratio, min_side=min_side
                )
                stem = f"{cat}_{img_path.stem}_{decision}"
                if rois:
                    crop_path = crop_root / split_name / f"{stem}_top1.jpg"
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    rois[0].crop.save(crop_path, quality=92)
                    out_rows.append(
                        {
                            "image": str(crop_path.resolve()),
                            "source_image": str(img_path),
                            "category": cat,
                            "split": split_name,
                            "crop_kind": "top1",
                            "label": label,
                            "defect_type": defect,
                            "edge_score": float(score),
                            "edge_thr": thr,
                            "roi_box": list(rois[0].box),
                            "roi_score": float(rois[0].score),
                            "prompt": prompt,
                            "response": short_response(decision, defect),
                        }
                    )
                if label == 0 and add_ok_low:
                    low = low_score_roi(pil, amap, pad_ratio=pad_ratio, min_side=min_side)
                    if low is not None:
                        crop_path = crop_root / split_name / f"{stem}_low.jpg"
                        crop_path.parent.mkdir(parents=True, exist_ok=True)
                        low.crop.save(crop_path, quality=92)
                        out_rows.append(
                            {
                                "image": str(crop_path.resolve()),
                                "source_image": str(img_path),
                                "category": cat,
                                "split": split_name,
                                "crop_kind": "low",
                                "label": 0,
                                "defect_type": "none",
                                "edge_score": float(score),
                                "edge_thr": thr,
                                "roi_box": list(low.box),
                                "roi_score": float(low.score),
                                "prompt": prompt,
                                "response": short_response("OK"),
                            }
                        )
                print(
                    f"[{split_name}] {cat} {img_path.name} gt={decision} "
                    f"score={score:.3f} thr={thr:.3f} n_out={len(out_rows)}"
                )

        out_jsonl = out_root / f"roi_sft_{split_name}.jsonl"
        with out_jsonl.open("w", encoding="utf-8") as f:
            for row in out_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_ok = sum(1 for x in out_rows if x["label"] == 0)
        n_ng = sum(1 for x in out_rows if x["label"] == 1)
        meta["splits"][split_name] = {
            "path": str(out_jsonl),
            "n": len(out_rows),
            "ok": n_ok,
            "ng": n_ng,
        }
        print(f"[build] {split_name}: n={len(out_rows)} OK={n_ok} NG={n_ng} -> {out_jsonl}")

    (out_root / "roi_sft_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    del ad
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
