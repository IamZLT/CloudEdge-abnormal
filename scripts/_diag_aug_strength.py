#!/usr/bin/env python3
"""Diagnostic: measure per-augmentation score shift for threshold-crossing conflict.

Loads the edge Qwen3.5-0.8B patch gallery once, calibrates a LOO threshold for a
category, then prints how each augmentation moves a handful of near-threshold
samples across the threshold. Goal: pick augmentations that actually produce
cross-threshold disagreement (real conflicts), instead of brightness 0.9/1.1
which barely moves the score.
"""
from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge.infer import DEFAULT_PATHS, _gallery_paths  # noqa: E402
from edge.methods.encoders import load_qwen35_vision_encoder  # noqa: E402
from edge.methods.gallery_ad import mvtec_test_split  # noqa: E402
from edge.methods.patch_gallery_ad import PatchGalleryAD  # noqa: E402
from src.collab_conflict import AugSpec, DEFAULT_AUGS, resolve_augs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / "edge_qwen35.yaml").read_text()) or {}
    edge_cfg = dict(cfg.get("edge") or {})
    data_root = ROOT / (cfg.get("data_root") or "datasets/mvtec")
    image_size = int(cfg.get("image_size") or 224)
    max_gallery = int(edge_cfg.get("max_gallery") or 16)
    layers = edge_cfg.get("layers") or [6, 8, 10, 12]
    model_path = edge_cfg.get("model_path") or DEFAULT_PATHS["qwen35"]

    _, encode_patches, _ = load_qwen35_vision_encoder(
        model_path,
        device=args.device,
        max_pixels=int(edge_cfg.get("max_pixels") or image_size * image_size),
        layers=layers,
    )
    ad = PatchGalleryAD(encode_patches, device=args.device, name="diag")
    gallery = _gallery_paths(data_root, args.category, max_gallery, args.seed)
    thr = ad.calibrate_threshold_loo(gallery, seed=args.seed, quantile=0.95)

    items = mvtec_test_split(data_root, args.category)
    # sample across the whole label/score spectrum: some near thr, some far
    rng = np.random.default_rng(args.seed + zlib.crc32(args.category.encode()) % 1000)
    if len(items) > args.limit:
        idx = sorted(rng.choice(len(items), args.limit, replace=False))
        items = [items[i] for i in idx]

    # extended augmentation bank (strength sweep)
    bank: list[AugSpec] = DEFAULT_AUGS + [
        AugSpec("brightness-0.6", "brightness", 0.6),
        AugSpec("brightness-0.7", "brightness", 0.7),
        AugSpec("brightness-0.8", "brightness", 0.8),
        AugSpec("brightness-1.2", "brightness", 1.2),
        AugSpec("brightness-1.3", "brightness", 1.3),
        AugSpec("brightness-1.5", "brightness", 1.5),
        AugSpec("contrast-0.6", "contrast", 0.6),
        AugSpec("contrast-1.5", "contrast", 1.5),
        AugSpec("blur-1.0", "blur", 1.0),
        AugSpec("blur-1.5", "blur", 1.5),
        AugSpec("noise-6", "noise", 6.0, seed=1),
        AugSpec("noise-12", "noise", 12.0, seed=1),
        AugSpec("noise-18", "noise", 18.0, seed=1),
    ]

    print(f"category={args.category} thr={thr:.4f} n={len(items)}")
    print(f"{'sample':>28} {'gt':>3} | " + " ".join(f"{a.name:>16}" for a in bank))
    for path, y in items:
        img = Image.open(path).convert("RGB")
        row = []
        for a in bank:
            s = float(ad.score_image(a.apply(img))[0])
            row.append(s)
        cells = " ".join(f"{s:16.4f}" for s in row)
        name = f"{path.parent.name}/{path.name}"
        print(f"{name:>28} {y:>3} | {cells}")

    # conflict summary per triple (identity + 2 variants) over the sampled set
    print("\n=== conflict ratio for candidate 3-node augment sets ===")
    candidates = {
        "weak (current)": ["identity", "brightness-0.9", "brightness-1.1"],
        "b-0.7/1.3": ["identity", "brightness-0.7", "brightness-1.3"],
        "b-0.8/1.2": ["identity", "brightness-0.8", "brightness-1.2"],
        "brightness-strong": ["identity", "brightness-0.6", "brightness-1.5"],
        "contrast-strong": ["identity", "contrast-0.6", "contrast-1.5"],
        "mixed": ["identity", "contrast-0.6", "noise-12"],
    }
    by_name = {a.name: a for a in bank}
    for label, names in candidates.items():
        augs = [by_name[n] for n in names]
        n_conf = 0
        for path, y in items:
            img = Image.open(path).convert("RGB")
            decs = []
            for a in augs:
                s = float(ad.score_image(a.apply(img))[0])
                decs.append("NG" if s >= thr else "OK")
            if ("NG" in decs) and ("OK" in decs):
                n_conf += 1
        print(f"  {label:20s} conflict={n_conf}/{len(items)} ({n_conf/len(items):.0%})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
