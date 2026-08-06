#!/usr/bin/env python3
"""Pixel-level metrics + per-category visualization across edge AD methods.

Methods (16-shot train/good gallery by default):
  clip / dinov3 / qwen35  — patch-token NN gallery → anomaly map
  padim_16shot            — Anomalib PaDiM from outputs/anomalib_16shot
  padim                   — full-good PaDiM from outputs/anomalib

Example:
  CUDA_VISIBLE_DEVICES=1 \\
    /home/zlt/miniconda3/envs/dinov3/bin/python scripts/bench_edge_pixel_viz.py \\
    --methods clip dinov3 --categories bottle screw --max-gallery 16

  # Qwen needs clip env (Qwen3VLVisionModel + safetensors):
  CUDA_VISIBLE_DEVICES=2 \\
    /home/zlt/miniconda3/envs/clip/bin/python scripts/bench_edge_pixel_viz.py \\
    --methods qwen35 --categories bottle --max-gallery 16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge.methods.gallery_ad import mvtec_test_split, mvtec_train_good
from edge.methods.pixel_metrics import mvtec_gt_mask, pick_viz_samples
from edge.methods.viz_compare import render_category_grid

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

METHOD_TITLES = {
    "clip_vitl14_mlpatch": "CLIP-ML",
    "dinov3_vitl16_mlpatch": "DINOv3-ML",
    "qwen35_0.8b_vision_mlpatch": "Qwen-ML",
    # legacy single-layer names (older runs)
    "clip_vitl14_patch": "CLIP",
    "dinov3_vitl16_patch": "DINOv3",
    "qwen35_0.8b_vision_patch": "Qwen-vis",
    "padim_resnet18_16shot": "PaDiM-16",
    "padim_resnet18": "PaDiM-full",
}

DEFAULT_LAYERS = {
    "clip": [12, 16, 20, 24],
    "dinov3": [12, 16, 20, 24],
    "qwen35": [6, 8, 10, 12],
}


def _write_reports(rows: list[dict], out_dir: Path, tag: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"edge_pixel_{tag}.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    methods = sorted({r["method"] for r in rows})
    cats = sorted({r["category"] for r in rows})
    lines = [
        f"# Edge methods — image + pixel metrics ({tag})",
        "",
        "Protocol: gallery = `train/good` (optionally k-shot); eval on `test/*` only.",
        "Pixel maps: patch-NN distance (CLIP/DINO/Qwen) or PaDiM anomaly_map.",
        "Pixel metrics computed at 256×256.",
        "",
        "## Mean over categories",
        "",
        "| Method | Image-AUROC | Pixel-AUROC | Pixel-F1 | Image-F1 | Latency ms | Peak MB |",
        "|--------|-------------|-------------|----------|----------|------------|---------|",
    ]

    def avg(sub, k):
        vals = [r[k] for r in sub if r.get(k) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    for m in methods:
        sub = [r for r in rows if r["method"] == m]
        lines.append(
            f"| {m} | {avg(sub,'image_auroc'):.4f} | {avg(sub,'pixel_auroc'):.4f} | "
            f"{avg(sub,'pixel_f1'):.4f} | {avg(sub,'f1'):.4f} | "
            f"{avg(sub,'infer_latency_ms_mean'):.1f} | {avg(sub,'peak_mem_mb'):.0f} |"
        )

    for metric, key in [("Image-AUROC", "image_auroc"), ("Pixel-AUROC", "pixel_auroc"), ("Pixel-F1", "pixel_f1")]:
        lines += ["", f"## Per-category {metric}", ""]
        header = "| Category | " + " | ".join(methods) + " |"
        sep = "|----------|" + "|".join(["------"] * len(methods)) + "|"
        lines += [header, sep]
        for c in cats:
            vals = []
            for m in methods:
                hit = next((r for r in rows if r["method"] == m and r["category"] == c), None)
                if hit and hit.get(key) is not None:
                    vals.append(f"{hit[key]:.4f}")
                else:
                    vals.append("—")
            lines.append(f"| {c} | " + " | ".join(vals) + " |")

    lines += ["", "## Visualization", "", f"Per-category grids: `{out_dir / 'viz' / tag}`", ""]
    md_path = out_dir / f"edge_pixel_{tag}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {json_path} and {md_path}")


def _merge_json_shards(out_dir: Path, tag: str) -> list[dict]:
    """Merge edge_pixel_{tag}__*.json shards + main file into one row list (latest method wins)."""
    by_key: dict[tuple[str, str], dict] = {}
    paths = sorted(out_dir.glob(f"edge_pixel_{tag}.json")) + sorted(out_dir.glob(f"edge_pixel_{tag}__*.json"))
    for p in paths:
        try:
            rows = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for r in rows:
            by_key[(r["method"], r["category"])] = r
    return list(by_key.values())


def run_patch_method(
    method: str,
    categories: list[str],
    data_root: Path,
    device: str,
    image_size: int,
    max_gallery: int | None,
    seed: int,
    viz_paths: dict[str, list[Path]],
    layers: list[int] | None = None,
    fusion_temperature: float = 0.5,
) -> tuple[list[dict], dict[str, dict[str, np.ndarray]]]:
    from edge.methods.encoders import load_clip_encoder, load_dinov3_encoder, load_qwen35_vision_encoder
    from edge.methods.patch_gallery_ad import PatchGalleryAD

    layer_ids = layers or DEFAULT_LAYERS.get(method)
    if method == "clip":
        _, encode_patches, meta = load_clip_encoder(
            DEFAULT_PATHS["clip"], device=device, image_size=image_size, layers=layer_ids
        )
        name = "clip_vitl14_mlpatch"
    elif method == "dinov3":
        _, encode_patches, meta = load_dinov3_encoder(
            DEFAULT_PATHS["dinov3"], device=device, image_size=image_size, layers=layer_ids
        )
        name = "dinov3_vitl16_mlpatch"
    elif method == "qwen35":
        _, encode_patches, meta = load_qwen35_vision_encoder(
            DEFAULT_PATHS["qwen35"],
            device=device,
            max_pixels=image_size * image_size,
            layers=layer_ids,
        )
        name = "qwen35_0.8b_vision_mlpatch"
    else:
        raise ValueError(method)

    ad = PatchGalleryAD(
        encode_patches,
        device=device,
        name=name,
        fusion_temperature=fusion_temperature,
    )
    rows = []
    maps_by_cat: dict[str, dict[str, np.ndarray]] = {}
    rng = np.random.default_rng(seed)
    for cat in categories:
        train = mvtec_train_good(data_root, cat)
        if max_gallery and len(train) > max_gallery:
            idx = rng.choice(len(train), size=max_gallery, replace=False)
            train = [train[i] for i in sorted(idx)]
        test = mvtec_test_split(data_root, cat)
        map_cache: dict[str, np.ndarray] = {}
        want = {str(p.resolve()) for p in viz_paths.get(cat, [])}
        print(f"[{name}] {cat}: gallery={len(train)} test={len(test)}")
        res = ad.evaluate(
            cat,
            train,
            test,
            flops_g=meta.get("flops_g"),
            params_m=meta.get("params_m"),
            notes=json.dumps(meta, ensure_ascii=False),
            extra={"split": f"train/good(k={max_gallery}) -> test/*", "encoder_meta": meta},
            map_cache=map_cache,
            cache_paths=want,
            seed=seed,
        )
        maps_by_cat[cat] = dict(map_cache)
        print(
            f"  imgAUROC={res.image_auroc:.4f} pixAUROC={res.pixel_auroc:.4f} "
            f"pixF1={res.pixel_f1:.4f} lat={res.infer_latency_ms_mean:.1f}ms"
        )
        rows.append(res.to_dict())
    return rows, {name: maps_by_cat}


def _save_maps(maps_store: dict[str, dict[str, dict[str, np.ndarray]]], maps_dir: Path) -> None:
    """Persist viz maps: maps_dir/{method}/{category}.npz with sidecar keys json."""
    maps_dir.mkdir(parents=True, exist_ok=True)
    for meth, by_cat in maps_store.items():
        mdir = maps_dir / meth
        mdir.mkdir(parents=True, exist_ok=True)
        for cat, path_maps in by_cat.items():
            payload = {}
            key_map = {}
            for i, (p, arr) in enumerate(path_maps.items()):
                k = f"m{i}"
                payload[k] = arr
                key_map[k] = p
            np.savez_compressed(mdir / f"{cat}.npz", **payload)
            (mdir / f"{cat}.keys.json").write_text(json.dumps(key_map, ensure_ascii=False), encoding="utf-8")


def _load_maps(maps_dir: Path) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    out: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    if not maps_dir.exists():
        return out
    for meth_dir in maps_dir.iterdir():
        if not meth_dir.is_dir():
            continue
        meth = meth_dir.name
        out[meth] = {}
        for npz in meth_dir.glob("*.npz"):
            cat = npz.stem
            key_path = meth_dir / f"{cat}.keys.json"
            if not key_path.exists():
                continue
            key_map = json.loads(key_path.read_text(encoding="utf-8"))
            data = np.load(npz)
            out[meth][cat] = {key_map[k]: data[k] for k in key_map}
    return out


def run_padim(
    categories: list[str],
    data_root: Path,
    anomalib_root: Path,
    device: str,
    method_name: str,
    viz_paths: dict[str, list[Path]],
) -> tuple[list[dict], dict[str, dict[str, np.ndarray]]]:
    from edge.methods.padim_ad import eval_padim_category

    rows = []
    maps_by_cat: dict[str, dict[str, np.ndarray]] = {}
    for cat in categories:
        map_cache: dict[str, np.ndarray] = {}
        print(f"[{method_name}] {cat}")
        res = eval_padim_category(
            cat,
            data_root,
            anomalib_root,
            device=device,
            method_name=method_name,
            map_cache=map_cache,
        )
        want = {str(p.resolve()) for p in viz_paths.get(cat, [])}
        slim = {}
        for vp in viz_paths.get(cat, []):
            for k, v in map_cache.items():
                if Path(k).name == vp.name and Path(k).parent.name == vp.parent.name:
                    slim[str(vp.resolve())] = v
                    break
            else:
                # try resolve match
                for k, v in map_cache.items():
                    if str(Path(k).resolve()) in want or str(vp.resolve()) == str(Path(k).resolve()):
                        slim[str(vp.resolve())] = v
                        break
        maps_by_cat[cat] = slim
        print(
            f"  imgAUROC={res.image_auroc:.4f} pixAUROC={res.pixel_auroc} "
            f"pixF1={res.pixel_f1} lat~{res.infer_latency_ms_mean:.1f}ms"
        )
        rows.append(res.to_dict())
    return rows, {method_name: maps_by_cat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["clip", "dinov3"])
    ap.add_argument("--categories", nargs="+", default=["bottle", "screw"])
    ap.add_argument("--data-root", default=str(ROOT / "datasets" / "mvtec"))
    ap.add_argument("--anomalib-root", default=str(ROOT / "outputs" / "anomalib"))
    ap.add_argument("--anomalib-16shot-root", default=str(ROOT / "outputs" / "anomalib_16shot"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--max-gallery", type=int, default=16)
    ap.add_argument("--n-viz", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "reports" / "edge_methods"))
    ap.add_argument("--tag", default="pixel16")
    ap.add_argument("--skip-viz", action="store_true")
    ap.add_argument("--viz-only", action="store_true", help="only rebuild grids from saved maps+json")
    ap.add_argument(
        "--shard",
        default=None,
        help="write metrics to edge_pixel_{tag}__{shard}.json to avoid parallel overwrite",
    )
    ap.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="ViT/Qwen block indices (1-based). Default: CLIP/DINO 12 16 20 24; Qwen 6 8 10 12",
    )
    ap.add_argument("--fusion-temp", type=float, default=0.5, help="softmax temperature over layers")
    args = ap.parse_args()

    methods = args.methods
    if methods == ["all"]:
        methods = ["clip", "dinov3", "qwen35", "padim_16shot", "padim"]
    cats = CATS if args.categories == ["all"] else args.categories
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    viz_dir = out_dir / "viz" / args.tag
    maps_dir = out_dir / "maps" / args.tag
    shard_tag = f"{args.tag}__{args.shard}" if args.shard else args.tag

    viz_paths: dict[str, list[Path]] = {}
    gt_by_path: dict[str, np.ndarray] = {}
    for cat in cats:
        picks = pick_viz_samples(data_root, cat, n=args.n_viz, seed=args.seed)
        viz_paths[cat] = picks
        for p in picks:
            gt_by_path[str(p.resolve())] = mvtec_gt_mask(p, target_hw=(256, 256))
        print(f"[viz picks] {cat}: {[f'{p.parent.name}/{p.name}' for p in picks]}")

    all_rows: list[dict] = []
    maps_store: dict[str, dict[str, dict[str, np.ndarray]]] = _load_maps(maps_dir)

    if not args.viz_only:
        for m in methods:
            if m == "padim":
                rows, store = run_padim(
                    cats, data_root, Path(args.anomalib_root), args.device, "padim_resnet18", viz_paths
                )
            elif m == "padim_16shot":
                rows, store = run_padim(
                    cats,
                    data_root,
                    Path(args.anomalib_16shot_root),
                    args.device,
                    "padim_resnet18_16shot",
                    viz_paths,
                )
            else:
                rows, store = run_patch_method(
                    m,
                    cats,
                    data_root,
                    args.device,
                    args.image_size,
                    args.max_gallery,
                    args.seed,
                    viz_paths,
                    layers=args.layers,
                    fusion_temperature=args.fusion_temp,
                )
            all_rows.extend(rows)
            for meth, by_cat in store.items():
                maps_store.setdefault(meth, {}).update(by_cat)
            _save_maps({meth: maps_store[meth] for meth in store}, maps_dir)
            _write_reports(all_rows, out_dir, shard_tag)

        if args.shard:
            # parallel-safe: keep shard only; caller merges later
            pass
        else:
            merged = _merge_json_shards(out_dir, args.tag)
            by_key = {(r["method"], r["category"]): r for r in merged}
            for r in all_rows:
                by_key[(r["method"], r["category"])] = r
            all_rows = list(by_key.values())
            _write_reports(all_rows, out_dir, args.tag)
    else:
        all_rows = _merge_json_shards(out_dir, args.tag)

    if not args.skip_viz:
        preferred = [
            "clip_vitl14_mlpatch",
            "dinov3_vitl16_mlpatch",
            "qwen35_0.8b_vision_mlpatch",
            "clip_vitl14_patch",
            "dinov3_vitl16_patch",
            "qwen35_0.8b_vision_patch",
            "padim_resnet18_16shot",
            "padim_resnet18",
        ]
        present = set(maps_store.keys())
        method_order = [m for m in preferred if m in present]
        for cat in cats:
            maps_by_method = {meth: maps_store.get(meth, {}).get(cat, {}) for meth in method_order}
            out_p = viz_dir / f"{cat}_compare.png"
            render_category_grid(
                cat,
                viz_paths[cat],
                gt_by_path,
                maps_by_method,
                method_order,
                out_p,
                titles=METHOD_TITLES,
            )
            print(f"Wrote {out_p}")

    if all_rows and not args.shard:
        _write_reports(all_rows, out_dir, args.tag)
    print("DONE")


if __name__ == "__main__":
    main()
