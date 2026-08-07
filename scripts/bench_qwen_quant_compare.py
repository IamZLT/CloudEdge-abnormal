#!/usr/bin/env python3
"""Compare HF Qwen3.5-0.8B vision AD vs mmproj-GGUF (quant package) weights.

Reports image/pixel metrics, latency, peak GPU memory, FLOPs, and disk size.
Edge AD only uses the vision tower; the Q4 LLM GGUF is counted in package disk
but is not loaded for this benchmark.

Example:
  conda activate clip
  CUDA_VISIBLE_DEVICES=0 python scripts/bench_qwen_quant_compare.py \\
    --categories bottle screw --max-gallery 16 --tag quant_cmp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util  # noqa: E402

from edge.methods.pixel_metrics import mvtec_gt_mask, pick_viz_samples  # noqa: E402


def _load_pixel_viz():
    path = ROOT / "scripts" / "bench_edge_pixel_viz.py"
    spec = importlib.util.spec_from_file_location("bench_edge_pixel_viz", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_pixel_viz = _load_pixel_viz()
METHOD_TITLES = _pixel_viz.METHOD_TITLES
_write_reports = _pixel_viz._write_reports
run_patch_method = _pixel_viz.run_patch_method
CATS = _pixel_viz.CATS


def _avg(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def _delta_table(hf_rows: list[dict], q_rows: list[dict]) -> list[str]:
    keys = [
        ("image_auroc", "Image-AUROC", 4),
        ("pixel_auroc", "Pixel-AUROC", 4),
        ("pixel_f1", "Pixel-F1", 4),
        ("f1", "Image-F1", 4),
        ("infer_latency_ms_mean", "Latency ms", 1),
        ("peak_mem_mb", "Peak MB", 0),
        ("flops_g", "FLOPs G", 3),
        ("params_m", "Params M", 2),
    ]
    lines = [
        "| Metric | HF (uncompressed) | mmproj GGUF | Δ (Q−HF) |",
        "|--------|-------------------|-------------|----------|",
    ]
    for k, title, nd in keys:
        a, b = _avg(hf_rows, k), _avg(q_rows, k)
        if np.isnan(a) and np.isnan(b):
            lines.append(f"| {title} | — | — | — |")
            continue
        d = b - a if (not np.isnan(a) and not np.isnan(b)) else float("nan")
        fmt = f"{{:.{nd}f}}"
        lines.append(
            f"| {title} | {fmt.format(a) if not np.isnan(a) else '—'} | "
            f"{fmt.format(b) if not np.isnan(b) else '—'} | "
            f"{fmt.format(d) if not np.isnan(d) else '—'} |"
        )
    return lines


def _disk_note(hf_rows: list[dict], q_rows: list[dict]) -> list[str]:
    def meta0(rows: list[dict]) -> dict:
        if not rows:
            return {}
        extra = rows[0].get("encoder_meta") or {}
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except Exception:
                extra = {}
        notes = rows[0].get("notes")
        if isinstance(notes, str):
            try:
                notes = json.loads(notes)
            except Exception:
                notes = {}
        if isinstance(notes, dict) and notes:
            return notes
        return extra if isinstance(extra, dict) else {}

    mh, mq = meta0(hf_rows), meta0(q_rows)
    lines = [
        "",
        "## Disk / weight source",
        "",
        "| Variant | Weight source | Vision disk MB | Full package MB |",
        "|---------|---------------|----------------|-----------------|",
        f"| HF | `{mh.get('weight_source', 'hf')}` | "
        f"{mh.get('disk_mb', float('nan')):.1f} | {mh.get('package_disk_mb', float('nan')):.1f} |",
        f"| Quant (mmproj) | `{mq.get('weight_source', 'gguf')}` | "
        f"{mq.get('disk_mb', float('nan')):.1f} | {mq.get('package_disk_mb', float('nan')):.1f} |",
        "",
        "- Edge AD loads **vision only**. HF package includes LLM safetensors; "
        "quant package = mmproj-F16 + Q4_K_M LLM (LLM not used here).",
        "- mmproj is F16 and maps near-bit-exact to HF `model.visual.*`; "
        "metric gaps should be ~0 if load succeeds.",
        "",
    ]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="HF vs mmproj-GGUF Qwen vision AD compare")
    ap.add_argument("--categories", nargs="+", default=["bottle", "screw"])
    ap.add_argument("--data-root", default=str(ROOT / "datasets" / "mvtec"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--max-gallery", type=int, default=16)
    ap.add_argument("--n-viz", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fusion-temp", type=float, default=0.5)
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "reports" / "qwen_quant"))
    ap.add_argument("--tag", default="quant_cmp")
    ap.add_argument("--skip-viz", action="store_true")
    args = ap.parse_args()

    cats = args.categories
    if cats == ["all"]:
        cats = list(CATS)

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_dir / "viz" / args.tag

    viz_paths: dict[str, list[Path]] = {}
    gt_by_path: dict[str, np.ndarray] = {}
    for cat in cats:
        picks = pick_viz_samples(data_root, cat, n=args.n_viz, seed=args.seed)
        viz_paths[cat] = picks
        for p in picks:
            gt_by_path[str(p.resolve())] = mvtec_gt_mask(p, target_hw=(256, 256))

    all_rows: list[dict] = []
    maps_store: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for method in ("qwen35", "qwen35_q"):
        rows, store = run_patch_method(
            method,
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
        maps_store.update(store)

    _write_reports(all_rows, out_dir, args.tag)

    hf_rows = [r for r in all_rows if "mmproj" not in r["method"]]
    q_rows = [r for r in all_rows if "mmproj" in r["method"]]
    md = [
        f"# Qwen3.5-0.8B HF vs mmproj-GGUF ({args.tag})",
        "",
        f"Categories: {', '.join(cats)} | gallery k={args.max_gallery} | "
        f"layers={args.layers or [6, 8, 10, 12]} | fusion_temp={args.fusion_temp}",
        "",
        "## Mean metrics",
        "",
        *_delta_table(hf_rows, q_rows),
        *_disk_note(hf_rows, q_rows),
        "## Per-category rows",
        "",
        "See also: "
        f"`edge_pixel_{args.tag}.json` / `edge_pixel_{args.tag}.md`.",
        "",
    ]
    cmp_path = out_dir / f"quant_compare_{args.tag}.md"
    cmp_path.write_text("\n".join(md), encoding="utf-8")
    (out_dir / f"quant_compare_{args.tag}.json").write_text(
        json.dumps(
            {
                "tag": args.tag,
                "categories": cats,
                "hf": hf_rows,
                "quant": q_rows,
                "delta_mean": {
                    k: _avg(q_rows, k) - _avg(hf_rows, k)
                    for k in (
                        "image_auroc",
                        "pixel_auroc",
                        "pixel_f1",
                        "infer_latency_ms_mean",
                        "peak_mem_mb",
                        "flops_g",
                    )
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {cmp_path}")

    if not args.skip_viz:
        from edge.methods.viz_compare import render_category_grid

        method_order = [
            "qwen35_0.8b_vision_mlpatch",
            "qwen35_0.8b_mmproj_mlpatch",
        ]
        for cat in cats:
            maps_by_method = {m: maps_store.get(m, {}).get(cat, {}) for m in method_order}
            out_p = viz_dir / f"{cat}_hf_vs_mmproj.png"
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

    print("DONE")


if __name__ == "__main__":
    main()
