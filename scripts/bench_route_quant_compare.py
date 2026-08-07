#!/usr/bin/env python3
"""Compare RouteAgent: HF bf16 full VL vs GGUF Q4 LLM + mmproj-F16.

Measures latency, parse success, GPU/RSS memory, and disk size.
Edge AD vision metrics are separate (scripts/bench_qwen_quant_compare.py).

Example:
  conda activate clip
  CUDA_VISIBLE_DEVICES=0 python scripts/bench_route_quant_compare.py \\
    --category bottle --n-samples 8 --tag route_q4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vlm.route_agent import (  # noqa: E402
    DEFAULT_GGUF_DIR,
    DEFAULT_MODEL,
    RouteAgent,
    RouteContext,
)


def _load_samples(data_root: Path, category: str, n: int, seed: int) -> list[Path]:
    """Pick mixed good/defect test images without importing sklearn-heavy edge.methods."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    test = data_root / category / "test"
    goods, defs = [], []
    for sub in sorted(test.iterdir()) if test.exists() else []:
        if not sub.is_dir():
            continue
        imgs = sorted(p for p in sub.iterdir() if p.suffix.lower() in exts)
        (goods if sub.name == "good" else defs).extend(imgs)
    rng = np.random.default_rng(seed)
    picks: list[Path] = []
    if goods:
        picks.append(goods[int(rng.integers(0, len(goods)))])
    for p in defs:
        if len(picks) >= n:
            break
        if p not in picks:
            picks.append(p)
    while len(picks) < n and goods:
        p = goods[int(rng.integers(0, len(goods)))]
        if p not in picks:
            picks.append(p)
        else:
            break
    return picks[:n]


def _run_backend(
    backend: str,
    samples: list[Path],
    *,
    category: str,
    device: str,
    model_path: str,
    gguf_dir: str,
    max_new_tokens: int,
    n_ctx: int,
    n_gpu_layers: int,
) -> dict:
    cfg = {
        "backend": backend,
        "device": device,
        "max_new_tokens": max_new_tokens,
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "verbose": False,
        "hard_block_on_outage": True,
    }
    if backend == "hf":
        cfg["model_path"] = model_path
    else:
        cfg["gguf_dir"] = gguf_dir
        cfg["model_path"] = gguf_dir

    t_load0 = time.perf_counter()
    agent = RouteAgent.from_config(cfg)
    load_s = time.perf_counter() - t_load0
    agent.mark_load_memory()

    rows = []
    for i, path in enumerate(samples):
        # synthetic edge context near threshold → forces real LLM path
        ctx = RouteContext(
            image=path,
            category=category,
            n_gallery=16 if i % 3 else 0,
            edge_score=0.52,
            edge_thr=0.50,
            edge_decision="NG",
            network_profile="fair",
            network={"profile": "fair", "rtt_ms": 40},
            hard_margin=0.05,
        )
        d = agent.decide(ctx)
        rows.append(
            {
                "image": str(path),
                "upload": d.upload,
                "confidence": d.confidence,
                "parse_ok": d.parse_ok,
                "source": d.source,
                "latency_ms": d.latency_ms,
                "peak_mem_mb": d.peak_mem_mb,
                "reason": d.reason,
                "raw": (d.raw or "")[:240],
            }
        )
        print(
            f"[{backend}] {path.parent.name}/{path.name}: "
            f"upload={d.upload} parse={d.parse_ok} lat={d.latency_ms:.0f}ms "
            f"peak={d.peak_mem_mb}"
        )

    lats = [r["latency_ms"] for r in rows]
    peaks = [r["peak_mem_mb"] for r in rows if r["peak_mem_mb"] is not None]
    return {
        "backend": backend,
        "load_s": load_s,
        "meta": dict(agent.meta),
        "n_samples": len(rows),
        "parse_ok_rate": float(np.mean([1.0 if r["parse_ok"] else 0.0 for r in rows])),
        "upload_rate": float(np.mean([1.0 if r["upload"] else 0.0 for r in rows])),
        "latency_ms_mean": float(np.mean(lats)) if lats else float("nan"),
        "latency_ms_std": float(np.std(lats)) if lats else float("nan"),
        "peak_mem_mb_mean": float(np.mean(peaks)) if peaks else None,
        "peak_mem_mb_max": float(np.max(peaks)) if peaks else None,
        "rows": rows,
    }


def _md(hf: dict, gguf: dict, tag: str) -> str:
    def fmt(v, nd=2):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.{nd}f}"
        return str(v)

    mh, mq = hf.get("meta") or {}, gguf.get("meta") or {}
    # Prefer GPU peak; fall back to RSS for CPU-only GGUF builds
    if gguf.get("peak_mem_mb_mean") is None and mq.get("process_rss_mb") is not None:
        gguf = dict(gguf)
        gguf["peak_mem_mb_mean"] = mq["process_rss_mb"]
        gguf["peak_mem_mb_max"] = mq["process_rss_mb"]

    pairs = [
        ("load_s", "Load s", 2),
        ("latency_ms_mean", "Latency ms", 1),
        ("peak_mem_mb_mean", "Peak mem MB*", 1),
        ("parse_ok_rate", "Parse OK rate", 3),
        ("upload_rate", "Upload rate", 3),
    ]
    lines = [
        f"# RouteAgent HF vs GGUF-Q4 ({tag})",
        "",
        "Full multimodal generate (image + CONTEXT JSON). "
        "GGUF = **Q4_K_M LLM decoder** + **mmproj-F16** vision.",
        "",
        "## Summary",
        "",
        "| Metric | HF bf16 | GGUF Q4+mmproj | Δ (Q−HF) |",
        "|--------|---------|----------------|----------|",
    ]
    for k, title, nd in pairs:
        a, b = hf.get(k), gguf.get(k)
        if a is None or b is None:
            lines.append(f"| {title} | {fmt(a, nd)} | {fmt(b, nd)} | — |")
            continue
        dlt = float(b) - float(a)
        lines.append(f"| {title} | {float(a):.{nd}f} | {float(b):.{nd}f} | {dlt:.{nd}f} |")

    lines += [
        "",
        "\\* Peak mem: HF = `torch` max_memory_allocated; "
        "GGUF CUDA = nvidia-smi **incremental** footprint at load; "
        "CPU-only llama.cpp falls back to process RSS.",
        "",
        "## Disk / weights",
        "",
        "| Variant | Source | Package MB | LLM MB | Vision MB | GPU offload |",
        "|---------|--------|------------|--------|-----------|-------------|",
        f"| HF | `{mh.get('weight_source')}` | {float(mh.get('package_disk_mb') or 0):.1f} | "
        f"(in package) | (in package) | torch |",
        f"| GGUF | `{mq.get('weight_source')}` | {float(mq.get('package_disk_mb') or 0):.1f} | "
        f"{float(mq.get('llm_disk_mb') or 0):.1f} | {float(mq.get('mmproj_disk_mb') or 0):.1f} | "
        f"n_gpu_layers={mq.get('n_gpu_layers')} reported={mq.get('gpu_offload_reported')} |",
        "",
        f"- HF params: {mh.get('params_m', '—')} M",
        f"- GGUF process RSS after load: {mq.get('process_rss_mb', '—')} MB",
        f"- GGUF GPU used after load: {mq.get('gpu_used_after_load_mb', '—')} MB",
        "",
        "## Notes",
        "",
        "- Edge AD (pixel metrics) still uses vision-only; this bench is **RouteAgent LLM path**.",
        "- If `gpu_offload_reported=false`, rebuild CUDA llama-cpp-python for GPU Q4 savings.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--data-root", default=str(ROOT / "datasets" / "mvtec"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--hf-model", default=DEFAULT_MODEL)
    ap.add_argument("--gguf-dir", default=DEFAULT_GGUF_DIR)
    ap.add_argument("--backends", nargs="+", default=["hf", "gguf"])
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "reports" / "qwen_quant"))
    ap.add_argument("--tag", default="route_q4")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = ap.parse_args()

    # optional config overrides
    if Path(args.config).exists():
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        ra = ((cfg.get("collab") or {}).get("route_agent") or {})
        if ra.get("model_path") and args.hf_model == DEFAULT_MODEL:
            args.hf_model = str(ra["model_path"])

    samples = _load_samples(Path(args.data_root), args.category, args.n_samples, args.seed)
    print(f"samples ({len(samples)}): {[f'{p.parent.name}/{p.name}' for p in samples]}")

    results = {}
    for b in args.backends:
        results[b] = _run_backend(
            b,
            samples,
            category=args.category,
            device=args.device,
            model_path=args.hf_model,
            gguf_dir=args.gguf_dir,
            max_new_tokens=args.max_new_tokens,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
        )
        # free GPU between backends
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"route_quant_{args.tag}.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    if "hf" in results and "gguf" in results:
        md = _md(results["hf"], results["gguf"], args.tag)
    else:
        md = f"# Route quant ({args.tag})\n\n```json\n{json.dumps(results, indent=2)}\n```\n"
    md_path = out_dir / f"route_quant_{args.tag}.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print("DONE")


if __name__ == "__main__":
    main()
