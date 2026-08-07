#!/usr/bin/env python3
"""Compare RouteAgent vision_mode: full (2nd encode) vs text (reuse AD CONTEXT).

After edge AD, CONTEXT already carries score/threshold/network. Default production
uses vision_mode=text to skip the second vision encode (GGUF mmproj / HF vision).

Example:
  conda activate clip
  CUDA_VISIBLE_DEVICES=0 python scripts/bench_route_vision_reuse.py \\
    --category bottle --n-samples 8 --tag route_reuse
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
    RouteAgent,
    RouteContext,
)


def _load_samples(data_root: Path, category: str, n: int, seed: int) -> list[Path]:
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


def _make_ctx(path: Path, category: str, i: int) -> RouteContext:
    # Mix confident / near-threshold / cold-start so LLM is actually exercised
    if i % 3 == 0:
        score, thr, n_gal = 0.52, 0.50, 16  # near thr → upload
    elif i % 3 == 1:
        score, thr, n_gal = 0.12, 0.50, 16  # confident OK → local
    else:
        score, thr, n_gal = 0.40, 0.50, 0  # cold start → upload
    return RouteContext(
        image=path,
        category=category,
        n_gallery=n_gal,
        edge_score=score,
        edge_thr=thr,
        edge_decision="NG" if score >= thr else "OK",
        network_profile="fair",
        network={"profile": "fair", "rtt_ms": 80.0, "bandwidth_mbps": 10.0},
        hard_margin=0.05,
    )


def _run_mode(
    agent: RouteAgent,
    samples: list[Path],
    *,
    category: str,
    vision_mode: str,
    warmup: int = 1,
) -> dict:
    agent.vision_mode = vision_mode
    # match RouteAgent default: snap rules when not full multimodal
    agent.enforce_context_rules = vision_mode not in {"full", "image", "multimodal", "mm", "vision"}
    agent.meta["vision_mode"] = vision_mode
    agent.meta["enforce_context_rules"] = agent.enforce_context_rules

    # warmup (not timed)
    if samples and warmup > 0:
        for w in range(min(warmup, len(samples))):
            agent.decide(_make_ctx(samples[w], category, w))

    rows = []
    for i, path in enumerate(samples):
        ctx = _make_ctx(path, category, i)
        dec = agent.decide(ctx)
        rows.append(
            {
                "path": str(path),
                "upload": dec.upload,
                "confidence": dec.confidence,
                "parse_ok": dec.parse_ok,
                "source": dec.source,
                "latency_ms": dec.latency_ms,
                "include_image": dec.include_image,
                "reason": dec.reason,
                "n_gallery": ctx.n_gallery,
                "score_margin": ctx.score_margin(),
            }
        )

    lats = [r["latency_ms"] for r in rows]
    return {
        "vision_mode": vision_mode,
        "include_image": bool(rows[0]["include_image"]) if rows else False,
        "n": len(rows),
        "mean_latency_ms": float(np.mean(lats)) if lats else float("nan"),
        "p50_latency_ms": float(np.median(lats)) if lats else float("nan"),
        "p90_latency_ms": float(np.percentile(lats, 90)) if lats else float("nan"),
        "parse_ok_rate": float(np.mean([r["parse_ok"] for r in rows])) if rows else 0.0,
        "upload_rate": float(np.mean([r["upload"] for r in rows])) if rows else 0.0,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--backend", default="gguf", choices=["gguf", "hf"])
    ap.add_argument("--gguf-dir", default=DEFAULT_GGUF_DIR)
    ap.add_argument("--model-path", default="/data2/zlt/anomaly_detection_llm/model_card/Qwen3.5-0.8B")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--tag", default="route_reuse")
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    data_root = Path(cfg.get("data_root") or ROOT / "datasets" / "mvtec")
    if not data_root.is_absolute():
        data_root = ROOT / data_root

    samples = _load_samples(data_root, args.category, args.n_samples, args.seed)
    if not samples:
        raise SystemExit(f"no samples under {data_root / args.category / 'test'}")

    ra_cfg = {
        "backend": args.backend,
        "device": args.device,
        "max_new_tokens": 96,
        "n_ctx": 4096,
        "n_gpu_layers": -1,
        "verbose": False,
        "hard_block_on_outage": True,
        "vision_mode": "full",  # start with full; switch in-process
    }
    if args.backend == "hf":
        ra_cfg["model_path"] = args.model_path
    else:
        ra_cfg["gguf_dir"] = args.gguf_dir
        ra_cfg["model_path"] = args.gguf_dir

    print(f"[bench] backend={args.backend} category={args.category} n={len(samples)}")
    t0 = time.perf_counter()
    agent = RouteAgent.from_config(ra_cfg)
    load_s = time.perf_counter() - t0
    agent.mark_load_memory()
    print(f"[bench] load_s={load_s:.2f} meta={ {k: agent.meta.get(k) for k in ('weight_source','gpu_footprint_mb','vision_mode')} }")

    # full first (baseline double-encode), then text (reuse)
    full = _run_mode(agent, samples, category=args.category, vision_mode="full", warmup=args.warmup)
    text = _run_mode(agent, samples, category=args.category, vision_mode="text", warmup=args.warmup)

    from src.vlm.route_agent import heuristic_upload

    agree = 0
    rule_full = 0
    rule_text = 0
    for i, (a, b) in enumerate(zip(full["rows"], text["rows"])):
        if a["upload"] == b["upload"]:
            agree += 1
        ctx = _make_ctx(samples[i], args.category, i)
        ruled = heuristic_upload(ctx)
        if a["upload"] == ruled:
            rule_full += 1
        if b["upload"] == ruled:
            rule_text += 1
    n = max(1, len(full["rows"]))
    speedup = (
        float(full["mean_latency_ms"] / text["mean_latency_ms"])
        if text["mean_latency_ms"] and text["mean_latency_ms"] > 0
        else float("nan")
    )
    reduce_pct = (
        float((full["mean_latency_ms"] - text["mean_latency_ms"]) / full["mean_latency_ms"] * 100.0)
        if full["mean_latency_ms"] and full["mean_latency_ms"] > 0
        else float("nan")
    )

    summary = {
        "tag": args.tag,
        "backend": args.backend,
        "category": args.category,
        "n_samples": len(samples),
        "load_s": load_s,
        "agent_meta": agent.meta,
        "full": {k: v for k, v in full.items() if k != "rows"},
        "text": {k: v for k, v in text.items() if k != "rows"},
        "upload_agreement_full_vs_text": agree / n,
        "rule_agreement_full": rule_full / n,
        "rule_agreement_text": rule_text / n,
        "speedup_vs_full": speedup,
        "latency_reduce_pct": reduce_pct,
        "rows_full": full["rows"],
        "rows_text": text["rows"],
    }

    out_dir = ROOT / "outputs" / "bench_route_vision_reuse"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}_{args.backend}_{args.category}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== RouteAgent vision reuse ===")
    print(f"backend          : {args.backend}")
    print(f"full  mean_ms    : {full['mean_latency_ms']:.1f}  (include_image={full['include_image']})")
    print(f"text  mean_ms    : {text['mean_latency_ms']:.1f}  (include_image={text['include_image']})")
    print(f"speedup          : {speedup:.2f}x")
    print(f"latency reduce   : {reduce_pct:.1f}%")
    print(f"full vs text agree: {agree}/{n} ({agree/n:.1%})")
    print(f"rule agree full/text: {rule_full}/{n} ({rule_full/n:.1%}) / {rule_text}/{n} ({rule_text/n:.1%})")
    print(f"parse_ok full/text: {full['parse_ok_rate']:.2f} / {text['parse_ok_rate']:.2f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
