#!/usr/bin/env python3
"""Multiscale ROI pipeline: build (full+ROI collage) → balance → train → eval.

Build the collage corpus from scratch (unlike run_qwen35_roi_balanced.py which
reuses the single-ROI corpus), then balance OK/NG 1:1 per category, hold out a
small val subset for periodic training eval, LoRA-train, and evaluate on test.

Env: conda activate clip
  CUDA_VISIBLE_DEVICES=3 python scripts/run_qwen35_roi_multiscale.py \
    --config configs/qwen35_roi_multiscale.yaml --device cuda:0
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_qwen35_roi_balanced import balance_rows, holdout_val  # noqa: E402
from scripts.run_qwen35_roi_newsplit_sft import (  # noqa: E402
    load_jsonl,
    stage_build,
    stage_eval,
    stage_train,
    write_jsonl,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/qwen35_roi_multiscale.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ok", type=int, default=20)
    parser.add_argument("--val-ng", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--stage", default="all", choices=["build", "train", "eval", "all"])
    args = parser.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = args.device or cfg.get("device") or "cuda:0"

    out_dir = Path(cfg.get("results_dir") or "outputs/qwen35_roi_multiscale")
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    build_jsonl = out_dir / "roi_sft_train_newsplit.jsonl"  # stage_build's fixed output name
    if args.stage in ("build", "all"):
        build_jsonl = stage_build(cfg, device, out_dir)

    if args.stage in ("train", "all"):
        # balance + hold out val
        rows = load_jsonl(build_jsonl)
        balanced = balance_rows(rows, seed=args.seed)
        train_rows, val_rows = holdout_val(balanced, n_ok=args.val_ok, n_ng=args.val_ng, seed=args.seed)
        bal_jsonl = out_dir / "roi_sft_train_ms_balanced.jsonl"
        val_jsonl = out_dir / "roi_sft_val_ms_balanced.jsonl"
        write_jsonl(bal_jsonl, train_rows)
        write_jsonl(val_jsonl, val_rows)
        print(f"[balance] train={len(train_rows)} val={len(val_rows)}")
        adapter = stage_train(
            cfg,
            device,
            out_dir,
            bal_jsonl,
            extra_args=["--val-jsonl", str(val_jsonl), "--eval-every", str(args.eval_every)],
        )
    else:
        adapter = out_dir / "adapter"

    if args.stage in ("eval", "all"):
        stage_eval(cfg, device, out_dir, adapter)


if __name__ == "__main__":
    main()
