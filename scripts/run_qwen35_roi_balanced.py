#!/usr/bin/env python3
"""Rebalance the OK/NG training corpus and retrain + re-eval the ROI LoRA model.

Why: the "balanced" prompt removed all edge leakage (score/thr/hint). Without
that cue the LoRA model fell back to the majority class — OK (3629 vs 755 NG,
a 4.8:1 imbalance) — and predicted "OK" for every test image.

Fix: downsample OK and upsample NG **per category** to a 1:1 ratio so the model
is forced to decide from the visual ROI alone. Reuses the existing build corpus
(roi_sft_train_newsplit.jsonl), so the expensive edge-scoring / ROI-crop stage is
not re-run.

Env: conda activate clip
  CUDA_VISIBLE_DEVICES=4 python scripts/run_qwen35_roi_balanced.py \
    --config configs/qwen35_roi_newsplit_sft.yaml --device cuda:4
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_qwen35_roi_newsplit_sft import (  # noqa: E402
    load_jsonl,
    stage_eval,
    stage_train,
    write_jsonl,
)


def balance_rows(rows: list[dict], seed: int = 42) -> list[dict]:
    """Per-category 1:1 OK/NG balance (downsample OK, repeat-upsample NG)."""
    rng = random.Random(seed)
    by: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[str(r["category"])][int(r["label"])].append(r)

    out: list[dict] = []
    stats = []
    for cat in sorted(by):
        ok = by[cat][0]
        ng = by[cat][1]
        target = (len(ok) + len(ng) + 1) // 2  # keep total size roughly unchanged
        ok_sel = rng.sample(ok, min(len(ok), target))
        ng_sel: list[dict] = []
        while len(ng_sel) < target:
            ng_sel.extend(rng.sample(ng, len(ng)))
        ng_sel = ng_sel[:target]
        out.extend(ok_sel + ng_sel)
        stats.append((cat, len(ok), len(ng), len(ok_sel), len(ng_sel)))
    rng.shuffle(out)

    print(f"{'cat':<14}{'OK':>6}{'NG':>6}{'OK_sel':>8}{'NG_sel':>8}")
    for cat, o, n, os_, ns_ in stats:
        print(f"{cat:<14}{o:>6}{n:>6}{os_:>8}{ns_:>8}")
    print(f"{'TOTAL':<14}{sum(s[1] for s in stats):>6}{sum(s[2] for s in stats):>6}"
          f"{sum(s[3] for s in stats):>8}{sum(s[4] for s in stats):>8}")
    return out


def holdout_val(rows: list[dict], n_ok: int = 60, n_ng: int = 60, seed: int = 42):
    """Randomly pull a small balanced validation subset; the rest stays for training."""
    rng = random.Random(seed)
    ok = [r for r in rows if int(r["label"]) == 0]
    ng = [r for r in rows if int(r["label"]) == 1]
    val_ok = rng.sample(ok, min(n_ok, len(ok)))
    val_ng = rng.sample(ng, min(n_ng, len(ng)))
    val = val_ok + val_ng
    rng.shuffle(val)
    val_ids = {id(r) for r in val}
    train = [r for r in rows if id(r) not in val_ids]
    return train, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/qwen35_roi_newsplit_sft.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ok", type=int, default=60, help="validation OK samples to hold out")
    parser.add_argument("--val-ng", type=int, default=60, help="validation NG samples to hold out")
    parser.add_argument("--eval-every", type=int, default=50, help="steps between quick val evals")
    parser.add_argument("--only-balance", action="store_true",
                        help="only write the balanced corpus, skip train/eval")
    args = parser.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = args.device or cfg.get("device") or "cuda:0"

    src_dir = Path(cfg.get("results_dir") or "outputs/qwen35_roi_newsplit_sft")
    if not src_dir.is_absolute():
        src_dir = ROOT / src_dir
    src_jsonl = src_dir / "roi_sft_train_newsplit.jsonl"
    meta_src = src_dir / "roi_sft_meta_newsplit.json"

    out_dir = src_dir.parent / (src_dir.name + "_balanced")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(src_jsonl)
    balanced = balance_rows(rows, seed=args.seed)
    train_rows, val_rows = holdout_val(balanced, n_ok=args.val_ok, n_ng=args.val_ng, seed=args.seed)

    bal_jsonl = out_dir / "roi_sft_train_balanced.jsonl"
    val_jsonl = out_dir / "roi_sft_val_balanced.jsonl"
    write_jsonl(bal_jsonl, train_rows)
    write_jsonl(val_jsonl, val_rows)
    print(f"[balance] train={len(train_rows)} val={len(val_rows)}")
    print(f"[balance] wrote {bal_jsonl}")
    print(f"[balance] wrote {val_jsonl}")

    # stage_eval reads vocab/thr_map from this meta file; copy it across.
    if meta_src.exists():
        shutil.copy(meta_src, out_dir / "roi_sft_meta_newsplit.json")

    if args.only_balance:
        return

    adapter = stage_train(
        cfg,
        device,
        out_dir,
        bal_jsonl,
        extra_args=["--val-jsonl", str(val_jsonl), "--eval-every", str(args.eval_every)],
    )
    stage_eval(cfg, device, out_dir, adapter)


if __name__ == "__main__":
    main()
