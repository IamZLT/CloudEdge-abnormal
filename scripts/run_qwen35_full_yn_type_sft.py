#!/usr/bin/env python3
"""Full-image + "Yes/No + defect type" LoRA pipeline: build → balance → train → eval.

Corpus: full image + fixed prompt; response is "No" (OK) or "Yes, {defect_type}"
(NG, using the real MVTec defect type). OK/NG balanced 1:1 per category.

Env: conda activate clip
  CUDA_VISIBLE_DEVICES=3 python scripts/run_qwen35_full_yn_type_sft.py \
    --config configs/qwen35_full_yn_type_sft.yaml --device cuda:0
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_qwen35_roi_balanced import balance_rows, holdout_val  # noqa: E402
from scripts.run_qwen35_roi_newsplit_sft import load_jsonl, write_jsonl  # noqa: E402


def build_corpus(cfg: dict, out_dir: Path) -> Path:
    train_path = Path(cfg["train_jsonl"])
    if not train_path.is_absolute():
        train_path = ROOT / train_path
    rows = load_jsonl(train_path)
    prompt = (cfg.get("prompt") or "").strip()

    out_rows = []
    for r in rows:
        label = int(r["label"])
        if label == 1:
            dt = str(r.get("defect_type") or "defect")
            response = f"Yes, {dt}"
        else:
            response = "No"
        out_rows.append(
            {
                "image": r["image"],
                "category": r["category"],
                "label": label,
                "defect_type": r.get("defect_type", "none"),
                "prompt": prompt,
                "response": response,
            }
        )

    out_path = out_dir / "full_yn_type_train.jsonl"
    write_jsonl(out_path, out_rows)
    ok = sum(1 for x in out_rows if x["label"] == 0)
    ng = sum(1 for x in out_rows if x["label"] == 1)
    print(f"[build] wrote {out_path} n={len(out_rows)} ok={ok} ng={ng}")
    return out_path


def stage_train(cfg: dict, device: str, out_dir: Path, train_jsonl: Path, extra_args: list[str] | None = None) -> Path:
    adapter = out_dir / "adapter"
    overlay = {
        "model": cfg["model"],
        "lora": cfg["lora"],
        "train_lora": {**cfg["train"], "device": device},
    }
    overlay_path = out_dir / "train_overlay.yaml"
    overlay_path.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "train_qwen35_sft.py"),
        "--config",
        str(overlay_path),
        "--mode",
        "lora",
        "--train-jsonl",
        str(train_jsonl),
        "--output-dir",
        str(adapter),
        "--device",
        device,
        "--train-section",
        "train_lora",
    ]
    if extra_args:
        cmd.extend(extra_args)
    print("[train]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    return adapter


def stage_eval(cfg: dict, device: str, out_dir: Path, adapter: Path):
    test_path = Path(cfg["test_jsonl"])
    if not test_path.is_absolute():
        test_path = ROOT / test_path
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "bench_full_yn_type.py"),
        "--test-jsonl",
        str(test_path),
        "--out-dir",
        str(out_dir),
        "--device",
        device,
        "--adapter-path",
        str(adapter),
    ]
    print("[eval]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/qwen35_full_yn_type_sft.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ok", type=int, default=20)
    parser.add_argument("--val-ng", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--stage", default="all", choices=["build", "train", "eval", "all"])
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = args.device or cfg.get("device") or "cuda:0"

    out_dir = Path(cfg.get("results_dir") or "outputs/qwen35_full_yn_type_sft")
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = out_dir / "full_yn_type_train.jsonl"
    if args.stage in ("build", "all"):
        corpus = build_corpus(cfg, out_dir)

    adapter = out_dir / "adapter"
    if args.stage in ("train", "all"):
        if not (adapter / "adapter_model.safetensors").exists():
            rows = load_jsonl(corpus)
            balanced = balance_rows(rows, seed=args.seed)
            train_rows, val_rows = holdout_val(balanced, n_ok=args.val_ok, n_ng=args.val_ng, seed=args.seed)
            bal_jsonl = out_dir / "full_yn_type_train_balanced.jsonl"
            val_jsonl = out_dir / "full_yn_type_val_balanced.jsonl"
            write_jsonl(bal_jsonl, train_rows)
            write_jsonl(val_jsonl, val_rows)
            print(f"[balance] train={len(train_rows)} val={len(val_rows)}")
            adapter = stage_train(
                cfg, device, out_dir, bal_jsonl,
                extra_args=["--val-jsonl", str(val_jsonl), "--eval-every", str(args.eval_every)],
            )
        else:
            print(f"[train] adapter exists, skip: {adapter}")

    if args.stage in ("eval", "all"):
        stage_eval(cfg, device, out_dir, adapter)


if __name__ == "__main__":
    main()
