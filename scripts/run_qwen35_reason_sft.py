#!/usr/bin/env python3
"""Reason fine-tuning from the REAL GPT annotations in mvtec_zero_shot_{train,test}.json.

The user's existing zero-shot annotation files contain multi-turn GPT conversations
whose 2nd GPT turn is a "Yes/No + reason" sentence (e.g. "Yes, there is an anomaly.
A small indentation ..." / "No, the capsule exhibits no anomalies. It displays ...").
We extract that as the SFT target instead of templated text.

Pipeline: build (extract reason corpus) -> balance -> LoRA train -> eval.

Prompt (simple, same as the best zero-shot Yes/No+reason setup):
  "Is there any anomaly or defect in the product shown in the image?\n
   Answer with Yes or No, and briefly explain the reason."

Env: conda activate clip
  CUDA_VISIBLE_DEVICES=4 python scripts/run_qwen35_reason_sft.py \
    --config configs/qwen35_reason_sft.yaml --device cuda:0 --stage all
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_qwen35_roi_balanced import balance_rows, holdout_val  # noqa: E402
from scripts.run_qwen35_roi_newsplit_sft import load_jsonl, write_jsonl  # noqa: E402

# mvtec_zero_shot image paths are relative to the mvtec_anomaly_llm dir, whose real
# location (via symlink) matches the absolute paths used in train.jsonl/test.jsonl.
LLM_BASE = (ROOT / "datasets/mvtec_anomaly_llm").resolve()

DEFAULT_PROMPT = (
    "Is there any anomaly or defect in the product shown in the image?\n"
    "Answer with Yes or No, and briefly explain the reason."
)


def load_zero_shot(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_reason(conv: list[dict], anomaly_flag: bool) -> tuple[int, str] | None:
    """Return (label, response) using the OFFICIAL label (metadata.anomaly) and the
    GPT "Yes/No + reason" sentence as the response text.

    Rules (so label and response never contradict):
      - Prefer the first GPT turn that starts with "Yes/No". If its decision agrees
        with `anomaly_flag`, use it verbatim (label=anomaly_flag).
      - If it disagrees (GPT misjudged), skip (None) to avoid label/text conflict.
      - If no "Yes/No" turn exists ("What anomaly is present?" style), build one
        from the 2nd GPT turn: "Yes, <body>" / "No, <body>".
      - Bare "Yes"/"No" turns (short 2-turn entries) have no reason -> skip.
    """
    label = 1 if anomaly_flag else 0
    gpt_turns = [m for m in conv if m.get("from") == "gpt"]
    body = None
    for m in gpt_turns:
        v = (m.get("value") or "").strip()
        m2 = re.match(r"^\s*(yes|no)\b[.,:;\s]*(.*)$", v, flags=re.I | re.S)
        if not m2:
            continue
        word = m2.group(1).lower()
        rest = m2.group(2).strip()
        if not rest:  # bare "Yes"/"No"
            continue
        if (1 if word == "yes" else 0) == label:
            return label, v  # verbatim, consistent
        return None  # GPT disagrees with the official label -> skip
    # no "Yes/No + reason" turn: synthesize from the anomaly-description turn.
    # Short 2-turn entries have only one bare GPT turn (no description) -> skip.
    if len(gpt_turns) < 2:
        return None
    body = gpt_turns[1]["value"].strip()
    if not body:
        return None
    return label, ("Yes" if label else "No") + ", " + body


def cat_from_image(rel: str) -> str:
    # rel like "train/capsule/poke/003.png" -> "capsule"
    parts = rel.split("/")
    return parts[1] if len(parts) > 1 else "unknown"


def type_from_image(rel: str, label: int) -> str:
    # rel like "train/capsule/poke/003.png" -> "poke"; good -> none
    parts = rel.split("/")
    if label == 0:
        return "none"
    return parts[2] if len(parts) > 2 else "defect"


def build_corpus(cfg: dict, out_dir: Path) -> Path:
    src = Path(cfg["zero_shot_train"])
    if not src.is_absolute():
        src = ROOT / src
    data = load_zero_shot(src)
    prompt = (cfg.get("prompt") or DEFAULT_PROMPT).strip()

    rows, skipped_bare, skipped_none = [], 0, 0
    for d in data:
        flag = bool(d["metadata"].get("anomaly"))
        r = extract_reason(d.get("conversations", []), flag)
        if r is None:
            skipped_none += 1
            continue
        label, response = r
        if response.strip().lower() in ("yes", "no"):
            skipped_bare += 1
            continue
        rel = d["image"]
        img = str(LLM_BASE / rel)
        cat = cat_from_image(rel)
        rows.append(
            {
                "image": img,
                "category": cat,
                "label": label,
                "defect_type": type_from_image(rel, label),
                "prompt": prompt,
                "response": response,
            }
        )

    out_path = out_dir / "reason_train.jsonl"
    write_jsonl(out_path, rows)
    ok = sum(1 for x in rows if x["label"] == 0)
    ng = sum(1 for x in rows if x["label"] == 1)
    print(f"[build] wrote {out_path} n={len(rows)} ok={ok} ng={ng} "
          f"(skipped bare={skipped_bare} no-reason={skipped_none})")
    return out_path


def build_test(cfg: dict, out_dir: Path) -> Path:
    src = Path(cfg["zero_shot_test"])
    if not src.is_absolute():
        src = ROOT / src
    data = load_zero_shot(src)
    rows = []
    for d in data:
        label = 1 if d["metadata"].get("anomaly") else 0
        rel = d["image"]
        img = str(LLM_BASE / rel)
        cat = cat_from_image(rel)
        gt_reason = extract_reason(d.get("conversations", []), bool(label))
        rows.append(
            {
                "image": img,
                "category": cat,
                "label": label,
                "defect_type": type_from_image(rel, label),
                "gt_reason": (gt_reason[1] if gt_reason else ""),
            }
        )
    out_path = out_dir / "reason_test.jsonl"
    write_jsonl(out_path, rows)
    ok = sum(1 for x in rows if x["label"] == 0)
    ng = sum(1 for x in rows if x["label"] == 1)
    print(f"[test] wrote {out_path} n={len(rows)} ok={ok} ng={ng}")
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
        sys.executable, "-u",
        str(ROOT / "scripts" / "train_qwen35_sft.py"),
        "--config", str(overlay_path),
        "--mode", "lora",
        "--train-jsonl", str(train_jsonl),
        "--output-dir", str(adapter),
        "--device", device,
        "--train-section", "train_lora",
    ]
    if extra_args:
        cmd.extend(extra_args)
    print("[train]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    return adapter


def stage_eval(cfg: dict, device: str, out_dir: Path, test_jsonl: Path, adapter: Path):
    cmd = [
        sys.executable, "-u",
        str(ROOT / "scripts" / "bench_zero_shot_yn_reason.py"),
        "--test-jsonl", str(test_jsonl),
        "--out-dir", str(out_dir),
        "--device", device,
        "--adapter-path", str(adapter),
    ]
    print("[eval]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/qwen35_reason_sft.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ok", type=int, default=20)
    parser.add_argument("--val-ng", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--stage", default="all", choices=["build", "train", "eval", "all"])
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = args.device or cfg.get("device") or "cuda:0"

    out_dir = Path(cfg.get("results_dir") or "outputs/qwen35_reason_sft")
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("build", "all"):
        build_corpus(cfg, out_dir)
        build_test(cfg, out_dir)

    corpus = out_dir / "reason_train.jsonl"
    test_jsonl = out_dir / "reason_test.jsonl"
    adapter = out_dir / "adapter"

    if args.stage in ("train", "all"):
        if not (adapter / "adapter_model.safetensors").exists():
            rows = load_jsonl(corpus)
            balanced = balance_rows(rows, seed=args.seed)
            train_rows, val_rows = holdout_val(balanced, n_ok=args.val_ok, n_ng=args.val_ng, seed=args.seed)
            bal_jsonl = out_dir / "reason_train_balanced.jsonl"
            val_jsonl = out_dir / "reason_val_balanced.jsonl"
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
        stage_eval(cfg, device, out_dir, test_jsonl, adapter)


if __name__ == "__main__":
    main()
