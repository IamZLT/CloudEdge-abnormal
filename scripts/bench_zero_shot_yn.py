#!/usr/bin/env python3
"""Zero-shot Yes/No anomaly check on the FULL image with the UN-FINETUNED 0.8B.

Prompt (fixed):
    "Is there any anomaly or defect in the product shown in the image?\n"
    "Answer with Yes or No."

No edge model, no ROI, no collage, no LoRA adapter — pure base-model judgement on
the whole image, to establish the zero-shot ceiling of Qwen3.5-0.8B.

Env: conda activate clip
  CUDA_VISIBLE_DEVICES=6 python scripts/bench_zero_shot_yn.py \
    --test-jsonl datasets/mvtec_anomaly_llm/test.jsonl \
    --out outputs/qwen35_zero_shot_yn/result.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vlm.qwen_client import QwenVLClient  # noqa: E402

DEFAULT_TEST_PROMPT = (
    "Is there any anomaly or defect in the product shown in the image?\n"
    "Answer with Yes or No."
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_yes_no(raw: str) -> int:
    """Return 1 (Yes/anomaly) or 0 (No/normal) from the model's free-form text."""
    s = (raw or "").strip().lower()
    first = s.split()[0].strip(".,;!?()") if s else ""
    if first == "yes":
        return 1
    if first == "no":
        return 0
    if s.startswith("yes"):
        return 1
    if s.startswith("no"):
        return 0
    # heuristic fallback
    if "yes" in s and "no" not in s:
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-jsonl", default="datasets/mvtec_anomaly_llm/test.jsonl")
    parser.add_argument("--out", default="outputs/qwen35_zero_shot_yn/result.json")
    parser.add_argument("--model-path", default="/data2/zlt/anomaly_detection_llm/model_card/Qwen3.5-0.8B")
    parser.add_argument("--adapter-path", default=None, help="optional LoRA adapter dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    test_path = Path(args.test_jsonl)
    if not test_path.is_absolute():
        test_path = ROOT / test_path
    rows = load_jsonl(test_path)
    if args.max_images:
        rows = rows[: args.max_images]

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = QwenVLClient(
        model_path=args.model_path,
        device=args.device,
        dtype="bfloat16",
        max_new_tokens=16,
        role="yn_sft" if args.adapter_path else "zero_shot_yn",
        prompt=DEFAULT_TEST_PROMPT,
        model_family="qwen3_5",
        adapter_path=args.adapter_path,
    )

    y_true, y_pred = [], []
    raw_samples = []
    t0 = time.perf_counter()
    for i, r in enumerate(rows):
        res = client.infer(r["image"])
        gt = int(r["label"])
        pred = parse_yes_no(res.raw)
        y_true.append(gt)
        y_pred.append(pred)
        if len(raw_samples) < 20:
            raw_samples.append({"category": r["category"], "gt": gt, "raw": res.raw})
        if (i + 1) % 50 == 0:
            acc = np.mean([a == b for a, b in zip(y_true, y_pred)])
            print(f"[{i+1}/{len(rows)}] acc={acc:.3f} elapsed={time.perf_counter()-t0:.0f}s")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    report = {
        "prompt": DEFAULT_TEST_PROMPT,
        "model": args.model_path,
        "adapter": args.adapter_path,
        "n": int(len(y_true)),
        "gt_ok": int(np.sum(y_true == 0)),
        "gt_ng": int(np.sum(y_true == 1)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "pred_yes": int(np.sum(y_pred == 1)),
        "pred_no": int(np.sum(y_pred == 0)),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "raw_samples": raw_samples,
    }
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n===== zero-shot Yes/No (full image) =====")
    print(f"N={report['n']}  GT OK/NG={report['gt_ok']}/{report['gt_ng']}")
    print(f"Acc={report['accuracy']:.4f}  F1={report['f1']:.4f}  P={report['precision']:.4f}  R={report['recall']:.4f}")
    print(f"pred Yes/No={report['pred_yes']}/{report['pred_no']}")
    print(f"confusion TP/FP/TN/FN = {tp}/{fp}/{tn}/{fn}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
