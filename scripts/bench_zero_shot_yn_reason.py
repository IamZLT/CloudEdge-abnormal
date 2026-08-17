#!/usr/bin/env python3
"""Zero-shot Yes/No + REASON check on the FULL image (un-finetuned 0.8B).

Prompt asks the model to answer Yes/No AND briefly explain why. Every sample's
full raw output is saved to a JSONL so the user can inspect the reasoning.

Output files (both under --out-dir):
  result.json     — aggregate metrics
  details.jsonl   — per-sample: image, category, gt, raw, decision, reason

Env: conda activate clip
  CUDA_VISIBLE_DEVICES=4 python scripts/bench_zero_shot_yn_reason.py \
    --test-jsonl datasets/mvtec_anomaly_llm/test.jsonl \
    --out-dir outputs/qwen35_zero_shot_yn_reason
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vlm.qwen_client import QwenVLClient  # noqa: E402

DEFAULT_PROMPT = (
    "Is there any anomaly or defect in the product shown in the image?\n"
    "Answer with Yes or No, and briefly explain the reason."
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_yn_reason(raw: str) -> dict:
    """Extract decision (Yes/No) and a short reason from free-form text."""
    s = (raw or "").strip()
    low = s.lower()

    # first word decides Yes/No
    first = s.split()[0].strip(".,;:!?()[]\"'") if s.split() else ""
    if first.lower() == "yes":
        decision = 1
    elif first.lower() == "no":
        decision = 0
    elif low.startswith("yes"):
        decision = 1
    elif low.startswith("no"):
        decision = 0
    elif "yes" in low and "no" not in low:
        decision = 1
    else:
        decision = 0

    # remainder after Yes/No keyword = reason
    m = re.match(r"^\s*(yes|no)\b[.,:;\s]*\s*(.*)$", s, flags=re.I | re.S)
    reason = m.group(2).strip() if m else s

    return {"decision": decision, "reason": reason, "raw": s}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-jsonl", default="datasets/mvtec_anomaly_llm/test.jsonl")
    parser.add_argument("--out-dir", default="outputs/qwen35_zero_shot_yn_reason")
    parser.add_argument("--model-path", default="/data2/zlt/anomaly_detection_llm/model_card/Qwen3.5-0.8B")
    parser.add_argument("--adapter-path", default=None, help="optional LoRA adapter dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--prompt", default=None, help="override the default prompt")
    args = parser.parse_args()

    test_path = Path(args.test_jsonl)
    if not test_path.is_absolute():
        test_path = ROOT / test_path
    rows = load_jsonl(test_path)
    if args.max_images:
        rows = rows[: args.max_images]

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = args.prompt or DEFAULT_PROMPT

    client = QwenVLClient(
        model_path=args.model_path,
        device=args.device,
        dtype="bfloat16",
        max_new_tokens=64,
        role="yn_reason_sft" if args.adapter_path else "yn_reason_zs",
        prompt=prompt,
        model_family="qwen3_5",
        adapter_path=args.adapter_path,
    )

    y_true, y_pred = [], []
    details = []
    t0 = time.perf_counter()
    for i, r in enumerate(rows):
        res = client.infer(r["image"])
        gt = int(r["label"])
        parsed = parse_yn_reason(res.raw)
        pred = parsed["decision"]
        y_true.append(gt)
        y_pred.append(pred)
        details.append(
            {
                "image": r["image"],
                "category": r["category"],
                "gt": gt,
                "gt_defect_type": r.get("defect_type", "none"),
                "pred": pred,
                "decision": "Yes" if pred else "No",
                "reason": parsed["reason"],
                "raw": res.raw,
            }
        )
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
        "prompt": prompt,
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
    }

    (out_dir / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "details.jsonl").open("w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("\n===== Yes/No + reason (full image) =====")
    print(f"N={report['n']}  GT OK/NG={report['gt_ok']}/{report['gt_ng']}")
    print(f"Acc={report['accuracy']:.4f}  F1={report['f1']:.4f}  P={report['precision']:.4f}  R={report['recall']:.4f}")
    print(f"pred Yes/No={report['pred_yes']}/{report['pred_no']}")
    print(f"confusion TP/FP/TN/FN = {tp}/{fp}/{tn}/{fn}")
    print(f"Wrote {out_dir / 'result.json'} and {out_dir / 'details.jsonl'}")


if __name__ == "__main__":
    main()
