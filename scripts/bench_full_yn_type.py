#!/usr/bin/env python3
"""Evaluate Yes/No + defect-type output (zero-shot or LoRA) on the full image.

Parsing:
  - FIRST word Yes/No  -> decision (OK/NG)
  - words after "Yes,"  -> predicted defect_type (matched against gt via norm_type)

Saves per-sample details (full raw + parsed type) so the user can inspect naming.

Env: conda activate clip
  CUDA_VISIBLE_DEVICES=3 python scripts/bench_full_yn_type.py \
    --test-jsonl datasets/mvtec_anomaly_llm/test.jsonl \
    --out-dir outputs/qwen35_full_yn_type_sft
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
    'Is there any anomaly or defect in the product shown in the image?\n'
    'Answer with "Yes" or "No". If Yes, name the defect type in a few words.'
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm_type(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def parse_yn_type(raw: str) -> dict:
    """decision (1/0) + predicted defect_type string from free-form text."""
    s = (raw or "").strip()
    low = s.lower()

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

    # extract words after "Yes," as the defect type (stop at punctuation/stopwords)
    m = re.match(r"^\s*(yes|no)\b[.,:;\s]*\s*(.*)$", s, flags=re.I | re.S)
    rest = m.group(2).strip() if m else ""
    rest = re.sub(r"[^a-zA-Z0-9\s\-]", " ", rest)  # drop punctuation
    toks = rest.split()
    # keep leading tokens that look like a defect name; cut at common filler words
    stop = {"the", "a", "an", "there", "is", "are", "it", "in", "on", "of", "product", "image", "appears", "visible"}
    kept = []
    for t in toks:
        if t.lower() in stop:
            break
        kept.append(t)
        if len(kept) >= 3:
            break
    pred_type = "_".join(kept).lower() if kept else "none"

    return {"decision": decision, "pred_type": norm_type(pred_type), "raw": s}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-jsonl", default="datasets/mvtec_anomaly_llm/test.jsonl")
    parser.add_argument("--out-dir", default="outputs/qwen35_full_yn_type_sft")
    parser.add_argument("--model-path", default="/data2/zlt/anomaly_detection_llm/model_card/Qwen3.5-0.8B")
    parser.add_argument("--adapter-path", default=None, help="optional LoRA adapter dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--prompt", default=None)
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
        max_new_tokens=32,
        role="yn_type_sft" if args.adapter_path else "yn_type_zs",
        prompt=prompt,
        model_family="qwen3_5",
        adapter_path=args.adapter_path,
    )

    y_true, y_pred = [], []
    type_true, type_pred = [], []
    details = []
    t0 = time.perf_counter()
    for i, r in enumerate(rows):
        res = client.infer(r["image"])
        gt = int(r["label"])
        gt_type = str(r.get("defect_type") or "none")
        parsed = parse_yn_type(res.raw)
        pred = parsed["decision"]
        y_true.append(gt)
        y_pred.append(pred)
        type_true.append(norm_type(gt_type))
        type_pred.append(parsed["pred_type"])
        details.append(
            {
                "image": r["image"],
                "category": r["category"],
                "gt": gt,
                "gt_defect_type": gt_type,
                "pred": pred,
                "pred_defect_type": parsed["pred_type"],
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

    # defect-type accuracy
    type_acc_all = float(np.mean([a == b for a, b in zip(type_true, type_pred)])) if type_true else 0.0
    ng_idx = [k for k, g in enumerate(y_true) if g == 1]
    type_acc_ng = (
        float(np.mean([type_true[k] == type_pred[k] for k in ng_idx])) if ng_idx else 0.0
    )
    both_ng = [k for k in ng_idx if y_pred[k] == 1]
    type_acc_ng_given_ng = (
        float(np.mean([type_true[k] == type_pred[k] for k in both_ng])) if both_ng else 0.0
    )
    from collections import Counter

    conf = Counter(
        f"{type_true[k]}->{type_pred[k]}" for k in ng_idx
    ).most_common(15)

    report = {
        "prompt": prompt,
        "model": args.model_path,
        "adapter": args.adapter_path,
        "n": int(len(y_true)),
        "gt_ok": int(np.sum(y_true == 0)),
        "gt_ng": int(np.sum(y_true == 1)),
        "decision": {
            "accuracy": float(np.mean(y_true == y_pred)),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "pred_yes": int(np.sum(y_pred == 1)),
            "pred_no": int(np.sum(y_pred == 0)),
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        },
        "defect_type": {
            "acc_all": type_acc_all,
            "acc_gt_ng": type_acc_ng,
            "acc_gt_ng_pred_ng": type_acc_ng_given_ng,
            "n_gt_ng": len(ng_idx),
            "n_both_ng": len(both_ng),
            "top_confusion": conf,
        },
    }

    (out_dir / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "details.jsonl").open("w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    d = report["decision"]
    t = report["defect_type"]
    print("\n===== Yes/No + defect type (full image) =====")
    print(f"Decision: Acc={d['accuracy']:.4f} F1={d['f1']:.4f} P={d['precision']:.4f} R={d['recall']:.4f}")
    print(f"          pred Yes/No={d['pred_yes']}/{d['pred_no']} conf={d['confusion']}")
    print(f"DefectType: acc_all={t['acc_all']:.4f} acc_gt_ng={t['acc_gt_ng']:.4f} "
          f"acc_ng_pred_ng={t['acc_gt_ng_pred_ng']:.4f} (n_ng={t['n_gt_ng']})")
    print(f"Wrote {out_dir / 'result.json'} and {out_dir / 'details.jsonl'}")


if __name__ == "__main__":
    main()
