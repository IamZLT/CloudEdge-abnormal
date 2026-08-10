#!/usr/bin/env python3
"""Compare zero-shot vs LoRA Qwen-VL / Qwen3.5 on SFT holdout set.

Env: conda activate base (Qwen3-VL) or clip (Qwen3.5-0.8B)
Example:
  CUDA_VISIBLE_DEVICES=3 python scripts/eval_qwen_vl_lora.py --config configs/qwen_vl_lora.yaml
  CUDA_VISIBLE_DEVICES=5 python scripts/eval_qwen_vl_lora.py --config configs/qwen35_lora.yaml --max-images 60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vlm import QwenVLClient
from src.vlm.qwen_client import resolve_vlm_model_class


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "n": int(len(y_true)),
    }


class LoRAQwenClient(QwenVLClient):
    """QwenVLClient with PEFT adapter loaded."""

    def __init__(self, model_path: str, adapter_path: str, **kwargs):
        from transformers import AutoProcessor

        self.model_path = str(model_path)
        self.adapter_path = str(adapter_path)
        self.device = kwargs.get("device", "cuda:0")
        self.role = kwargs.get("role", "lora")
        self.max_new_tokens = int(kwargs.get("max_new_tokens", 128))
        self.prompt = kwargs.get("prompt")
        model_family = kwargs.get("model_family")

        dtype_name = str(kwargs.get("dtype", "bfloat16")).lower()
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
        }.get(dtype_name, torch.bfloat16)

        device = self.device
        if str(device).lower() == "auto" or ("," in str(device) and "cuda" in str(device)):
            device_map = "auto"
        elif str(device).startswith("cuda"):
            device_map = device
        else:
            device_map = None

        ModelCls, fam = resolve_vlm_model_class(model_path, model_family)
        self.model_family = fam
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        base = ModelCls.from_pretrained(
            model_path,
            dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()
        self._input_device = next(self.model.parameters()).device


def run_client(client, rows: list[dict], tag: str) -> dict:
    y_true, y_pred = [], []
    details = []
    for i, row in enumerate(rows):
        # use sample prompt to match training
        if getattr(client, "prompt", None) is None or row.get("prompt"):
            client.prompt = row.get("prompt") or client.prompt
        res = client.infer(row["image"])
        pred = 1 if res.decision == "NG" else 0
        y_true.append(int(row["label"]))
        y_pred.append(pred)
        details.append(
            {
                "image": row["image"],
                "category": row.get("category"),
                "gt": int(row["label"]),
                "pred": pred,
                "decision": res.decision,
                "confidence": res.confidence,
                "defect_type": res.defect_type,
                "reason": res.reason,
                "raw": res.raw,
                "latency_ms": res.latency_ms,
                "correct": pred == int(row["label"]),
            }
        )
        print(
            f"[{tag} {i+1}/{len(rows)}] {row.get('category')} "
            f"gt={'NG' if row['label'] else 'OK'} pred={res.decision} "
            f"ok={pred == int(row['label'])} | {res.reason[:50]}"
        )
    return {"metrics": metrics(y_true, y_pred), "details": details}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/qwen_vl_lora.yaml"))
    parser.add_argument("--holdout-jsonl", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-zeroshot", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_root = Path(cfg.get("results_dir", "outputs/qwen_vl_lora"))
    holdout_path = Path(args.holdout_jsonl or (out_root / "sft_holdout.jsonl"))
    adapter = Path(args.adapter or (out_root / "adapter"))
    rows = load_jsonl(holdout_path)
    if args.max_images is not None and args.max_images > 0:
        # balanced-ish subsample
        ok = [r for r in rows if r["label"] == 0]
        ng = [r for r in rows if r["label"] == 1]
        n_ok = min(len(ok), args.max_images // 2)
        n_ng = min(len(ng), args.max_images - n_ok)
        rows = ok[:n_ok] + ng[:n_ng]

    model_cfg = cfg["model"]
    model_path = model_cfg["model_path"]
    model_family = model_cfg.get("model_family")
    device = args.device or cfg.get("train", {}).get("device", "cuda:0")
    prompt = cfg.get("prompt")
    dtype = model_cfg.get("dtype", "bfloat16")

    report = {
        "holdout_path": str(holdout_path),
        "n_eval": len(rows),
        "base_model": model_path,
        "model_family": model_family,
        "adapter": str(adapter) if adapter.exists() else None,
    }

    if not args.skip_zeroshot:
        print("[eval] zero-shot base model")
        zs = QwenVLClient(
            model_path=model_path,
            device=device,
            dtype=dtype,
            max_new_tokens=128,
            role="zeroshot",
            prompt=prompt,
            model_family=model_family,
        )
        report["zeroshot"] = run_client(zs, rows, "zs")
        # free memory before loading adapter
        del zs
        torch.cuda.empty_cache()

    if not adapter.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter}. Train first.")

    print("[eval] LoRA adapter")
    lora = LoRAQwenClient(
        model_path=model_path,
        adapter_path=str(adapter),
        device=device,
        dtype=dtype,
        max_new_tokens=128,
        role="lora",
        prompt=prompt,
        model_family=model_family,
    )
    report["lora"] = run_client(lora, rows, "lora")

    out_json = out_root / "eval_holdout.json"
    out_md = out_root / "eval_holdout.md"
    out_root.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    zs_m = (report.get("zeroshot") or {}).get("metrics")
    lo_m = report["lora"]["metrics"]
    lines = [
        "# Qwen-VL LoRA vs Zero-shot (holdout)",
        "",
        f"- Base: `{model_path}`",
        f"- Family: `{model_family or 'auto'}`",
        f"- Adapter: `{adapter}`",
        f"- N: {len(rows)}",
        "",
        "| Model | Acc | F1 | P | R |",
        "|-------|-----|----|---|---|",
    ]
    if zs_m:
        lines.append(
            f"| Zero-shot | {zs_m['accuracy']:.4f} | {zs_m['f1']:.4f} | {zs_m['precision']:.4f} | {zs_m['recall']:.4f} |"
        )
    lines.append(
        f"| LoRA | {lo_m['accuracy']:.4f} | {lo_m['f1']:.4f} | {lo_m['precision']:.4f} | {lo_m['recall']:.4f} |"
    )
    if zs_m:
        lines += ["", f"- ΔF1 (LoRA - Zero-shot): **{lo_m['f1'] - zs_m['f1']:+.4f}**", ""]

    # Cross-model reference (4B / 8B historical holdout N=60)
    lines += [
        "## Reference (same protocol, N=60)",
        "",
        "| Model | ZS Acc | ZS F1 | LoRA Acc | LoRA F1 |",
        "|-------|--------|-------|----------|---------|",
        "| Qwen3-VL-4B | 0.60 | 0.33 | 0.83 | 0.85 |",
        "| Qwen3-VL-8B | 0.68 | 0.54 | 0.85 | 0.85 |",
    ]
    if zs_m:
        lines.append(
            f"| Qwen3.5-0.8B | {zs_m['accuracy']:.2f} | {zs_m['f1']:.2f} | "
            f"{lo_m['accuracy']:.2f} | {lo_m['f1']:.2f} |"
        )
    else:
        lines.append(
            f"| Qwen3.5-0.8B | — | — | {lo_m['accuracy']:.2f} | {lo_m['f1']:.2f} |"
        )
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
