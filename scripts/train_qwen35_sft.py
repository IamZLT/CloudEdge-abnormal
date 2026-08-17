#!/usr/bin/env python3
"""Train Qwen3.5 VLM with LoRA or full decoder FT (new script).

Env: conda activate clip
Examples:
  python scripts/train_qwen35_sft.py --mode lora --train-jsonl ... --output-dir ...
  python scripts/train_qwen35_sft.py --mode full --train-jsonl ... --output-dir ... --lr 2e-5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from transformers import AutoProcessor, Trainer, TrainerCallback, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "train_qwen_vl_lora", ROOT / "scripts" / "train_qwen_vl_lora.py"
)
_tr = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_tr)
OkNgJsonlDataset = _tr.OkNgJsonlDataset
collate_fn = _tr.collate_fn
load_causal_vlm = _tr.load_causal_vlm
load_jsonl = _tr.load_jsonl
maybe_freeze_vision = _tr.maybe_freeze_vision
resolve_model_family = _tr.resolve_model_family


def count_trainable(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def prepare_full_ft(model, freeze_vision: bool = True):
    """Train all non-vision params; optionally freeze vision tower."""
    if freeze_vision:
        maybe_freeze_vision(model)
    # ensure LM / projector params are trainable
    for name, param in model.named_parameters():
        lname = name.lower()
        if any(k in lname for k in ("visual", "vision", "vit")):
            continue
        param.requires_grad = True


def _generate_decision(model, processor, image_path: str, prompt: str, max_new_tokens: int):
    """Run one VLM generation and return the raw text + parsed decision dict."""
    from src.vlm.parse import parse_vlm_json

    pil = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        kw = {"text": [text], "images": image_inputs, "padding": True, "return_tensors": "pt"}
        if video_inputs:
            kw["videos"] = video_inputs
        inputs = processor(**kw)
    except Exception:
        # Fallback without qwen_vl_utils: pass PIL directly
        inputs = processor(
            text=[text],
            images=[pil],
            padding=True,
            return_tensors="pt",
        )
    target = next(model.parameters()).device
    inputs = {k: v.to(target) if hasattr(v, "to") else v for k, v in inputs.items()}

    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    in_len = inputs["input_ids"].shape[-1]
    raw = processor.batch_decode(
        generated[:, in_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return raw, parse_vlm_json(raw)


def run_quick_eval(model, processor, rows: list[dict], max_new_tokens: int) -> dict:
    """Lightweight decision-only eval on a small fixed subset (used during training)."""
    y_true: list[int] = []
    y_pred: list[int] = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for r in rows:
                _, parsed = _generate_decision(model, processor, r["image"], r["prompt"], max_new_tokens)
                gt = int(r["label"])
                pred = 1 if str(parsed.get("decision", "")).upper() == "NG" else 0
                y_true.append(gt)
                y_pred.append(pred)
    finally:
        if was_training:
            model.train()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "n": int(len(y_true)),
        "acc": float(np.mean(y_true == y_pred)) if y_true.size else 0.0,
        "f1": float(f1),
        "prec": float(prec),
        "rec": float(rec),
        "pred_ok": int(np.sum(y_pred == 0)),
        "pred_ng": int(np.sum(y_pred == 1)),
    }


class PeriodicEvalCallback(TrainerCallback):
    """Every `eval_every` optimizer steps, run a quick decision eval on a held-out subset."""

    def __init__(self, model, processor, val_rows, eval_every: int, max_eval: int, max_new_tokens: int = 96):
        self.model = model
        self.processor = processor
        self.val_rows = val_rows
        self.eval_every = max(1, int(eval_every))
        self.max_eval = int(max_eval) if max_eval else None
        self.max_new_tokens = int(max_new_tokens)
        self._last_step = -1

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_every != 0 or state.global_step == self._last_step:
            return control
        self._last_step = state.global_step
        rows = self.val_rows if self.max_eval is None else self.val_rows[: self.max_eval]
        m = run_quick_eval(self.model, self.processor, rows, self.max_new_tokens)
        print(
            f"\n[val @step {state.global_step}] n={m['n']} acc={m['acc']:.3f} f1={m['f1']:.3f} "
            f"P={m['prec']:.3f} R={m['rec']:.3f} pred_ok={m['pred_ok']} pred_ng={m['pred_ng']}\n"
        )
        return control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/qwen35_ft_conditions.yaml"))
    parser.add_argument("--mode", choices=["lora", "full"], required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--train-section", default=None, help="config key override, e.g. train_full_1cat")
    parser.add_argument("--val-jsonl", default=None, help="held-out subset for periodic eval during training")
    parser.add_argument("--eval-every", type=int, default=0, help="steps between quick val evals (0 = off)")
    parser.add_argument("--eval-max", type=int, default=None, help="cap val samples per eval")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_cfg = cfg["model"]
    lora_cfg = cfg.get("lora") or {}

    if args.train_section:
        train_cfg = dict(cfg[args.train_section])
    elif args.mode == "lora":
        train_cfg = dict(cfg.get("train_lora") or {})
    else:
        train_cfg = dict(cfg.get("train_full") or {})

    if args.lr is not None:
        train_cfg["learning_rate"] = args.lr
    if args.epochs is not None:
        train_cfg["num_epochs"] = args.epochs
    if args.grad_accum is not None:
        train_cfg["grad_accum"] = args.grad_accum
    if args.device:
        train_cfg["device"] = args.device

    train_jsonl = Path(args.train_jsonl)
    if not train_jsonl.is_absolute():
        train_jsonl = ROOT / train_jsonl
    if not train_jsonl.exists():
        raise FileNotFoundError(train_jsonl)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_cfg["model_path"]
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }.get(str(model_cfg.get("dtype", "bfloat16")).lower(), torch.bfloat16)
    family = resolve_model_family(model_cfg)

    print(f"[train] mode={args.mode} family={family}")
    print(f"[train] data={train_jsonl}")
    print(f"[train] out={output_dir}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    device = str(train_cfg.get("device", "cuda:0"))
    if device == "auto" or ("," in device and "cuda" in device):
        device_map = "auto"
    elif device.startswith("cuda"):
        device_map = device
    else:
        device_map = None

    model = load_causal_vlm(model_path, model_family=family, dtype=dtype, device_map=device_map)

    if args.mode == "lora":
        from peft import LoraConfig, get_peft_model

        if bool(lora_cfg.get("freeze_vision", True)):
            maybe_freeze_vision(model)
        peft_config = LoraConfig(
            r=int(lora_cfg.get("r", 8)),
            lora_alpha=int(lora_cfg.get("alpha", 16)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(lora_cfg.get("target_modules") or ["q_proj", "v_proj"]),
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        prepare_full_ft(model, freeze_vision=bool(train_cfg.get("freeze_vision", True)))
        tr, tot = count_trainable(model)
        print(f"trainable params: {tr:,} || all params: {tot:,} || trainable%: {100*tr/tot:.4f}")

    rows = load_jsonl(train_jsonl)
    dataset = OkNgJsonlDataset(rows, processor, max_image_size=int(train_cfg.get("max_image_size", 512)))
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id

    use_gc = bool(train_cfg.get("gradient_checkpointing", True))
    if use_gc:
        model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    # Avoid mid-run optimizer checkpoints (full FT ~1.6GB weights + optimizer can fill disk).
    args_tr = TrainingArguments(
        output_dir=str(output_dir / "runs"),
        num_train_epochs=float(train_cfg.get("num_epochs", 1)),
        per_device_train_batch_size=int(train_cfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("grad_accum", 8)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        save_strategy="no",
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        remove_unused_columns=False,
        report_to=[],
        dataloader_num_workers=0,
        gradient_checkpointing=use_gc,
    )
    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=dataset,
        data_collator=lambda feats: collate_fn(feats, pad_token_id=pad_id),
    )

    callbacks = []
    if args.val_jsonl:
        val_path = Path(args.val_jsonl)
        if not val_path.is_absolute():
            val_path = ROOT / val_path
        val_rows = load_jsonl(val_path)
        print(f"[train] periodic val: {val_path} n={len(val_rows)} every={args.eval_every} max={args.eval_max}")
        callbacks.append(
            PeriodicEvalCallback(model, processor, val_rows, args.eval_every, args.eval_max)
        )
    trainer.add_callback(callbacks[0]) if callbacks else None

    trainer.train()

    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    meta = {
        "mode": args.mode,
        "base_model": model_path,
        "model_family": family,
        "output_dir": str(output_dir),
        "train_jsonl": str(train_jsonl),
        "n_train": len(rows),
        "train": train_cfg,
        "lora": lora_cfg if args.mode == "lora" else None,
    }
    (output_dir / "finetune_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[train] saved -> {output_dir}")


if __name__ == "__main__":
    main()
