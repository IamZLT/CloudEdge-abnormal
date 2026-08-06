#!/usr/bin/env python3
"""LoRA SFT Qwen3-VL for OK/NG JSON defect inspection.

Env: conda activate base
Prereq: python scripts/build_vlm_sft_data.py --config configs/qwen_vl_lora.yaml

Example:
  CUDA_VISIBLE_DEVICES=3 python scripts/train_qwen_vl_lora.py --config configs/qwen_vl_lora.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class OkNgJsonlDataset(Dataset):
    def __init__(self, rows: list[dict], processor, max_image_size: int = 512):
        self.rows = rows
        self.processor = processor
        self.max_image_size = max_image_size

    def __len__(self):
        return len(self.rows)

    def _resize(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = min(self.max_image_size / max(w, h), 1.0)
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BICUBIC)
        return img

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = self._resize(Image.open(row["image"]).convert("RGB"))
        prompt = row["prompt"]
        response = row["response"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            },
        ]

        # full conversation text (with assistant answer)
        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # prompt-only text (to know where to mask labels)
        prompt_text = self.processor.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )

        full_inputs = self.processor(
            text=[full_text],
            images=[image],
            padding=False,
            return_tensors="pt",
        )
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=[image],
            padding=False,
            return_tensors="pt",
        )

        input_ids = full_inputs["input_ids"][0]
        attention_mask = full_inputs["attention_mask"][0]
        prompt_len = int(prompt_inputs["input_ids"].shape[-1])
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        # keep vision tensors if present
        for k, v in full_inputs.items():
            if k in {"input_ids", "attention_mask"}:
                continue
            item[k] = v[0] if hasattr(v, "dim") and v.dim() > 0 and v.shape[0] == 1 else v
        return item


def collate_fn(features: list[dict], pad_token_id: int):
    # pad sequence fields
    keys = set().union(*[f.keys() for f in features])
    batch = {}
    seq_keys = {"input_ids", "attention_mask", "labels"}
    max_len = max(f["input_ids"].shape[-1] for f in features)

    for k in seq_keys:
        if k not in keys:
            continue
        pad_val = -100 if k == "labels" else (pad_token_id if k == "input_ids" else 0)
        tensors = []
        for f in features:
            t = f[k]
            if t.shape[-1] < max_len:
                pad = torch.full((max_len - t.shape[-1],), pad_val, dtype=t.dtype)
                t = torch.cat([t, pad], dim=-1)
            tensors.append(t)
        batch[k] = torch.stack(tensors, dim=0)

    # stack other tensors when shapes match; else list
    for k in keys - seq_keys:
        vals = [f[k] for f in features if k in f]
        if not vals:
            continue
        if torch.is_tensor(vals[0]):
            try:
                batch[k] = torch.stack(vals, dim=0)
            except RuntimeError:
                batch[k] = vals
        else:
            batch[k] = vals
    return batch


def maybe_freeze_vision(model):
    for name, param in model.named_parameters():
        lname = name.lower()
        if any(k in lname for k in ("visual", "vision", "vit")):
            param.requires_grad = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/qwen_vl_lora.yaml"))
    parser.add_argument("--train-jsonl", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_root = Path(cfg.get("results_dir", "outputs/qwen_vl_lora"))
    train_jsonl = Path(args.train_jsonl or (out_root / "sft_train.jsonl"))
    if not train_jsonl.exists():
        raise FileNotFoundError(f"Missing {train_jsonl}. Run build_vlm_sft_data.py first.")

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["train"]
    model_path = model_cfg["model_path"]
    output_dir = Path(args.output_dir or (out_root / "adapter"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] model={model_path}")
    print(f"[train] data={train_jsonl}")
    print(f"[train] out={output_dir}")

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }.get(str(model_cfg.get("dtype", "bfloat16")).lower(), torch.bfloat16)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    device = str(train_cfg.get("device", "auto"))
    if device == "auto" or ("," in device and "cuda" in device):
        device_map = "auto"
    elif device.startswith("cuda"):
        device_map = device
    else:
        device_map = None

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
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

    rows = load_jsonl(train_jsonl)
    dataset = OkNgJsonlDataset(rows, processor, max_image_size=int(train_cfg.get("max_image_size", 512)))
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id

    use_gc = bool(train_cfg.get("gradient_checkpointing", False))
    if use_gc:
        model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    args_tr = TrainingArguments(
        output_dir=str(output_dir / "runs"),
        num_train_epochs=float(train_cfg.get("num_epochs", 1)),
        per_device_train_batch_size=int(train_cfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("grad_accum", 8)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-4)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        save_steps=int(train_cfg.get("save_steps", 200)),
        save_total_limit=2,
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
    trainer.train()

    # save adapter + processor config pointer
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    meta = {
        "base_model": model_path,
        "adapter_dir": str(output_dir),
        "train_jsonl": str(train_jsonl),
        "lora": lora_cfg,
        "train": train_cfg,
        "n_train": len(rows),
    }
    (output_dir / "finetune_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[train] saved adapter -> {output_dir}")


if __name__ == "__main__":
    main()
