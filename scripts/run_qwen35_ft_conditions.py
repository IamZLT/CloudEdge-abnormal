#!/usr/bin/env python3
"""Run Qwen3.5-0.8B FT condition matrix and write a comparison report.

Conditions (default):
  zs_5cat         — zero-shot on 5-cat holdout N=60
  lora_5cat_ref   — reuse outputs/qwen35_lora/adapter
  full_5cat       — full decoder FT on 5-cat train
  lora_screw      — LoRA on screw-only train
  full_screw      — full decoder FT on screw-only train

Env: conda activate clip
  CUDA_VISIBLE_DEVICES=3 python scripts/run_qwen35_ft_conditions.py \
    --config configs/qwen35_ft_conditions.yaml
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vlm import QwenVLClient


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "n": int(len(y_true)),
        "pred_ok": int(sum(1 for p in y_pred if p == 0)),
        "pred_ng": int(sum(1 for p in y_pred if p == 1)),
    }


def subsample_balanced(rows: list[dict], max_images: int | None) -> list[dict]:
    if max_images is None or max_images <= 0:
        return list(rows)
    ok = [r for r in rows if int(r["label"]) == 0]
    ng = [r for r in rows if int(r["label"]) == 1]
    n_ok = min(len(ok), max_images // 2)
    n_ng = min(len(ng), max_images - n_ok)
    return ok[:n_ok] + ng[:n_ng]


def filter_category(rows: list[dict], category: str) -> list[dict]:
    return [r for r in rows if r.get("category") == category]


def prepare_data(cfg: dict, out_root: Path) -> dict[str, Path]:
    src_train = Path(cfg["source_train_jsonl"])
    src_hold = Path(cfg["source_holdout_jsonl"])
    if not src_train.is_absolute():
        src_train = ROOT / src_train
    if not src_hold.is_absolute():
        src_hold = ROOT / src_hold
    cat = str(cfg.get("single_category") or "screw")

    train_all = load_jsonl(src_train)
    hold_all = load_jsonl(src_hold)
    train_1 = filter_category(train_all, cat)
    hold_1 = filter_category(hold_all, cat)

    paths = {
        "train_5cat": out_root / "data" / "sft_train_5cat.jsonl",
        "hold_5cat": out_root / "data" / "sft_holdout_5cat.jsonl",
        "train_1cat": out_root / "data" / f"sft_train_{cat}.jsonl",
        "hold_1cat": out_root / "data" / f"sft_holdout_{cat}.jsonl",
    }
    write_jsonl(paths["train_5cat"], train_all)
    write_jsonl(paths["hold_5cat"], hold_all)
    write_jsonl(paths["train_1cat"], train_1)
    write_jsonl(paths["hold_1cat"], hold_1)
    meta = {
        "n_train_5cat": len(train_all),
        "n_hold_5cat": len(hold_all),
        f"n_train_{cat}": len(train_1),
        f"n_hold_{cat}": len(hold_1),
        "single_category": cat,
    }
    (out_root / "data" / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[data] {meta}")
    return paths


def run_train(
    *,
    mode: str,
    train_jsonl: Path,
    output_dir: Path,
    config: Path,
    device: str,
    train_section: str,
    python: str,
) -> None:
    if (output_dir / "finetune_meta.json").exists() and (
        (output_dir / "adapter_model.safetensors").exists()
        or (output_dir / "model.safetensors").exists()
        or any(output_dir.glob("model-*.safetensors"))
        or (output_dir / "adapter_config.json").exists()
    ):
        print(f"[train] skip existing {output_dir}")
        return
    cmd = [
        python,
        str(ROOT / "scripts" / "train_qwen35_sft.py"),
        "--config",
        str(config),
        "--mode",
        mode,
        "--train-jsonl",
        str(train_jsonl),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--train-section",
        train_section,
    ]
    print("[train]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def eval_condition(
    *,
    name: str,
    rows: list[dict],
    model_path: str,
    adapter_path: str | None,
    model_family: str | None,
    dtype: str,
    device: str,
    prompt: str,
) -> dict:
    print(f"[eval] {name} n={len(rows)} model={model_path} adapter={adapter_path}")
    client = QwenVLClient(
        model_path=model_path,
        device=device,
        dtype=dtype,
        max_new_tokens=128,
        role=name,
        prompt=prompt,
        model_family=model_family,
        adapter_path=adapter_path,
    )
    y_true, y_pred = [], []
    details = []
    for i, row in enumerate(rows):
        if row.get("prompt"):
            client.prompt = row["prompt"]
        else:
            client.prompt = prompt
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
                "reason": res.reason,
                "raw": res.raw,
                "correct": pred == int(row["label"]),
            }
        )
        print(
            f"[{name} {i+1}/{len(rows)}] {row.get('category')} "
            f"gt={'NG' if row['label'] else 'OK'} pred={res.decision} "
            f"ok={pred == int(row['label'])}"
        )
    del client
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"metrics": metrics(y_true, y_pred), "details": details}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/qwen35_ft_conditions.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--only", default=None, help="comma-separated condition ids")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out_root = Path(cfg.get("results_dir") or "outputs/qwen35_ft_conditions")
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    device = args.device or cfg.get("device") or "cuda:0"
    model_cfg = cfg["model"]
    prompt = cfg.get("prompt") or ""
    cat = str(cfg.get("single_category") or "screw")
    conds = list(cfg.get("conditions") or [])
    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        conds = [c for c in conds if c in want]

    paths = prepare_data(cfg, out_root)
    ckpt = {
        "full_5cat": out_root / "ckpts" / "full_5cat",
        "lora_screw": out_root / "ckpts" / "lora_screw",
        "full_screw": out_root / "ckpts" / f"full_{cat}",
    }
    ref_lora = Path(cfg.get("ref_lora_5cat") or "outputs/qwen35_lora/adapter")
    if not ref_lora.is_absolute():
        ref_lora = ROOT / ref_lora

    if not args.skip_train:
        if "full_5cat" in conds:
            run_train(
                mode="full",
                train_jsonl=paths["train_5cat"],
                output_dir=ckpt["full_5cat"],
                config=cfg_path,
                device=device,
                train_section="train_full",
                python=args.python,
            )
        if "lora_screw" in conds or f"lora_{cat}" in conds:
            run_train(
                mode="lora",
                train_jsonl=paths["train_1cat"],
                output_dir=ckpt["lora_screw"],
                config=cfg_path,
                device=device,
                train_section="train_lora_1cat",
                python=args.python,
            )
        if "full_screw" in conds or f"full_{cat}" in conds:
            run_train(
                mode="full",
                train_jsonl=paths["train_1cat"],
                output_dir=ckpt["full_screw"],
                config=cfg_path,
                device=device,
                train_section="train_full_1cat",
                python=args.python,
            )

    if args.skip_eval:
        print("[eval] skipped")
        return

    eval_cfg = cfg.get("eval") or {}
    rows_5 = subsample_balanced(load_jsonl(paths["hold_5cat"]), int(eval_cfg.get("max_images_5cat") or 60))
    rows_1 = subsample_balanced(load_jsonl(paths["hold_1cat"]), int(eval_cfg.get("max_images_1cat") or 32))

    report = {
        "device": device,
        "single_category": cat,
        "n_eval_5cat": len(rows_5),
        "n_eval_1cat": len(rows_1),
        "conditions": {},
    }

    base = model_cfg["model_path"]
    fam = model_cfg.get("model_family")
    dtype = model_cfg.get("dtype", "bfloat16")

    def _save_partial():
        (out_root / "condition_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if "zs_5cat" in conds:
        report["conditions"]["zs_5cat"] = {
            "scope": "5cat",
            **eval_condition(
                name="zs_5cat",
                rows=rows_5,
                model_path=base,
                adapter_path=None,
                model_family=fam,
                dtype=dtype,
                device=device,
                prompt=prompt,
            ),
        }
        _save_partial()

    if "lora_5cat_ref" in conds:
        report["conditions"]["lora_5cat_ref"] = {
            "scope": "5cat",
            "adapter": str(ref_lora),
            **eval_condition(
                name="lora_5cat_ref",
                rows=rows_5,
                model_path=base,
                adapter_path=str(ref_lora),
                model_family=fam,
                dtype=dtype,
                device=device,
                prompt=prompt,
            ),
        }
        _save_partial()

    if "full_5cat" in conds:
        report["conditions"]["full_5cat"] = {
            "scope": "5cat",
            "ckpt": str(ckpt["full_5cat"]),
            **eval_condition(
                name="full_5cat",
                rows=rows_5,
                model_path=str(ckpt["full_5cat"]),
                adapter_path=None,
                model_family=fam,
                dtype=dtype,
                device=device,
                prompt=prompt,
            ),
        }
        _save_partial()

    if "zs_screw" in conds or f"zs_{cat}" in conds:
        key = "zs_screw" if "zs_screw" in conds else f"zs_{cat}"
        report["conditions"][key] = {
            "scope": "1cat",
            "category": cat,
            **eval_condition(
                name=key,
                rows=rows_1,
                model_path=base,
                adapter_path=None,
                model_family=fam,
                dtype=dtype,
                device=device,
                prompt=prompt,
            ),
        }
        _save_partial()

    if "lora_screw" in conds or f"lora_{cat}" in conds:
        key = "lora_screw" if "lora_screw" in conds else f"lora_{cat}"
        report["conditions"][key] = {
            "scope": "1cat",
            "category": cat,
            "adapter": str(ckpt["lora_screw"]),
            **eval_condition(
                name=key,
                rows=rows_1,
                model_path=base,
                adapter_path=str(ckpt["lora_screw"]),
                model_family=fam,
                dtype=dtype,
                device=device,
                prompt=prompt,
            ),
        }
        _save_partial()

    if "full_screw" in conds or f"full_{cat}" in conds:
        key = "full_screw" if "full_screw" in conds else f"full_{cat}"
        report["conditions"][key] = {
            "scope": "1cat",
            "category": cat,
            "ckpt": str(ckpt["full_screw"]),
            **eval_condition(
                name=key,
                rows=rows_1,
                model_path=str(ckpt["full_screw"]),
                adapter_path=None,
                model_family=fam,
                dtype=dtype,
                device=device,
                prompt=prompt,
            ),
        }
        _save_partial()

    # also eval 1-cat models on 5-cat to see specialization collapse (optional quick)
    # skip for time unless present

    lines = [
        "# Qwen3.5-0.8B FT condition comparison",
        "",
        f"- Device: `{device}`",
        f"- 5-cat eval N: {len(rows_5)} (balanced holdout)",
        f"- 1-cat (`{cat}`) eval N: {len(rows_1)} (balanced holdout)",
        "",
        "| Condition | Scope | Acc | F1 | P | R | pred OK/NG |",
        "|-----------|-------|-----|----|---|---|------------|",
    ]
    for name, block in report["conditions"].items():
        m = block["metrics"]
        lines.append(
            f"| {name} | {block.get('scope')} | {m['accuracy']:.4f} | {m['f1']:.4f} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['pred_ok']}/{m['pred_ng']} |"
        )
    lines += [
        "",
        "## Setup",
        "",
        "- `full_*`: freeze vision, train all other params, lr=2e-5",
        "- `lora_*`: r=8 α=16, lr=1e-4",
        "- `*_screw`: train only screw split; eval on screw holdout",
        "- `lora_5cat_ref`: previous adapter under `outputs/qwen35_lora/adapter`",
        "",
        f"JSON: `{out_root / 'condition_report.json'}`",
        "",
    ]
    md = out_root / "condition_report.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    _save_partial()
    print("\n" + "\n".join(lines))
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
