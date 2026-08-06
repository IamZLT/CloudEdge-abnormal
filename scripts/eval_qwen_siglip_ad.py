#!/usr/bin/env python3
"""Proxy AD eval: Qwen3-VL SigLIP-2 vision encoder + normal-gallery scoring.

Why this is a proxy (not drop-in WinCLIP/PromptAD):
  WinCLIP/PromptAD need a CLIP *joint* image-text space. Qwen3-VL only exposes
  the continued-trained SigLIP-2 *vision* tower; its text path is an LLM, not a
  CLIP text encoder. So we evaluate the visual association branch analogue:
  build a gallery from normal train images and score tests by NN distance.

Env: conda activate base
Example:
  CUDA_VISIBLE_DEVICES=3 python scripts/eval_qwen_siglip_ad.py --shots 0 --device cuda:0
  # shots=0 => use all train/good; shots=k => k-shot few-shot gallery
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
CATS = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def best_f1(labels: np.ndarray, scores: np.ndarray) -> float:
    best = 0.0
    for t in np.quantile(scores, np.linspace(0.05, 0.95, 37)):
        best = max(best, float(f1_score(labels, (scores >= t).astype(int), zero_division=0)))
    return best


@torch.no_grad()
def encode_image(visual, processor, image: Image.Image, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Return L2-normalized global feature [D] (mean-pooled patch tokens)."""
    msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "."}]}]
    inputs = processor.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    pv = inputs["pixel_values"].to(device=device, dtype=dtype)
    grid = inputs["image_grid_thw"].to(device=device)
    out = visual(pv, grid_thw=grid)
    tokens = out[0] if isinstance(out, tuple) else out
    # tokens: [N_tokens, hidden]
    feat = tokens.float().mean(dim=0)
    feat = feat / (feat.norm() + 1e-8)
    return feat.cpu()


def encode_paths(visual, processor, paths: list[Path], device: str, dtype: torch.dtype) -> torch.Tensor:
    feats = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        feats.append(encode_image(visual, processor, img, device, dtype))
    return torch.stack(feats, dim=0) if feats else torch.empty(0)


def score_against_gallery(query: torch.Tensor, gallery: torch.Tensor) -> float:
    """Higher = more anomalous. 1 - max cosine similarity to normal gallery."""
    if gallery.numel() == 0:
        return 0.0
    sims = gallery @ query
    return float(1.0 - sims.max().item())


def eval_category(
    visual,
    processor,
    data_root: Path,
    category: str,
    shots: int,
    seed: int,
    device: str,
    dtype: torch.dtype,
) -> dict:
    train_good = list_images(data_root / category / "train" / "good")
    if not train_good:
        raise FileNotFoundError(f"no train/good for {category}")
    rng = np.random.default_rng(seed)
    if shots and shots > 0:
        idx = rng.choice(len(train_good), size=min(shots, len(train_good)), replace=False)
        gallery_paths = [train_good[i] for i in idx]
    else:
        gallery_paths = train_good

    gallery = encode_paths(visual, processor, gallery_paths, device, dtype)

    labels, scores, paths_out = [], [], []
    test_root = data_root / category / "test"
    for sub in sorted(test_root.iterdir()):
        if not sub.is_dir():
            continue
        y = 0 if sub.name == "good" else 1
        for p in list_images(sub):
            feat = encode_image(visual, processor, Image.open(p).convert("RGB"), device, dtype)
            s = score_against_gallery(feat, gallery)
            labels.append(y)
            scores.append(s)
            paths_out.append(str(p))

    labels_a = np.asarray(labels, dtype=int)
    scores_a = np.asarray(scores, dtype=float)
    auroc = float(roc_auc_score(labels_a, scores_a)) if len(np.unique(labels_a)) > 1 else float("nan")
    f1 = best_f1(labels_a, scores_a)
    return {
        "category": category,
        "n_gallery": len(gallery_paths),
        "n_test": int(len(labels_a)),
        "image_auroc": auroc,
        "f1": f1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data2/zlt/anomaly_detection_llm/model_card/Qwen3-VL-4B-Instruct")
    ap.add_argument("--data-root", default=str(ROOT / "datasets" / "mvtec"))
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--shots", type=int, default=0, help="0=all normal train; k=k-shot")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoConfig, AutoProcessor, Qwen3VLVisionModel

    cats = args.categories or CATS
    data_root = Path(args.data_root)
    device = args.device
    dtype = torch.bfloat16

    print(f"Loading Qwen3-VL vision tower only from {args.model} ...")
    cfg = AutoConfig.from_pretrained(args.model)
    visual = Qwen3VLVisionModel(cfg.vision_config).to(device=device, dtype=dtype)
    # checkpoint keys are model.visual.*; VisionModel expects unprefixed keys
    try:
        from safetensors import safe_open
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit("safetensors required") from exc

    state = {}
    for shard in sorted(Path(args.model).glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("model.visual."):
                    state[k[len("model.visual.") :]] = f.get_tensor(k)
    missing, unexpected = visual.load_state_dict(state, strict=False)
    print(f"loaded visual tensors={len(state)} missing={len(missing)} unexpected={len(unexpected)}")
    visual.eval()
    processor = AutoProcessor.from_pretrained(args.model)
    print(f"Vision params: {sum(p.numel() for p in visual.parameters()) / 1e6:.1f}M")

    rows = []
    for cat in cats:
        print(f"== {cat} ==")
        row = eval_category(visual, processor, data_root, cat, args.shots, args.seed, device, dtype)
        print(
            f"  gallery={row['n_gallery']} test={row['n_test']} "
            f"AUROC={row['image_auroc']:.4f} F1={row['f1']:.4f}"
        )
        rows.append(row)

    mean_auroc = float(np.mean([r["image_auroc"] for r in rows]))
    mean_f1 = float(np.mean([r["f1"] for r in rows]))
    summary = {
        "method": "Qwen3-VL SigLIP-2 visual gallery NN (WinCLIP-visual proxy)",
        "model": args.model,
        "shots": args.shots,
        "mean_image_auroc": mean_auroc,
        "mean_f1": mean_f1,
        "rows": rows,
        "note": (
            "Not a drop-in WinCLIP/PromptAD: those need CLIP text-image alignment. "
            "This measures what the Qwen SigLIP-2 vision tower alone can do with a normal gallery."
        ),
        "reference": {
            "WinCLIP_paper_mvtec_iAUROC": 0.9181,
            "WinCLIP_reimpl_mvtec_iAUROC": 0.7017,
            "PaDiM_edge_B1_iAUROC": 0.9145,
        },
    }

    out = Path(args.out) if args.out else ROOT / "outputs" / "reports" / f"qwen_siglip_ad_shots{args.shots}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = out.with_suffix(".md")
    lines = [
        f"# Qwen SigLIP-2 visual-gallery AD (shots={args.shots})",
        "",
        summary["note"],
        "",
        f"- mean Image-AUROC: **{mean_auroc:.4f}**",
        f"- mean F1: **{mean_f1:.4f}**",
        "",
        "| Category | n_gallery | n_test | Image-AUROC | F1 |",
        "|----------|-----------|--------|-------------|----|",
    ]
    for r in rows:
        lines.append(
            f"| {r['category']} | {r['n_gallery']} | {r['n_test']} | {r['image_auroc']:.4f} | {r['f1']:.4f} |"
        )
    lines += [
        "",
        "## References",
        "- WinCLIP paper MVTec mean i-AUROC ≈ 0.918",
        "- WinCLIP local reimpl ≈ 0.702",
        "- PaDiM edge (B1) ≈ 0.9145",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out} and {md}")
    print(f"MEAN Image-AUROC={mean_auroc:.4f} F1={mean_f1:.4f}")


if __name__ == "__main__":
    main()
