#!/usr/bin/env python3
"""ROI short-label LoRA + edge-gated holdout bench for Qwen3.5-0.8B.

Pipeline:
  1) build ROI SFT crops (if missing)
  2) LoRA train on ROI short JSON
  3) eval modes on balanced holdout:
       - edge_only
       - roi_vlm (always VLM on top1 crop)
       - gated   (edge outside band; VLM inside hard_margin)

Env: conda activate clip
  CUDA_VISIBLE_DEVICES=3 python scripts/run_qwen35_roi_gated.py \
    --config configs/qwen35_roi_gated.yaml
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge.methods.gallery_ad import best_f1_threshold, mvtec_train_good
from edge.methods.patch_gallery_ad import PatchGalleryAD
from edge.methods.roi_crops import amap_to_rois
from src.vlm import QwenVLClient


def calibrate_thr_on_train(
    ad: PatchGalleryAD,
    train_rows: list[dict],
) -> float:
    """Pick score thr by best-F1 on train split only (no holdout leak).

    Falls back to high quantile of OK scores when best-F1 collapses to ~0
    (common if gallery self-distance is tiny and all scores > 0 look NG).
    """
    scores, labels = [], []
    for r in train_rows:
        s, _ = ad.score_image(Image.open(r["image"]).convert("RGB"))
        scores.append(float(s))
        labels.append(int(r["label"]))
    if not scores:
        return 0.5
    arr_s = np.asarray(scores, dtype=np.float64)
    arr_y = np.asarray(labels, dtype=np.int64)
    f1, _p, _r, thr = best_f1_threshold(arr_y, arr_s)
    thr = float(thr)
    ok_scores = arr_s[arr_y == 0]
    if thr <= 1e-6 and ok_scores.size:
        # Use OK high-quantile so clear OK stay OK; uncertain/high → band/VLM
        thr = float(np.quantile(ok_scores, 0.9))
        print(f"[cache] thr fallback ok_q90={thr:.4f} (best_f1 thr~0, f1={f1:.3f})")
    else:
        print(f"[cache] thr train_best_f1={thr:.4f} (f1={f1:.3f})")
    return thr


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


def build_edge_ad(cfg: dict, device: str) -> PatchGalleryAD:
    from edge.methods.encoders import load_qwen35_vision_encoder

    edge = cfg.get("edge") or {}
    _, encode_patches, _ = load_qwen35_vision_encoder(
        edge["model_path"],
        device=device,
        max_pixels=int(edge.get("max_pixels") or 50176),
        layers=edge.get("layers"),
    )
    return PatchGalleryAD(
        encode_patches,
        device=device,
        name="roi_gated_eval",
        fusion_temperature=float(edge.get("fusion_temp") or 0.5),
    )


def cache_holdout_edge(
    rows: list[dict],
    cfg: dict,
    device: str,
    out_dir: Path,
) -> list[dict]:
    data_root = Path(cfg.get("data_root") or "datasets/mvtec_anomaly_detection")
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    edge = cfg.get("edge") or {}
    roi_cfg = cfg.get("roi") or {}
    seed = int(cfg.get("seed") or 42)
    max_gallery = int(edge.get("max_gallery") or 16)
    top_k = int(roi_cfg.get("top_k") or 1)
    pad_ratio = float(roi_cfg.get("pad_ratio") or 0.4)
    min_side = int(roi_cfg.get("min_side") or 96)

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[str(r.get("category") or "unknown")].append(r)

    ad = build_edge_ad(cfg, device)
    caches = []
    crop_dir = out_dir / "eval_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    # optional train rows for thr calibration (same seed split as SFT train)
    train_src = Path(cfg.get("source_train_jsonl") or "")
    if train_src and not train_src.is_absolute():
        train_src = ROOT / train_src
    train_by_cat: dict[str, list[dict]] = defaultdict(list)
    if train_src.exists():
        for r in load_jsonl(train_src):
            train_by_cat[str(r.get("category") or "unknown")].append(r)

    for cat, cat_rows in by_cat.items():
        train = mvtec_train_good(data_root, cat)
        if len(train) > max_gallery:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(train), size=max_gallery, replace=False)
            train = [train[i] for i in sorted(idx)]
        ad.build_gallery(train, seed=seed)
        if train_by_cat.get(cat):
            thr = calibrate_thr_on_train(ad, train_by_cat[cat])
            thr_src = "train_best_f1"
        else:
            g_scores = [float(ad.score_image(Image.open(p).convert("RGB"))[0]) for p in train]
            thr = float(np.quantile(g_scores, float(edge.get("thr_quantile") or 0.95)))
            thr_src = "gallery_q"
            # gallery self-scores can collapse to ~0; fall back to mid of query scores later if needed
            if thr <= 1e-6:
                thr = float(np.median(g_scores) + 1e-3)
        print(f"[cache] {cat}: gallery={len(train)} thr={thr:.4f} ({thr_src}) n={len(cat_rows)}")

        for r in cat_rows:
            img_path = Path(r["image"])
            pil = Image.open(img_path).convert("RGB")
            score, amap = ad.score_image(pil)
            rois = amap_to_rois(pil, amap, top_k=top_k, pad_ratio=pad_ratio, min_side=min_side)
            crop_path = None
            if rois:
                crop_path = crop_dir / f"{cat}_{img_path.stem}_top1.jpg"
                rois[0].crop.save(crop_path, quality=92)
            caches.append(
                {
                    "image": str(img_path),
                    "crop": str(crop_path) if crop_path else None,
                    "category": cat,
                    "label": int(r["label"]),
                    "edge_score": float(score),
                    "edge_thr": thr,
                    "roi_box": list(rois[0].box) if rois else None,
                }
            )

    del ad
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    path = out_dir / "eval_edge_cache.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for e in caches:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"[cache] wrote {path} n={len(caches)}")
    return caches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/qwen35_roi_gated.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out_dir = Path(cfg.get("results_dir") or "outputs/qwen35_roi_gated")
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or cfg.get("device") or "cuda:0"

    train_jsonl = out_dir / "roi_sft_train.jsonl"
    if not args.skip_build and not train_jsonl.exists():
        cmd = [
            args.python,
            str(ROOT / "scripts" / "build_roi_short_sft.py"),
            "--config",
            str(cfg_path),
            "--device",
            device,
            "--split",
            "train",
        ]
        print("[build]", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(ROOT))
    elif not train_jsonl.exists():
        raise FileNotFoundError(train_jsonl)

    adapter = out_dir / "adapter"
    if not args.skip_train and not (adapter / "adapter_model.safetensors").exists():
        # write a tiny train overlay config for train_qwen35_sft
        train_cfg_path = out_dir / "train_overlay.yaml"
        overlay = {
            "model": cfg["model"],
            "lora": cfg["lora"],
            "train_lora": {**cfg["train"], "device": device},
        }
        train_cfg_path.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
        cmd = [
            args.python,
            str(ROOT / "scripts" / "train_qwen35_sft.py"),
            "--config",
            str(train_cfg_path),
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
        print("[train]", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(ROOT))
    elif not (adapter / "adapter_model.safetensors").exists():
        raise FileNotFoundError(adapter)

    if args.skip_eval:
        print("[eval] skipped")
        return

    hold_src = Path(cfg["source_holdout_jsonl"])
    if not hold_src.is_absolute():
        hold_src = ROOT / hold_src
    max_images = args.max_images
    if max_images is None:
        max_images = int((cfg.get("eval") or {}).get("max_images") or 60)
    rows = subsample_balanced(load_jsonl(hold_src), max_images)

    cache_path = out_dir / "eval_edge_cache.jsonl"
    if cache_path.exists():
        caches = load_jsonl(cache_path)
        # keep only selected rows by source image
        want = {r["image"] for r in rows}
        caches = [c for c in caches if c["image"] in want]
        if len(caches) != len(rows):
            caches = cache_holdout_edge(rows, cfg, device, out_dir)
    else:
        caches = cache_holdout_edge(rows, cfg, device, out_dir)

    margin = float((cfg.get("edge") or {}).get("hard_margin") or 0.04)
    prompt = cfg.get("prompt_roi") or ""
    model_cfg = cfg["model"]

    print(f"[eval] loading ROI LoRA adapter {adapter}")
    client = QwenVLClient(
        model_path=model_cfg["model_path"],
        device=device,
        dtype=model_cfg.get("dtype", "bfloat16"),
        max_new_tokens=64,
        role="roi_gated",
        prompt=prompt,
        model_family=model_cfg.get("model_family"),
        adapter_path=str(adapter),
    )

    modes = {
        "edge_only": [],
        "roi_vlm": [],
        "gated": [],
    }
    details = {k: [] for k in modes}

    for i, e in enumerate(caches):
        gt = int(e["label"])
        score = float(e["edge_score"])
        thr = float(e["edge_thr"])
        edge_pred = 1 if score >= thr else 0
        band = abs(score - thr) < margin

        # ROI VLM
        if e.get("crop"):
            client.prompt = prompt
            res = client.infer(e["crop"])
            vlm_pred = 1 if res.decision == "NG" else 0
            vlm_raw = res.raw
            vlm_dec = res.decision
            vlm_reason = res.reason
        else:
            vlm_pred = edge_pred
            vlm_raw = ""
            vlm_dec = "OK" if edge_pred == 0 else "NG"
            vlm_reason = "no_crop_fallback_edge"

        gated_pred = vlm_pred if band else edge_pred
        gated_src = "vlm" if band else "edge"

        for name, pred in (
            ("edge_only", edge_pred),
            ("roi_vlm", vlm_pred),
            ("gated", gated_pred),
        ):
            modes[name].append((gt, pred))
            details[name].append(
                {
                    "image": e["image"],
                    "category": e["category"],
                    "gt": gt,
                    "pred": pred,
                    "edge_score": score,
                    "edge_thr": thr,
                    "band": band,
                    "edge_pred": edge_pred,
                    "vlm_pred": vlm_pred,
                    "vlm_decision": vlm_dec,
                    "vlm_reason": vlm_reason,
                    "vlm_raw": vlm_raw,
                    "gated_src": gated_src if name == "gated" else None,
                    "correct": pred == gt,
                }
            )
        print(
            f"[eval {i+1}/{len(caches)}] {e['category']} gt={'NG' if gt else 'OK'} "
            f"edge={'NG' if edge_pred else 'OK'} vlm={vlm_dec} "
            f"gated={'NG' if gated_pred else 'OK'}({gated_src}) band={band} "
            f"score={score:.3f} thr={thr:.3f}"
        )

    del client
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    report = {
        "n_eval": len(caches),
        "hard_margin": margin,
        "adapter": str(adapter),
        "modes": {},
    }
    for name, pairs in modes.items():
        y_true = [a for a, _ in pairs]
        y_pred = [b for _, b in pairs]
        report["modes"][name] = {
            "metrics": metrics(y_true, y_pred),
            "details": details[name],
            "n_band" if name == "gated" else "n": (
                sum(1 for d in details["gated"] if d["band"]) if name == "gated" else len(pairs)
            ),
        }
    if "gated" in report["modes"]:
        report["modes"]["gated"]["n_band"] = sum(1 for d in details["gated"] if d["band"])
        report["modes"]["gated"]["n_edge"] = sum(1 for d in details["gated"] if not d["band"])

    out_json = out_dir / "bench_roi_gated.json"
    out_md = out_dir / "bench_roi_gated.md"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Edge-gated ROI short-label VLM (Qwen3.5-0.8B)",
        "",
        f"- N: {len(caches)} (balanced holdout)",
        f"- Adapter: `{adapter}`",
        f"- Gate band: |score - thr| < {margin}",
        "",
        "| Mode | Acc | F1 | P | R | pred OK/NG |",
        "|------|-----|----|---|---|------------|",
    ]
    for name in ("edge_only", "roi_vlm", "gated"):
        m = report["modes"][name]["metrics"]
        lines.append(
            f"| {name} | {m['accuracy']:.4f} | {m['f1']:.4f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['pred_ok']}/{m['pred_ng']} |"
        )
    g = report["modes"]["gated"]
    lines += [
        "",
        f"- Gated calls: edge={g.get('n_edge')} / vlm_band={g.get('n_band')}",
        f"- ΔF1 (gated − edge_only): "
        f"**{g['metrics']['f1'] - report['modes']['edge_only']['metrics']['f1']:+.4f}**",
        f"- ΔF1 (gated − roi_vlm): "
        f"**{g['metrics']['f1'] - report['modes']['roi_vlm']['metrics']['f1']:+.4f}**",
        "",
        "## Notes",
        "",
        "- `edge_only`: patch-gallery score vs gallery quantile thr",
        "- `roi_vlm`: always ROI crop → short-label LoRA VLM",
        "- `gated`: trust edge outside band; VLM only on uncertain band",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
