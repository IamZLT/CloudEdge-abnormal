#!/usr/bin/env python3
"""v2 ROI short-label LoRA: inject edge_score into prompt + improved gate.

Reuses v1 crop files; rewrites JSONL prompts. New outputs under
outputs/qwen35_roi_gated_v2/ (does not overwrite v1).

Env: conda activate clip
  CUDA_VISIBLE_DEVICES=3 python scripts/run_qwen35_roi_gated_v2.py \
    --config configs/qwen35_roi_gated_v2.yaml
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


def edge_hint(score: float, thr: float, margin: float) -> str:
    if score < thr - margin:
        return "likely_OK"
    if score > thr + margin:
        return "likely_NG"
    return "uncertain"


def make_prompt(tmpl: str, row: dict, thr: float, margin: float) -> str:
    score = float(row.get("edge_score") or 0.0)
    roi = float(row.get("roi_score") or score)
    return tmpl.format(
        category=row.get("category") or "unknown",
        edge_score=score,
        edge_thr=float(thr),
        edge_hint=edge_hint(score, thr, margin),
        crop_kind=row.get("crop_kind") or "top1",
        roi_score=roi,
    )


def calibrate_thr_from_scores(labels: list[int], scores: list[float]) -> float:
    arr_s = np.asarray(scores, dtype=np.float64)
    arr_y = np.asarray(labels, dtype=np.int64)
    f1, _p, _r, thr = best_f1_threshold(arr_y, arr_s)
    thr = float(thr)
    ok = arr_s[arr_y == 0]
    if thr <= 1e-6 and ok.size:
        thr = float(np.quantile(ok, 0.9))
        print(f"[thr] fallback ok_q90={thr:.4f} (best_f1~0, f1={f1:.3f})")
    else:
        print(f"[thr] train_best_f1={thr:.4f} f1={f1:.3f}")
    return thr


def build_v2_train(cfg: dict, out_dir: Path) -> Path:
    """Rewrite v1 ROI jsonl with score-aware prompts + per-category thr."""
    v1 = Path(cfg["v1_train_jsonl"])
    if not v1.is_absolute():
        v1 = ROOT / v1
    rows = load_jsonl(v1)
    margin = float((cfg.get("edge") or {}).get("hard_margin") or 0.05)
    tmpl = cfg.get("prompt_template") or ""

    # per-category thr from v1 top1 rows (image-level scores)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("crop_kind") == "top1":
            by_cat[str(r.get("category"))].append(r)
    thr_map = {}
    for cat, cat_rows in by_cat.items():
        # unique source images
        seen = {}
        for r in cat_rows:
            seen[r["source_image"]] = r
        uniq = list(seen.values())
        thr_map[cat] = calibrate_thr_from_scores(
            [int(r["label"]) for r in uniq],
            [float(r["edge_score"]) for r in uniq],
        )

    out_rows = []
    for r in rows:
        cat = str(r.get("category"))
        thr = thr_map.get(cat, 0.2)
        nr = dict(r)
        nr["edge_thr"] = thr
        nr["prompt"] = make_prompt(tmpl, nr, thr, margin)
        out_rows.append(nr)

    out_path = out_dir / "roi_sft_train_v2.jsonl"
    write_jsonl(out_path, out_rows)
    meta = {
        "n": len(out_rows),
        "ok": sum(1 for r in out_rows if r["label"] == 0),
        "ng": sum(1 for r in out_rows if r["label"] == 1),
        "thr_map": thr_map,
        "source": str(v1),
    }
    (out_dir / "roi_sft_meta_v2.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[build-v2] {meta}")
    return out_path


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
        name="roi_gated_v2",
        fusion_temperature=float(edge.get("fusion_temp") or 0.5),
    )


def cache_eval(rows: list[dict], cfg: dict, device: str, out_dir: Path) -> list[dict]:
    data_root = Path(cfg.get("data_root") or "datasets/mvtec_anomaly_detection")
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    edge = cfg.get("edge") or {}
    roi_cfg = cfg.get("roi") or {}
    seed = int(cfg.get("seed") or 42)
    max_gallery = int(edge.get("max_gallery") or 16)
    margin = float(edge.get("hard_margin") or 0.05)
    tmpl = cfg.get("prompt_template") or ""

    train_src = Path(cfg["source_train_jsonl"])
    if not train_src.is_absolute():
        train_src = ROOT / train_src
    train_by = defaultdict(list)
    for r in load_jsonl(train_src):
        train_by[str(r.get("category"))].append(r)

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[str(r.get("category"))].append(r)

    ad = build_edge_ad(cfg, device)
    crop_dir = out_dir / "eval_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    caches = []

    for cat, cat_rows in by_cat.items():
        gallery = mvtec_train_good(data_root, cat)
        if len(gallery) > max_gallery:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(gallery), size=max_gallery, replace=False)
            gallery = [gallery[i] for i in sorted(idx)]
        ad.build_gallery(gallery, seed=seed)

        # thr on train SFT images
        tr = train_by.get(cat) or []
        scores, labels = [], []
        for r in tr:
            s, _ = ad.score_image(Image.open(r["image"]).convert("RGB"))
            scores.append(float(s))
            labels.append(int(r["label"]))
        thr = calibrate_thr_from_scores(labels, scores) if scores else 0.2
        print(f"[cache] {cat} thr={thr:.4f} n_eval={len(cat_rows)}")

        for r in cat_rows:
            img_path = Path(r["image"])
            pil = Image.open(img_path).convert("RGB")
            score, amap = ad.score_image(pil)
            rois = amap_to_rois(
                pil,
                amap,
                top_k=int(roi_cfg.get("top_k") or 1),
                pad_ratio=float(roi_cfg.get("pad_ratio") or 0.4),
                min_side=int(roi_cfg.get("min_side") or 96),
            )
            crop_path = None
            roi_score = float(score)
            if rois:
                crop_path = crop_dir / f"{cat}_{img_path.stem}_top1.jpg"
                rois[0].crop.save(crop_path, quality=92)
                roi_score = float(rois[0].score)
            row_like = {
                "category": cat,
                "edge_score": float(score),
                "roi_score": roi_score,
                "crop_kind": "top1",
            }
            caches.append(
                {
                    "image": str(img_path),
                    "crop": str(crop_path) if crop_path else None,
                    "category": cat,
                    "label": int(r["label"]),
                    "edge_score": float(score),
                    "edge_thr": thr,
                    "roi_score": roi_score,
                    "prompt": make_prompt(tmpl, row_like, thr, margin),
                }
            )

    del ad
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    path = out_dir / "eval_edge_cache.jsonl"
    write_jsonl(path, caches)
    return caches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/qwen35_roi_gated_v2.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(cfg.get("results_dir") or "outputs/qwen35_roi_gated_v2")
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or cfg.get("device") or "cuda:0"
    margin = float((cfg.get("edge") or {}).get("hard_margin") or 0.05)

    train_jsonl = out_dir / "roi_sft_train_v2.jsonl"
    if not args.skip_build or not train_jsonl.exists():
        train_jsonl = build_v2_train(cfg, out_dir)

    adapter = out_dir / "adapter"
    if not args.skip_train and not (adapter / "adapter_model.safetensors").exists():
        overlay = {
            "model": cfg["model"],
            "lora": cfg["lora"],
            "train_lora": {**cfg["train"], "device": device},
        }
        overlay_path = out_dir / "train_overlay.yaml"
        overlay_path.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
        cmd = [
            args.python,
            str(ROOT / "scripts" / "train_qwen35_sft.py"),
            "--config",
            str(overlay_path),
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

    if args.skip_eval:
        return

    hold = Path(cfg["source_holdout_jsonl"])
    if not hold.is_absolute():
        hold = ROOT / hold
    max_images = args.max_images
    if max_images is None:
        max_images = int((cfg.get("eval") or {}).get("max_images") or 60)
    rows = subsample_balanced(load_jsonl(hold), max_images)

    cache_path = out_dir / "eval_edge_cache.jsonl"
    if cache_path.exists():
        caches = load_jsonl(cache_path)
        want = {r["image"] for r in rows}
        caches = [c for c in caches if c["image"] in want]
        if len(caches) != len(rows):
            caches = cache_eval(rows, cfg, device, out_dir)
    else:
        caches = cache_eval(rows, cfg, device, out_dir)

    model_cfg = cfg["model"]
    print(f"[eval] load adapter {adapter}")
    client = QwenVLClient(
        model_path=model_cfg["model_path"],
        device=device,
        dtype=model_cfg.get("dtype", "bfloat16"),
        max_new_tokens=64,
        role="roi_v2",
        model_family=model_cfg.get("model_family"),
        adapter_path=str(adapter),
    )

    mode_preds = {"edge_only": [], "roi_vlm": [], "gated": [], "gated_or_disagree": []}
    details = {k: [] for k in mode_preds}

    for i, e in enumerate(caches):
        gt = int(e["label"])
        score = float(e["edge_score"])
        thr = float(e["edge_thr"])
        edge_pred = 1 if score >= thr else 0
        band = abs(score - thr) < margin

        if e.get("crop"):
            client.prompt = e["prompt"]
            res = client.infer(e["crop"])
            vlm_pred = 1 if res.decision == "NG" else 0
            vlm_dec, vlm_reason, vlm_raw = res.decision, res.reason, res.raw
        else:
            vlm_pred = edge_pred
            vlm_dec = "OK" if edge_pred == 0 else "NG"
            vlm_reason, vlm_raw = "no_crop", ""

        gated = vlm_pred if band else edge_pred
        # if edge vs vlm disagree, prefer VLM (score-aware) even outside band
        gated_or = vlm_pred if (band or vlm_pred != edge_pred) else edge_pred
        src_g = "vlm" if band else "edge"
        src_d = "vlm" if (band or vlm_pred != edge_pred) else "edge"

        for name, pred, src in (
            ("edge_only", edge_pred, "edge"),
            ("roi_vlm", vlm_pred, "vlm"),
            ("gated", gated, src_g),
            ("gated_or_disagree", gated_or, src_d),
        ):
            mode_preds[name].append((gt, pred))
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
                    "src": src,
                    "correct": pred == gt,
                }
            )
        print(
            f"[eval {i+1}/{len(caches)}] {e['category']} gt={'NG' if gt else 'OK'} "
            f"edge={'NG' if edge_pred else 'OK'} vlm={vlm_dec} "
            f"gated={'NG' if gated else 'OK'} disagree={'NG' if gated_or else 'OK'} "
            f"band={band} score={score:.3f} thr={thr:.3f}"
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
        "reference": {
            "full_image_lora": {"acc": 0.57, "f1": 0.70},
            "roi_vlm_v1": {"acc": 0.68, "f1": 0.75},
            "edge_only_v1": {"acc": 0.72, "f1": 0.77},
        },
    }
    for name, pairs in mode_preds.items():
        y_true = [a for a, _ in pairs]
        y_pred = [b for _, b in pairs]
        report["modes"][name] = {"metrics": metrics(y_true, y_pred), "details": details[name]}
    report["modes"]["gated"]["n_band"] = sum(1 for d in details["gated"] if d["band"])
    report["modes"]["gated"]["n_edge"] = sum(1 for d in details["gated"] if not d["band"])
    report["modes"]["gated_or_disagree"]["n_vlm"] = sum(
        1 for d in details["gated_or_disagree"] if d["src"] == "vlm"
    )

    out_json = out_dir / "bench_roi_gated_v2.json"
    out_md = out_dir / "bench_roi_gated_v2.md"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# ROI short-label v2 (edge score in prompt)",
        "",
        f"- N: {len(caches)}",
        f"- Adapter: `{adapter}`",
        f"- Gate margin: {margin}",
        "",
        "| Mode | Acc | F1 | P | R | pred OK/NG |",
        "|------|-----|----|---|---|------------|",
    ]
    for name in ("edge_only", "roi_vlm", "gated", "gated_or_disagree"):
        m = report["modes"][name]["metrics"]
        lines.append(
            f"| {name} | {m['accuracy']:.4f} | {m['f1']:.4f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['pred_ok']}/{m['pred_ng']} |"
        )
    lines += [
        "",
        "## Reference",
        "",
        "| Prior | Acc | F1 |",
        "|-------|-----|----|",
        "| full-image LoRA | 0.57 | 0.70 |",
        "| roi_vlm v1 | 0.68 | 0.75 |",
        "| edge_only v1 | 0.72 | 0.77 |",
        "",
        f"- gated band calls: {report['modes']['gated'].get('n_band')} / "
        f"edge {report['modes']['gated'].get('n_edge')}",
        f"- gated_or_disagree VLM uses: {report['modes']['gated_or_disagree'].get('n_vlm')}",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
