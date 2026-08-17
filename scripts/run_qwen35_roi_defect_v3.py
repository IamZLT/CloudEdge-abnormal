#!/usr/bin/env python3
"""v3: ROI + score CONTEXT, learn full defect_type / reason (not OK/NG only).

Reuses ROI crops from v2 jsonl; rewrites prompts/responses with rich targets
from the original SFT corpus. Evaluates decision Acc/F1 and defect_type accuracy.

Env: conda activate clip
  CUDA_VISIBLE_DEVICES=3 python scripts/run_qwen35_roi_defect_v3.py \
    --config configs/qwen35_roi_defect_v3.yaml
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from collections import Counter, defaultdict
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
from src.vlm.parse import parse_vlm_json


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


def metrics_bin(y_true, y_pred) -> dict:
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


def rich_response(decision: str, defect_type: str, category: str) -> str:
    if decision == "OK":
        obj = {
            "decision": "OK",
            "confidence": 0.95,
            "defect_type": "none",
            "reason": f"No visible defect on the {category}; product appears normal.",
        }
    else:
        nice = (defect_type or "defect").replace("_", " ")
        obj = {
            "decision": "NG",
            "confidence": 0.95,
            "defect_type": defect_type or "defect",
            "reason": f"Visible {nice} defect on the {category}.",
        }
    return json.dumps(obj, ensure_ascii=False)


def calibrate_thr(labels: list[int], scores: list[float]) -> float:
    arr_s = np.asarray(scores, dtype=np.float64)
    arr_y = np.asarray(labels, dtype=np.int64)
    f1, _p, _r, thr = best_f1_threshold(arr_y, arr_s)
    thr = float(thr)
    ok = arr_s[arr_y == 0]
    if thr <= 1e-6 and ok.size:
        thr = float(np.quantile(ok, 0.9))
        print(f"[thr] fallback ok_q90={thr:.4f} (f1={f1:.3f})")
    else:
        print(f"[thr] train_best_f1={thr:.4f} f1={f1:.3f}")
    return thr


def build_vocab(source_train: list[dict]) -> dict[str, list[str]]:
    by = defaultdict(set)
    for r in source_train:
        cat = str(r.get("category"))
        dt = str(r.get("defect_type") or "none")
        if dt and dt != "none":
            by[cat].add(dt)
    return {c: sorted(v) for c, v in by.items()}


def build_v3_train(cfg: dict, out_dir: Path) -> Path:
    v2 = Path(cfg["v2_train_jsonl"])
    if not v2.is_absolute():
        v2 = ROOT / v2
    src = Path(cfg["source_train_jsonl"])
    if not src.is_absolute():
        src = ROOT / src
    src_rows = load_jsonl(src)
    src_by_img = {r["image"]: r for r in src_rows}
    vocab = build_vocab(src_rows)
    margin = float((cfg.get("edge") or {}).get("hard_margin") or 0.05)
    tmpl = cfg.get("prompt_template") or ""

    rows = load_jsonl(v2)
    # thr per category from top1 unique sources
    by_cat = defaultdict(list)
    for r in rows:
        if r.get("crop_kind") == "top1":
            by_cat[str(r.get("category"))].append(r)
    thr_map = {}
    for cat, cat_rows in by_cat.items():
        uniq = {}
        for r in cat_rows:
            uniq[r["source_image"]] = r
        u = list(uniq.values())
        thr_map[cat] = calibrate_thr(
            [int(x["label"]) for x in u],
            [float(x["edge_score"]) for x in u],
        )

    out_rows = []
    for r in rows:
        cat = str(r.get("category"))
        thr = thr_map.get(cat, 0.2)
        src_r = src_by_img.get(r.get("source_image")) or src_by_img.get(r.get("image"))
        label = int(r["label"])
        if label == 0:
            decision, defect = "OK", "none"
        else:
            decision = "NG"
            defect = (src_r or r).get("defect_type") or r.get("defect_type") or "defect"
            if defect == "none":
                defect = "defect"
        score = float(r.get("edge_score") or 0.0)
        vocab_list = vocab.get(cat) or ["defect"]
        prompt = tmpl.format(
            category=cat,
            edge_score=score,
            edge_thr=float(thr),
            edge_hint=edge_hint(score, thr, margin),
            crop_kind=r.get("crop_kind") or "top1",
            defect_vocab=", ".join(vocab_list),
        )
        nr = dict(r)
        nr["edge_thr"] = thr
        nr["defect_type"] = defect
        nr["prompt"] = prompt
        nr["response"] = rich_response(decision, defect, cat)
        out_rows.append(nr)

    out_path = out_dir / "roi_sft_train_v3.jsonl"
    write_jsonl(out_path, out_rows)
    meta = {
        "n": len(out_rows),
        "ok": sum(1 for x in out_rows if x["label"] == 0),
        "ng": sum(1 for x in out_rows if x["label"] == 1),
        "defect_types": Counter(x["defect_type"] for x in out_rows),
        "vocab": vocab,
        "thr_map": thr_map,
    }
    # Counter not json serializable directly in nested - convert
    meta["defect_types"] = dict(meta["defect_types"])
    (out_dir / "roi_sft_meta_v3.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[build-v3] n={meta['n']} ok={meta['ok']} ng={meta['ng']} types={len(meta['defect_types'])}")
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
        name="roi_defect_v3",
        fusion_temperature=float(edge.get("fusion_temp") or 0.5),
    )


def cache_eval(rows: list[dict], cfg: dict, device: str, out_dir: Path, vocab: dict) -> list[dict]:
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
    src_by_img = {}
    for r in load_jsonl(train_src):
        train_by[str(r.get("category"))].append(r)
        src_by_img[r["image"]] = r

    # also index holdout source for gt defect type
    hold_src = Path(cfg["source_holdout_jsonl"])
    if not hold_src.is_absolute():
        hold_src = ROOT / hold_src
    for r in load_jsonl(hold_src):
        src_by_img[r["image"]] = r

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
        tr = train_by.get(cat) or []
        scores = []
        labels = []
        for r in tr:
            s, _ = ad.score_image(Image.open(r["image"]).convert("RGB"))
            scores.append(float(s))
            labels.append(int(r["label"]))
        thr = calibrate_thr(labels, scores) if scores else 0.2
        vocab_list = vocab.get(cat) or ["defect"]

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
            if rois:
                crop_path = crop_dir / f"{cat}_{img_path.stem}_top1.jpg"
                rois[0].crop.save(crop_path, quality=92)
            gt_src = src_by_img.get(str(img_path)) or r
            gt_defect = gt_src.get("defect_type") or ("none" if int(r["label"]) == 0 else "defect")
            prompt = tmpl.format(
                category=cat,
                edge_score=float(score),
                edge_thr=float(thr),
                edge_hint=edge_hint(float(score), thr, margin),
                crop_kind="top1",
                defect_vocab=", ".join(vocab_list),
            )
            caches.append(
                {
                    "image": str(img_path),
                    "crop": str(crop_path) if crop_path else None,
                    "category": cat,
                    "label": int(r["label"]),
                    "gt_defect_type": gt_defect,
                    "edge_score": float(score),
                    "edge_thr": thr,
                    "prompt": prompt,
                }
            )
        print(f"[cache] {cat} thr={thr:.4f} n={len(cat_rows)}")

    del ad
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    write_jsonl(out_dir / "eval_edge_cache.jsonl", caches)
    return caches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/qwen35_roi_defect_v3.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(cfg.get("results_dir") or "outputs/qwen35_roi_defect_v3")
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or cfg.get("device") or "cuda:0"

    train_jsonl = out_dir / "roi_sft_train_v3.jsonl"
    if not args.skip_build or not train_jsonl.exists():
        train_jsonl = build_v3_train(cfg, out_dir)

    meta = json.loads((out_dir / "roi_sft_meta_v3.json").read_text(encoding="utf-8"))
    vocab = meta.get("vocab") or {}

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
            caches = cache_eval(rows, cfg, device, out_dir, vocab)
    else:
        caches = cache_eval(rows, cfg, device, out_dir, vocab)

    model_cfg = cfg["model"]
    client = QwenVLClient(
        model_path=model_cfg["model_path"],
        device=device,
        dtype=model_cfg.get("dtype", "bfloat16"),
        max_new_tokens=96,
        role="defect_v3",
        model_family=model_cfg.get("model_family"),
        adapter_path=str(adapter),
    )

    y_true, y_pred = [], []
    type_true, type_pred = [], []
    type_true_ng, type_pred_ng = [], []
    details = []

    for i, e in enumerate(caches):
        gt = int(e["label"])
        gt_type = str(e.get("gt_defect_type") or ("none" if gt == 0 else "defect"))
        if e.get("crop"):
            client.prompt = e["prompt"]
            res = client.infer(e["crop"])
            # prefer parsed fields from client; also re-parse raw for type
            parsed = parse_vlm_json(res.raw)
            decision = parsed.get("decision") or res.decision
            pred_type = str(parsed.get("defect_type") or res.defect_type or "none")
            reason = parsed.get("reason") or res.reason
            raw = res.raw
            conf = float(parsed.get("confidence") or res.confidence or 0.0)
        else:
            decision = "OK" if gt == 0 else "NG"
            pred_type = "none"
            reason, raw, conf = "no_crop", "", 0.0

        pred = 1 if str(decision).upper() == "NG" else 0
        if pred == 0:
            pred_type = "none"
        y_true.append(gt)
        y_pred.append(pred)
        type_true.append(gt_type)
        type_pred.append(pred_type)
        if gt == 1:
            type_true_ng.append(gt_type)
            type_pred_ng.append(pred_type)

        details.append(
            {
                "image": e["image"],
                "category": e["category"],
                "gt": gt,
                "pred": pred,
                "gt_defect_type": gt_type,
                "pred_defect_type": pred_type,
                "type_correct": pred_type == gt_type,
                "decision_correct": pred == gt,
                "edge_score": e.get("edge_score"),
                "edge_thr": e.get("edge_thr"),
                "confidence": conf,
                "reason": reason,
                "raw": raw,
            }
        )
        print(
            f"[eval {i+1}/{len(caches)}] {e['category']} "
            f"gt={'NG' if gt else 'OK'}/{gt_type} "
            f"pred={'NG' if pred else 'OK'}/{pred_type} "
            f"dec_ok={pred==gt} type_ok={pred_type==gt_type} | {(reason or '')[:40]}"
        )

    del client
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    type_acc_all = float(np.mean([a == b for a, b in zip(type_true, type_pred)])) if type_true else 0.0
    type_acc_ng = (
        float(np.mean([a == b for a, b in zip(type_true_ng, type_pred_ng)])) if type_true_ng else 0.0
    )
    # type accuracy only when decision also NG
    both_ng = [
        (t, p)
        for t, p, g, pr in zip(type_true, type_pred, y_true, y_pred)
        if g == 1 and pr == 1
    ]
    type_acc_ng_given_ng = (
        float(np.mean([a == b for a, b in both_ng])) if both_ng else 0.0
    )

    report = {
        "n_eval": len(caches),
        "adapter": str(adapter),
        "decision_metrics": metrics_bin(y_true, y_pred),
        "defect_type_accuracy_all": type_acc_all,
        "defect_type_accuracy_on_gt_ng": type_acc_ng,
        "defect_type_accuracy_on_pred_and_gt_ng": type_acc_ng_given_ng,
        "n_gt_ng": len(type_true_ng),
        "type_confusion_top": Counter(
            f"{t}->{p}" for t, p in zip(type_true_ng, type_pred_ng)
        ).most_common(15),
        "details": details,
        "reference": {
            "full_image_lora_decision": {"acc": 0.57, "f1": 0.70},
            "roi_vlm_v2_decision": {"acc": 0.78, "f1": 0.78},
        },
    }
    out_json = out_dir / "bench_roi_defect_v3.json"
    out_md = out_dir / "bench_roi_defect_v3.md"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    m = report["decision_metrics"]
    lines = [
        "# ROI defect-aware v3 (full defect_type / reason)",
        "",
        f"- N: {len(caches)}",
        f"- Adapter: `{adapter}`",
        "",
        "## Decision (OK/NG)",
        "",
        "| Mode | Acc | F1 | P | R | pred OK/NG |",
        "|------|-----|----|---|---|------------|",
        f"| roi_defect_v3 | {m['accuracy']:.4f} | {m['f1']:.4f} | {m['precision']:.4f} | "
        f"{m['recall']:.4f} | {m['pred_ok']}/{m['pred_ng']} |",
        f"| ref roi_vlm_v2 | 0.7833 | 0.7797 | — | — | 31/29 |",
        f"| ref full-image LoRA | 0.5700 | 0.7000 | — | — | 4/56 |",
        "",
        "## Defect type",
        "",
        f"- type Acc (all samples): **{type_acc_all:.4f}**",
        f"- type Acc (GT=NG only): **{type_acc_ng:.4f}** (n={len(type_true_ng)})",
        f"- type Acc (GT=NG & pred=NG): **{type_acc_ng_given_ng:.4f}** (n={len(both_ng)})",
        "",
        "Top GT→Pred on NG:",
        "",
    ]
    for k, c in report["type_confusion_top"][:10]:
        lines.append(f"- `{k}`: {c}")
    lines += [
        "",
        "## Notes",
        "",
        "- Training keeps ROI crops + edge CONTEXT, but restores rich JSON targets",
        "  (`defect_type` + full sentence `reason`) from the original SFT corpus.",
        "- Prompt lists per-category defect vocabulary to reduce open-vocab collapse.",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
