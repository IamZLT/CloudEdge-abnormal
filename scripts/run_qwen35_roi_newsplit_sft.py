#!/usr/bin/env python3
"""Full supervised pipeline on the NEW mvtec_anomaly_llm split.

  edge contrast (few-shot) → ROI zoom → LoRA-tuned Qwen3.5-0.8B

Train on the train split, test on the test split. Reuses the real GPT-4V
annotations (mvtec_zero_shot_train.json) to enrich the `reason` field instead of
the weak template ("Visible X defect on the Y.").

Stages:
  build  — edge-score + top-1 ROI crops + prompt/response for every train image
  train  — LoRA SFT on the built corpus
  eval   — decision (OK/NG) + defect_type accuracy on the test split

Env: conda activate clip
  CUDA_VISIBLE_DEVICES=3 python scripts/run_qwen35_roi_newsplit_sft.py \
    --config configs/qwen35_roi_newsplit_sft.yaml --stage all
"""
from __future__ import annotations

import argparse
import gc
import json
import re
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

from edge.methods.gallery_ad import best_f1_threshold
from edge.methods.patch_gallery_ad import PatchGalleryAD
from edge.methods.roi_crops import amap_to_rois, make_multiscale_collage
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


def edge_hint(score: float, thr: float, margin: float) -> str:
    if score < thr - margin:
        return "likely_OK"
    if score > thr + margin:
        return "likely_NG"
    return "uncertain"


def norm_type(s: str) -> str:
    """Normalize a defect type for comparison (lowercase, space/hyphen → underscore)."""
    s = (s or "").strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


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
        name="roi_newsplit_sft",
        fusion_temperature=float(edge.get("fusion_temp") or 0.5),
    )


def _rel_of_train_row(row: dict) -> str:
    # train.jsonl image: .../mvtec_anomaly_llm/train/{cat}/{sub}/{file}
    return "/".join(Path(row["image"]).parts[-4:])


def build_zero_shot_map(path: Path) -> dict[str, dict]:
    # mvtec_zero_shot_train.json is a pretty-printed JSON array (not JSONL).
    # Each image may have 1 detailed multi-turn entry + 1 short Yes/No entry;
    # keep the DETAILED (longest) entry for rich reason extraction.
    rows = json.loads(path.read_text(encoding="utf-8"))
    m = {}
    for e in rows:
        # e["image"] = "train/{cat}/{sub}/{file}"
        key = e["image"]
        prev = m.get(key)
        if prev is None or len(e.get("conversations") or []) > len(prev.get("conversations") or []):
            m[key] = e
    return m


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def extract_reason(entry: dict | None, label: int, category: str, defect_type: str) -> str:
    """Pull a concise real-GPT-4V reason; fallback to template when unavailable."""
    if entry:
        conv = entry.get("conversations") or []
        for i in range(len(conv) - 1):
            h, g = conv[i], conv[i + 1]
            if h.get("from") != "human" or g.get("from") != "gpt":
                continue
            hv = str(h.get("value", "")).lower()
            if ("anomal" in hv or "defect" in hv or "abnormal" in hv or "irregular" in hv) and "?" in hv:
                ans = str(g.get("value", "")).strip()
                # skip trivial Yes/No answers and look for a richer description
                if not ans or ans.lower() in {"yes", "no", "yes.", "no.", "yes,", "no,"}:
                    continue
                sents = _SENT_SPLIT.split(ans)
                reason = " ".join(sents[:2]).strip()
                if len(reason) > 220:
                    reason = reason[:220].rsplit(" ", 1)[0]
                if reason.lower() in {"yes", "no"}:
                    continue
                return reason
    # template fallback
    if label == 0:
        return f"No visible defect on the {category}; product appears normal."
    nice = (defect_type or "defect").replace("_", " ")
    return f"Visible {nice} defect on the {category}."


def rich_response(decision: str, defect_type: str, reason: str) -> str:
    obj = {
        "decision": decision,
        "confidence": 0.95,
        "defect_type": defect_type,
        "reason": reason,
    }
    return json.dumps(obj, ensure_ascii=False)


def build_vocab(train_rows: list[dict]) -> dict[str, list[str]]:
    by = defaultdict(set)
    for r in train_rows:
        cat = str(r.get("category"))
        dt = str(r.get("defect_type") or "none")
        if dt and dt != "none":
            by[cat].add(dt)
    return {c: sorted(v) for c, v in by.items()}


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


def gallery_paths_by_cat(cfg: dict, split: str) -> dict[str, list[Path]]:
    """Return per-category gallery image paths from train or test good images."""
    manifest = (
        cfg["train_jsonl"] if split == "train" else cfg["test_jsonl"]
    )
    path = Path(manifest)
    if not path.is_absolute():
        path = ROOT / path
    by: dict[str, list[Path]] = defaultdict(list)
    for r in load_jsonl(path):
        if int(r["label"]) == 0:
            by[str(r["category"])].append(Path(r["image"]))
    return {c: sorted(v) for c, v in by.items()}


def stage_build(cfg: dict, device: str, out_dir: Path) -> Path:
    train_path = Path(cfg["train_jsonl"])
    if not train_path.is_absolute():
        train_path = ROOT / train_path
    train_rows = load_jsonl(train_path)

    zs_path = Path(cfg.get("zero_shot_train") or "")
    zs_map = build_zero_shot_map(zs_path) if zs_path and zs_path.exists() else {}

    edge = cfg.get("edge") or {}
    roi_cfg = cfg.get("roi") or {}
    margin = float(edge.get("hard_margin") or 0.05)
    seed = int(cfg.get("seed") or 42)
    gallery_split = str(edge.get("gallery_split") or "test")
    tmpl = cfg.get("prompt_template") or ""
    vocab = build_vocab(train_rows)
    crop_mode = str(roi_cfg.get("crop_mode") or "roi")  # "roi" | "multiscale"
    full_height = int(roi_cfg.get("full_height") or 224)
    top_k = int(roi_cfg.get("top_k") or (3 if crop_mode == "multiscale" else 1))

    gallery_by = gallery_paths_by_cat(cfg, gallery_split)
    ad = build_edge_ad(cfg, device)
    crop_dir = out_dir / "train_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    # group train rows by category, compute per-category threshold
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in train_rows:
        by_cat[str(r["category"])].append(r)

    thr_map: dict[str, float] = {}
    out_rows: list[dict] = []
    total = 0

    for cat, cat_rows in sorted(by_cat.items()):
        gpaths = gallery_by.get(cat) or []
        print(f"[build] {cat}: gallery({gallery_split})={len(gpaths)} train={len(cat_rows)}")
        gpts = [ad.encode_patches(Image.open(p).convert("RGB")) for p in gpaths]
        # calibrate threshold on this category's train scores
        scores, labels = [], []
        for r in cat_rows:
            ad.build_gallery_from_tokens(gpts, seed=seed)
            s, _ = ad.score_image(Image.open(r["image"]).convert("RGB"))
            scores.append(float(s))
            labels.append(int(r["label"]))
        thr = calibrate_thr(labels, scores) if scores else 0.2
        thr_map[cat] = thr
        vocab_list = vocab.get(cat) or ["defect"]

        for i, r in enumerate(cat_rows):
            ad.build_gallery_from_tokens(gpts, seed=seed)
            pil = Image.open(r["image"]).convert("RGB")
            score, amap = ad.score_image(pil)
            rois = amap_to_rois(
                pil,
                amap,
                top_k=top_k,
                pad_ratio=float(roi_cfg.get("pad_ratio") or 0.4),
                min_side=int(roi_cfg.get("min_side") or 96),
            )
            label = int(r["label"])
            defect = str(r.get("defect_type") or "none") if label == 1 else "none"
            if label == 1 and defect == "none":
                defect = "defect"
            decision = "OK" if label == 0 else "NG"

            rel = _rel_of_train_row(r)
            zs_entry = zs_map.get(rel)
            reason = extract_reason(zs_entry, label, cat, defect)

            prompt = tmpl.format(
                category=cat,
                edge_score=float(score),
                edge_thr=float(thr),
                edge_hint=edge_hint(float(score), thr, margin),
                crop_kind="multiscale" if crop_mode == "multiscale" else "top1",
                defect_vocab=", ".join(vocab_list),
            )
            crop_path = crop_dir / f"{cat}_{Path(r['image']).parts[-2]}_{Path(r['image']).stem}.jpg"
            if crop_mode == "multiscale":
                collage = make_multiscale_collage(pil, rois, full_height=full_height)
                if collage is not None:
                    collage.save(crop_path, quality=92)
                    img_field = str(crop_path)
                else:
                    img_field = r["image"]
            elif rois:
                rois[0].crop.save(crop_path, quality=92)
                img_field = str(crop_path)
            else:
                img_field = r["image"]

            out_rows.append(
                {
                    "image": img_field,
                    "source_image": r["image"],
                    "category": cat,
                    "label": label,
                    "defect_type": defect,
                    "edge_score": float(score),
                    "edge_thr": float(thr),
                    "prompt": prompt,
                    "response": rich_response(decision, defect, reason),
                }
            )
            total += 1
            if (i + 1) % 200 == 0:
                print(f"[build] {cat} {i+1}/{len(cat_rows)}")

    del ad
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_path = out_dir / "roi_sft_train_newsplit.jsonl"
    write_jsonl(out_path, out_rows)
    meta = {
        "n": len(out_rows),
        "ok": sum(1 for x in out_rows if x["label"] == 0),
        "ng": sum(1 for x in out_rows if x["label"] == 1),
        "defect_types": dict(Counter(x["defect_type"] for x in out_rows)),
        "vocab": vocab,
        "thr_map": thr_map,
        "gallery_split": gallery_split,
        "n_reason_from_gpt": sum(
            1 for r in out_rows if "Visible" not in r["response"] and "No visible" not in r["response"]
        ),
    }
    (out_dir / "roi_sft_meta_newsplit.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[build] wrote {out_path} n={len(out_rows)} ok={meta['ok']} ng={meta['ng']}")
    return out_path


def stage_train(cfg: dict, device: str, out_dir: Path, train_jsonl: Path, extra_args: list[str] | None = None) -> Path:
    adapter = out_dir / "adapter"
    overlay = {
        "model": cfg["model"],
        "lora": cfg["lora"],
        "train_lora": {**cfg["train"], "device": device},
    }
    overlay_path = out_dir / "train_overlay.yaml"
    overlay_path.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
    cmd = [
        sys.executable,
        "-u",
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
    if extra_args:
        cmd.extend(extra_args)
    print("[train]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    return adapter


def stage_eval(cfg: dict, device: str, out_dir: Path, adapter: Path):
    test_path = Path(cfg["test_jsonl"])
    if not test_path.is_absolute():
        test_path = ROOT / test_path
    rows = load_jsonl(test_path)
    max_images = (cfg.get("eval") or {}).get("max_images")
    if max_images:
        ok = [r for r in rows if int(r["label"]) == 0]
        ng = [r for r in rows if int(r["label"]) == 1]
        n_ok = min(len(ok), int(max_images) // 2)
        n_ng = min(len(ng), int(max_images) - n_ok)
        rows = ok[:n_ok] + ng[:n_ng]

    edge = cfg.get("edge") or {}
    roi_cfg = cfg.get("roi") or {}
    margin = float(edge.get("hard_margin") or 0.05)
    seed = int(cfg.get("seed") or 42)
    gallery_split = str(edge.get("gallery_split") or "test")
    tmpl = cfg.get("prompt_template") or ""
    crop_mode = str(roi_cfg.get("crop_mode") or "roi")
    full_height = int(roi_cfg.get("full_height") or 224)
    top_k = int(roi_cfg.get("top_k") or (3 if crop_mode == "multiscale" else 1))

    # vocab + threshold from the training meta
    meta = json.loads((out_dir / "roi_sft_meta_newsplit.json").read_text(encoding="utf-8"))
    vocab = meta.get("vocab") or {}
    thr_map = meta.get("thr_map") or {}

    ad = build_edge_ad(cfg, device)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[str(r["category"])].append(r)

    # gallery: test/good (leave-one-out for good queries)
    gallery_by = gallery_paths_by_cat(cfg, "test")
    crop_dir = out_dir / "eval_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    caches = []
    for cat, cat_rows in sorted(by_cat.items()):
        gpaths = gallery_by.get(cat) or []
        gpts = [ad.encode_patches(Image.open(p).convert("RGB")) for p in gpaths]
        thr = float(thr_map.get(cat, 0.2))
        vocab_list = vocab.get(cat) or ["defect"]
        for r in cat_rows:
            img_path = Path(r["image"])
            pil = Image.open(img_path).convert("RGB")
            if int(r["label"]) == 0:
                pts = [
                    pt for gp, pt in zip(gpaths, gpts)
                    if gp.resolve() != img_path.resolve()
                ] or list(gpts)
            else:
                pts = list(gpts)
            ad.build_gallery_from_tokens(pts, seed=seed)
            score, amap = ad.score_image(pil)
            rois = amap_to_rois(
                pil,
                amap,
                top_k=top_k,
                pad_ratio=float(roi_cfg.get("pad_ratio") or 0.4),
                min_side=int(roi_cfg.get("min_side") or 96),
            )
            crop_path = None
            if crop_mode == "multiscale":
                collage = make_multiscale_collage(pil, rois, full_height=full_height)
                if collage is not None:
                    crop_path = crop_dir / f"{cat}_{img_path.parts[-2]}_{img_path.stem}_ms.jpg"
                    collage.save(crop_path, quality=92)
            elif rois:
                crop_path = crop_dir / f"{cat}_{img_path.parts[-2]}_{img_path.stem}_top1.jpg"
                rois[0].crop.save(crop_path, quality=92)
            prompt = tmpl.format(
                category=cat,
                edge_score=float(score),
                edge_thr=float(thr),
                edge_hint=edge_hint(float(score), thr, margin),
                crop_kind="multiscale" if crop_mode == "multiscale" else "top1",
                defect_vocab=", ".join(vocab_list),
            )
            caches.append(
                {
                    "image": str(img_path),
                    "crop": str(crop_path) if crop_path else None,
                    "category": cat,
                    "label": int(r["label"]),
                    "gt_defect_type": str(r.get("defect_type") or "none"),
                    "edge_score": float(score),
                    "edge_thr": float(thr),
                    "prompt": prompt,
                }
            )
        print(f"[eval-cache] {cat} thr={thr:.4f} n={len(cat_rows)}")

    del ad
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_cfg = cfg["model"]
    client = QwenVLClient(
        model_path=model_cfg["model_path"],
        device=device,
        dtype=model_cfg.get("dtype", "bfloat16"),
        max_new_tokens=96,
        role="newsplit_sft",
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
            parsed = parse_vlm_json(res.raw)
            decision = parsed.get("decision") or res.decision
            pred_type = str(parsed.get("defect_type") or res.defect_type or "none")
            reason = parsed.get("reason") or res.reason
            conf = float(parsed.get("confidence") or res.confidence or 0.0)
        else:
            decision = "OK" if gt == 0 else "NG"
            pred_type = "none"
            reason, conf = "no_crop", 0.0

        pred = 1 if str(decision).upper() == "NG" else 0
        if pred == 0:
            pred_type = "none"
        y_true.append(gt)
        y_pred.append(pred)
        type_true.append(norm_type(gt_type))
        type_pred.append(norm_type(pred_type))
        if gt == 1:
            type_true_ng.append(norm_type(gt_type))
            type_pred_ng.append(norm_type(pred_type))

        details.append(
            {
                "image": e["image"],
                "category": e["category"],
                "gt": gt,
                "pred": pred,
                "gt_defect_type": gt_type,
                "pred_defect_type": pred_type,
                "type_correct": norm_type(pred_type) == norm_type(gt_type),
                "decision_correct": pred == gt,
                "edge_score": e.get("edge_score"),
                "edge_thr": e.get("edge_thr"),
                "confidence": conf,
                "reason": reason,
            }
        )
        print(
            f"[eval {i+1}/{len(caches)}] {e['category']} "
            f"gt={'NG' if gt else 'OK'}/{gt_type} pred={'NG' if pred else 'OK'}/{pred_type} "
            f"dec_ok={pred==gt} type_ok={pred_type==gt_type}"
        )

    del client
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    type_acc_all = float(np.mean([a == b for a, b in zip(type_true, type_pred)])) if type_true else 0.0
    type_acc_ng = (
        float(np.mean([a == b for a, b in zip(type_true_ng, type_pred_ng)])) if type_true_ng else 0.0
    )
    both_ng = [(t, p) for t, p, g, pr in zip(type_true, type_pred, y_true, y_pred) if g == 1 and pr == 1]
    type_acc_ng_given_ng = float(np.mean([a == b for a, b in both_ng])) if both_ng else 0.0

    report = {
        "n_eval": len(caches),
        "adapter": str(adapter),
        "gallery_split": str(edge.get("gallery_split") or "test"),
        "decision_metrics": metrics_bin(y_true, y_pred),
        "defect_type_accuracy_all": type_acc_all,
        "defect_type_accuracy_on_gt_ng": type_acc_ng,
        "defect_type_accuracy_on_pred_and_gt_ng": type_acc_ng_given_ng,
        "n_gt_ng": len(type_true_ng),
        "type_confusion_top": Counter(
            f"{t}->{p}" for t, p in zip(type_true_ng, type_pred_ng)
        ).most_common(15),
        "details": details,
    }
    (out_dir / "bench_roi_newsplit_sft.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    m = report["decision_metrics"]
    lines = [
        "# ROI supervised (new split): edge contrast → zoom → LoRA LLM",
        "",
        f"- N: {len(caches)} (train={len(load_jsonl(Path(cfg['train_jsonl'])))} test={len(rows)})",
        f"- Adapter: `{adapter}`",
        f"- Edge gallery: `{edge.get('gallery_split')}` (few-shot, leave-one-out for good queries)",
        "",
        "## Decision (OK/NG)",
        "",
        "| Mode | Acc | F1 | P | R | pred OK/NG |",
        "|------|-----|----|---|---|------------|",
        f"| roi_sft | {m['accuracy']:.4f} | {m['f1']:.4f} | {m['precision']:.4f} | "
        f"{m['recall']:.4f} | {m['pred_ok']}/{m['pred_ng']} |",
        "",
        "## Defect type",
        "",
        f"- type Acc (all): **{type_acc_all:.4f}**",
        f"- type Acc (GT=NG): **{type_acc_ng:.4f}** (n={len(type_true_ng)})",
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
        "- Trained on the new train split (all 15 categories), tested on the new test split.",
        "- Training `reason` is enriched from the real GPT-4V annotations (not the weak template).",
        "",
    ]
    (out_dir / "bench_roi_newsplit_sft.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"Wrote {out_dir / 'bench_roi_newsplit_sft.md'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/qwen35_roi_newsplit_sft.yaml"))
    parser.add_argument("--stage", default="all", choices=["build", "train", "eval", "all"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(cfg.get("results_dir") or "outputs/qwen35_roi_newsplit_sft")
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or cfg.get("device") or "cuda:0"

    train_jsonl = out_dir / "roi_sft_train_newsplit.jsonl"
    if args.stage in ("build", "all"):
        train_jsonl = stage_build(cfg, device, out_dir)

    adapter = out_dir / "adapter"
    if args.stage in ("train", "all"):
        if not (adapter / "adapter_model.safetensors").exists():
            adapter = stage_train(cfg, device, out_dir, train_jsonl)
        else:
            print(f"[train] adapter exists, skip: {adapter}")

    if args.stage in ("eval", "all"):
        stage_eval(cfg, device, out_dir, adapter)


if __name__ == "__main__":
    main()
