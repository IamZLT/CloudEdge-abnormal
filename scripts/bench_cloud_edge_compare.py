#!/usr/bin/env python3
"""Cloud–edge full-test-set comparison: edge-only vs cloud-only vs collab.

Three schemes, each runnable independently (so they can be launched on separate
free GPUs in parallel), each writing its own JSON to --out:

    --scheme edge    Qwen3.5-0.8B multi-layer patch gallery (train/good bank,
                     leave-one-out threshold). Image-level detection only.
    --scheme cloud   Qwen3-VL-4B (+LoRA) direct OK/NG verdict per image.
    --scheme collab  edge AD → hand-written CRR routing (uncertainty + live geo
                     network + cloud load) → cloud VLM arbitration with LoRA
                     domain allowlist.

Base metrics collected for every scheme:
    F1 / precision / recall / accuracy / image-AUROC, latency (ms), FLOPs,
    params, peak GPU memory, plus collab-only overhead (upload rate, cloud
    review rate, network time, end-to-end latency).

Example (parallel):
  CUDA_VISIBLE_DEVICES=0 python scripts/bench_cloud_edge_compare.py --scheme edge
  CUDA_VISIBLE_DEVICES=6 python scripts/bench_cloud_edge_compare.py --scheme cloud
  CUDA_VISIBLE_DEVICES=1 python scripts/bench_cloud_edge_compare.py --scheme collab
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics as st
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge.infer import DEFAULT_PATHS  # noqa: E402
from edge.methods.encoders import load_qwen35_vision_encoder  # noqa: E402
from edge.methods.patch_gallery_ad import PatchGalleryAD  # noqa: E402
from src.collab_routing import CloudState, RouteSignal, build_router  # noqa: E402
from src.network_geo import live_network_dict, make_geo_simulator  # noqa: E402
from src.vlm import QwenVLClient  # noqa: E402

OUT_DIR = ROOT / "outputs" / "reports" / "cloud_edge_compare"

# SFT "Yes/No + reason" prompt — MUST match the training prompt of
# outputs/qwen35_reason_sft/adapter (configs/qwen35_reason_sft.yaml).
SFT_PROMPT = (
    "Is there any anomaly or defect in the product shown in the image?\n"
    "Answer with Yes or No, and briefly explain the reason."
)


def parse_yn_reason(raw: str) -> tuple[int, str]:
    """First word Yes/No → decision (1/0) + remainder as reason."""
    import re

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
    m = re.match(r"^\s*(yes|no)\b[.,:;\s]*\s*(.*)$", s, flags=re.I | re.S)
    reason = m.group(2).strip() if m else s
    return decision, reason

CATS = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def llm_train_good(data_root: Path, category: str) -> list[Path]:
    """mvtec_anomaly_llm layout: {split}/{cat}/{defect}, not {cat}/{split}/{defect}."""
    return _list_images(data_root / "train" / category / "good")


def llm_test_split(data_root: Path, category: str) -> list[tuple[Path, int]]:
    test_root = data_root / "test" / category
    items: list[tuple[Path, int]] = []
    for sub in sorted(test_root.iterdir()):
        if not sub.is_dir():
            continue
        y = 0 if sub.name == "good" else 1
        for p in _list_images(sub):
            items.append((p, y))
    return items


def llm_gallery_paths(data_root: Path, category: str, max_gallery: int, seed: int) -> list[Path]:
    train = llm_train_good(data_root, category)
    if not train:
        raise FileNotFoundError(f"no train/good under {data_root / 'train' / category}")
    if max_gallery and len(train) > max_gallery:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(train), size=max_gallery, replace=False)
        train = [train[i] for i in sorted(idx)]
    return train


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _agg(xs: list[float]) -> dict[str, Any] | None:
    if not xs:
        return None
    return {
        "n": len(xs),
        "mean": float(st.mean(xs)),
        "p50": float(st.median(xs)),
        "p95": float(np.percentile(xs, 95)),
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


def _det(yt: list[int], yp: list[int], scores: list[float] | None = None) -> dict[str, Any]:
    yt = np.asarray(yt, dtype=int)
    yp = np.asarray(yp, dtype=int)
    out = {
        "n": int(len(yt)),
        "f1": float(f1_score(yt, yp, zero_division=0)),
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "accuracy": float(accuracy_score(yt, yp)),
    }
    if scores is not None and len(np.unique(yt)) > 1:
        try:
            out["image_auroc"] = float(roc_auc_score(yt, np.asarray(scores, dtype=float)))
        except ValueError:
            out["image_auroc"] = float("nan")
    else:
        out["image_auroc"] = float("nan")
    return out


def _macro(per_cat: dict[str, dict[str, Any]]) -> dict[str, Any]:
    keys = ["f1", "precision", "recall", "accuracy", "image_auroc"]
    out: dict[str, Any] = {}
    for k in keys:
        vals = [v[k] for v in per_cat.values() if v.get(k) is not None and not (isinstance(v[k], float) and np.isnan(v[k]))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def _count_params(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def _thop_flops(model: torch.nn.Module, *inputs: Any) -> float | None:
    try:
        from thop import profile

        macs, _ = profile(model, inputs=inputs, verbose=False)
        return float(macs) / 1e9
    except Exception:
        return None


def _sync(device: str) -> None:
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.synchronize()


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- edge
def run_edge(
    *,
    cfg: dict[str, Any],
    categories: list[str],
    data_root: Path,
    device: str,
    max_gallery: int,
    seed: int,
    warmup: int = 5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    edge_cfg = dict(cfg.get("edge") or {})
    image_size = int(cfg.get("image_size") or 224)
    layers = edge_cfg.get("layers") or [6, 8, 10, 12]
    fusion_temp = float(edge_cfg.get("fusion_temp") or 0.5)
    model_path = edge_cfg.get("model_path") or DEFAULT_PATHS["qwen35"]

    print(f"[edge] load Qwen3.5 vision @ {device} ...")
    t0 = time.perf_counter()
    _, encode_patches, meta = load_qwen35_vision_encoder(
        model_path,
        device=device,
        max_pixels=int(edge_cfg.get("max_pixels") or image_size * image_size),
        layers=layers,
    )
    load_s = time.perf_counter() - t0
    print(f"[edge] loaded in {load_s:.1f}s")

    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        _sync(device)

    rows: list[dict[str, Any]] = []
    per_cat: dict[str, dict[str, Any]] = {}
    for cat in categories:
        ad = PatchGalleryAD(
            encode_patches, device=device, name="qwen35_0.8b_vision_mlpatch", fusion_temperature=fusion_temp
        )
        gallery = llm_gallery_paths(data_root, cat, max_gallery, seed)
        thr = ad.calibrate_threshold_loo(
            gallery, seed=seed, quantile=float(edge_cfg.get("thr_quantile") or 0.95)
        )
        test_items = llm_test_split(data_root, cat)
        print(f"[edge] {cat}: gallery={len(gallery)} thr={thr:.4f} n_test={len(test_items)}")

        for i in range(min(warmup, len(test_items))):
            _ = ad.score_image(Image.open(test_items[i][0]).convert("RGB"))
        _sync(device)

        labels, preds, scores, lats = [], [], [], []
        for path, y in test_items:
            img = Image.open(path).convert("RGB")
            _sync(device)
            t1 = time.perf_counter()
            score, _amap = ad.score_image(img)
            _sync(device)
            ms = (time.perf_counter() - t1) * 1000.0
            decision = "NG" if float(score) >= thr else "OK"
            labels.append(y)
            preds.append(1 if decision == "NG" else 0)
            scores.append(float(score))
            lats.append(ms)
            rows.append(
                {
                    "category": cat, "path": str(path), "label": int(y),
                    "score": float(score), "thr": float(thr), "decision": decision,
                    "pred": 1 if decision == "NG" else 0, "latency_ms": float(ms),
                }
            )
        per_cat[cat] = _det(labels, preds, scores)
        per_cat[cat]["thr"] = float(thr)
        per_cat[cat]["n_gallery"] = len(gallery)
        per_cat[cat]["latency_ms_mean"] = float(np.mean(lats))

    peak = None
    if torch.cuda.is_available() and device.startswith("cuda"):
        peak = float(torch.cuda.max_memory_allocated() / (1024**2))

    _free()
    base_metrics = {
        "params_m": float(meta.get("params_m") or 0.0),
        "flops_g": float(meta.get("flops_g") or 0.0),
        "peak_mem_mb": peak,
        "load_s": float(load_s),
        "backbone": meta.get("backbone"),
    }
    return {"per_category": per_cat, "macro": _macro(per_cat), "rows": rows}, base_metrics


# --------------------------------------------------------------------------- sft_edge
def run_sft_edge(
    *,
    categories: list[str],
    data_root: Path,
    device: str,
    model_path: str,
    adapter_path: str,
    prompt: str | None,
    max_new_tokens: int = 64,
    max_pixels: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Edge = full Qwen3.5-0.8B + reason LoRA, generative Yes/No + reason.

    This is the SFT detection method (NOT the patch-gallery kNN). Output per image
    is a generated Yes/No + reason sentence; decision = first word.

    max_pixels bounds the vision-tower pixel budget (e.g. 224*224) so the edge
    matches the architecture's image_size, instead of running at native resolution.
    """
    print(f"[sft_edge] load Qwen3.5-0.8B + reason LoRA @ {device} ...")
    t0 = time.perf_counter()
    client = QwenVLClient(
        model_path=model_path,
        device=device,
        dtype="bfloat16",
        max_new_tokens=max_new_tokens,
        role="sft_edge",
        prompt=prompt or SFT_PROMPT,
        model_family="qwen3_5",
        adapter_path=adapter_path,
        max_pixels=max_pixels,
    )
    load_s = time.perf_counter() - t0
    params_m = _count_params(client.model)
    # LLM FLOPs ≈ 2·N per token (causal dense).
    vision_params_m = 100.6  # Qwen3.5-0.8B vision tower (measured earlier)
    llm_params_m = params_m - vision_params_m
    print(f"[sft_edge] loaded in {load_s:.1f}s params={params_m:.1f}M max_pixels={max_pixels}")

    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        _sync(device)

    rows: list[dict[str, Any]] = []
    per_cat: dict[str, dict[str, Any]] = {}
    for cat in categories:
        test_items = llm_test_split(data_root, cat)
        print(f"[sft_edge] {cat}: n_test={len(test_items)}")
        labels, preds, lats = [], [], []
        for path, y in test_items:
            vlm = client.infer(Path(path))
            decision, reason = parse_yn_reason(vlm.raw)
            pred = decision
            labels.append(y)
            preds.append(pred)
            lats.append(float(vlm.latency_ms or 0.0))
            rows.append(
                {
                    "category": cat, "path": str(path), "label": int(y),
                    "decision": "Yes" if decision else "No", "pred": pred,
                    "reason": reason, "raw": vlm.raw,
                    "latency_ms": float(vlm.latency_ms or 0.0),
                }
            )
        per_cat[cat] = _det(labels, preds)
        per_cat[cat]["latency_ms_mean"] = float(np.mean(lats)) if lats else 0.0

    peak = None
    if torch.cuda.is_available() and device.startswith("cuda"):
        peak = float(torch.cuda.max_memory_allocated() / (1024**2))

    _free()
    base_metrics = {
        "params_m": float(params_m),
        "vision_params_m": float(vision_params_m),
        "llm_params_m": float(llm_params_m),
        "vision_flops_g": 36.41,  # vision tower forward @224x224 (FlopCounterMode, 196 token)
        "llm_flops_per_token_g": float(llm_params_m * 2.0 / 1e3),  # ≈ 2·N per token (GFLOPs)
        "max_pixels": max_pixels,
        "peak_mem_mb": peak,
        "load_s": float(load_s),
        "backbone": "Qwen3.5-0.8B+reason-LoRA (generative)",
    }
    return {"per_category": per_cat, "macro": _macro(per_cat), "rows": rows}, base_metrics


# --------------------------------------------------------------------------- cloud
def run_cloud(
    *,
    cloud_cfg: dict[str, Any],
    categories: list[str],
    data_root: Path,
    device: str,
    prompt: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    print(f"[cloud] load Qwen3-VL-4B (+LoRA) @ {device} ...")
    t0 = time.perf_counter()
    cloud = QwenVLClient(
        model_path=cloud_cfg["model_path"],
        adapter_path=cloud_cfg.get("adapter_path"),
        device=device,
        dtype=cloud_cfg.get("dtype", "bfloat16"),
        max_new_tokens=int(cloud_cfg.get("max_new_tokens", 160)),
        role="cloud",
        prompt=prompt,
    )
    load_s = time.perf_counter() - t0
    print(f"[cloud] loaded in {load_s:.1f}s")

    params_m = _count_params(cloud.model)
    # thop profile of the Qwen3-VL vision tower can hang on dynamic grid_thw,
    # so estimate vision FLOPs analytically: ≈ 2 · params · visual_tokens.
    # Note: PeftModel wraps base → vision tower lives at base_model.model.model.visual.
    vision_flops_g = None
    try:
        base = getattr(cloud.model, "base_model", None)
        base = getattr(base, "model", None) or base
        visual = base.model.visual
        v_params = sum(p.numel() for p in visual.parameters()) / 1e6
        n_tok = (224 // 14) ** 2  # patch 14 → 16×16 grid (pre-merge)
        vision_flops_g = float(2.0 * v_params * n_tok / 1e3)
    except Exception:
        vision_flops_g = None

    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        _sync(device)

    rows: list[dict[str, Any]] = []
    per_cat: dict[str, dict[str, Any]] = {}
    for cat in categories:
        test_items = llm_test_split(data_root, cat)
        print(f"[cloud] {cat}: n_test={len(test_items)}")
        labels, preds, scores, lats = [], [], [], []
        for path, y in test_items:
            vlm = cloud.infer(Path(path))
            decision = str(vlm.decision).upper() if vlm.decision else "OK"
            pred = 1 if decision == "NG" else 0
            conf = float(vlm.confidence)
            score = conf if pred else 1.0 - conf
            labels.append(y)
            preds.append(pred)
            scores.append(score)
            lats.append(float(vlm.latency_ms or 0.0))
            rows.append(
                {
                    "category": cat, "path": str(path), "label": int(y),
                    "decision": decision, "confidence": conf, "pred": pred,
                    "latency_ms": float(vlm.latency_ms or 0.0),
                    "defect_type": vlm.defect_type, "reason": vlm.reason,
                }
            )
        per_cat[cat] = _det(labels, preds, scores)
        per_cat[cat]["latency_ms_mean"] = float(np.mean(lats)) if lats else 0.0

    peak = None
    if torch.cuda.is_available() and device.startswith("cuda"):
        peak = float(torch.cuda.max_memory_allocated() / (1024**2))

    _free()
    base_metrics = {
        "params_m": float(params_m),
        "vision_flops_g": vision_flops_g,
        "llm_flops_per_token_g": float(params_m * 2.0 / 1e3),  # ≈ 2·N per generated token (GFLOPs)
        "peak_mem_mb": peak,
        "load_s": float(load_s),
        "backbone": "Qwen3-VL-4B+LoRA",
    }
    return {"per_category": per_cat, "macro": _macro(per_cat), "rows": rows}, base_metrics


# --------------------------------------------------------------------------- collab
def run_collab(
    *,
    cfg: dict[str, Any],
    cloud_cfg: dict[str, Any],
    categories: list[str],
    data_root: Path,
    device: str,
    prompt: str | None,
    scenario: str,
    seed: int,
    max_gallery: int,
    min_cloud_conf: float,
    warmup: int = 5,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    collab = dict(cfg.get("collab") or {})
    edge_cfg = dict(cfg.get("edge") or {})
    image_size = int(cfg.get("image_size") or 224)
    layers = edge_cfg.get("layers") or [6, 8, 10, 12]
    fusion_temp = float(edge_cfg.get("fusion_temp") or 0.5)
    model_path = edge_cfg.get("model_path") or DEFAULT_PATHS["qwen35"]
    cloud_categories = {str(c).strip() for c in (cloud_cfg.get("categories") or [])}
    hard_margin = float(collab.get("hard_margin") or collab.get("thr_margin") or 0.05)
    up_bytes = int(collab.get("upload_bytes_hard") or 80000)

    # ---- phase 1: edge AD (unload before cloud)
    print(f"[collab] edge load @ {device} ...")
    t0 = time.perf_counter()
    _, encode_patches, edge_meta = load_qwen35_vision_encoder(
        model_path,
        device=device,
        max_pixels=int(edge_cfg.get("max_pixels") or image_size * image_size),
        layers=layers,
    )
    edge_load_s = time.perf_counter() - t0

    edge_rows: list[dict[str, Any]] = []
    for cat in categories:
        ad = PatchGalleryAD(encode_patches, device=device, name="qwen35_0.8b_vision_mlpatch", fusion_temperature=fusion_temp)
        gallery = llm_gallery_paths(data_root, cat, max_gallery, seed)
        thr = ad.calibrate_threshold_loo(gallery, seed=seed, quantile=float(edge_cfg.get("thr_quantile") or 0.95))
        test_items = llm_test_split(data_root, cat)
        for i in range(min(warmup, len(test_items))):
            _ = ad.score_image(Image.open(test_items[i][0]).convert("RGB"))
        _sync(device)
        for path, y in test_items:
            img = Image.open(path).convert("RGB")
            _sync(device)
            t1 = time.perf_counter()
            score, _amap = ad.score_image(img)
            _sync(device)
            ms = (time.perf_counter() - t1) * 1000.0
            decision = "NG" if float(score) >= thr else "OK"
            edge_rows.append(
                {
                    "category": cat, "path": str(path), "label": int(y),
                    "edge_score": float(score), "edge_thr": float(thr),
                    "edge_decision": decision, "edge_pred": 1 if decision == "NG" else 0,
                    "edge_ms": float(ms), "n_gallery": len(gallery),
                }
            )
        print(f"[collab] edge {cat}: thr={thr:.4f} n={len(test_items)}")

    _free()
    print("[collab] edge unloaded")

    # ---- phase 2: CRR routing + cloud arbitration
    router = build_router("cost_risk", {**collab, "route_policy": "cost_risk"})
    adm = dict(collab.get("cloud_admission") or {})
    max_inflight = int(adm.get("max_inflight", 2))
    sim = make_geo_simulator(collab, scenario, seed=seed, edge_id=f"edge-{scenario}")

    print(f"[collab] cloud load @ {device} ...")
    t1 = time.perf_counter()
    cloud = QwenVLClient(
        model_path=cloud_cfg["model_path"],
        adapter_path=cloud_cfg.get("adapter_path"),
        device=device,
        dtype=cloud_cfg.get("dtype", "bfloat16"),
        max_new_tokens=int(cloud_cfg.get("max_new_tokens", 160)),
        role="cloud",
        prompt=prompt,
    )
    cloud_load_s = time.perf_counter() - t1
    cloud_params_m = _count_params(cloud.model)
    print(f"[collab] cloud loaded in {cloud_load_s:.1f}s")

    cloud_state = CloudState(inflight=0, queue=0, max_inflight=max_inflight)
    rows: list[dict[str, Any]] = []
    for i, er in enumerate(edge_rows):
        net = live_network_dict(sim)
        sig = RouteSignal(
            category=er["category"],
            n_gallery=int(er["n_gallery"]),
            edge_score=float(er["edge_score"]),
            edge_thr=float(er["edge_thr"]),
            edge_decision=str(er["edge_decision"]),
            network_profile=str(net.get("profile") or "geo"),
            network=net,
            hard_margin=hard_margin,
            edge_node_id=f"edge-{scenario}",
            conflict=0.0,
        )
        verd = router.decide(sig, cloud_state)
        upload = bool(verd.upload)

        net_ms = 0.0
        net_ok = False
        if upload:
            out = sim.try_upload(up_bytes)
            net_ms = float(out.rtt_ms) + float(out.tx_ms)
            net_ok = bool(out.ok)

        cloud_ms = 0.0
        cloud_decision = None
        cloud_conf = None
        cloud_used = False
        cloud_fallback = False
        if upload and net_ok:
            use_vlm = bool(cloud_categories) and er["category"] in cloud_categories
            if use_vlm:
                cloud_used = True
                vlm = cloud.infer(Path(er["path"]))
                cloud_ms = float(vlm.latency_ms or 0.0)
                cloud_conf = float(vlm.confidence)
                if vlm.parse_ok and vlm.decision in {"OK", "NG"} and cloud_conf >= min_cloud_conf:
                    cloud_decision = str(vlm.decision).upper()
                else:
                    cloud_fallback = True
                    cloud_decision = er["edge_decision"]
            else:
                # out-of-domain category → don't trust LoRA, keep edge decision
                cloud_decision = er["edge_decision"]

        final_decision = cloud_decision if (upload and net_ok and cloud_decision) else er["edge_decision"]
        final_pred = 1 if final_decision == "NG" else 0
        path_type = "CLOUD_REVIEW" if (upload and net_ok) else ("LOCAL" if not upload else "LOCAL_NET_FALLBACK")
        total_ms = float(er["edge_ms"]) + net_ms + cloud_ms

        rows.append(
            {
                "category": er["category"], "path": er["path"], "label": int(er["label"]),
                "edge_score": er["edge_score"], "edge_thr": er["edge_thr"],
                "edge_decision": er["edge_decision"], "edge_pred": er["edge_pred"],
                "edge_ms": er["edge_ms"], "upload": upload, "net_ok": net_ok,
                "net_ms": net_ms, "cloud_ms": cloud_ms, "cloud_used": cloud_used,
                "cloud_fallback": cloud_fallback, "cloud_conf": cloud_conf,
                "path_type": path_type, "final_decision": final_decision,
                "final_pred": final_pred, "total_ms": total_ms, "crr_reason": verd.reason,
            }
        )
        print(
            f"[{scenario} {i}] {er['category']}/{Path(er['path']).name} "
            f"{path_type} up={upload} | edge={er['edge_ms']:.0f} net={net_ms:.0f} "
            f"cloud={cloud_ms:.0f} | edge={er['edge_decision']} final={final_decision} "
            f"gt={'NG' if er['label'] else 'OK'}"
        )

    _free()

    yt = [r["label"] for r in rows]
    y_edge = [r["edge_pred"] for r in rows]
    y_final = [r["final_pred"] for r in rows]
    edge_scores = [r["edge_score"] for r in rows]
    up = [r for r in rows if r["upload"]]
    cloud_rev = [r for r in rows if r["path_type"] == "CLOUD_REVIEW"]

    base_metrics = {
        "edge_params_m": float(edge_meta.get("params_m") or 0.0),
        "edge_flops_g": float(edge_meta.get("flops_g") or 0.0),
        "cloud_params_m": float(cloud_params_m),
        "edge_load_s": float(edge_load_s),
        "cloud_load_s": float(cloud_load_s),
    }

    summary = {
        "scenario": scenario,
        "n": len(rows),
        "upload_rate": len(up) / max(1, len(rows)),
        "cloud_review_rate": len(cloud_rev) / max(1, len(rows)),
        "cloud_used_rate": sum(1 for r in rows if r["cloud_used"]) / max(1, len(rows)),
        "cloud_fallback_rate": sum(1 for r in rows if r["cloud_fallback"]) / max(1, len(rows)),
        "fallback_rate": sum(1 for r in rows if r["path_type"] == "LOCAL_NET_FALLBACK") / max(1, len(rows)),
        "edge_det": _det(yt, y_edge, edge_scores),
        "final_det": _det(yt, y_final),
        "edge_ms": _agg([r["edge_ms"] for r in rows]),
        "net_ms_when_upload": _agg([r["net_ms"] for r in up]),
        "cloud_ms_when_review": _agg([r["cloud_ms"] for r in cloud_rev]),
        "total_ms_all": _agg([r["total_ms"] for r in rows]),
        "total_ms_local": _agg([r["total_ms"] for r in rows if r["path_type"] == "LOCAL"]),
        "total_ms_cloud_review": _agg([r["total_ms"] for r in cloud_rev]),
    }
    return summary, base_metrics, rows


# --------------------------------------------------------------------------- collab_sft
def run_collab_sft(
    *,
    cfg: dict[str, Any],
    cloud_cfg: dict[str, Any],
    categories: list[str],
    data_root: Path,
    device: str,
    prompt: str | None,
    sft_model: str,
    sft_adapter: str,
    scenario: str,
    seed: int,
    max_gallery: int,
    min_cloud_conf: float,
    warmup: int = 5,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Collab where edge DETECTION = SFT generative (Yes/No+reason), but upload
    ROUTING signal = vision-tower kNN score (CRR margin logic unchanged).

    Phases:
      1. vision tower → kNN edge_score (upload signal)
      2. full 0.8B + reason LoRA → edge Yes/No (edge detection result)
      3. CRR route (on kNN score) + cloud VLM review (domain allowlist)
    Final = cloud review when uploaded & trusted, else SFT edge Yes/No.
    """
    collab = dict(cfg.get("collab") or {})
    edge_cfg = dict(cfg.get("edge") or {})
    image_size = int(cfg.get("image_size") or 224)
    layers = edge_cfg.get("layers") or [6, 8, 10, 12]
    fusion_temp = float(edge_cfg.get("fusion_temp") or 0.5)
    model_path = edge_cfg.get("model_path") or DEFAULT_PATHS["qwen35"]
    cloud_categories = {str(c).strip() for c in (cloud_cfg.get("categories") or [])}
    hard_margin = float(collab.get("hard_margin") or collab.get("thr_margin") or 0.05)
    up_bytes = int(collab.get("upload_bytes_hard") or 80000)

    # ---- phase 1: vision tower kNN → edge_score (upload signal)
    print(f"[collab_sft] phase1: vision-tower kNN @ {device} ...")
    t0 = time.perf_counter()
    _, encode_patches, edge_meta = load_qwen35_vision_encoder(
        model_path,
        device=device,
        max_pixels=int(edge_cfg.get("max_pixels") or image_size * image_size),
        layers=layers,
    )
    edge_load_s = time.perf_counter() - t0

    kNN_rows: list[dict[str, Any]] = []
    for cat in categories:
        ad = PatchGalleryAD(encode_patches, device=device, name="qwen35_0.8b_vision_mlpatch", fusion_temperature=fusion_temp)
        gallery = llm_gallery_paths(data_root, cat, max_gallery, seed)
        thr = ad.calibrate_threshold_loo(gallery, seed=seed, quantile=float(edge_cfg.get("thr_quantile") or 0.95))
        test_items = llm_test_split(data_root, cat)
        for i in range(min(warmup, len(test_items))):
            _ = ad.score_image(Image.open(test_items[i][0]).convert("RGB"))
        _sync(device)
        for path, y in test_items:
            img = Image.open(path).convert("RGB")
            _sync(device)
            t1 = time.perf_counter()
            score, _amap = ad.score_image(img)
            _sync(device)
            ms = (time.perf_counter() - t1) * 1000.0
            kNN_rows.append(
                {
                    "category": cat, "path": str(path), "label": int(y),
                    "edge_score": float(score), "edge_thr": float(thr),
                    "knn_ms": float(ms), "n_gallery": len(gallery),
                }
            )
        print(f"[collab_sft] kNN {cat}: thr={thr:.4f} n={len(test_items)}")

    _free()
    print("[collab_sft] vision tower unloaded")

    # ---- phase 2: full 0.8B + reason LoRA → edge Yes/No
    print(f"[collab_sft] phase2: SFT edge (full 0.8B + LoRA) @ {device} ...")
    t2 = time.perf_counter()
    sft = QwenVLClient(
        model_path=sft_model,
        device=device,
        dtype="bfloat16",
        max_new_tokens=64,
        role="sft_edge",
        prompt=SFT_PROMPT,  # reason LoRA is trained on THIS prompt, not cloud JSON prompt
        model_family="qwen3_5",
        adapter_path=sft_adapter,
        max_pixels=int(edge_cfg.get("max_pixels") or image_size * image_size),
    )
    sft_load_s = time.perf_counter() - t2
    sft_params_m = _count_params(sft.model)
    print(f"[collab_sft] SFT loaded in {sft_load_s:.1f}s params={sft_params_m:.1f}M")

    # map path -> SFT decision (kNN_rows order matches llm_test_split order)
    sft_map: dict[str, tuple[int, str, float]] = {}
    for er in kNN_rows:
        vlm = sft.infer(Path(er["path"]))
        decision, reason = parse_yn_reason(vlm.raw)
        sft_map[er["path"]] = (decision, reason, float(vlm.latency_ms or 0.0))
    _free()
    print("[collab_sft] SFT edge unloaded")

    # ---- phase 3: CRR routing (kNN score) + cloud review
    router = build_router("cost_risk", {**collab, "route_policy": "cost_risk"})
    adm = dict(collab.get("cloud_admission") or {})
    max_inflight = int(adm.get("max_inflight", 2))
    sim = make_geo_simulator(collab, scenario, seed=seed, edge_id=f"edge-{scenario}")

    print(f"[collab_sft] cloud load @ {device} ...")
    t3 = time.perf_counter()
    cloud = QwenVLClient(
        model_path=cloud_cfg["model_path"],
        adapter_path=cloud_cfg.get("adapter_path"),
        device=device,
        dtype=cloud_cfg.get("dtype", "bfloat16"),
        max_new_tokens=int(cloud_cfg.get("max_new_tokens", 160)),
        role="cloud",
        prompt=prompt,
    )
    cloud_load_s = time.perf_counter() - t3
    cloud_params_m = _count_params(cloud.model)
    print(f"[collab_sft] cloud loaded in {cloud_load_s:.1f}s")

    cloud_state = CloudState(inflight=0, queue=0, max_inflight=max_inflight)
    rows: list[dict[str, Any]] = []
    for i, er in enumerate(kNN_rows):
        sft_dec, sft_reason, sft_ms = sft_map[er["path"]]
        edge_decision = "NG" if sft_dec == 1 else "OK"

        net = live_network_dict(sim)
        sig = RouteSignal(
            category=er["category"],
            n_gallery=int(er["n_gallery"]),
            edge_score=float(er["edge_score"]),
            edge_thr=float(er["edge_thr"]),
            edge_decision=edge_decision,
            network_profile=str(net.get("profile") or "geo"),
            network=net,
            hard_margin=hard_margin,
            edge_node_id=f"edge-{scenario}",
            conflict=0.0,
        )
        verd = router.decide(sig, cloud_state)
        upload = bool(verd.upload)

        net_ms = 0.0
        net_ok = False
        if upload:
            out = sim.try_upload(up_bytes)
            net_ms = float(out.rtt_ms) + float(out.tx_ms)
            net_ok = bool(out.ok)

        cloud_ms = 0.0
        cloud_decision = None
        cloud_conf = None
        cloud_used = False
        cloud_fallback = False
        if upload and net_ok:
            use_vlm = bool(cloud_categories) and er["category"] in cloud_categories
            if use_vlm:
                cloud_used = True
                vlm = cloud.infer(Path(er["path"]))
                cloud_ms = float(vlm.latency_ms or 0.0)
                cloud_conf = float(vlm.confidence)
                if vlm.parse_ok and vlm.decision in {"OK", "NG"} and cloud_conf >= min_cloud_conf:
                    cloud_decision = str(vlm.decision).upper()
                else:
                    cloud_fallback = True
                    cloud_decision = edge_decision
            else:
                # out-of-domain → keep SFT edge decision
                cloud_decision = edge_decision

        final_decision = cloud_decision if (upload and net_ok and cloud_decision) else edge_decision
        final_pred = 1 if final_decision == "NG" else 0
        path_type = "CLOUD_REVIEW" if (upload and net_ok) else ("LOCAL" if not upload else "LOCAL_NET_FALLBACK")
        edge_ms = float(er["knn_ms"]) + sft_ms
        total_ms = edge_ms + net_ms + cloud_ms

        rows.append(
            {
                "category": er["category"], "path": er["path"], "label": int(er["label"]),
                "edge_score": er["edge_score"], "edge_thr": er["edge_thr"],
                "knn_ms": er["knn_ms"], "sft_ms": sft_ms, "edge_ms": edge_ms,
                "edge_decision": edge_decision, "sft_reason": sft_reason,
                "edge_pred": 1 if edge_decision == "NG" else 0,
                "upload": upload, "net_ok": net_ok, "net_ms": net_ms,
                "cloud_ms": cloud_ms, "cloud_used": cloud_used,
                "cloud_fallback": cloud_fallback, "cloud_conf": cloud_conf,
                "path_type": path_type, "final_decision": final_decision,
                "final_pred": final_pred, "total_ms": total_ms, "crr_reason": verd.reason,
            }
        )
        print(
            f"[{scenario} {i}] {er['category']}/{Path(er['path']).name} "
            f"{path_type} up={upload} | knn={er['knn_ms']:.0f}+sft={sft_ms:.0f} "
            f"net={net_ms:.0f} cloud={cloud_ms:.0f} | edge={edge_decision} "
            f"final={final_decision} gt={'NG' if er['label'] else 'OK'}"
        )

    _free()

    yt = [r["label"] for r in rows]
    y_edge = [r["edge_pred"] for r in rows]
    y_final = [r["final_pred"] for r in rows]
    edge_scores = [r["edge_score"] for r in rows]
    up = [r for r in rows if r["upload"]]
    cloud_rev = [r for r in rows if r["path_type"] == "CLOUD_REVIEW"]

    base_metrics = {
        "edge_knn_params_m": float(edge_meta.get("params_m") or 0.0),
        "edge_knn_flops_g": float(edge_meta.get("flops_g") or 0.0),
        "edge_sft_params_m": float(sft_params_m),
        "edge_sft_llm_flops_per_token_g": float((sft_params_m - 100.6) * 2.0 / 1e3),
        "cloud_params_m": float(cloud_params_m),
        "edge_load_s": float(edge_load_s),
        "sft_load_s": float(sft_load_s),
        "cloud_load_s": float(cloud_load_s),
    }

    summary = {
        "scenario": scenario,
        "n": len(rows),
        "upload_rate": len(up) / max(1, len(rows)),
        "cloud_review_rate": len(cloud_rev) / max(1, len(rows)),
        "cloud_used_rate": sum(1 for r in rows if r["cloud_used"]) / max(1, len(rows)),
        "cloud_fallback_rate": sum(1 for r in rows if r["cloud_fallback"]) / max(1, len(rows)),
        "fallback_rate": sum(1 for r in rows if r["path_type"] == "LOCAL_NET_FALLBACK") / max(1, len(rows)),
        "edge_det": _det(yt, y_edge, edge_scores),
        "final_det": _det(yt, y_final),
        "knn_ms": _agg([r["knn_ms"] for r in rows]),
        "sft_ms": _agg([r["sft_ms"] for r in rows]),
        "edge_ms": _agg([r["edge_ms"] for r in rows]),
        "net_ms_when_upload": _agg([r["net_ms"] for r in up]),
        "cloud_ms_when_review": _agg([r["cloud_ms"] for r in cloud_rev]),
        "total_ms_all": _agg([r["total_ms"] for r in rows]),
        "total_ms_local": _agg([r["total_ms"] for r in rows if r["path_type"] == "LOCAL"]),
        "total_ms_cloud_review": _agg([r["total_ms"] for r in cloud_rev]),
    }
    return summary, base_metrics, rows


def _save(name: str, payload: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / name
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"wrote {p}")
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=["edge", "sft_edge", "cloud", "collab", "collab_sft"])
    ap.add_argument("--data-root", default=str(ROOT / "datasets" / "mvtec_anomaly_llm"))
    ap.add_argument("--categories", default="all")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--cloud-config", default=str(ROOT / "configs" / "hybrid_lora.yaml"))
    ap.add_argument("--scenario", default="good")
    ap.add_argument("--max-gallery", type=int, default=16)
    ap.add_argument("--min-cloud-conf", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sft-model", default="/data2/zlt/anomaly_detection_llm/model_card/Qwen3.5-0.8B")
    ap.add_argument("--sft-adapter", default=str(ROOT / "outputs" / "qwen35_reason_sft" / "adapter"))
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    cloud_yaml = _load_yaml(Path(args.cloud_config))
    data_root = Path(args.data_root)
    cats = CATS if args.categories == "all" else [c.strip() for c in args.categories.split(",") if c.strip()]

    if args.scheme == "edge":
        result, base = run_edge(
            cfg=cfg, categories=cats, data_root=data_root, device=args.device,
            max_gallery=args.max_gallery, seed=args.seed,
        )
        _save("edge.json", {"scheme": "edge", "base": base, "macro": result["macro"], "per_category": result["per_category"], "rows": result["rows"]})
        print(json.dumps({"base": base, "macro": result["macro"]}, indent=2, ensure_ascii=False))

    elif args.scheme == "sft_edge":
        image_size = int(cfg.get("image_size") or 224)
        result, base = run_sft_edge(
            categories=cats, data_root=data_root, device=args.device,
            model_path=args.sft_model, adapter_path=args.sft_adapter,
            prompt=None, max_pixels=image_size * image_size,
        )
        _save("sft_edge.json", {"scheme": "sft_edge", "base": base, "macro": result["macro"], "per_category": result["per_category"], "rows": result["rows"]})
        print(json.dumps({"base": base, "macro": result["macro"]}, indent=2, ensure_ascii=False))

    elif args.scheme == "cloud":
        result, base = run_cloud(
            cloud_cfg=dict(cloud_yaml.get("cloud") or {}), categories=cats,
            data_root=data_root, device=args.device, prompt=cloud_yaml.get("prompt"),
        )
        _save("cloud.json", {"scheme": "cloud", "base": base, "macro": result["macro"], "per_category": result["per_category"], "rows": result["rows"]})
        print(json.dumps({"base": base, "macro": result["macro"]}, indent=2, ensure_ascii=False))

    elif args.scheme == "collab":
        summary, base, rows = run_collab(
            cfg=cfg, cloud_cfg=dict(cloud_yaml.get("cloud") or {}), categories=cats,
            data_root=data_root, device=args.device, prompt=cloud_yaml.get("prompt"),
            scenario=args.scenario, seed=args.seed, max_gallery=args.max_gallery,
            min_cloud_conf=args.min_cloud_conf,
        )
        _save("collab.json", {"scheme": "collab", "base": base, "summary": summary, "rows": rows})
        print(json.dumps({"base": base, "summary": summary}, indent=2, ensure_ascii=False))

    elif args.scheme == "collab_sft":
        summary, base, rows = run_collab_sft(
            cfg=cfg, cloud_cfg=dict(cloud_yaml.get("cloud") or {}), categories=cats,
            data_root=data_root, device=args.device, prompt=cloud_yaml.get("prompt"),
            sft_model=args.sft_model, sft_adapter=args.sft_adapter,
            scenario=args.scenario, seed=args.seed, max_gallery=args.max_gallery,
            min_cloud_conf=args.min_cloud_conf,
        )
        _save("collab_sft.json", {"scheme": "collab_sft", "base": base, "summary": summary, "rows": rows})
        print(json.dumps({"base": base, "summary": summary}, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
