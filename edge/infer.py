"""Edge inference entry — default: Qwen3.5-0.8B multi-layer patch gallery.

Examples:
  CUDA_VISIBLE_DEVICES=0 python -m edge.infer \\
    --image datasets/mvtec/bottle/test/broken_large/000.png --category bottle

  # override method
  python -m edge.infer --method padim --category bottle --image ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CFG = ROOT / "configs" / "edge_qwen35.yaml"
DEFAULT_PATHS = {
    "clip": "/data2/zlt/anomaly_detection_llm/model_card/clip-vit-large-patch14",
    "dinov3": "/data2/zlt/anomaly_detection_llm/model_card/dinov3-vitl16-pretrain-lvd1689m",
    "qwen35": "/data2/zlt/anomaly_detection_llm/model_card/Qwen3.5-0.8B",
    "qwen35_q": str(ROOT / "model_card" / "qwen3.5VL-0.8B-q"),
}


def _load_cfg(path: Path | None) -> dict:
    p = path or DEFAULT_CFG
    if p.exists():
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {}


def _gallery_paths(data_root: Path, category: str, max_gallery: int | None, seed: int) -> list[Path]:
    from edge.methods.gallery_ad import mvtec_train_good

    train = mvtec_train_good(data_root, category)
    if not train:
        raise FileNotFoundError(f"no train/good under {data_root / category}")
    if max_gallery and len(train) > max_gallery:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(train), size=max_gallery, replace=False)
        train = [train[i] for i in sorted(idx)]
    return train


def _infer_patch_gallery(
    method: str,
    image: Path,
    category: str,
    cfg: dict,
    device: str,
) -> dict:
    from edge.methods.encoders import (
        load_clip_encoder,
        load_dinov3_encoder,
        load_qwen35_vision_encoder,
    )
    from edge.methods.patch_gallery_ad import PatchGalleryAD

    edge = cfg.get("edge") or {}
    data_root = Path(cfg.get("data_root") or ROOT / "datasets" / "mvtec")
    image_size = int(cfg.get("image_size") or 224)
    seed = int(cfg.get("seed") or 42)
    max_gallery = edge.get("max_gallery", 16)
    layers = edge.get("layers")
    fusion_temp = float(edge.get("fusion_temp") or 0.5)
    model_path = edge.get("model_path") or DEFAULT_PATHS.get(method)

    if method == "qwen35":
        _, encode_patches, meta = load_qwen35_vision_encoder(
            model_path or DEFAULT_PATHS["qwen35"],
            device=device,
            max_pixels=int(edge.get("max_pixels") or image_size * image_size),
            layers=layers,
        )
        name = "qwen35_0.8b_vision_mlpatch"
    elif method in {"qwen35_q", "qwen35_quant", "qwen35_mmproj"}:
        hf = DEFAULT_PATHS["qwen35"]
        mmproj = edge.get("mmproj_gguf") or DEFAULT_PATHS["qwen35_q"]
        _, encode_patches, meta = load_qwen35_vision_encoder(
            hf,
            device=device,
            max_pixels=int(edge.get("max_pixels") or image_size * image_size),
            layers=layers,
            mmproj_gguf=mmproj,
            config_path=hf,
        )
        name = "qwen35_0.8b_mmproj_mlpatch"
    elif method == "clip":
        _, encode_patches, meta = load_clip_encoder(
            model_path or DEFAULT_PATHS["clip"], device=device, image_size=image_size, layers=layers
        )
        name = "clip_vitl14_mlpatch"
    elif method == "dinov3":
        _, encode_patches, meta = load_dinov3_encoder(
            model_path or DEFAULT_PATHS["dinov3"], device=device, image_size=image_size, layers=layers
        )
        name = "dinov3_vitl16_mlpatch"
    else:
        raise ValueError(f"unsupported patch method: {method}")

    ad = PatchGalleryAD(encode_patches, device=device, name=name, fusion_temperature=fusion_temp)
    gallery = _gallery_paths(data_root, category, max_gallery, seed)

    # calibrate threshold via leave-one-out on train/good if not provided
    # (gallery self-scores are ~0 → degenerate; LOO gives a real normal-reference)
    thr = edge.get("threshold")
    if thr is None:
        thr = ad.calibrate_threshold_loo(
            gallery,
            seed=seed,
            quantile=float(edge.get("thr_quantile") or 0.95),
        )
        build_s = 0.0
    else:
        thr = float(thr)
        build_s = ad.build_gallery(gallery, seed=seed)

    img = Image.open(image).convert("RGB")
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    score, amap = ad.score_image(img)
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000

    decision = "NG" if score >= thr else "OK"
    margin = float(edge.get("hard_margin") or edge.get("thr_margin") or 0.05)
    hard = abs(score - thr) < margin
    out = {
        "method": name,
        "decision": decision,
        "score": float(score),
        "threshold": thr,
        "hard_flag": hard,
        "latency_ms": latency_ms,
        "gallery_build_s": float(build_s),
        "n_gallery": len(gallery),
        "layers": meta.get("layers"),
        "fusion_temp": fusion_temp,
        "map_hw": list(amap.shape),
        "path": "CLOUD_REVIEW" if hard else "LOCAL",
        "category": category,
        "image": str(image),
    }
    route = _maybe_route_agent(
        cfg,
        image=image,
        category=category,
        n_gallery=len(gallery),
        edge_score=float(score),
        edge_thr=float(thr),
        edge_decision=decision,
        hard_margin=margin,
        heuristic_hard=hard,
    )
    if route is not None:
        out["route"] = route
        out["hard_flag"] = bool(route.get("upload"))
        out["path"] = "CLOUD_REVIEW" if out["hard_flag"] else "LOCAL"
    return out


def _maybe_route_agent(
    cfg: dict,
    *,
    image: Path,
    category: str,
    n_gallery: int,
    edge_score: float,
    edge_thr: float,
    edge_decision: str,
    hard_margin: float,
    heuristic_hard: bool,
) -> dict | None:
    """Optionally call Qwen3.5 RouteAgent; return route dict or None if disabled."""
    collab = cfg.get("collab") or {}
    ra_cfg = dict(collab.get("route_agent") or {})
    if cfg.get("_disable_route_agent"):
        return None
    if not ra_cfg.get("enabled", False):
        return None
    try:
        from src.vlm.route_agent import RouteAgent, RouteContext, resolve_network_profile

        profile, net = resolve_network_profile(collab)
        # outage hard-gate: no LLM load
        if bool(ra_cfg.get("hard_block_on_outage", True)) and profile == "outage":
            return {
                "upload": False,
                "confidence": 1.0,
                "reason": "hard_gate: network outage — stay local",
                "latency_ms": 0.0,
                "parse_ok": True,
                "source": "hard_gate",
                "raw": "",
                "network_profile": profile,
                "peak_mem_mb": None,
                "heuristic_hard": bool(heuristic_hard),
            }
        # lazy singleton on module to avoid reloading weights per call in same process
        agent = getattr(_maybe_route_agent, "_agent", None)
        agent_key = (ra_cfg.get("model_path"), ra_cfg.get("device"), ra_cfg.get("dtype"))
        if agent is None or getattr(_maybe_route_agent, "_agent_key", None) != agent_key:
            agent = RouteAgent.from_config(ra_cfg)
            _maybe_route_agent._agent = agent  # type: ignore[attr-defined]
            _maybe_route_agent._agent_key = agent_key  # type: ignore[attr-defined]
        ctx = RouteContext(
            image=image,
            category=category,
            n_gallery=n_gallery,
            edge_score=edge_score,
            edge_thr=edge_thr,
            edge_decision=edge_decision,
            network_profile=profile,
            network=net,
            hard_margin=hard_margin,
        )
        dec = agent.decide(ctx)
        d = dec.to_dict()
        d["heuristic_hard"] = bool(heuristic_hard)
        return d
    except Exception as e:
        return {
            "upload": bool(heuristic_hard),
            "confidence": 0.0,
            "reason": f"route_agent_error: {e}",
            "source": "heuristic_fallback",
            "parse_ok": False,
            "network_profile": str((collab.get("network") or {}).get("profile") or "fair"),
            "heuristic_hard": bool(heuristic_hard),
        }


def _infer_padim(image: Path, category: str, cfg: dict, device: str) -> dict:
    """Fallback: Anomalib PaDiM edge ckpt (legacy)."""
    from anomalib.data import PredictDataset
    from anomalib.engine import Engine
    from anomalib.models import Padim

    pad = cfg.get("padim") or {}
    anomalib_root = Path(pad.get("anomalib_root") or ROOT / "outputs" / "anomalib")
    meta_path = anomalib_root / category / "edge" / "train_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    backbone = meta.get("backbone") or pad.get("backbone") or "resnet18"
    ckpt = meta.get("checkpoint")
    if not ckpt or not Path(ckpt).exists():
        ckpt = str(next((anomalib_root / category / "edge").rglob("*.ckpt")))

    try:
        from src.offline_timm import enable as enable_offline_timm

        enable_offline_timm(backbone)
    except Exception:
        pass

    model = Padim(backbone=backbone)
    engine = Engine(accelerator="gpu" if device.startswith("cuda") else "cpu", devices=1)
    ds = PredictDataset(path=str(image), image_size=(256, 256))
    t0 = time.perf_counter()
    preds = engine.predict(model=model, dataset=ds, ckpt_path=ckpt)
    latency_ms = (time.perf_counter() - t0) * 1000
    batch = preds[0]
    score = float(batch.pred_score.detach().cpu().reshape(-1)[0])
    # use train_meta metrics threshold if present
    thr = 0.5
    tm = meta.get("metrics") or {}
    decision = "NG" if score >= thr else "OK"
    return {
        "method": "padim_resnet18",
        "decision": decision,
        "score": score,
        "threshold": thr,
        "hard_flag": abs(score - thr) < 0.05,
        "latency_ms": latency_ms,
        "path": "LOCAL",
        "category": category,
        "image": str(image),
        "checkpoint": ckpt,
        "note": "legacy PaDiM; threshold placeholder 0.5 unless calibrated externally",
        "train_meta_metrics": tm,
    }


def main():
    p = argparse.ArgumentParser(description="Edge AD infer (default: Qwen3.5-0.8B ML patch)")
    p.add_argument("--config", default=str(DEFAULT_CFG))
    p.add_argument("--image", required=True)
    p.add_argument("--category", default=None, help="MVTec category (default from config or path)")
    p.add_argument(
        "--method",
        default=None,
        choices=["qwen35", "qwen35_q", "clip", "dinov3", "padim"],
        help="default from config.edge.method (=qwen35); qwen35_q = mmproj GGUF vision",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--max-gallery", type=int, default=None)
    p.add_argument("--no-route-agent", action="store_true", help="skip Qwen3.5 upload router")
    p.add_argument("--network-profile", default=None, help="override collab.network.profile")
    args = p.parse_args()

    cfg = _load_cfg(Path(args.config))
    # merge route_agent defaults from configs/default.yaml when missing
    if "collab" not in cfg or not (cfg.get("collab") or {}).get("route_agent"):
        default_cfg = _load_cfg(ROOT / "configs" / "default.yaml")
        collab = dict(cfg.get("collab") or {})
        d_collab = dict(default_cfg.get("collab") or {})
        if "network" not in collab and "network" in d_collab:
            collab["network"] = d_collab["network"]
        if "route_agent" not in collab and "route_agent" in d_collab:
            collab["route_agent"] = d_collab["route_agent"]
        cfg["collab"] = collab
    if args.no_route_agent:
        cfg["_disable_route_agent"] = True
    if args.network_profile:
        collab = dict(cfg.get("collab") or {})
        net = dict(collab.get("network") or {})
        net["profile"] = args.network_profile
        collab["network"] = net
        cfg["collab"] = collab
    edge = dict(cfg.get("edge") or {})
    if args.max_gallery is not None:
        edge["max_gallery"] = args.max_gallery
        cfg["edge"] = edge

    method = (args.method or edge.get("method") or "qwen35").lower()
    if method in {"qwen35_mlpatch", "qwen", "qwen3.5", "qwen35_vision"}:
        method = "qwen35"
    if method in {"qwen35_quant", "qwen35_mmproj"}:
        method = "qwen35_q"
    device = args.device or edge.get("device") or cfg.get("device") or "cuda:0"
    image = Path(args.image)
    if not image.exists():
        raise FileNotFoundError(image)

    category = args.category or cfg.get("category")
    if not category:
        # .../mvtec/<cat>/test/...
        parts = image.resolve().parts
        if "test" in parts:
            category = parts[parts.index("test") - 1]
        else:
            raise ValueError("please pass --category")

    if method == "padim":
        out = _infer_padim(image, category, cfg, device)
    else:
        out = _infer_patch_gallery(method, image, category, cfg, device)

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
