"""Vision encoders for edge feature-gallery AD."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image


def count_params_m(module: torch.nn.Module) -> float:
    return sum(p.numel() for p in module.parameters()) / 1e6


def try_flops_g(fn: Callable, example_args: tuple, device: str) -> float | None:
    """Estimate GFLOPs via thop; return None if unsupported."""
    try:
        from thop import profile

        # wrap callable as nn.Module if needed
        class _Wrap(torch.nn.Module):
            def __init__(self, f):
                super().__init__()
                self.f = f

            def forward(self, *args):
                return self.f(*args)

        # only works for nn.Module with tensor inputs — callers pass model+tensor
        model, inputs = example_args  # type: ignore
        model = model.to(device)
        inputs = tuple(x.to(device) if torch.is_tensor(x) else x for x in inputs)
        macs, _ = profile(model, inputs=inputs, verbose=False)
        return float(macs) / 1e9
    except Exception:
        return None


def load_clip_encoder(
    model_path: str | Path,
    device: str = "cuda:0",
    image_size: int = 224,
) -> tuple[Callable[[Image.Image], torch.Tensor], dict[str, Any]]:
    from transformers import CLIPImageProcessor, CLIPVisionModel

    model = CLIPVisionModel.from_pretrained(str(model_path)).to(device).eval()
    processor = CLIPImageProcessor.from_pretrained(str(model_path))
    # force square size for fair FLOPs
    processor.size = {"height": image_size, "width": image_size}
    processor.crop_size = {"height": image_size, "width": image_size}

    @torch.inference_mode()
    def encode(img: Image.Image) -> torch.Tensor:
        inputs = processor(images=img, return_tensors="pt")
        pv = inputs["pixel_values"].to(device)
        out = model(pixel_values=pv)
        # pooled CLS
        feat = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0]
        return feat.squeeze(0).detach()

    # FLOPs on vision tower
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    flops = None
    try:
        from thop import profile

        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        flops = float(macs) / 1e9
    except Exception:
        flops = None

    meta = {
        "backbone": "CLIP-ViT-L/14",
        "model_path": str(model_path),
        "image_size": image_size,
        "params_m": count_params_m(model),
        "flops_g": flops,
        "device": device,
    }
    return encode, meta


def load_dinov3_encoder(
    model_path: str | Path,
    device: str = "cuda:0",
    image_size: int = 224,
) -> tuple[Callable[[Image.Image], torch.Tensor], dict[str, Any]]:
    from transformers import AutoImageProcessor, AutoModel

    model = AutoModel.from_pretrained(str(model_path)).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(str(model_path))

    @torch.inference_mode()
    def encode(img: Image.Image) -> torch.Tensor:
        # resize externally for fixed size FLOPs fairness
        img_r = img.resize((image_size, image_size), Image.BICUBIC)
        inputs = processor(images=img_r, return_tensors="pt")
        pv = inputs["pixel_values"].to(device)
        # some processors ignore our resize; force
        if pv.shape[-1] != image_size:
            pv = torch.nn.functional.interpolate(pv, size=(image_size, image_size), mode="bilinear", align_corners=False)
        out = model(pixel_values=pv)
        feat = out.last_hidden_state[:, 0]  # CLS
        return feat.squeeze(0).detach()

    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    flops = None
    try:
        from thop import profile

        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        flops = float(macs) / 1e9
    except Exception:
        flops = None

    meta = {
        "backbone": "DINOv3-ViT-L/16",
        "model_path": str(model_path),
        "image_size": image_size,
        "params_m": count_params_m(model),
        "flops_g": flops,
        "device": device,
    }
    return encode, meta


def load_qwen35_vision_encoder(
    model_path: str | Path,
    device: str = "cuda:0",
    max_pixels: int = 224 * 224,
) -> tuple[Callable[[Image.Image], torch.Tensor], dict[str, Any]]:
    """Load Qwen3.5-0.8B vision tower only (SigLIP-style ViT).

    Current transformers may not ship `qwen3_5` yet; we remap `model.visual.*`
    into `Qwen3VLVisionModel` (compatible geometry) and preprocess with the
    Qwen3-VL processor under a fixed pixel budget.
    """
    import json

    from safetensors import safe_open
    from transformers import AutoProcessor, Qwen3VLVisionModel
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

    model_path = Path(model_path)
    raw = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    vc = dict(raw["vision_config"])
    vc.pop("model_type", None)
    cfg = Qwen3VLVisionConfig(**vc)
    visual = Qwen3VLVisionModel(cfg).to(device=device, dtype=torch.bfloat16)

    state: dict[str, torch.Tensor] = {}
    for shard in sorted(model_path.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("model.visual."):
                    state[k[len("model.visual.") :]] = f.get_tensor(k)
    missing, unexpected = visual.load_state_dict(state, strict=False)
    loader_note = (
        f"remapped model.visual.* -> Qwen3VLVisionModel; "
        f"loaded={len(state)} missing={len(missing)} unexpected={len(unexpected)}"
    )
    visual.eval()

    # Prefer local Qwen3-VL processor (same patch packing) if Qwen3.5 processor unsupported
    proc_path = model_path
    try:
        processor = AutoProcessor.from_pretrained(str(proc_path), trust_remote_code=True)
    except Exception:
        proc_path = Path("/data2/zlt/anomaly_detection_llm/model_card/Qwen3-VL-4B-Instruct")
        processor = AutoProcessor.from_pretrained(str(proc_path))
        loader_note += f" | processor={proc_path.name}"

    if hasattr(processor, "image_processor"):
        ip = processor.image_processor
        if hasattr(ip, "max_pixels"):
            ip.max_pixels = max_pixels
        if hasattr(ip, "min_pixels"):
            ip.min_pixels = min(65536, max_pixels)

    @torch.inference_mode()
    def encode(img: Image.Image) -> torch.Tensor:
        msgs = [
            {
                "role": "user",
                "content": [{"type": "image", "image": img}, {"type": "text", "text": "."}],
            }
        ]
        inputs = processor.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        pv = inputs["pixel_values"].to(device=device, dtype=torch.bfloat16)
        grid = inputs["image_grid_thw"].to(device)
        out = visual(pv, grid_thw=grid)
        if isinstance(out, (tuple, list)):
            tokens = out[0]
        elif hasattr(out, "last_hidden_state"):
            tokens = out.last_hidden_state
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            tokens = out.pooler_output
        else:
            # BaseModelOutputWithDeepstackFeatures / ModelOutput
            tokens = out[0] if hasattr(out, "__getitem__") else out
        if hasattr(tokens, "last_hidden_state"):
            tokens = tokens.last_hidden_state
        if tokens.ndim == 3:
            feat = tokens.float().mean(dim=1).squeeze(0)
        else:
            feat = tokens.float().mean(dim=0)
        return feat.detach()

    # FLOPs at the configured pixel budget (dynamic-res encoder; profile one real forward)
    flops_g = None
    try:
        from thop import profile
        from PIL import Image as _Image

        _img = _Image.new("RGB", (224, 224), color=(128, 128, 128))
        _msgs = [{"role": "user", "content": [{"type": "image", "image": _img}, {"type": "text", "text": "."}]}]
        _inp = processor.apply_chat_template(
            _msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        _pv = _inp["pixel_values"].to(device=device, dtype=torch.bfloat16)
        _grid = _inp["image_grid_thw"].to(device)
        macs, _ = profile(visual, inputs=(_pv, _grid), verbose=False)
        flops_g = float(macs) / 1e9
    except Exception:
        flops_g = None

    meta = {
        "backbone": "Qwen3.5-0.8B-vision",
        "model_path": str(model_path),
        "max_pixels": max_pixels,
        "params_m": count_params_m(visual),
        "flops_g": flops_g,
        "device": device,
        "loader_note": loader_note,
        "vision_depth": int(cfg.depth),
        "vision_hidden": int(cfg.hidden_size),
    }
    return encode, meta
