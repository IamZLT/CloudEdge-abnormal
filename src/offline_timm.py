"""Offline pretrained weight helper for timm (HF hub may be unreachable)."""
from __future__ import annotations

import os
from pathlib import Path

_WEIGHTS_ROOT = Path(
    os.environ.get("CLOUD_EDGE_TIMM_WEIGHTS", "model_card/timm")
)

LOCAL_WEIGHTS = {
    "resnet18": _WEIGHTS_ROOT / "resnet18.tv_in1k/model.safetensors",
    "wide_resnet50_2": _WEIGHTS_ROOT / "wide_resnet50_2.tv2_in1k/model.safetensors",
    "vit_base_patch16_dinov3_qkvb": (
        _WEIGHTS_ROOT
        / "vit_base_patch16_dinov3_qkvb.eupe_lvd1689m/model.safetensors"
    ),
    "resnet50": Path.home() / ".cache/torch/hub/checkpoints/resnet50-0676ba61.pth",
}

_ORIG = None


def enable(backbone: str | None = None):
    global _ORIG
    import timm

    if _ORIG is None:
        _ORIG = timm.create_model

    def create_model(model_name, *args, **kwargs):
        key = str(model_name).split(".")[0]
        weight = LOCAL_WEIGHTS.get(key)
        if kwargs.get("pretrained", False) and weight is not None and Path(weight).exists():
            overlay = dict(kwargs.get("pretrained_cfg_overlay") or {})
            overlay["file"] = str(weight)
            kwargs["pretrained_cfg_overlay"] = overlay
        return _ORIG(model_name, *args, **kwargs)

    timm.create_model = create_model


def disable():
    global _ORIG
    if _ORIG is not None:
        import timm

        timm.create_model = _ORIG
        _ORIG = None
