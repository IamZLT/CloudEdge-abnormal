"""Vision encoders for edge feature-gallery AD (multi-layer patch tokens)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image


# Default mid→late ViT layers (1..N after each block; hidden_states[k])
DEFAULT_VIT_LAYERS = [12, 16, 20, 24]
# Qwen3.5-0.8B vision depth=12 → mid-late blocks (1-based)
DEFAULT_QWEN_LAYERS = [6, 8, 10, 12]


@dataclass
class PatchTokens:
    """Spatial patch tokens — possibly multi-layer (pre-merge for Qwen)."""

    layer_tokens: list[torch.Tensor]  # each [N, D], same N
    grid_hw: tuple[int, int]  # (H, W) with H*W == N
    layer_ids: list[int] = field(default_factory=list)

    @property
    def tokens(self) -> torch.Tensor:
        """Last selected layer (compat)."""
        return self.layer_tokens[-1]


EncodeFn = Callable[[Image.Image], torch.Tensor]
EncodePatchesFn = Callable[[Image.Image], PatchTokens]


def count_params_m(module: torch.nn.Module) -> float:
    return sum(p.numel() for p in module.parameters()) / 1e6


def _strip_cls_reg(hs: torch.Tensor, n_skip: int) -> torch.Tensor:
    """hs: [1, 1+reg+N, D] or [1, N, D] → [N, D] float CPU-ready."""
    return hs[0, n_skip:].float()


def load_clip_encoder(
    model_path: str | Path,
    device: str = "cuda:0",
    image_size: int = 224,
    layers: list[int] | None = None,
) -> tuple[EncodeFn, EncodePatchesFn, dict[str, Any]]:
    from transformers import CLIPImageProcessor, CLIPVisionModel

    model = CLIPVisionModel.from_pretrained(str(model_path)).to(device).eval()
    processor = CLIPImageProcessor.from_pretrained(str(model_path))
    processor.size = {"height": image_size, "width": image_size}
    processor.crop_size = {"height": image_size, "width": image_size}
    patch = int(getattr(model.config, "patch_size", 14))
    grid = image_size // patch
    n_layers = int(model.config.num_hidden_layers)
    layer_ids = list(layers or DEFAULT_VIT_LAYERS)
    for lid in layer_ids:
        if lid < 1 or lid > n_layers:
            raise ValueError(f"CLIP layer {lid} out of range 1..{n_layers}")

    @torch.inference_mode()
    def _hidden_states(img: Image.Image) -> tuple[torch.Tensor, ...]:
        inputs = processor(images=img, return_tensors="pt")
        pv = inputs["pixel_values"].to(device)
        out = model(pixel_values=pv, output_hidden_states=True)
        return out.hidden_states  # [0]=embed, [1..N]=after layer

    @torch.inference_mode()
    def encode(img: Image.Image) -> torch.Tensor:
        hs = _hidden_states(img)
        return hs[-1][:, 0].squeeze(0).detach()

    @torch.inference_mode()
    def encode_patches(img: Image.Image) -> PatchTokens:
        hs = _hidden_states(img)
        toks = []
        for lid in layer_ids:
            t = _strip_cls_reg(hs[lid], n_skip=1)  # drop CLS
            assert t.shape[0] == grid * grid, f"CLIP L{lid}: {t.shape[0]} != {grid*grid}"
            toks.append(t.detach())
        return PatchTokens(layer_tokens=toks, grid_hw=(grid, grid), layer_ids=list(layer_ids))

    flops = None
    try:
        from thop import profile

        dummy = torch.randn(1, 3, image_size, image_size, device=device)
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
        "patch_grid": [grid, grid],
        "layers": layer_ids,
        "feature_mode": "multi_layer_patch",
    }
    return encode, encode_patches, meta


def load_dinov3_encoder(
    model_path: str | Path,
    device: str = "cuda:0",
    image_size: int = 224,
    layers: list[int] | None = None,
) -> tuple[EncodeFn, EncodePatchesFn, dict[str, Any]]:
    from transformers import AutoImageProcessor, AutoModel

    model = AutoModel.from_pretrained(str(model_path)).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(str(model_path))
    patch = int(getattr(model.config, "patch_size", 16))
    n_reg = int(getattr(model.config, "num_register_tokens", 0) or 0)
    grid = image_size // patch
    n_layers = int(model.config.num_hidden_layers)
    layer_ids = list(layers or DEFAULT_VIT_LAYERS)
    for lid in layer_ids:
        if lid < 1 or lid > n_layers:
            raise ValueError(f"DINOv3 layer {lid} out of range 1..{n_layers}")

    @torch.inference_mode()
    def _hidden_states(img: Image.Image) -> tuple[torch.Tensor, ...]:
        img_r = img.resize((image_size, image_size), Image.BICUBIC)
        inputs = processor(images=img_r, return_tensors="pt")
        pv = inputs["pixel_values"].to(device)
        if pv.shape[-1] != image_size:
            pv = torch.nn.functional.interpolate(
                pv, size=(image_size, image_size), mode="bilinear", align_corners=False
            )
        out = model(pixel_values=pv, output_hidden_states=True)
        return out.hidden_states

    @torch.inference_mode()
    def encode(img: Image.Image) -> torch.Tensor:
        hs = _hidden_states(img)
        return hs[-1][:, 0].squeeze(0).detach()

    @torch.inference_mode()
    def encode_patches(img: Image.Image) -> PatchTokens:
        hs = _hidden_states(img)
        toks = []
        n_skip = 1 + n_reg  # CLS + registers
        for lid in layer_ids:
            t = _strip_cls_reg(hs[lid], n_skip=n_skip)
            assert t.shape[0] == grid * grid, f"DINO L{lid}: {t.shape[0]} != {grid*grid}"
            toks.append(t.detach())
        return PatchTokens(layer_tokens=toks, grid_hw=(grid, grid), layer_ids=list(layer_ids))

    flops = None
    try:
        from thop import profile

        dummy = torch.randn(1, 3, image_size, image_size, device=device)
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
        "num_register_tokens": n_reg,
        "patch_grid": [grid, grid],
        "layers": layer_ids,
        "feature_mode": "multi_layer_patch",
    }
    return encode, encode_patches, meta


def _visual_state_from_hf_safetensors(model_path: Path) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    state: dict[str, torch.Tensor] = {}
    for shard in sorted(model_path.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("model.visual."):
                    state[k[len("model.visual.") :]] = f.get_tensor(k)
    return state


def _visual_state_from_mmproj_gguf(gguf_path: Path) -> dict[str, torch.Tensor]:
    """Map llama.cpp clip mmproj (F16) tensors → Qwen3VLVisionModel state_dict keys.

    GGUF stores dims reversed vs torch; reshape with reversed(shape). Weights match the
    HF Qwen3.5-0.8B vision tower (bit-exact within fp16).
    """
    import numpy as np
    from gguf import GGUFReader

    reader = GGUFReader(str(gguf_path))

    def _tensor(name: str) -> torch.Tensor:
        for t in reader.tensors:
            if t.name == name:
                shape = tuple(int(x) for x in t.shape)
                arr = np.asarray(t.data).reshape(tuple(reversed(shape)))
                return torch.from_numpy(arr.copy())
        raise KeyError(f"missing gguf tensor: {name}")

    state: dict[str, torch.Tensor] = {}
    # patch embed: two temporal kernels (16,16,3,C) → (C,3,2,16,16)
    w0 = _tensor("v.patch_embd.weight")
    w1 = _tensor("v.patch_embd.weight.1")
    state["patch_embed.proj.weight"] = torch.stack([w0, w1], dim=2)
    state["patch_embed.proj.bias"] = _tensor("v.patch_embd.bias")
    state["pos_embed.weight"] = _tensor("v.position_embd.weight")

    # discover block count
    block_ids = sorted(
        {
            int(t.name.split(".")[2])
            for t in reader.tensors
            if t.name.startswith("v.blk.") and t.name.split(".")[2].isdigit()
        }
    )
    for i in block_ids:
        p = f"v.blk.{i}"
        state[f"blocks.{i}.attn.qkv.weight"] = _tensor(f"{p}.attn_qkv.weight")
        state[f"blocks.{i}.attn.qkv.bias"] = _tensor(f"{p}.attn_qkv.bias")
        state[f"blocks.{i}.attn.proj.weight"] = _tensor(f"{p}.attn_out.weight")
        state[f"blocks.{i}.attn.proj.bias"] = _tensor(f"{p}.attn_out.bias")
        state[f"blocks.{i}.mlp.linear_fc1.weight"] = _tensor(f"{p}.ffn_up.weight")
        state[f"blocks.{i}.mlp.linear_fc1.bias"] = _tensor(f"{p}.ffn_up.bias")
        state[f"blocks.{i}.mlp.linear_fc2.weight"] = _tensor(f"{p}.ffn_down.weight")
        state[f"blocks.{i}.mlp.linear_fc2.bias"] = _tensor(f"{p}.ffn_down.bias")
        state[f"blocks.{i}.norm1.weight"] = _tensor(f"{p}.ln1.weight")
        state[f"blocks.{i}.norm1.bias"] = _tensor(f"{p}.ln1.bias")
        state[f"blocks.{i}.norm2.weight"] = _tensor(f"{p}.ln2.weight")
        state[f"blocks.{i}.norm2.bias"] = _tensor(f"{p}.ln2.bias")

    # merger (mm.0 / mm.2) — unused by pre-merger AD, skip if shapes differ
    try:
        state["merger.linear_fc1.weight"] = _tensor("mm.0.weight")
        state["merger.linear_fc1.bias"] = _tensor("mm.0.bias")
        state["merger.linear_fc2.weight"] = _tensor("mm.2.weight")
        state["merger.linear_fc2.bias"] = _tensor("mm.2.bias")
    except KeyError:
        pass
    return state


def load_qwen35_vision_encoder(
    model_path: str | Path,
    device: str = "cuda:0",
    max_pixels: int = 224 * 224,
    layers: list[int] | None = None,
    mmproj_gguf: str | Path | None = None,
    config_path: str | Path | None = None,
) -> tuple[EncodeFn, EncodePatchesFn, dict[str, Any]]:
    """Qwen3.5-0.8B vision tower — multi-layer **pre-merger** patch tokens.

    Spatial merge is skipped for AD maps so resolution stays grid_thw (e.g. 14×14).

    Args:
      model_path: HF dir (config + optional safetensors) used for vision_config / processor.
      mmproj_gguf: optional llama.cpp mmproj GGUF (F16). When set, vision weights load from
        GGUF instead of HF safetensors (quant package vision tower).
    """
    import json

    import torch.nn.functional as F
    from transformers import AutoProcessor, Qwen3VLVisionModel
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

    model_path = Path(model_path)
    cfg_src = Path(config_path) if config_path else model_path
    if not (cfg_src / "config.json").exists() and model_path.name.endswith(".gguf"):
        raise FileNotFoundError(
            f"need HF config.json for vision_config; got {cfg_src}. "
            "Pass config_path=.../Qwen3.5-0.8B"
        )
    raw = json.loads((cfg_src / "config.json").read_text(encoding="utf-8"))
    vc = dict(raw["vision_config"])
    vc.pop("model_type", None)
    cfg = Qwen3VLVisionConfig(**vc)
    visual = Qwen3VLVisionModel(cfg)

    if mmproj_gguf is not None:
        gguf_path = Path(mmproj_gguf)
        if gguf_path.is_dir():
            cands = sorted(gguf_path.glob("*mmproj*.gguf")) + sorted(gguf_path.glob("*.gguf"))
            if not cands:
                raise FileNotFoundError(f"no mmproj gguf under {gguf_path}")
            gguf_path = next((p for p in cands if "mmproj" in p.name.lower()), cands[0])
        state = _visual_state_from_mmproj_gguf(gguf_path)
        weight_src = f"mmproj_gguf:{gguf_path.name}"
        disk_bytes = gguf_path.stat().st_size
        package_dir = gguf_path.parent
        package_disk_bytes = sum(p.stat().st_size for p in package_dir.glob("*.gguf"))
    else:
        state = _visual_state_from_hf_safetensors(model_path)
        weight_src = "hf_safetensors"
        # vision-only nbytes (fair vs mmproj); also keep full HF package size
        disk_bytes = int(sum(t.numel() * t.element_size() for t in state.values()))
        package_disk_bytes = sum(p.stat().st_size for p in model_path.glob("*.safetensors"))
        gguf_path = None
        package_dir = model_path

    missing, unexpected = visual.load_state_dict(state, strict=False)
    # load_state_dict may reintroduce fp32 weights — cast after load (transformers 5.x)
    visual = visual.to(device=device, dtype=torch.bfloat16)
    loader_note = (
        f"weights={weight_src} -> Qwen3VLVisionModel; "
        f"loaded={len(state)} missing={len(missing)} unexpected={len(unexpected)}; "
        f"AD uses pre-merger tokens"
    )
    visual.eval()

    proc_path = cfg_src if (cfg_src / "preprocessor_config.json").exists() or (cfg_src / "processor_config.json").exists() else model_path
    try:
        processor = AutoProcessor.from_pretrained(str(proc_path), trust_remote_code=True)
    except Exception:
        proc_path = Path("model_card/Qwen3-VL-4B-Instruct")
        processor = AutoProcessor.from_pretrained(str(proc_path))
        loader_note += f" | processor={proc_path.name}"

    if hasattr(processor, "image_processor"):
        ip = processor.image_processor
        # Keep AD near ~224^2. Newer Qwen2VLImageProcessor uses size.{shortest,longest}_edge
        # as min/max pixel *area* (not max_pixels attrs).
        min_pixels = min(3136, int(max_pixels))
        if hasattr(ip, "max_pixels"):
            ip.max_pixels = int(max_pixels)
        if hasattr(ip, "min_pixels"):
            ip.min_pixels = min_pixels
        if hasattr(ip, "size"):
            try:
                ip.size = {"shortest_edge": min_pixels, "longest_edge": int(max_pixels)}
            except Exception:
                pass
            loader_note += f" | resize_pixels=[{min_pixels},{int(max_pixels)}]"

    n_blocks = len(visual.blocks)
    layer_ids = list(layers or DEFAULT_QWEN_LAYERS)
    for lid in layer_ids:
        if lid < 1 or lid > n_blocks:
            raise ValueError(f"Qwen vision layer {lid} out of range 1..{n_blocks}")
    layer_set = set(layer_ids)
    merge = int(getattr(cfg, "spatial_merge_size", 2) or 2)

    def _unpack_merge_order(tokens: torch.Tensor, h: int, w: int, m: int) -> torch.Tensor:
        """Convert Qwen merge-packed tokens [H*W, D] → spatial row-major [H*W, D].

        Packing matches pos_embed: (h/m, m, w/m, m) → permute → (h/m, w/m, m, m).
        """
        d = tokens.shape[-1]
        x = tokens.reshape(h // m, w // m, m, m, d)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (h/m, m, w/m, m, D) = spatial
        return x.reshape(h * w, d)

    @torch.inference_mode()
    def _pre_merge_layers(img: Image.Image) -> tuple[dict[int, torch.Tensor], tuple[int, int]]:
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

        hidden_states = visual.patch_embed(pv)
        pos_embeds = visual.fast_pos_embed_interpolate(grid)
        # transformers 5.x may return fp32 pos embeds → keep activation dtype = model dtype
        pos_embeds = pos_embeds.to(dtype=hidden_states.dtype)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb = visual.rot_pos_emb(grid).to(dtype=hidden_states.dtype)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        cu_seqlens = torch.repeat_interleave(grid[:, 1] * grid[:, 2], grid[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        t, h, w = [int(x) for x in grid[0].tolist()]
        captured: dict[int, torch.Tensor] = {}
        for layer_num, blk in enumerate(visual.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            lid = layer_num + 1
            if lid in layer_set:
                tok = hidden_states.float().detach()
                # single-image path: unpack merge order → row-major spatial
                if t == 1 and tok.shape[0] == h * w and h % merge == 0 and w % merge == 0:
                    tok = _unpack_merge_order(tok, h, w, merge)
                captured[lid] = tok

        # pre-merger spatial grid (no spatial_merge applied)
        return captured, (h, w)

    @torch.inference_mode()
    def encode(img: Image.Image) -> torch.Tensor:
        captured, _ = _pre_merge_layers(img)
        # mean of last selected layer
        return captured[layer_ids[-1]].mean(dim=0)

    @torch.inference_mode()
    def encode_patches(img: Image.Image) -> PatchTokens:
        captured, (h, w) = _pre_merge_layers(img)
        toks = []
        n = h * w
        for lid in layer_ids:
            t = captured[lid]
            if t.shape[0] != n:
                # multi-image batching unlikely; take first image slice
                t = t[:n]
            assert t.shape[0] == n, f"Qwen L{lid}: {t.shape[0]} != {n}"
            toks.append(t)
        return PatchTokens(layer_tokens=toks, grid_hw=(h, w), layer_ids=list(layer_ids))

    flops_g = None
    try:
        from thop import profile

        _img = Image.new("RGB", (224, 224), color=(128, 128, 128))
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
        "backbone": "Qwen3.5-0.8B-vision" + ("-mmproj-gguf" if mmproj_gguf else ""),
        "model_path": str(model_path),
        "mmproj_gguf": str(gguf_path) if gguf_path is not None else None,
        "weight_source": weight_src,
        "disk_mb": float(disk_bytes) / (1024**2),
        "package_disk_mb": float(package_disk_bytes) / (1024**2),
        "package_dir": str(package_dir),
        "max_pixels": max_pixels,
        "params_m": count_params_m(visual),
        "flops_g": flops_g,
        "device": device,
        "loader_note": loader_note,
        "vision_depth": int(cfg.depth),
        "vision_hidden": int(cfg.hidden_size),
        "spatial_merge_size": merge,
        "layers": layer_ids,
        "feature_mode": "multi_layer_pre_merger_patch",
        "note": "AD maps use pre-merger tokens (e.g. 14x14), not post-merger 7x7",
    }
    return encode, encode_patches, meta


def load_qwen35_mmproj_gguf(
    mmproj_gguf: str | Path,
    *,
    config_path: str | Path = "model_card/Qwen3.5-0.8B",
    device: str = "cuda:0",
    max_pixels: int = 224 * 224,
    layers: list[int] | None = None,
    processor_path: str | Path | None = None,
) -> tuple[EncodeFn, EncodePatchesFn, dict[str, Any]]:
    """Convenience: load quantized package vision tower from mmproj GGUF (F16)."""
    cfg_path = Path(config_path)
    encode, encode_patches, meta = load_qwen35_vision_encoder(
        cfg_path,
        device=device,
        max_pixels=max_pixels,
        layers=layers,
        mmproj_gguf=mmproj_gguf,
        config_path=cfg_path,
    )
    # optional alternate processor
    if processor_path is not None:
        meta["processor_path"] = str(processor_path)
    return encode, encode_patches, meta
