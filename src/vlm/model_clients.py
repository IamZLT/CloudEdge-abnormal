from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .parse import parse_vlm_json
from .qwen_client import QwenVLClient, VLMResult


DEFAULT_PROMPT = (
    "Inspect the product image and decide whether it is defective. "
    'Reply ONLY JSON: {"decision":"OK" or "NG","confidence":0-1,'
    '"defect_type":str,"reason":str}'
)


def _torch_dtype(name: str):
    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "auto": "auto",
    }.get(str(name).lower(), torch.bfloat16)


def _device_kwargs(device: str) -> tuple[dict[str, Any], str | None]:
    value = str(device)
    if value.lower() == "auto" or ("," in value and "cuda" in value):
        return {"device_map": "auto"}, None
    if value.startswith("cuda"):
        return {"device_map": value}, None
    return {}, value


def _open_rgb(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            return source.convert("RGB")
    return image.convert("RGB")


def _timed_start(device: torch.device) -> tuple[bool, float]:
    use_cuda = torch.cuda.is_available() and device.type == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    return use_cuda, time.perf_counter()


def _finish_result(
    raw: str,
    started: float,
    use_cuda: bool,
    device: torch.device,
    role: str,
    model_path: str,
) -> VLMResult:
    if use_cuda:
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - started) * 1000.0
    parsed = parse_vlm_json(raw)
    peak = float(torch.cuda.max_memory_allocated(device) / (1024**2)) if use_cuda else None
    return VLMResult(
        decision=parsed["decision"],
        confidence=float(parsed["confidence"]),
        defect_type=parsed.get("defect_type", "none"),
        reason=parsed.get("reason", ""),
        latency_ms=float(latency_ms),
        parse_ok=bool(parsed.get("parse_ok", False)),
        raw=raw,
        role=role,
        model_path=model_path,
        peak_mem_mb=peak,
    )


class _RoutingMixin:
    def is_hard(self, result: VLMResult, conf_low: float, conf_high: float, use_band: bool = True) -> bool:
        if not result.parse_ok or result.confidence < conf_low:
            return True
        return bool(use_band and conf_low <= result.confidence < conf_high)


class TransformersVLMClient(_RoutingMixin):
    """Generic Transformers multimodal interface, currently used by Qwen3.5."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_new_tokens: int = 128,
        role: str = "cloud",
        prompt: str | None = None,
        enable_thinking: bool = False,
        **_: Any,
    ):
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.model_path = str(model_path)
        self.role = role
        self.max_new_tokens = int(max_new_tokens)
        self.prompt = prompt or DEFAULT_PROMPT
        self.enable_thinking = bool(enable_thinking)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"model_path not found: {self.model_path}")

        load_kwargs, to_device = _device_kwargs(device)
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_path,
            dtype=_torch_dtype(dtype),
            trust_remote_code=True,
            **load_kwargs,
        )
        if to_device is not None:
            self.model = self.model.to(to_device)
        self.model.eval()
        self._input_device = next(self.model.parameters()).device

    @torch.inference_mode()
    def infer(self, image: str | Path | Image.Image) -> VLMResult:
        pil = _open_rgb(image)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text", "text": self.prompt},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=self.enable_thinking,
        )
        inputs = {k: v.to(self._input_device) if hasattr(v, "to") else v for k, v in inputs.items()}
        use_cuda, started = _timed_start(self._input_device)
        generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new_tokens = generated[:, inputs["input_ids"].shape[-1]:]
        raw = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return _finish_result(raw, started, use_cuda, self._input_device, self.role, self.model_path)


def _internvl_tiles(image: Image.Image, image_size: int = 448, max_tiles: int = 6) -> torch.Tensor:
    from torchvision import transforms as T
    from torchvision.transforms.functional import InterpolationMode

    width, height = image.size
    aspect = width / height
    ratios = sorted(
        {(i, j) for n in range(1, max_tiles + 1) for i in range(1, n + 1)
         for j in range(1, n + 1) if 1 <= i * j <= max_tiles},
        key=lambda x: x[0] * x[1],
    )
    target = min(
        ratios,
        key=lambda r: (abs(aspect - r[0] / r[1]),
                       -int(width * height > 0.5 * image_size * image_size * r[0] * r[1])),
    )
    grid_w, grid_h = target
    resized = image.resize((image_size * grid_w, image_size * grid_h))
    tiles = []
    for index in range(grid_w * grid_h):
        left = (index % grid_w) * image_size
        top = (index // grid_w) * image_size
        tiles.append(resized.crop((left, top, left + image_size, top + image_size)))
    if len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size)))
    transform = T.Compose([
        T.Lambda(lambda x: x.convert("RGB")),
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return torch.stack([transform(tile) for tile in tiles])


class InternVLClient(_RoutingMixin):
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_new_tokens: int = 128,
        role: str = "cloud",
        prompt: str | None = None,
        max_tiles: int = 6,
        **_: Any,
    ):
        from transformers import AutoConfig, AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        self.model_path = str(model_path)
        self.role = role
        self.prompt = prompt or DEFAULT_PROMPT
        self.max_new_tokens = int(max_new_tokens)
        self.max_tiles = int(max_tiles)
        self.dtype = _torch_dtype(dtype)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"model_path not found: {self.model_path}")
        load_kwargs, to_device = _device_kwargs(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False,
            fix_mistral_regex=True,
        )
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        class_ref = config.auto_map["AutoModel"]
        model_class = get_class_from_dynamic_module(class_ref, self.model_path)
        # InternVL3.5's remote code targets Transformers 4.x. Transformers 5.x
        # expects this mapping during allocator warm-up, as it does for MiniCPM.
        if not hasattr(model_class, "all_tied_weights_keys"):
            model_class.all_tied_weights_keys = {}
        self.model = model_class.from_pretrained(
            self.model_path, config=config, dtype=self.dtype, trust_remote_code=True,
            low_cpu_mem_usage=True, attn_implementation="eager",
            **load_kwargs,
        )
        if to_device is not None:
            self.model = self.model.to(to_device)
        self.model.eval()
        self._input_device = next(self.model.parameters()).device

    @torch.inference_mode()
    def infer(self, image: str | Path | Image.Image) -> VLMResult:
        pixels = _internvl_tiles(_open_rgb(image), max_tiles=self.max_tiles)
        pixels = pixels.to(device=self._input_device, dtype=self.dtype)
        use_cuda, started = _timed_start(self._input_device)
        raw = self.model.chat(
            self.tokenizer,
            pixels,
            f"<image>\n{self.prompt}",
            {"max_new_tokens": self.max_new_tokens, "do_sample": False},
        )
        return _finish_result(str(raw), started, use_cuda, self._input_device, self.role, self.model_path)


class MiniCPMVClient(_RoutingMixin):
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_new_tokens: int = 128,
        role: str = "cloud",
        prompt: str | None = None,
        **_: Any,
    ):
        from transformers import AutoConfig, AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        self.model_path = str(model_path)
        self.role = role
        self.prompt = prompt or DEFAULT_PROMPT
        self.max_new_tokens = int(max_new_tokens)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"model_path not found: {self.model_path}")
        load_kwargs, to_device = _device_kwargs(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        class_ref = config.auto_map["AutoModel"]
        model_class = get_class_from_dynamic_module(class_ref, self.model_path)
        # MiniCPM-V 4.5 targets Transformers 4.x. Transformers 5.x (required by
        # Qwen3.5) reads this new mapping during quantized allocator warm-up.
        if not hasattr(model_class, "all_tied_weights_keys"):
            model_class.all_tied_weights_keys = {}
        self.model = model_class.from_pretrained(
            self.model_path,
            config=config,
            dtype=_torch_dtype(dtype),
            trust_remote_code=True,
            attn_implementation="sdpa",
            **load_kwargs,
        )
        if to_device is not None:
            self.model = self.model.to(to_device)
        self.model.eval()
        self._input_device = next(self.model.parameters()).device

    @torch.inference_mode()
    def infer(self, image: str | Path | Image.Image) -> VLMResult:
        pil = _open_rgb(image)
        use_cuda, started = _timed_start(self._input_device)
        raw = self.model.chat(
            msgs=[{"role": "user", "content": [pil, self.prompt]}],
            tokenizer=self.tokenizer,
            enable_thinking=False,
            stream=False,
            sampling=False,
            max_new_tokens=self.max_new_tokens,
        )
        return _finish_result(str(raw), started, use_cuda, self._input_device, self.role, self.model_path)


def detect_vlm_backend(model_path: str | Path) -> str:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json not found under model_path: {model_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = str(config.get("model_type", "")).lower()
    if model_type == "qwen3_vl":
        return "qwen3_vl"
    if model_type == "qwen3_5":
        return "transformers"
    if model_type == "internvl_chat":
        return "internvl"
    if model_type == "minicpmv":
        return "minicpm"
    raise ValueError(f"unsupported VLM model_type={model_type!r} in {config_path}")


def create_vlm_client(model_path: str, backend: str = "auto", **kwargs: Any):
    selected = detect_vlm_backend(model_path) if backend in {"", "auto", None} else backend.lower()
    clients = {
        "qwen3_vl": QwenVLClient,
        "qwen3.5": TransformersVLMClient,
        "qwen3_5": TransformersVLMClient,
        "transformers": TransformersVLMClient,
        "internvl": InternVLClient,
        "minicpm": MiniCPMVClient,
    }
    if selected not in clients:
        raise ValueError(f"unsupported VLM backend={selected!r}; choices={sorted(clients)}")
    return clients[selected](model_path=model_path, **kwargs)
