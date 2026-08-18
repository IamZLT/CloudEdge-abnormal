from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image


@dataclass
class VLMResult:
    decision: str
    confidence: float
    defect_type: str
    reason: str
    latency_ms: float
    parse_ok: bool
    raw: str
    role: str
    model_path: str
    peak_mem_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "defect_type": self.defect_type,
            "reason": self.reason,
            "latency_ms": self.latency_ms,
            "parse_ok": self.parse_ok,
            "role": self.role,
            "model_path": self.model_path,
            "peak_mem_mb": self.peak_mem_mb,
            "raw": self.raw,
        }


def resolve_vlm_model_class(model_path: str, model_family: str | None = None):
    """Return (ModelClass, family) for Qwen3-VL or Qwen3.5."""
    fam = str(model_family or "").strip().lower()
    path = str(model_path or "").lower()
    if not fam:
        if "qwen3.5" in path or "qwen3_5" in path or "qwen35" in path:
            fam = "qwen3_5"
        else:
            fam = "qwen3_vl"
    if fam in {"qwen3_5", "qwen35", "qwen3.5"}:
        from transformers import Qwen3_5ForConditionalGeneration

        return Qwen3_5ForConditionalGeneration, "qwen3_5"
    from transformers import Qwen3VLForConditionalGeneration

    return Qwen3VLForConditionalGeneration, "qwen3_vl"


class QwenVLClient:
    """Thin wrapper around local Qwen3-VL / Qwen3.5 checkpoints (+ optional LoRA adapter)."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_new_tokens: int = 128,
        role: str = "edge",
        prompt: str | None = None,
        adapter_path: str | None = None,
        model_family: str | None = None,
        max_pixels: int | None = None,
    ):
        from transformers import AutoProcessor

        self.model_path = str(model_path)
        self.adapter_path = str(adapter_path) if adapter_path else None
        self.device = device
        self.role = role
        self.max_new_tokens = int(max_new_tokens)
        self.prompt = prompt or (
            'Inspect the product image. Reply ONLY JSON: '
            '{"decision":"OK"|"NG","confidence":0-1,"defect_type":str,"reason":str}'
        )

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"model_path not found: {self.model_path}")
        if self.adapter_path and not Path(self.adapter_path).exists():
            raise FileNotFoundError(f"adapter_path not found: {self.adapter_path}")

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "auto": "auto",
        }.get(str(dtype).lower(), torch.bfloat16)

        ModelCls, fam = resolve_vlm_model_class(self.model_path, model_family)
        self.model_family = fam

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        # Optional resize: bound the image pixel area so the vision tower runs at a
        # fixed budget (e.g. 224*224) instead of native resolution. None = no resize.
        if max_pixels is not None and hasattr(self.processor, "image_processor"):
            ip = self.processor.image_processor
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
            print(f"[{role}] resize: max_pixels={int(max_pixels)}")
        # device="auto" / "cuda:0,1" → shard across visible GPUs (useful when single card <16GB free)
        if str(device).lower() == "auto" or ("," in str(device) and "cuda" in str(device)):
            device_map = "auto"
            to_device = None
        elif str(device).startswith("cuda"):
            device_map = device
            to_device = None
        else:
            device_map = None
            to_device = device

        self.model = ModelCls.from_pretrained(
            self.model_path,
            dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        if to_device is not None:
            self.model = self.model.to(to_device)

        if self.adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            print(f"[{role}] loaded LoRA adapter: {self.adapter_path}")

        self.model.eval()
        self._input_device = next(self.model.parameters()).device

    @torch.inference_mode()
    def infer(self, image: str | Path | Image.Image) -> VLMResult:
        from src.vlm.parse import parse_vlm_json

        if isinstance(image, (str, Path)):
            pil = Image.open(image).convert("RGB")
        else:
            pil = image.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        # Prefer chat template + process_vision_info when available
        try:
            from qwen_vl_utils import process_vision_info

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            kwargs = {
                "text": [text],
                "images": image_inputs,
                "padding": True,
                "return_tensors": "pt",
            }
            if video_inputs:
                kwargs["videos"] = video_inputs
            inputs = self.processor(**kwargs)
        except Exception:
            # Fallback without qwen_vl_utils: pass PIL directly
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=[pil],
                padding=True,
                return_tensors="pt",
            )

        target = getattr(self, "_input_device", None)
        if target is None:
            target = next(self.model.parameters()).device
        inputs = {k: v.to(target) if hasattr(v, "to") else v for k, v in inputs.items()}

        use_cuda = torch.cuda.is_available() and target.type == "cuda"
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(target)
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        if use_cuda:
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # decode only new tokens
        in_len = inputs["input_ids"].shape[-1]
        out_ids = generated[:, in_len:]
        raw = self.processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        parsed = parse_vlm_json(raw)

        peak = None
        if use_cuda:
            peak = float(torch.cuda.max_memory_allocated(target) / (1024**2))

        return VLMResult(
            decision=parsed["decision"],
            confidence=float(parsed["confidence"]),
            defect_type=parsed.get("defect_type", "none"),
            reason=parsed.get("reason", ""),
            latency_ms=float(latency_ms),
            parse_ok=bool(parsed.get("parse_ok", False)),
            raw=raw,
            role=self.role,
            model_path=self.adapter_path or self.model_path,
            peak_mem_mb=peak,
        )

    def is_hard(self, result: VLMResult, conf_low: float, conf_high: float, use_band: bool = True) -> bool:
        """Hard / uncertain sample → upload to cloud."""
        c = float(result.confidence)
        if not result.parse_ok:
            return True
        if c < conf_low:
            return True
        if use_band and conf_low <= c < conf_high:
            return True
        return False
