"""Qwen3.5 edge RouteAgent — decide whether to upload a sample to the cloud.

Uses full multimodal generate (image + structured context). Network outage is a
hard gate (no LLM call). Parse failures fall back to hard_margin heuristic.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image

DEFAULT_MODEL = "/data2/zlt/anomaly_detection_llm/model_card/Qwen3.5-0.8B"

DEFAULT_ROUTE_PROMPT = """You are an edge routing agent for industrial defect inspection.
Given the product image and CONTEXT, decide whether to upload to a stronger cloud VLM.

Reply with ONLY one JSON object (no markdown):
{"upload": true or false, "confidence": float 0-1, "reason": "short English"}

Rules (priority order):
1. If network_profile is outage or link is unavailable → upload=false.
2. If n_gallery==0 (cold start, no local normals) and network is usable → upload=true.
3. If n_gallery>0 and local decision is confident (score far from threshold) → upload=false.
4. If n_gallery>0 but uncertain (near threshold / low confidence) → upload=true.
5. Prefer local when the edge score is clearly OK or clearly NG.
"""


@dataclass
class RouteContext:
    image: str | Path | Image.Image
    category: str
    n_gallery: int
    edge_score: float
    edge_thr: float
    edge_decision: str  # OK | NG
    network_profile: str = "fair"
    network: dict[str, Any] = field(default_factory=dict)
    hard_margin: float = 0.05

    def score_margin(self) -> float:
        return abs(float(self.edge_score) - float(self.edge_thr))

    def context_text(self) -> str:
        net = dict(self.network or {})
        net.setdefault("profile", self.network_profile)
        return (
            f"CONTEXT:\n"
            f"- category: {self.category}\n"
            f"- n_gallery: {int(self.n_gallery)}\n"
            f"- edge_decision: {self.edge_decision}\n"
            f"- edge_score: {float(self.edge_score):.6f}\n"
            f"- edge_threshold: {float(self.edge_thr):.6f}\n"
            f"- score_margin: {self.score_margin():.6f}\n"
            f"- hard_margin: {float(self.hard_margin):.6f}\n"
            f"- network_profile: {self.network_profile}\n"
            f"- network: {json.dumps(net, ensure_ascii=False)}\n"
        )


@dataclass
class RouteDecision:
    upload: bool
    confidence: float
    reason: str
    latency_ms: float = 0.0
    parse_ok: bool = True
    source: str = "llm"  # llm | hard_gate | heuristic_fallback
    raw: str = ""
    network_profile: str = "fair"
    peak_mem_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_route_json(text: str) -> dict[str, Any]:
    """Extract upload-routing JSON from model text."""
    raw = (text or "").strip()
    if not raw:
        return {"upload": None, "confidence": 0.0, "reason": "empty_response", "parse_ok": False, "raw": text}

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.S | re.I)
    blob = fence.group(1) if fence else raw
    m = re.search(r"\{.*\}", blob, flags=re.S)
    payload: dict[str, Any] = {}
    parse_ok = False
    if m:
        try:
            payload = json.loads(m.group(0))
            parse_ok = True
        except json.JSONDecodeError:
            payload = {}

    upload = payload.get("upload", None)
    if isinstance(upload, str):
        upload = upload.strip().lower() in {"1", "true", "yes", "y"}
    elif upload is not None:
        upload = bool(upload)

    try:
        confidence = float(payload.get("confidence", 0.5 if parse_ok else 0.3))
    except (TypeError, ValueError):
        confidence = 0.3
    confidence = max(0.0, min(1.0, confidence))
    reason = str(payload.get("reason", "") or "")
    return {
        "upload": upload,
        "confidence": confidence,
        "reason": reason,
        "parse_ok": parse_ok and upload is not None,
        "raw": text,
    }


def heuristic_upload(ctx: RouteContext) -> bool:
    """Fallback: near-threshold or empty gallery → upload when network usable."""
    if str(ctx.network_profile).lower() == "outage":
        return False
    if int(ctx.n_gallery) <= 0:
        return True
    return ctx.score_margin() < float(ctx.hard_margin)


class RouteAgent:
    """Full Qwen3.5 multimodal agent for cloud-upload routing."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_new_tokens: int = 96,
        prompt: str | None = None,
        hard_block_on_outage: bool = True,
    ):
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        self.model_path = str(model_path)
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.prompt = prompt or DEFAULT_ROUTE_PROMPT
        self.hard_block_on_outage = bool(hard_block_on_outage)

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"model_path not found: {self.model_path}")

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "auto": "auto",
        }.get(str(dtype).lower(), torch.bfloat16)

        if str(device).lower() == "auto" or ("," in str(device) and "cuda" in str(device)):
            device_map = "auto"
            to_device = None
        elif str(device).startswith("cuda"):
            device_map = device
            to_device = None
        else:
            device_map = None
            to_device = device

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        if to_device is not None:
            self.model = self.model.to(to_device)
        self.model.eval()
        self._input_device = next(self.model.parameters()).device

    @classmethod
    def from_config(cls, route_cfg: dict | None = None, collab_cfg: dict | None = None) -> "RouteAgent":
        cfg = dict(route_cfg or {})
        if not cfg and collab_cfg:
            cfg = dict((collab_cfg or {}).get("route_agent") or {})
        return cls(
            model_path=str(cfg.get("model_path") or DEFAULT_MODEL),
            device=str(cfg.get("device") or "cuda:0"),
            dtype=str(cfg.get("dtype") or "bfloat16"),
            max_new_tokens=int(cfg.get("max_new_tokens") or 96),
            prompt=cfg.get("prompt"),
            hard_block_on_outage=bool(cfg.get("hard_block_on_outage", True)),
        )

    def decide(self, ctx: RouteContext) -> RouteDecision:
        profile = str(ctx.network_profile or "fair").lower()
        if self.hard_block_on_outage and profile == "outage":
            return RouteDecision(
                upload=False,
                confidence=1.0,
                reason="hard_gate: network outage — stay local",
                latency_ms=0.0,
                parse_ok=True,
                source="hard_gate",
                network_profile=profile,
            )

        if isinstance(ctx.image, (str, Path)):
            pil = Image.open(ctx.image).convert("RGB")
        else:
            pil = ctx.image.convert("RGB")

        user_text = f"{self.prompt.strip()}\n\n{ctx.context_text()}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        # Prefer apply_chat_template return_dict path (works for Qwen3.5)
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=[text], images=[pil], padding=True, return_tensors="pt")

        target = self._input_device
        inputs = {k: v.to(target) if hasattr(v, "to") else v for k, v in inputs.items()}

        use_cuda = torch.cuda.is_available() and getattr(target, "type", None) == "cuda"
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(target)
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        if use_cuda:
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        in_len = inputs["input_ids"].shape[-1]
        raw = self.processor.batch_decode(
            generated[:, in_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        parsed = parse_route_json(raw)
        peak = float(torch.cuda.max_memory_allocated(target) / (1024**2)) if use_cuda else None

        if not parsed.get("parse_ok") or parsed.get("upload") is None:
            upload = heuristic_upload(ctx)
            return RouteDecision(
                upload=upload,
                confidence=float(parsed.get("confidence") or 0.3),
                reason=parsed.get("reason") or "heuristic_fallback: parse_failed",
                latency_ms=float(latency_ms),
                parse_ok=False,
                source="heuristic_fallback",
                raw=raw,
                network_profile=profile,
                peak_mem_mb=peak,
            )

        return RouteDecision(
            upload=bool(parsed["upload"]),
            confidence=float(parsed["confidence"]),
            reason=str(parsed.get("reason") or ""),
            latency_ms=float(latency_ms),
            parse_ok=True,
            source="llm",
            raw=raw,
            network_profile=profile,
            peak_mem_mb=peak,
        )


def resolve_network_profile(collab_cfg: dict | None = None, override: str | None = None) -> tuple[str, dict]:
    """Return (profile_name, network_dict) from collab.network config."""
    from src.network_sim import resolve_profile

    net_cfg = dict((collab_cfg or {}).get("network") or {})
    if override:
        net_cfg["profile"] = override
    if "profile" not in net_cfg:
        net_cfg["profile"] = "fair"
    prof = resolve_profile(net_cfg)
    return prof.name, prof.to_dict()
