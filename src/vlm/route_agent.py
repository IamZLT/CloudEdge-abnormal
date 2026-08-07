"""Qwen3.5 edge RouteAgent — decide whether to upload a sample to the cloud.

Backends:
  - hf   : transformers Qwen3_5ForConditionalGeneration (bf16/fp16)
  - gguf : llama.cpp Q4_K_M LLM + mmproj-F16 vision (MTMDChatHandler)

Network outage is a hard gate (no LLM call). Parse failures fall back to
hard_margin heuristic.
"""
from __future__ import annotations

import base64
import io
import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image

DEFAULT_MODEL = "/data2/zlt/anomaly_detection_llm/model_card/Qwen3.5-0.8B"
DEFAULT_GGUF_DIR = str(Path(__file__).resolve().parents[2] / "model_card" / "qwen3.5VL-0.8B-q")

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


def _resolve_gguf_paths(
    model_path: str | Path,
    mmproj_gguf: str | Path | None = None,
) -> tuple[Path, Path]:
    """Return (llm_gguf, mmproj_gguf). model_path may be a dir or a .gguf file."""
    p = Path(model_path)
    if p.is_dir():
        llm_cands = sorted(
            x for x in p.glob("*.gguf") if "mmproj" not in x.name.lower() and "Q4" in x.name.upper()
        ) or sorted(x for x in p.glob("*.gguf") if "mmproj" not in x.name.lower())
        if not llm_cands:
            raise FileNotFoundError(f"no LLM gguf under {p}")
        llm = llm_cands[0]
        if mmproj_gguf is None:
            mm = sorted(p.glob("*mmproj*.gguf"))
            if not mm:
                raise FileNotFoundError(f"no mmproj gguf under {p}")
            mmproj = mm[0]
        else:
            mmproj = Path(mmproj_gguf)
    else:
        llm = p
        if mmproj_gguf is None:
            mm = sorted(p.parent.glob("*mmproj*.gguf"))
            if not mm:
                raise FileNotFoundError(f"need mmproj_gguf alongside {p}")
            mmproj = mm[0]
        else:
            mmproj = Path(mmproj_gguf)
    if not llm.exists():
        raise FileNotFoundError(llm)
    if not mmproj.exists():
        raise FileNotFoundError(mmproj)
    return llm, mmproj


def _pil_to_data_uri(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt.upper() in {"JPG", "JPEG"} else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


def _gpu_used_mb(device_index: int | None = None) -> float | None:
    """Device-level used memory (MB). Tracks llama.cpp CUDA better than torch peak."""
    try:
        idx = 0 if device_index is None else int(device_index)
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={idx}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        return float(out.splitlines()[0])
    except Exception:
        return None


def _cuda_index_from_device(device: str) -> int | None:
    s = str(device)
    if s.startswith("cuda:"):
        try:
            return int(s.split(":", 1)[1])
        except ValueError:
            return 0
    if s == "cuda":
        return 0
    return None


class RouteAgent:
    """Qwen3.5 multimodal agent for cloud-upload routing (HF or GGUF/Q4)."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_new_tokens: int = 96,
        prompt: str | None = None,
        hard_block_on_outage: bool = True,
        backend: str = "hf",
        mmproj_gguf: str | Path | None = None,
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
        verbose: bool = False,
    ):
        self.backend = str(backend or "hf").lower()
        if self.backend in {"q4", "quant", "llama", "llama_cpp", "llama.cpp"}:
            self.backend = "gguf"
        self.model_path = str(model_path)
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.prompt = prompt or DEFAULT_ROUTE_PROMPT
        self.hard_block_on_outage = bool(hard_block_on_outage)
        self.mmproj_gguf = str(mmproj_gguf) if mmproj_gguf else None
        self.n_gpu_layers = int(n_gpu_layers)
        self.n_ctx = int(n_ctx)
        self.verbose = bool(verbose)

        self.processor = None
        self.model = None
        self.llm = None
        self._input_device = None
        self._cuda_index = _cuda_index_from_device(device)
        self.meta: dict[str, Any] = {"backend": self.backend}

        if self.backend == "gguf":
            self._init_gguf()
        elif self.backend == "hf":
            self._init_hf(dtype=dtype)
        else:
            raise ValueError(f"unsupported route_agent backend: {backend} (use hf|gguf)")

    def _init_hf(self, dtype: str = "bfloat16") -> None:
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"model_path not found: {self.model_path}")

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "auto": "auto",
        }.get(str(dtype).lower(), torch.bfloat16)

        if str(self.device).lower() == "auto" or ("," in str(self.device) and "cuda" in str(self.device)):
            device_map = "auto"
            to_device = None
        elif str(self.device).startswith("cuda"):
            device_map = self.device
            to_device = None
        else:
            device_map = None
            to_device = self.device

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
        n_params = sum(p.numel() for p in self.model.parameters())
        self.meta.update(
            {
                "weight_source": "hf_safetensors",
                "params_m": n_params / 1e6,
                "dtype": str(dtype),
                "package_disk_mb": sum(p.stat().st_size for p in Path(self.model_path).glob("*.safetensors"))
                / (1024**2),
            }
        )

    def _init_gguf(self) -> None:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import MTMDChatHandler

        try:
            from llama_cpp import llama_cpp as _ll

            gpu_ok = bool(_ll.llama_supports_gpu_offload())
        except Exception:
            gpu_ok = False

        path = self.model_path
        if Path(path).is_dir() is False and not str(path).endswith(".gguf"):
            # allow pointing at HF path by mistake → fall back to default quant dir
            path = DEFAULT_GGUF_DIR
        llm_path, mmproj = _resolve_gguf_paths(path, self.mmproj_gguf)

        # Prefer reported capability; if False, still honor explicit n_gpu_layers>0
        # (some wheels mis-report) but default to CPU when offload unsupported.
        if gpu_ok:
            n_gpu = self.n_gpu_layers
        elif self.n_gpu_layers == 0:
            n_gpu = 0
        else:
            # try requested offload; llama.cpp falls back to CPU if CUDA missing
            n_gpu = self.n_gpu_layers

        mem_before = _gpu_used_mb(self._cuda_index)
        handler = MTMDChatHandler(clip_model_path=str(mmproj))
        self.llm = Llama(
            model_path=str(llm_path),
            chat_handler=handler,
            n_gpu_layers=n_gpu,
            n_ctx=self.n_ctx,
            verbose=self.verbose,
        )
        mem_after = _gpu_used_mb(self._cuda_index)
        self.model_path = str(llm_path)
        self.mmproj_gguf = str(mmproj)
        pkg = llm_path.parent
        footprint = None
        if mem_before is not None and mem_after is not None:
            footprint = max(0.0, float(mem_after) - float(mem_before))
        self.meta.update(
            {
                "weight_source": f"gguf_q4+mmproj:{llm_path.name}+{mmproj.name}",
                "llm_gguf": str(llm_path),
                "mmproj_gguf": str(mmproj),
                "llm_disk_mb": llm_path.stat().st_size / (1024**2),
                "mmproj_disk_mb": mmproj.stat().st_size / (1024**2),
                "package_disk_mb": sum(p.stat().st_size for p in pkg.glob("*.gguf")) / (1024**2),
                "n_gpu_layers": n_gpu,
                "gpu_offload_reported": gpu_ok,
                "n_ctx": self.n_ctx,
                "gpu_used_before_load_mb": mem_before,
                "gpu_used_after_load_mb": mem_after,
                # fair vs HF torch peak: incremental VRAM of this process load
                "gpu_footprint_mb": footprint,
            }
        )
        # CPU-only builds: track process RSS as "peak" proxy for decoder footprint
        try:
            import resource

            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            # Linux ru_maxrss is KB
            self.meta["process_rss_mb"] = float(rss_mb)
        except Exception:
            pass

    @classmethod
    def from_config(cls, route_cfg: dict | None = None, collab_cfg: dict | None = None) -> "RouteAgent":
        cfg = dict(route_cfg or {})
        if not cfg and collab_cfg:
            cfg = dict((collab_cfg or {}).get("route_agent") or {})
        # default backend: quantized GGUF (Q4 LLM + mmproj-F16)
        backend = str(cfg.get("backend") or "gguf").lower()
        root = Path(__file__).resolve().parents[2]

        def _resolve(p: str | Path | None) -> str | None:
            if p is None:
                return None
            path = Path(p)
            if not path.is_absolute():
                path = (root / path).resolve()
            return str(path)

        model_path = str(cfg.get("model_path") or DEFAULT_MODEL)
        mmproj = cfg.get("mmproj_gguf")
        if backend in {"gguf", "q4", "quant", "llama", "llama_cpp"}:
            model_path = _resolve(cfg.get("gguf_dir") or cfg.get("model_path") or DEFAULT_GGUF_DIR) or DEFAULT_GGUF_DIR
            mmproj = _resolve(mmproj)
        return cls(
            model_path=model_path,
            device=str(cfg.get("device") or "cuda:0"),
            dtype=str(cfg.get("dtype") or "bfloat16"),
            max_new_tokens=int(cfg.get("max_new_tokens") or 96),
            prompt=cfg.get("prompt"),
            hard_block_on_outage=bool(cfg.get("hard_block_on_outage", True)),
            backend=backend,
            mmproj_gguf=mmproj,
            n_gpu_layers=int(cfg.get("n_gpu_layers", -1)),
            n_ctx=int(cfg.get("n_ctx") or 4096),
            verbose=bool(cfg.get("verbose", False)),
        )

    def _decide_hf(self, pil: Image.Image, user_text: str) -> tuple[str, float, float | None]:
        assert self.processor is not None and self.model is not None
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil},
                    {"type": "text", "text": user_text},
                ],
            }
        ]
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
        mem0 = _gpu_used_mb(self._cuda_index) if use_cuda else None
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
        if use_cuda:
            peak_torch = float(torch.cuda.max_memory_allocated(target) / (1024**2))
            mem1 = _gpu_used_mb(self._cuda_index)
            peak = peak_torch
            if mem0 is not None and mem1 is not None:
                self.meta["gpu_used_delta_mb"] = mem1 - mem0
        else:
            peak = None
        return raw, latency_ms, peak

    def _decide_gguf(self, pil: Image.Image, user_text: str) -> tuple[str, float, float | None]:
        assert self.llm is not None
        uri = _pil_to_data_uri(pil)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": uri}},
                ],
            }
        ]
        mem0 = _gpu_used_mb(self._cuda_index)
        t0 = time.perf_counter()
        out = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=0.0,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        raw = out["choices"][0]["message"].get("content") or ""
        mem1 = _gpu_used_mb(self._cuda_index)
        peak = None
        if mem0 is not None and mem1 is not None:
            self.meta["gpu_used_before_mb"] = mem0
            self.meta["gpu_used_after_mb"] = mem1
            self.meta["gpu_used_delta_mb"] = mem1 - mem0
        # Prefer load footprint (incremental VRAM); else RSS for CPU builds
        if self.meta.get("gpu_footprint_mb") is not None and float(self.meta["gpu_footprint_mb"]) > 32:
            peak = float(self.meta["gpu_footprint_mb"])
        elif self.meta.get("process_rss_mb") is not None:
            peak = float(self.meta["process_rss_mb"])
        return raw, latency_ms, peak

    def mark_load_memory(self) -> None:
        """Record GPU used MB right after model load (for GGUF peak reporting)."""
        self.meta["gpu_used_after_load_mb"] = _gpu_used_mb(self._cuda_index)

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
        if self.backend == "gguf":
            raw, latency_ms, peak = self._decide_gguf(pil, user_text)
        else:
            raw, latency_ms, peak = self._decide_hf(pil, user_text)

        parsed = parse_route_json(raw)
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
