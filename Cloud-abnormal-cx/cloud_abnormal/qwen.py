from __future__ import annotations

import hashlib
import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from .dino import torch_dtype


@dataclass
class Region:
    bbox: list[float]
    confidence: float
    defect: str = "unknown"


@dataclass
class QwenOpinion:
    anomaly_probability: float = 0.0
    regions: list[Region] = field(default_factory=list)
    defect_type: str = "none"
    reason: str = ""


def _recover_fields(text: str) -> dict:
    """Recover useful fields when the model emits syntactically invalid JSON."""
    def string_field(name: str, default: str) -> str:
        match = re.search(rf'"{name}"\s*:\s*"((?:\\.|[^"\\])*)"', text, flags=re.DOTALL)
        if not match:
            return default
        try:
            return json.loads(f'"{match.group(1)}"')
        except json.JSONDecodeError:
            return match.group(1)

    probability_match = re.search(
        r'"anomaly_probability"\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+))', text
    )
    probability = float(probability_match.group(1)) if probability_match else 0.0
    regions = []
    object_pattern = re.compile(r'\{[^{}]*"bbox"\s*:\s*\[([^\]]+)\][^{}]*\}', re.DOTALL)
    for match in object_pattern.finditer(text):
        numbers = re.findall(r'-?(?:\d+(?:\.\d*)?|\.\d+)', match.group(1))
        if len(numbers) != 4:
            continue
        block = match.group(0)
        confidence_match = re.search(
            r'"confidence"\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+))', block
        )
        defect_match = re.search(r'"defect"\s*:\s*"((?:\\.|[^"\\])*)"', block)
        regions.append({
            "bbox": [float(value) for value in numbers],
            "confidence": float(confidence_match.group(1)) if confidence_match else probability,
            "defect": defect_match.group(1) if defect_match else "unknown",
        })
    return {
        "anomaly_probability": probability,
        "defect_type": string_field("defect_type", "unknown"),
        "regions": regions,
        "reason": string_field("reason", "Qwen response required tolerant parsing"),
        "parse_fallback": True,
    }


def _extract_json(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        warnings.warn("Qwen returned no JSON object; using DINO-only fallback for this image.")
        return _recover_fields(cleaned)
    candidate = cleaned[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as original_error:
        # Repair common generation mistakes: trailing commas and a missing comma
        # between a completed value and the next known object key.
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        keys = r'(?:anomaly_probability|defect_type|regions|reason|bbox|confidence|defect)'
        value_end = r'(?:[}\]"]|-?(?:\d+(?:\.\d*)?|\.\d+)|true|false|null)'
        repaired = re.sub(rf'({value_end})\s*("{keys}"\s*:)', r"\1, \2", repaired)
        try:
            data = json.loads(repaired)
            data["parse_repaired"] = True
            return data
        except json.JSONDecodeError:
            warnings.warn(
                f"Malformed Qwen JSON ({original_error}); recovered available fields and continued evaluation."
            )
            return _recover_fields(candidate)


class FrozenQwenInspector:
    def __init__(
        self,
        model_path: str,
        device: str,
        dtype: str,
        max_new_tokens: int,
        cache_dir: str,
    ) -> None:
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Qwen model not found: {path}")
        self.processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(path), local_files_only=True, dtype=torch_dtype(dtype), device_map=device
        ).eval()
        self.model.requires_grad_(False)
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("Qwen parameter freezing failed")
        self.max_new_tokens = max_new_tokens
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, category: str, target: Path, references: list[Path]) -> Path:
        stat = target.stat()
        signature = f"{target.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:" + ":".join(map(str, references))
        digest = hashlib.sha256(signature.encode()).hexdigest()[:24]
        return self.cache_dir / category / f"{digest}.json"

    @torch.inference_mode()
    def inspect(self, category: str, target: Path, references: list[Path]) -> QwenOpinion:
        cache = self._cache_path(category, target, references)
        if cache.exists():
            try:
                return self._to_opinion(json.loads(cache.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError) as exc:
                warnings.warn(f"Ignoring unreadable Qwen cache {cache}: {exc}")
        content: list[dict] = [{
            "type": "text",
            "text": (
                f"You are a conservative industrial visual quality inspector for the product class '{category}'. "
                "The following images labeled NORMAL REFERENCE show acceptable products. Compare the final TARGET "
                "image against them. Detect scratches, cracks, dents, contamination, missing/excess parts, deformation, "
                "color/texture irregularity, or any structural inconsistency. Do not mark normal lighting or viewpoint "
                "variation as defects. Small defects matter. Coordinates use [x1,y1,x2,y2] on a 0..1000 canvas."
            ),
        }]
        for index, path in enumerate(references):
            content.extend([
                {"type": "text", "text": f"NORMAL REFERENCE {index + 1}:"},
                {"type": "image", "image": str(path)},
            ])
        content.extend([
            {"type": "text", "text": "TARGET IMAGE TO INSPECT:"},
            {"type": "image", "image": str(target)},
            {"type": "text", "text": (
                "Return JSON only: {\"anomaly_probability\": number 0..1, \"defect_type\": string, "
                "\"regions\": [{\"bbox\":[x1,y1,x2,y2],\"confidence\":number 0..1,\"defect\":string}], "
                "\"reason\": string}. Use at most 3 regions and a reason of at most 20 words. "
                "Use an empty regions list for a normal target. Keep the JSON compact and on one line."
            )},
        ])
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = [out[len(inp) :] for inp, out in zip(inputs["input_ids"], generated)]
        response = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        data = _extract_json(response)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return self._to_opinion(data)

    @staticmethod
    def _to_opinion(data: dict) -> QwenOpinion:
        if not isinstance(data, dict):
            warnings.warn("Qwen result is not an object; using a neutral semantic opinion.")
            return QwenOpinion(reason="Invalid Qwen result type")

        def probability(value: object, default: float = 0.0) -> float:
            try:
                return float(np.clip(float(value), 0, 1))
            except (TypeError, ValueError):
                return default

        regions = []
        region_items = data.get("regions", [])
        if not isinstance(region_items, list):
            region_items = []
        for item in region_items:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox", [])
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                safe_bbox = [float(np.clip(float(value), 0, 1000)) for value in bbox]
            except (TypeError, ValueError):
                continue
            regions.append(Region(
                safe_bbox,
                probability(item.get("confidence", 0)),
                str(item.get("defect", "unknown")),
            ))
        return QwenOpinion(
            probability(data.get("anomaly_probability", 0)),
            regions,
            str(data.get("defect_type", "unknown")),
            str(data.get("reason", "")),
        )


def opinion_map(opinion: QwenOpinion, shape: tuple[int, int], blur_fraction: float) -> np.ndarray:
    height, width = shape
    result = np.zeros(shape, dtype=np.float32)
    for region in opinion.regions:
        x1, y1, x2, y2 = region.bbox
        xa, xb = sorted((int(x1 * width / 1000), int(x2 * width / 1000)))
        ya, yb = sorted((int(y1 * height / 1000), int(y2 * height / 1000)))
        xa, xb = np.clip([xa, xb], 0, width)
        ya, yb = np.clip([ya, yb], 0, height)
        if xb > xa and yb > ya:
            result[ya:yb, xa:xb] = np.maximum(result[ya:yb, xa:xb], region.confidence)
    if result.any():
        result = gaussian_filter(result, sigma=max(shape) * blur_fraction)
        if result.max() > 0:
            result /= result.max()
        result *= opinion.anomaly_probability
    return result
