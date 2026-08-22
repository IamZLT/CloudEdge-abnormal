from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from src.vlm.model_clients import create_vlm_client

from .schemas import ReviewEvidence


REVIEW_PROMPT = """You are the cloud DetectionReviewAgent for industrial anomaly inspection.
The evidence board has four panels:
A: original query with the PatchCore candidate region boxed;
B: PatchCore heatmap (a hint, not ground truth);
C: the highest-scoring local crop;
D: the nearest normal training reference.

Independently compare A/C against D. Ignore harmless illumination, pose and background
differences. Mark NG only for a real structural, surface, contamination, missing-part,
or geometric defect. region_agreement is true only when your evidence is spatially
consistent with the boxed/hot region. A confident OK is allowed when the highlighted
difference is normal variation.

Reply with ONLY one JSON object:
{"decision":"OK" or "NG","confidence":0-1,"defect_type":"none" or short string,
 "region_agreement":true or false,"reason":"short evidence-based explanation"}
"""


def _extra_fields(raw: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", raw or "", flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


@dataclass
class DetectionReviewer:
    client: Any

    @classmethod
    def from_model(
        cls,
        model_path: str | Path,
        backend: str = "transformers",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_new_tokens: int = 160,
    ) -> "DetectionReviewer":
        return cls(create_vlm_client(
            model_path=str(model_path),
            backend=backend,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            role="detection_review_agent",
            prompt=REVIEW_PROMPT,
        ))

    def review(self, board: str | Path | Image.Image) -> ReviewEvidence:
        result = self.client.infer(board)
        extra = _extra_fields(result.raw)
        agreement = extra.get("region_agreement", False)
        if isinstance(agreement, str):
            agreement = agreement.strip().lower() in {"true", "yes", "1"}
        return ReviewEvidence(
            decision=str(result.decision).upper(),
            confidence=float(result.confidence),
            defect_type=str(result.defect_type),
            reason=str(result.reason),
            region_agreement=bool(agreement),
            parse_ok=bool(result.parse_ok) and str(result.decision).upper() in {"OK", "NG"},
            latency_ms=float(result.latency_ms),
            peak_mem_mb=result.peak_mem_mb,
            raw=str(result.raw),
        )
