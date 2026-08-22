from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from src.vlm.model_clients import create_vlm_client

from .schemas import ExpertEvidence, ReviewEvidence, VerificationEvidence


VERIFY_PROMPT = """You are the conservative second-pass verifier in an industrial anomaly system.
PatchCore is the primary detector. A first VLM reviewer proposed OVERRIDING it.
Your task is not to classify from scratch: decide whether the override is supported by
unambiguous visual evidence. False alarms are costly. Normal texture, lighting, pose,
background, reflections, and harmless reference mismatch are NOT defects.

Evidence-board panels: A=query with candidate box, B=PatchCore heatmap (hint only),
C=highest-scoring crop, D=nearest normal training reference.

Context:
- category: {category}
- PatchCore: {expert_decision}, anomaly probability={expert_probability:.4f}
- first reviewer: {review_decision}, confidence={review_confidence:.4f}
- first reviewer reason: {review_reason}

Confirm only if A/C versus D shows a concrete defect at the boxed/hot region and the
final decision should be {review_decision}. When evidence is ambiguous, reject the
override and retain PatchCore's {expert_decision} decision.

Reply with ONLY one JSON object:
{{"confirm_override":true or false,"decision":"OK" or "NG","confidence":0-1,
"region_agreement":true or false,"reason":"short evidence-based explanation"}}
"""


def _json_object(raw: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", raw or "", flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


@dataclass
class ConflictVerifier:
    client: Any

    @classmethod
    def from_model(
        cls,
        model_path: str | Path,
        backend: str = "transformers",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_new_tokens: int = 160,
    ) -> "ConflictVerifier":
        return cls(create_vlm_client(
            model_path=str(model_path),
            backend=backend,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            role="conflict_verifier",
            prompt="The per-sample verification prompt will be supplied at inference time.",
        ))

    def verify(
        self,
        board: str | Path | Image.Image,
        expert: ExpertEvidence,
        review: ReviewEvidence,
    ) -> VerificationEvidence:
        self.client.prompt = VERIFY_PROMPT.format(
            category=expert.category,
            expert_decision=expert.decision,
            expert_probability=expert.probability,
            review_decision=review.decision,
            review_confidence=review.confidence,
            review_reason=review.reason[:400],
        )
        result = self.client.infer(board)
        extra = _json_object(result.raw)
        decision = str(extra.get("decision", result.decision)).upper()
        try:
            confidence = min(1.0, max(0.0, float(extra.get("confidence", result.confidence))))
        except (TypeError, ValueError):
            confidence = 0.0
        parse_ok = (
            bool(result.parse_ok)
            and "confirm_override" in extra
            and decision in {"OK", "NG"}
        )
        return VerificationEvidence(
            confirm_override=_as_bool(extra.get("confirm_override", False)),
            decision=decision,
            confidence=confidence,
            region_agreement=_as_bool(extra.get("region_agreement", False)),
            reason=str(extra.get("reason", result.reason)),
            parse_ok=parse_ok,
            latency_ms=float(result.latency_ms),
            peak_mem_mb=result.peak_mem_mb,
            raw=str(result.raw),
        )


@dataclass
class VerificationGate:
    min_confidence: float = 0.90
    require_region_agreement: bool = True
    allow_overrides: bool = True

    def accept(
        self,
        expert: ExpertEvidence,
        review: ReviewEvidence,
        verification: VerificationEvidence | None,
    ) -> tuple[bool, str]:
        if not self.allow_overrides:
            return False, "advisory_only"
        if verification is None:
            return False, "verification_missing"
        if not verification.parse_ok:
            return False, "verification_parse_failed"
        if not verification.confirm_override:
            return False, "override_rejected"
        if verification.confidence < self.min_confidence:
            return False, "verification_confidence_low"
        if self.require_region_agreement and not verification.region_agreement:
            return False, "verification_region_disagreement"
        if verification.decision != review.decision:
            return False, "reviewer_verifier_disagreement"
        if verification.decision == expert.decision:
            return False, "verification_matches_expert"
        return True, "override_confirmed"
