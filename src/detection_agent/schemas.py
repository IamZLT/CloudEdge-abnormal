from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class RegionEvidence:
    bbox_xyxy: tuple[int, int, int, int]
    score: float
    grid_rc: tuple[int, int]


@dataclass
class ExpertEvidence:
    image_path: str
    category: str
    score: float
    threshold: float
    probability: float
    decision: str
    patch_scores: np.ndarray = field(repr=False)
    reference_path: str = ""
    reference_similarity: float | None = None
    regions: list[RegionEvidence] = field(default_factory=list)
    concentration: float = 0.0
    latency_ms: float = 0.0

    def to_dict(self, include_patch_scores: bool = False) -> dict[str, Any]:
        result = asdict(self)
        result["patch_scores"] = self.patch_scores.tolist() if include_patch_scores else None
        return result


@dataclass
class ReviewEvidence:
    decision: str
    confidence: float
    defect_type: str
    reason: str
    region_agreement: bool
    parse_ok: bool
    latency_ms: float
    peak_mem_mb: float | None = None
    raw: str = ""

    @property
    def anomaly_score(self) -> float:
        if not self.parse_ok:
            return float("nan")
        return self.confidence if self.decision == "NG" else 1.0 - self.confidence

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["anomaly_score"] = self.anomaly_score
        return result


@dataclass
class VerificationEvidence:
    confirm_override: bool
    decision: str
    confidence: float
    region_agreement: bool
    reason: str
    parse_ok: bool
    latency_ms: float
    peak_mem_mb: float | None = None
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentDecision:
    final_score: float
    prediction: int
    decision: str
    reviewed: bool
    review_applied: bool
    fallback_reason: str
    expert: ExpertEvidence
    review: ReviewEvidence | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_score": self.final_score,
            "prediction": self.prediction,
            "decision": self.decision,
            "reviewed": self.reviewed,
            "review_applied": self.review_applied,
            "fallback_reason": self.fallback_reason,
            "expert": self.expert.to_dict(),
            "review": None if self.review is None else self.review.to_dict(),
        }
