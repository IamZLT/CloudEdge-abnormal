from __future__ import annotations

import math
from dataclasses import dataclass

from .schemas import ExpertEvidence, ReviewEvidence


def _clip_probability(value: float) -> float:
    return min(1.0 - 1e-5, max(1e-5, float(value)))


def logit(value: float) -> float:
    value = _clip_probability(value)
    return math.log(value / (1.0 - value))


def sigmoid(value: float) -> float:
    value = min(30.0, max(-30.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def calibrate_patchcore_score(score: float, threshold: float) -> float:
    scale = max(abs(float(threshold)) * 0.25, 1e-6)
    return sigmoid((float(score) - float(threshold)) / scale)


@dataclass
class FusionOutcome:
    score: float
    applied: bool
    reason: str


@dataclass
class ConservativeFusion:
    min_confidence: float = 0.85
    raise_weight: float = 0.35
    lower_weight: float = 0.15
    max_lower_abs_logit: float = 1.25
    require_region_agreement: bool = True

    def fuse(self, expert: ExpertEvidence, review: ReviewEvidence | None) -> FusionOutcome:
        if review is None:
            return FusionOutcome(expert.probability, False, "not_reviewed")
        if not review.parse_ok:
            return FusionOutcome(expert.probability, False, "review_parse_failed")
        if review.confidence < self.min_confidence:
            return FusionOutcome(expert.probability, False, "review_confidence_low")
        if self.require_region_agreement and not review.region_agreement:
            return FusionOutcome(expert.probability, False, "region_disagreement")

        expert_logit = logit(expert.probability)
        review_logit = logit(review.anomaly_score)
        if review.decision == "NG":
            final = expert_logit + self.raise_weight * max(review_logit, 0.0)
            return FusionOutcome(sigmoid(final), True, "high_confidence_ng_evidence")
        if abs(expert_logit) <= self.max_lower_abs_logit:
            final = expert_logit + self.lower_weight * min(review_logit, 0.0)
            return FusionOutcome(sigmoid(final), True, "bounded_ok_correction")
        return FusionOutcome(expert.probability, False, "expert_too_confident_to_lower")
