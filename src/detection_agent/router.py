from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fusion import logit
from .schemas import ExpertEvidence


@dataclass
class BudgetRouter:
    review_budget: float = 0.30
    concentration_weight: float = 0.15

    def priority(self, evidence: ExpertEvidence) -> float:
        # Smaller is more urgent. Diffuse maps receive a modest priority boost.
        confidence_margin = abs(logit(evidence.probability))
        dispersion = 1.0 - min(1.0, max(0.0, evidence.concentration))
        return confidence_margin - self.concentration_weight * dispersion

    def select(self, evidence: list[ExpertEvidence]) -> set[int]:
        if not evidence or self.review_budget <= 0:
            return set()
        count = min(len(evidence), max(1, int(round(len(evidence) * self.review_budget))))
        order = np.argsort([self.priority(item) for item in evidence])
        return {int(index) for index in order[:count]}
