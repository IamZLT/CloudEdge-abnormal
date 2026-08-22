from __future__ import annotations

from dataclasses import dataclass

from .fusion import ConservativeFusion
from .schemas import AgentDecision, ExpertEvidence, ReviewEvidence


@dataclass
class DetectionAgent:
    fusion: ConservativeFusion

    def decide(
        self,
        expert: ExpertEvidence,
        *,
        reviewed: bool,
        review: ReviewEvidence | None = None,
    ) -> AgentDecision:
        outcome = self.fusion.fuse(expert, review if reviewed else None)
        prediction = int(outcome.score >= 0.5)
        return AgentDecision(
            final_score=float(outcome.score),
            prediction=prediction,
            decision="NG" if prediction else "OK",
            reviewed=bool(reviewed),
            review_applied=bool(outcome.applied),
            fallback_reason=outcome.reason,
            expert=expert,
            review=review,
        )
