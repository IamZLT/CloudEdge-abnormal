"""Pluggable cloud–edge collaboration routing interfaces.

Swap algorithms via ``collab.route_policy`` (see ``registry.build_router``).
All routers share the same input/output contracts so benches can A/B compare.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


@dataclass
class RouteSignal:
    """Per-sample edge evidence + link snapshot for upload routing."""

    category: str
    n_gallery: int
    edge_score: float
    edge_thr: float
    edge_decision: str  # OK | NG
    network_profile: str = "fair"
    network: dict[str, Any] = field(default_factory=dict)
    hard_margin: float = 0.05  # h0
    edge_node_id: str | None = None
    recent_cloud: float = 0.0  # fairness counter for admission
    conflict: float = 0.0  # multi-node disagreement 0..1 (0 = unanimous)

    def score_margin(self) -> float:
        return abs(float(self.edge_score) - float(self.edge_thr))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CloudState:
    """Shared cloud reviewer load (fleet-level)."""

    inflight: int = 0
    queue: int = 0
    max_inflight: int = 2

    @property
    def load(self) -> int:
        return int(self.inflight) + int(self.queue)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RouteVerdict:
    """Single-sample routing decision with explainable features."""

    upload: bool
    utility: float
    reason: str
    algorithm: str
    features: dict[str, Any] = field(default_factory=dict)
    admit_score: float | None = None  # filled by admit()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdmitCandidate:
    """One node's request in a multi-edge admission window."""

    signal: RouteSignal
    verdict: RouteVerdict
    request_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "edge_node_id": self.signal.edge_node_id,
            "verdict": self.verdict.to_dict(),
        }


@dataclass
class AdmitResult:
    accepted: list[AdmitCandidate]
    rejected: list[AdmitCandidate]
    algorithm: str

    def accepted_ids(self) -> list[str]:
        return [c.request_id or (c.signal.edge_node_id or "") for c in self.accepted]

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "accepted": [c.to_dict() for c in self.accepted],
            "rejected": [c.to_dict() for c in self.rejected],
        }


class CollabRouter(ABC):
    """Cloud–edge upload router + optional multi-node admission."""

    name: str = "base"

    @abstractmethod
    def decide(self, signal: RouteSignal, cloud: CloudState | None = None) -> RouteVerdict:
        """Decide whether this sample should attempt cloud upload."""

    def admit(
        self,
        candidates: list[AdmitCandidate],
        *,
        max_inflight: int | None = None,
        cloud: CloudState | None = None,
    ) -> AdmitResult:
        """Select which upload-wanting candidates may use the cloud this window.

        Default: keep candidates with upload=True, rank by utility, take Top-K.
        Algorithms may override (e.g. CRR value-density ranking).
        """
        k = int(
            max_inflight
            if max_inflight is not None
            else (cloud.max_inflight if cloud is not None else 2)
        )
        wanting = [c for c in candidates if c.verdict.upload]
        local = [c for c in candidates if not c.verdict.upload]
        ranked = sorted(wanting, key=lambda c: float(c.verdict.utility), reverse=True)
        for c in ranked:
            c.verdict.admit_score = float(c.verdict.utility)
        accepted = ranked[: max(0, k)]
        rejected = ranked[max(0, k) :] + local
        return AdmitResult(accepted=accepted, rejected=rejected, algorithm=self.name)

    def decide_upload(self, signal: RouteSignal, cloud: CloudState | None = None) -> bool:
        return bool(self.decide(signal, cloud).upload)
