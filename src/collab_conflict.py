"""Multi-node conflict detection + arbitration for cloud–edge collaboration.

Hand-written algorithm (no LLM). N edge nodes judge the **same** image under
different deterministic augmentations (simulating camera / lighting drift across
sites). Disagreement among nodes = conflict; a conflict is resolved by cloud
arbitration when the link is usable, otherwise fail-safe (conservative NG +
divert for human review).

The LLM is used only for *detection* (edge AD + cloud VLM review), never for the
conflict / routing / arbitration decision itself.

Design notes (vs. the legacy ``src/collab.py`` simulation):
  - ``src/collab.py`` faked a second node by adding Gaussian noise to the same
    model's scores; here each node is a real, deterministic, per-image variant.
  - ``src/collab.py`` had an ``or True`` that made resolution always succeed;
    here ``arbitrate`` distinguishes cloud-arbitrated vs fail-safe-divert.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class AugSpec:
    """One deterministic image transform applied to a node's query image."""

    name: str
    kind: str  # none | brightness | contrast | blur | noise
    factor: float = 1.0
    seed: int = 0

    def apply(self, img: Image.Image) -> Image.Image:
        if self.kind == "none":
            return img
        if self.kind == "brightness":
            return ImageEnhance.Brightness(img).enhance(self.factor)
        if self.kind == "contrast":
            return ImageEnhance.Contrast(img).enhance(self.factor)
        if self.kind == "blur":
            return img.filter(ImageFilter.GaussianBlur(radius=self.factor))
        if self.kind == "noise":
            arr = np.asarray(img.convert("RGB"), dtype=np.float32)
            rng = np.random.default_rng(self.seed)
            noise = rng.normal(0.0, self.factor, arr.shape)
            return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
        raise ValueError(f"unknown augment kind: {self.kind}")


# Default per-node variants. Node 0 = clean reference; the rest simulate site drift.
# Brightness drift (esp. darkening) is the dominant cross-threshold mover for the
# patch gallery; blur/noise barely shift the normalized patch distance.
DEFAULT_AUGS: list[AugSpec] = [
    AugSpec("identity", "none"),
    AugSpec("brightness-0.85", "brightness", 0.85),
    AugSpec("brightness-0.9", "brightness", 0.9),
    AugSpec("brightness-1.1", "brightness", 1.1),
    AugSpec("brightness-1.15", "brightness", 1.15),
    AugSpec("brightness-0.7", "brightness", 0.7),
    AugSpec("brightness-1.3", "brightness", 1.3),
    AugSpec("contrast-0.9", "contrast", 0.9),
    AugSpec("contrast-1.1", "contrast", 1.1),
    AugSpec("contrast-0.6", "contrast", 0.6),
    AugSpec("contrast-1.5", "contrast", 1.5),
    AugSpec("blur-0.8", "blur", 0.8),
    AugSpec("noise-5", "noise", 5.0, seed=7),
]


def resolve_augs(
    names: list[str] | None = None,
    n_nodes: int | None = None,
) -> list[AugSpec]:
    """Resolve a list of augment specs from names and/or a target node count."""
    by_name = {s.name: s for s in DEFAULT_AUGS}
    if names:
        base: list[AugSpec] = []
        for n in names:
            key = str(n).strip()
            if key not in by_name:
                raise ValueError(f"unknown augment: {key!r}; choose {sorted(by_name)}")
            base.append(by_name[key])
    else:
        base = list(DEFAULT_AUGS)
    if n_nodes is not None:
        base = [base[i % len(base)] for i in range(int(n_nodes))]
    return base


@dataclass
class MultiNodeConsensus:
    """Aggregated decision of N nodes judging the same image."""

    n_nodes: int
    node_scores: list[float]
    node_decisions: list[str]  # OK | NG
    thr: float
    majority_decision: str
    n_ng: int
    conflict: bool
    conflict_score: float  # graded 0..1 (0 = unanimous, 1 = full split)
    vote_entropy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "node_scores": [float(s) for s in self.node_scores],
            "node_decisions": list(self.node_decisions),
            "thr": float(self.thr),
            "majority_decision": self.majority_decision,
            "n_ng": self.n_ng,
            "conflict": self.conflict,
            "conflict_score": float(self.conflict_score),
            "vote_entropy": float(self.vote_entropy),
        }

    def weighted_vote_decision(self) -> str:
        """Hand-written tiebreaker: each node votes with margin |score - thr|.

        The node furthest from the threshold is the most confident and carries
        the most weight. Used when no cloud arbiter is available.
        """
        if self.n_nodes <= 1:
            return self.majority_decision
        w_ng = 0.0
        w_ok = 0.0
        for s, d in zip(self.node_scores, self.node_decisions):
            w = abs(float(s) - float(self.thr)) + 1e-6
            if d == "NG":
                w_ng += w
            else:
                w_ok += w
        return "NG" if w_ng >= w_ok else "OK"


def multi_node_consensus(
    score_fn: Callable[[Image.Image], float],
    image: Image.Image,
    thr: float,
    *,
    augs: list[AugSpec] | None = None,
    n_nodes: int | None = None,
) -> MultiNodeConsensus:
    """Score the image under each node's augmentation and aggregate."""
    specs = list(augs) if augs is not None else resolve_augs(None, n_nodes)
    scores: list[float] = []
    decisions: list[str] = []
    for spec in specs:
        s = float(score_fn(spec.apply(image)))
        scores.append(s)
        decisions.append("NG" if s >= thr else "OK")

    n = len(scores)
    n_ng = int(sum(1 for d in decisions if d == "NG"))
    n_ok = n - n_ng
    majority = "NG" if n_ng > n_ok else "OK"
    conflict = (n_ng > 0) and (n_ok > 0)

    # Graded disagreement in [0, 1]: (1 - majority_frac) normalized by (1 - 1/n).
    maj_frac = max(n_ng, n_ok) / max(1, n)
    if n > 1:
        conflict_score = float((1.0 - maj_frac) / (1.0 - 1.0 / n))
    else:
        conflict_score = 0.0

    ent = 0.0
    for p in (n_ng / n, n_ok / n):
        if p > 0:
            ent -= p * np.log2(p)

    return MultiNodeConsensus(
        n_nodes=n,
        node_scores=scores,
        node_decisions=decisions,
        thr=float(thr),
        majority_decision=majority,
        n_ng=n_ng,
        conflict=conflict,
        conflict_score=float(conflict_score),
        vote_entropy=float(ent),
    )


@dataclass
class ArbitrationResult:
    """Final decision after resolving a (possibly conflicting) consensus."""

    path: str  # LOCAL | CLOUD_ARBITRATED | FAIL_SAFE_DIVERT
    decision: str  # final OK | NG
    resolved: bool
    provisional: bool  # True => divert for human review (fail-safe)
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "decision": self.decision,
            "resolved": self.resolved,
            "provisional": self.provisional,
            "reason": self.reason,
        }


def arbitrate(
    consensus: MultiNodeConsensus,
    *,
    cloud_decision: str | None = None,
    upload_ok: bool = False,
) -> ArbitrationResult:
    """Resolve a multi-node consensus into a final decision.

    - Cloud reachable + valid decision → adopt cloud (higher-fidelity reviewer /
      tiebreaker), regardless of conflict.
    - Conflict + no cloud → fail-safe: conservative NG + divert (never release an
      uncertain part).
    - No conflict + no cloud → local majority.
    """
    if upload_ok and cloud_decision in {"OK", "NG"}:
        return ArbitrationResult(
            path="CLOUD_ARBITRATED",
            decision=cloud_decision,
            resolved=True,
            provisional=False,
            reason="cloud-arbitrated",
        )
    if consensus.conflict:
        return ArbitrationResult(
            path="FAIL_SAFE_DIVERT",
            decision="NG",
            resolved=True,
            provisional=True,
            reason="fail-safe-divert",
        )
    return ArbitrationResult(
        path="LOCAL",
        decision=consensus.majority_decision,
        resolved=True,
        provisional=False,
        reason="unanimous-local",
    )
