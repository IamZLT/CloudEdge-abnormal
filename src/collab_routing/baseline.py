"""Baseline margin router (legacy heuristic_upload).

Rules:
1. outage -> local
2. n_gallery == 0 -> upload
3. else upload iff score_margin < hard_margin
"""
from __future__ import annotations

from typing import Any

from src.collab_routing.base import CloudState, CollabRouter, RouteSignal, RouteVerdict


class BaselineMarginRouter(CollabRouter):
    """Original near-threshold / cold-start heuristic (no link cost)."""

    name = "baseline"

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = dict(cfg or {})

    def decide(self, signal: RouteSignal, cloud: CloudState | None = None) -> RouteVerdict:
        profile = str(signal.network_profile or "").lower()
        net = dict(signal.network or {})
        outage = profile == "outage" or bool(net.get("outage"))
        m = signal.score_margin()
        h = float(signal.hard_margin)
        n_g = int(signal.n_gallery)

        features = {
            "score_margin": m,
            "hard_margin": h,
            "n_gallery": n_g,
            "outage": outage,
            "c_cloud": None,
        }

        if outage:
            return RouteVerdict(
                upload=False,
                utility=-1.0,
                reason="baseline: outage — stay local",
                algorithm=self.name,
                features=features,
            )
        if n_g <= 0:
            return RouteVerdict(
                upload=True,
                utility=1.0,
                reason="baseline: cold start (n_gallery=0) — upload",
                algorithm=self.name,
                features=features,
            )
        upload = m < h
        return RouteVerdict(
            upload=upload,
            utility=(1.0 - m / max(h, 1e-6)) if upload else -(m - h),
            reason=(
                f"baseline: near threshold (margin={m:.4f} < {h:.4f}) — upload"
                if upload
                else f"baseline: confident local (margin={m:.4f} >= {h:.4f})"
            ),
            algorithm=self.name,
            features=features,
        )
