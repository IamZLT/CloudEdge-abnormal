"""Cost–Risk Routing (CRR). See docs/cloud_edge_collab_algorithms.md."""
from __future__ import annotations

from typing import Any

from src.collab_routing.base import (
    AdmitCandidate,
    AdmitResult,
    CloudState,
    CollabRouter,
    RouteSignal,
    RouteVerdict,
    clip,
)


class CostRiskRouter(CollabRouter):
    """Utility U = u_unc - w_n*c_net - w_c*c_cloud (+ cold-start)."""

    name = "cost_risk"

    def __init__(self, cfg: dict[str, Any] | None = None):
        c = dict(cfg or {})
        cr = dict(c.get("cost_risk") or c)
        adm = dict(c.get("cloud_admission") or {})
        self.n_ref = float(cr.get("n_ref", 16))
        self.w_n = float(cr.get("w_n", 0.8))
        self.w_c = float(cr.get("w_c", 0.5))
        self.w_g = float(cr.get("w_g", 0.6))
        self.w_conf = float(cr.get("w_conf", 0.7))
        self.rtt_ref = float(cr.get("rtt_ref_ms", 80.0))
        self.bw_ref = float(cr.get("bw_ref_mbps", 50.0))
        self.weak_c_net = float(cr.get("weak_c_net", 0.75))
        self.force_unc = float(cr.get("force_unc", 0.85))
        self.h_clip_lo = float(cr.get("h_clip_lo", 0.5))
        self.h_clip_hi = float(cr.get("h_clip_hi", 2.0))
        self.max_inflight = int(adm.get("max_inflight", c.get("max_inflight", 2)))
        self.fairness_gamma = float(adm.get("fairness_gamma", 0.1))

    def _link_cost(self, signal: RouteSignal) -> float:
        net = dict(signal.network or {})
        rtt = float(net.get("rtt_ms") or net.get("prop_rtt_ms") or self.rtt_ref)
        bw = float(net.get("bandwidth_mbps") or self.bw_ref)
        loss = float(net.get("loss_prob") or 0.0)
        return clip(
            0.4 * (rtt / max(self.rtt_ref, 1e-6))
            + 0.4 * (self.bw_ref / max(bw, 1e-3))
            + 0.2 * loss,
            0.0,
            1.0,
        )

    def decide(self, signal: RouteSignal, cloud: CloudState | None = None) -> RouteVerdict:
        profile = str(signal.network_profile or "").lower()
        net = dict(signal.network or {})
        outage = profile == "outage" or bool(net.get("outage"))
        n_g = int(signal.n_gallery)
        h0 = float(signal.hard_margin)
        m = signal.score_margin()
        conflict = clip(float(getattr(signal, "conflict", 0.0) or 0.0), 0.0, 1.0)

        if outage:
            return RouteVerdict(
                upload=False,
                utility=float("-inf"),
                reason="crr: outage — stay local",
                algorithm=self.name,
                features={
                    "outage": True,
                    "score_margin": m,
                    "n_gallery": n_g,
                    "conflict": conflict,
                },
            )

        scale = clip(self.n_ref / max(n_g, 1), self.h_clip_lo, self.h_clip_hi)
        h_eff = h0 * scale
        u_unc = clip(1.0 - m / max(h_eff, 1e-6), 0.0, 1.0)
        c_net = self._link_cost(signal)

        if cloud is None:
            c_cloud = 0.0
            k = self.max_inflight
        else:
            k = int(cloud.max_inflight or self.max_inflight)
            c_cloud = clip(cloud.load / max(k, 1), 0.0, 1.0)

        # Conflict is a strong upload trigger (cloud is the tiebreaker).
        U = u_unc + self.w_conf * conflict - self.w_n * c_net - self.w_c * c_cloud
        cold = n_g <= 0
        if cold:
            U += self.w_g

        features = {
            "score_margin": m,
            "h0": h0,
            "h_eff": h_eff,
            "u_unc": u_unc,
            "conflict": conflict,
            "c_net": c_net,
            "c_cloud": c_cloud,
            "n_gallery": n_g,
            "utility": U,
            "w_n": self.w_n,
            "w_c": self.w_c,
            "w_g": self.w_g,
            "w_conf": self.w_conf,
            "cold_start": cold,
        }

        # Cold start: always attempt cloud when link is up (doc §8).
        if cold:
            return RouteVerdict(
                upload=True,
                utility=float(U if U > 0 else 0.01 + self.w_g),
                reason=f"crr: cold start (n_gallery=0) — upload (U={U:.3f})",
                algorithm=self.name,
                features=features,
            )

        # Weak-link guard: suppressed only when NOT strongly conflicted; conflict
        # lets the utility trade conflict benefit against link cost directly.
        if c_net >= self.weak_c_net and u_unc < self.force_unc and conflict < 0.5:
            return RouteVerdict(
                upload=False,
                utility=U,
                reason=(
                    f"crr: weak-link guard (c_net={c_net:.3f}, u_unc={u_unc:.3f}) — local"
                ),
                algorithm=self.name,
                features=features,
            )

        upload = U > 0.0
        return RouteVerdict(
            upload=upload,
            utility=float(U),
            reason=(
                f"crr: U={U:.3f} (unc={u_unc:.3f} + conf={conflict:.2f} "
                f"- net={c_net:.3f} - cloud={c_cloud:.3f})"
                + (" — upload" if upload else " — local")
            ),
            algorithm=self.name,
            features=features,
        )

    def admit(
        self,
        candidates: list[AdmitCandidate],
        *,
        max_inflight: int | None = None,
        cloud: CloudState | None = None,
    ) -> AdmitResult:
        if max_inflight is not None:
            k = int(max_inflight)
        elif cloud is not None:
            # Available slots = max_inflight − current inflight (never oversubscribe).
            k = max(0, int(cloud.max_inflight) - int(cloud.inflight))
        else:
            k = self.max_inflight
        wanting = [c for c in candidates if c.verdict.upload]
        local = [c for c in candidates if not c.verdict.upload]
        for c in wanting:
            c_net = float((c.verdict.features or {}).get("c_net") or self._link_cost(c.signal))
            recent = float(c.signal.recent_cloud)
            score = float(c.verdict.utility) / (1e-6 + c_net) - self.fairness_gamma * recent
            c.verdict.admit_score = score
        ranked = sorted(wanting, key=lambda c: float(c.verdict.admit_score or 0.0), reverse=True)
        accepted = ranked[: max(0, k)]
        rejected = ranked[max(0, k) :] + local
        return AdmitResult(accepted=accepted, rejected=rejected, algorithm=self.name)
