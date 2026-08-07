"""Cloud–edge network condition simulator (latency / bandwidth / loss / timeout).

Two modes:
  1) **Legacy profiles** (good/fair/weak/outage) — static mean + Gaussian jitter
  2) **Physical geo env** — see ``src/network_env.py`` (distance + temporal dynamics)

Used for accounting-only evaluation (no real sleep). Hard-example uploads that
fail due to loss or timeout fall back to the edge-local decision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class NetworkProfile:
    name: str = "fair"
    rtt_ms: float = 80.0
    rtt_jitter_ms: float = 30.0
    bandwidth_mbps: float = 10.0
    loss_prob: float = 0.02
    timeout_ms: float = 2000.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UploadOutcome:
    ok: bool
    rtt_ms: float = 0.0
    tx_ms: float = 0.0
    total_net_ms: float = 0.0
    failed_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PROFILES: dict[str, NetworkProfile] = {
    "good": NetworkProfile(
        name="good",
        rtt_ms=20.0,
        rtt_jitter_ms=5.0,
        bandwidth_mbps=50.0,
        loss_prob=0.0,
        timeout_ms=2000.0,
    ),
    "fair": NetworkProfile(
        name="fair",
        rtt_ms=80.0,
        rtt_jitter_ms=30.0,
        bandwidth_mbps=10.0,
        loss_prob=0.02,
        timeout_ms=2000.0,
    ),
    "weak": NetworkProfile(
        name="weak",
        rtt_ms=200.0,
        rtt_jitter_ms=80.0,
        bandwidth_mbps=1.0,
        loss_prob=0.15,
        timeout_ms=1500.0,
    ),
    "outage": NetworkProfile(
        name="outage",
        rtt_ms=500.0,
        rtt_jitter_ms=0.0,
        bandwidth_mbps=0.1,
        loss_prob=1.0,
        timeout_ms=500.0,
    ),
}


def resolve_profile(network_cfg: dict | None = None) -> NetworkProfile:
    """Build a NetworkProfile from collab.network / live geo snapshot dict."""
    cfg = dict(network_cfg or {})
    name = str(cfg.get("profile") or cfg.get("name") or "fair").lower()
    if name in {"geo", "physical", "env"}:
        # Live snapshot from NetworkEnvironment.to_profile_dict()
        return NetworkProfile(
            name="geo",
            rtt_ms=float(cfg.get("rtt_ms", 80.0)),
            rtt_jitter_ms=float(cfg.get("rtt_jitter_ms", 10.0)),
            bandwidth_mbps=float(cfg.get("bandwidth_mbps", 10.0)),
            loss_prob=float(cfg.get("loss_prob", 0.02)),
            timeout_ms=float(cfg.get("timeout_ms", 3000.0)),
        )
    if name == "custom":
        base = NetworkProfile(name="custom")
    else:
        if name not in PROFILES:
            raise ValueError(f"unknown network profile: {name}; choose {list(PROFILES)}|geo|custom")
        base = PROFILES[name]
    # allow overrides
    return NetworkProfile(
        name=name if name != "custom" else "custom",
        rtt_ms=float(cfg.get("rtt_ms", base.rtt_ms)),
        rtt_jitter_ms=float(cfg.get("rtt_jitter_ms", base.rtt_jitter_ms)),
        bandwidth_mbps=float(cfg.get("bandwidth_mbps", base.bandwidth_mbps)),
        loss_prob=float(cfg.get("loss_prob", base.loss_prob)),
        timeout_ms=float(cfg.get("timeout_ms", base.timeout_ms)),
    )


@dataclass
class NetworkSimulator:
    profile: NetworkProfile = field(default_factory=lambda: PROFILES["fair"])
    seed: int = 42
    # Optional physical environment binding (preferred for multi-edge fleet)
    env: Any | None = None
    edge_id: str | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self.last_link: dict[str, Any] | None = None

    @classmethod
    def from_config(cls, network_cfg: dict | None = None) -> "NetworkSimulator":
        cfg = dict(network_cfg or {})
        return cls(profile=resolve_profile(cfg), seed=int(cfg.get("seed", 42)))

    @classmethod
    def from_env(cls, env: Any, edge_id: str, seed: int | None = None) -> "NetworkSimulator":
        """Bind this simulator to a physical NetworkEnvironment edge site."""
        link = env.sample_link(edge_id)
        prof = NetworkProfile(
            name="outage" if link.outage else "geo",
            rtt_ms=float(link.rtt_ms),
            rtt_jitter_ms=float(link.rtt_jitter_ms),
            bandwidth_mbps=float(link.bandwidth_mbps),
            loss_prob=float(link.loss_prob),
            timeout_ms=float(link.timeout_ms),
        )
        sim = cls(profile=prof, seed=int(seed if seed is not None else getattr(env, "seed", 42)))
        sim.env = env
        sim.edge_id = edge_id
        sim.last_link = link.to_dict()
        return sim

    def refresh_profile_from_env(self) -> NetworkProfile:
        """Update ``profile`` from the live geo/temporal link (for UI snapshots)."""
        if self.env is None or self.edge_id is None:
            return self.profile
        link = self.env.sample_link(self.edge_id)
        self.profile = NetworkProfile(
            name="outage" if link.outage else "geo",
            rtt_ms=float(link.rtt_ms),
            rtt_jitter_ms=float(link.rtt_jitter_ms),
            bandwidth_mbps=float(link.bandwidth_mbps),
            loss_prob=float(link.loss_prob),
            timeout_ms=float(link.timeout_ms),
        )
        self.last_link = link.to_dict()
        return self.profile

    def try_upload(self, payload_bytes: int, rng: np.random.Generator | None = None) -> UploadOutcome:
        """Simulate one hard-example upload attempt."""
        if self.env is not None and self.edge_id is not None:
            out, link = self.env.try_upload(self.edge_id, int(payload_bytes))
            self.last_link = link.to_dict()
            self.profile = NetworkProfile(
                name="outage" if link.outage else "geo",
                rtt_ms=float(link.rtt_ms),
                rtt_jitter_ms=float(link.rtt_jitter_ms),
                bandwidth_mbps=float(link.bandwidth_mbps),
                loss_prob=float(link.loss_prob),
                timeout_ms=float(link.timeout_ms),
            )
            return out

        g = rng if rng is not None else self._rng
        p = self.profile
        # link drop / outage
        if g.random() < p.loss_prob:
            return UploadOutcome(ok=False, failed_reason="loss")

        rtt = max(0.0, float(g.normal(p.rtt_ms, p.rtt_jitter_ms)))
        bw = max(1e-6, float(p.bandwidth_mbps))
        tx = (float(payload_bytes) * 8.0) / (bw * 1e6) * 1000.0
        total = rtt + tx
        if total > p.timeout_ms:
            return UploadOutcome(
                ok=False,
                rtt_ms=rtt,
                tx_ms=tx,
                total_net_ms=total,
                failed_reason="timeout",
            )
        return UploadOutcome(ok=True, rtt_ms=rtt, tx_ms=tx, total_net_ms=total)

    def summarize(self, outcomes: list[UploadOutcome]) -> dict[str, Any]:
        if not outcomes:
            return {
                "profile": self.profile.to_dict(),
                "n_attempts": 0,
                "cloud_upload_success_rate": float("nan"),
                "mean_upload_rtt_ms": float("nan"),
                "mean_tx_ms": float("nan"),
                "mean_total_net_ms": float("nan"),
                "n_loss": 0,
                "n_timeout": 0,
            }
        oks = [o for o in outcomes if o.ok]
        return {
            "profile": self.profile.to_dict(),
            "n_attempts": len(outcomes),
            "n_success": len(oks),
            "cloud_upload_success_rate": float(len(oks) / len(outcomes)),
            "mean_upload_rtt_ms": float(np.mean([o.rtt_ms for o in outcomes])),
            "mean_tx_ms": float(np.mean([o.tx_ms for o in outcomes])),
            "mean_total_net_ms": float(np.mean([o.total_net_ms for o in outcomes])),
            "n_loss": int(sum(1 for o in outcomes if o.failed_reason == "loss")),
            "n_timeout": int(sum(1 for o in outcomes if o.failed_reason == "timeout")),
        }


def apply_collab_uploads(
    *,
    hard_mask: np.ndarray,
    edge_scores: np.ndarray,
    cloud_scores: np.ndarray,
    edge_lat_ms: list[float] | np.ndarray,
    cloud_lat_ms: list[float] | np.ndarray,
    upload_bytes_hard: int,
    sim: NetworkSimulator,
    legacy_extra_ms: float = 0.0,
) -> tuple[np.ndarray, list[float], np.ndarray, list[UploadOutcome]]:
    """Apply network sim on hard samples for collaborative path S.

    Returns:
      s_scores, s_lat_ms, cloud_ok_mask (True where hard & upload succeeded), outcomes
    """
    hard_mask = np.asarray(hard_mask, dtype=bool)
    edge_scores = np.asarray(edge_scores, dtype=float)
    cloud_scores = np.asarray(cloud_scores, dtype=float)
    edge_lat = list(edge_lat_ms)
    cloud_lat = list(cloud_lat_ms)
    n = len(edge_scores)
    s_scores = edge_scores.copy()
    s_lat = list(edge_lat)
    cloud_ok = np.zeros(n, dtype=bool)
    outcomes: list[UploadOutcome] = []

    for i in range(n):
        if not hard_mask[i]:
            continue
        out = sim.try_upload(upload_bytes_hard)
        outcomes.append(out)
        if out.ok:
            s_scores[i] = cloud_scores[i]
            s_lat[i] = float(edge_lat[i]) + float(cloud_lat[i]) + out.total_net_ms + float(legacy_extra_ms)
            cloud_ok[i] = True
        else:
            # fallback: edge local (service kept)
            s_scores[i] = edge_scores[i]
            s_lat[i] = float(edge_lat[i])
            cloud_ok[i] = False
    return s_scores, s_lat, cloud_ok, outcomes
