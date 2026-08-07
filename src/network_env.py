"""Physical + temporal cloud–edge network environment.

Models a shared cloud site and multiple geographically distributed edge sites.
Link quality is derived from:

1. **Physics / geography**
   - Haversine great-circle distance between edge and cloud
   - Fiber route stretch (real cables are longer than the geodesic)
   - Propagation RTT ≈ 2 · (path_km / v_fiber)  with v_fiber ≈ 2e5 km/s
   - Access technology baseline (enterprise fiber / broadband / 5G / weak backhaul)

2. **Time-varying impairment** (per edge, continuous-time)
   - Ornstein–Uhlenbeck congestion process (mean-reverting random walk)
   - Diurnal load (sinusoidal, busier in local working hours)
   - Short burst congestion (Poisson arrivals)
   - Rare longer outages

Accounting-only: no real ``sleep``; callers use returned RTT/tx for metrics.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from src.network_sim import UploadOutcome

# Earth radius (km)
_EARTH_R_KM = 6371.0
# Speed of light in fiber ≈ c/1.5 ≈ 2e5 km/s → 200 km per ms (one-way)
_DEFAULT_FIBER_KM_PER_MS = 200.0


# ---------------------------------------------------------------------------
# Geography presets (approximate WGS84; industrial demo defaults in China)
# ---------------------------------------------------------------------------
CITY_PRESETS: dict[str, tuple[float, float]] = {
    "Shanghai": (31.2304, 121.4737),
    "Suzhou": (31.2989, 120.5853),
    "Hangzhou": (30.2741, 120.1551),
    "Nanjing": (32.0603, 118.7969),
    "Beijing": (39.9042, 116.4074),
    "Tianjin": (39.3434, 117.3616),
    "Shenzhen": (22.5431, 114.0579),
    "Guangzhou": (23.1291, 113.2644),
    "Dongguan": (23.0205, 113.7518),
    "Chengdu": (30.5728, 104.0668),
    "Chongqing": (29.4316, 106.9123),
    "Xi'an": (34.3416, 108.9398),
    "Wuhan": (30.5928, 114.3055),
    "Zhengzhou": (34.7466, 113.6253),
    "Urumqi": (43.8256, 87.6168),
    "Harbin": (45.8038, 126.5340),
    "Kunming": (25.0389, 102.7183),
    "Lhasa": (29.6520, 91.1721),
}


@dataclass(frozen=True)
class AccessTech:
    """Last-mile / campus uplink characteristics."""

    name: str
    bandwidth_mbps: float
    access_rtt_ms: float
    base_loss: float
    jitter_frac: float = 0.08  # fraction of RTT as short-term jitter σ


ACCESS_PRESETS: dict[str, AccessTech] = {
    "fiber_enterprise": AccessTech("fiber_enterprise", 100.0, 1.0, 0.001, 0.05),
    "broadband": AccessTech("broadband", 20.0, 8.0, 0.01, 0.10),
    "5g": AccessTech("5g", 40.0, 18.0, 0.02, 0.15),
    "weak_backhaul": AccessTech("weak_backhaul", 5.0, 35.0, 0.05, 0.22),
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * _EARTH_R_KM * math.asin(min(1.0, math.sqrt(a)))


def resolve_city(city: str | None, lat: float | None, lon: float | None) -> tuple[str, float, float]:
    if lat is not None and lon is not None:
        name = str(city or "custom")
        return name, float(lat), float(lon)
    key = str(city or "Shanghai")
    # case-insensitive match
    for k, (la, lo) in CITY_PRESETS.items():
        if k.lower() == key.lower():
            return k, la, lo
    if key not in CITY_PRESETS:
        raise KeyError(f"unknown city {city!r}; choose {sorted(CITY_PRESETS)} or pass lat/lon")
    la, lo = CITY_PRESETS[key]
    return key, la, lo


@dataclass
class GeoSite:
    id: str
    name: str
    city: str
    lat: float
    lon: float
    access: str = "broadband"
    tz_offset_hours: float = 8.0  # China default CST

    def access_tech(self) -> AccessTech:
        return ACCESS_PRESETS.get(self.access, ACCESS_PRESETS["broadband"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "city": self.city,
            "lat": self.lat,
            "lon": self.lon,
            "access": self.access,
            "tz_offset_hours": self.tz_offset_hours,
        }


@dataclass
class LinkInstant:
    """Snapshot of one edge→cloud link at a given time."""

    edge_id: str
    t: float
    distance_geo_km: float
    distance_fiber_km: float
    prop_rtt_ms: float
    access_rtt_ms: float
    processing_rtt_ms: float
    congestion: float
    diurnal: float
    burst: float
    outage: bool
    rtt_ms: float
    rtt_jitter_ms: float
    bandwidth_mbps: float
    loss_prob: float
    timeout_ms: float
    city: str = ""
    access: str = ""
    # compatibility with old NetworkProfile fields
    profile: str = "geo"
    name: str = "geo"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_profile_dict(self) -> dict[str, Any]:
        """Shape expected by older UI / RouteAgent CONTEXT."""
        return {
            "profile": "geo" if not self.outage else "outage",
            "name": "outage" if self.outage else "geo",
            "rtt_ms": float(self.rtt_ms),
            "rtt_jitter_ms": float(self.rtt_jitter_ms),
            "bandwidth_mbps": float(self.bandwidth_mbps),
            "loss_prob": float(self.loss_prob),
            "timeout_ms": float(self.timeout_ms),
            "edge_id": self.edge_id,
            "city": self.city,
            "access": self.access,
            "distance_geo_km": float(self.distance_geo_km),
            "distance_fiber_km": float(self.distance_fiber_km),
            "prop_rtt_ms": float(self.prop_rtt_ms),
            "congestion": float(self.congestion),
            "diurnal": float(self.diurnal),
            "burst": float(self.burst),
            "outage": bool(self.outage),
            "seed": None,
        }


@dataclass
class _EdgeDynState:
    """Internal continuous-time state for one edge link."""

    congestion: float = 0.0  # OU state (dimensionless)
    burst_until: float = 0.0
    outage_until: float = 0.0
    force_outage: bool = False
    last_t: float | None = None


@dataclass
class NetworkEnvironment:
    """Shared physical network world: 1 cloud + N edges with live dynamics."""

    cloud: GeoSite
    edges: dict[str, GeoSite] = field(default_factory=dict)
    seed: int = 42
    route_stretch: float = 1.5
    fiber_km_per_ms: float = _DEFAULT_FIBER_KM_PER_MS
    processing_rtt_ms: float = 2.0
    timeout_ms: float = 3000.0
    # dynamics hyperparameters
    ou_theta: float = 0.12  # mean reversion / second (wall clock)
    ou_mu: float = 0.15
    ou_sigma: float = 0.25
    diurnal_amp: float = 0.30
    burst_rate_per_hour: float = 0.6
    burst_duration_s: float = 25.0
    burst_severity: float = 1.2
    outage_rate_per_day: float = 0.05
    outage_duration_s: float = 45.0
    cong_rtt_gain_ms: float = 40.0  # extra RTT per unit congestion
    cong_bw_exp: float = 1.35
    # optional virtual time base; None → wall clock
    t0: float | None = None
    time_scale: float = 1.0  # >1 accelerates diurnal/OU for demos

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._dyn: dict[str, _EdgeDynState] = {eid: _EdgeDynState() for eid in self.edges}
        if self.t0 is None:
            self.t0 = time.time()

    # ---- construction -----------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict | None = None, *, edge_ids: list[str] | None = None) -> "NetworkEnvironment":
        """Build env from ``collab.network_env`` (or full yaml collab block)."""
        raw = dict(cfg or {})
        # allow passing full collab dict
        if "network_env" in raw and "cloud" not in raw:
            raw = dict(raw.get("network_env") or {})

        cloud_cfg = dict(raw.get("cloud") or {})
        c_city, c_lat, c_lon = resolve_city(
            cloud_cfg.get("city"), cloud_cfg.get("lat"), cloud_cfg.get("lon")
        )
        cloud = GeoSite(
            id=str(cloud_cfg.get("id") or "cloud-0"),
            name=str(cloud_cfg.get("name") or f"Cloud-{c_city}"),
            city=c_city,
            lat=c_lat,
            lon=c_lon,
            access=str(cloud_cfg.get("access") or "fiber_enterprise"),
            tz_offset_hours=float(cloud_cfg.get("tz_offset_hours", 8.0)),
        )

        default_cities = list(
            raw.get("default_edge_cities")
            or ["Suzhou", "Shenzhen", "Chengdu", "Beijing", "Urumqi", "Guangzhou"]
        )
        default_access = list(
            raw.get("default_edge_access")
            or ["fiber_enterprise", "broadband", "5g", "broadband", "weak_backhaul"]
        )
        explicit = list(raw.get("edges") or [])
        ids = list(edge_ids or [])
        if not ids:
            n = int(raw.get("num_edges") or max(len(explicit), 3))
            ids = [str((explicit[i] or {}).get("id") if i < len(explicit) else f"edge-{i}") for i in range(n)]
            for i in range(n):
                if i < len(explicit) and explicit[i].get("id"):
                    ids[i] = str(explicit[i]["id"])

        edges: dict[str, GeoSite] = {}
        for i, eid in enumerate(ids):
            ov = dict(explicit[i]) if i < len(explicit) and isinstance(explicit[i], dict) else {}
            city = ov.get("city") or default_cities[i % len(default_cities)]
            city, lat, lon = resolve_city(city, ov.get("lat"), ov.get("lon"))
            access = str(ov.get("access") or default_access[i % len(default_access)])
            edges[eid] = GeoSite(
                id=eid,
                name=str(ov.get("name") or f"Edge-{city}"),
                city=city,
                lat=lat,
                lon=lon,
                access=access,
                tz_offset_hours=float(ov.get("tz_offset_hours", cloud.tz_offset_hours)),
            )

        dyn = dict(raw.get("dynamics") or {})
        return cls(
            cloud=cloud,
            edges=edges,
            seed=int(raw.get("seed", 42)),
            route_stretch=float(raw.get("route_stretch", 1.5)),
            fiber_km_per_ms=float(raw.get("fiber_km_per_ms", _DEFAULT_FIBER_KM_PER_MS)),
            processing_rtt_ms=float(raw.get("processing_rtt_ms", 2.0)),
            timeout_ms=float(raw.get("timeout_ms", 3000.0)),
            ou_theta=float(dyn.get("ou_theta", raw.get("ou_theta", 0.12))),
            ou_mu=float(dyn.get("ou_mu", raw.get("ou_mu", 0.15))),
            ou_sigma=float(dyn.get("ou_sigma", raw.get("ou_sigma", 0.25))),
            diurnal_amp=float(dyn.get("diurnal_amp", raw.get("diurnal_amp", 0.30))),
            burst_rate_per_hour=float(dyn.get("burst_rate_per_hour", raw.get("burst_rate_per_hour", 0.6))),
            burst_duration_s=float(dyn.get("burst_duration_s", raw.get("burst_duration_s", 25.0))),
            burst_severity=float(dyn.get("burst_severity", raw.get("burst_severity", 1.2))),
            outage_rate_per_day=float(dyn.get("outage_rate_per_day", raw.get("outage_rate_per_day", 0.05))),
            outage_duration_s=float(dyn.get("outage_duration_s", raw.get("outage_duration_s", 45.0))),
            cong_rtt_gain_ms=float(dyn.get("cong_rtt_gain_ms", raw.get("cong_rtt_gain_ms", 40.0))),
            cong_bw_exp=float(dyn.get("cong_bw_exp", raw.get("cong_bw_exp", 1.35))),
            time_scale=float(raw.get("time_scale", 1.0)),
        )

    # ---- time / dynamics --------------------------------------------------

    def now(self) -> float:
        return time.time()

    def _sim_dt(self, prev: float | None, t: float) -> float:
        if prev is None:
            return 0.0
        return max(0.0, (t - prev) * float(self.time_scale))

    def _ensure_dyn(self, edge_id: str) -> _EdgeDynState:
        if edge_id not in self._dyn:
            self._dyn[edge_id] = _EdgeDynState()
        return self._dyn[edge_id]

    def _advance(self, edge_id: str, t: float) -> _EdgeDynState:
        st = self._ensure_dyn(edge_id)
        dt = self._sim_dt(st.last_t, t)
        st.last_t = t
        if dt <= 0:
            return st

        # Ornstein–Uhlenbeck: dX = θ(μ−X)dt + σ dW
        theta, mu, sigma = self.ou_theta, self.ou_mu, self.ou_sigma
        # exact OU step
        exp = math.exp(-theta * dt)
        st.congestion = (
            st.congestion * exp
            + mu * (1.0 - exp)
            + sigma * math.sqrt(max(1e-12, (1.0 - exp * exp) / (2.0 * theta))) * float(self._rng.normal())
        )
        st.congestion = float(np.clip(st.congestion, -0.2, 3.5))

        # Poisson burst / outage arrivals over dt
        burst_p = 1.0 - math.exp(-self.burst_rate_per_hour / 3600.0 * dt)
        if t >= st.burst_until and self._rng.random() < burst_p:
            st.burst_until = t + float(self.burst_duration_s) * (0.5 + self._rng.random())

        outage_p = 1.0 - math.exp(-self.outage_rate_per_day / 86400.0 * dt)
        if t >= st.outage_until and self._rng.random() < outage_p:
            st.outage_until = t + float(self.outage_duration_s) * (0.5 + self._rng.random())

        return st

    def _diurnal(self, site: GeoSite, t: float) -> float:
        """0..~amp congestion contribution; peaks near local 14:00."""
        local = (t + site.tz_offset_hours * 3600.0) % 86400.0
        # phase so peak ~ 14:00
        hour = local / 3600.0
        phase = 2.0 * math.pi * (hour - 14.0) / 24.0
        # night quieter
        return float(self.diurnal_amp * (0.5 + 0.5 * math.cos(phase)))

    # ---- public link API --------------------------------------------------

    def distance_km(self, edge_id: str) -> tuple[float, float]:
        edge = self.edges[edge_id]
        geo = haversine_km(edge.lat, edge.lon, self.cloud.lat, self.cloud.lon)
        fiber = geo * float(self.route_stretch)
        return geo, fiber

    def sample_link(self, edge_id: str, t: float | None = None) -> LinkInstant:
        if edge_id not in self.edges:
            raise KeyError(f"unknown edge site: {edge_id}")
        t = float(self.now() if t is None else t)
        edge = self.edges[edge_id]
        st = self._advance(edge_id, t)
        tech = edge.access_tech()
        geo_km, fiber_km = self.distance_km(edge_id)
        prop = 2.0 * (fiber_km / max(1e-6, self.fiber_km_per_ms))

        diurnal = self._diurnal(edge, t)
        in_burst = t < st.burst_until
        burst = float(self.burst_severity if in_burst else 0.0)
        outage = bool(st.force_outage or t < st.outage_until)

        cong = max(0.0, float(st.congestion) + diurnal + burst)
        base_rtt = prop + tech.access_rtt_ms + float(self.processing_rtt_ms)
        rtt = base_rtt + self.cong_rtt_gain_ms * cong
        jitter = max(0.5, rtt * tech.jitter_frac * (1.0 + 0.5 * cong))

        bw = tech.bandwidth_mbps / ((1.0 + cong) ** self.cong_bw_exp)
        bw = max(0.05, float(bw))
        loss = min(0.95, tech.base_loss + 0.04 * cong + (0.25 if in_burst else 0.0))
        if outage:
            loss = 1.0
            bw = 0.05
            rtt = max(rtt, 500.0)

        return LinkInstant(
            edge_id=edge_id,
            t=t,
            distance_geo_km=geo_km,
            distance_fiber_km=fiber_km,
            prop_rtt_ms=prop,
            access_rtt_ms=tech.access_rtt_ms,
            processing_rtt_ms=float(self.processing_rtt_ms),
            congestion=float(st.congestion),
            diurnal=diurnal,
            burst=burst,
            outage=outage,
            rtt_ms=float(rtt),
            rtt_jitter_ms=float(jitter),
            bandwidth_mbps=bw,
            loss_prob=float(loss),
            timeout_ms=float(self.timeout_ms),
            city=edge.city,
            access=tech.name,
            profile="outage" if outage else "geo",
            name="outage" if outage else "geo",
        )

    def try_upload(
        self,
        edge_id: str,
        payload_bytes: int,
        t: float | None = None,
    ) -> tuple[UploadOutcome, LinkInstant]:
        """One upload attempt on the live physical link (no wall sleep)."""
        link = self.sample_link(edge_id, t=t)
        if link.outage or self._rng.random() < link.loss_prob:
            reason = "outage" if link.outage else "loss"
            return (
                UploadOutcome(ok=False, failed_reason=reason, rtt_ms=float(link.rtt_ms)),
                link,
            )

        rtt = max(0.0, float(self._rng.normal(link.rtt_ms, link.rtt_jitter_ms)))
        bw = max(1e-6, float(link.bandwidth_mbps))
        # mild bandwidth flicker
        bw *= float(np.clip(self._rng.normal(1.0, 0.05), 0.7, 1.3))
        tx = (float(payload_bytes) * 8.0) / (bw * 1e6) * 1000.0
        total = rtt + tx
        if total > link.timeout_ms:
            return (
                UploadOutcome(
                    ok=False,
                    rtt_ms=rtt,
                    tx_ms=tx,
                    total_net_ms=total,
                    failed_reason="timeout",
                ),
                link,
            )
        return (
            UploadOutcome(ok=True, rtt_ms=rtt, tx_ms=tx, total_net_ms=total),
            link,
        )

    def set_force_outage(self, edge_id: str, on: bool) -> None:
        self._ensure_dyn(edge_id).force_outage = bool(on)

    def summary(self, t: float | None = None) -> dict[str, Any]:
        t = float(self.now() if t is None else t)
        links = {eid: self.sample_link(eid, t=t).to_dict() for eid in self.edges}
        return {
            "mode": "physical_geo_temporal",
            "cloud": self.cloud.to_dict(),
            "seed": self.seed,
            "route_stretch": self.route_stretch,
            "fiber_km_per_ms": self.fiber_km_per_ms,
            "timeout_ms": self.timeout_ms,
            "time_scale": self.time_scale,
            "n_edges": len(self.edges),
            "links": links,
        }


def default_env_config(num_edges: int = 3) -> dict[str, Any]:
    """Sensible default block for configs/default.yaml."""
    return {
        "enabled": True,
        "seed": 42,
        "route_stretch": 1.5,
        "fiber_km_per_ms": 200.0,
        "processing_rtt_ms": 2.0,
        "timeout_ms": 3000.0,
        "time_scale": 8.0,  # accelerate dynamics for interactive demos
        "cloud": {"city": "Shanghai", "name": "Cloud-Shanghai", "access": "fiber_enterprise"},
        "default_edge_cities": ["Suzhou", "Shenzhen", "Chengdu", "Beijing", "Urumqi"],
        "default_edge_access": ["fiber_enterprise", "broadband", "5g", "broadband", "weak_backhaul"],
        "num_edges": int(num_edges),
        "dynamics": {
            "ou_theta": 0.12,
            "ou_mu": 0.15,
            "ou_sigma": 0.25,
            "diurnal_amp": 0.30,
            "burst_rate_per_hour": 0.6,
            "burst_duration_s": 25.0,
            "outage_rate_per_day": 0.05,
            "outage_duration_s": 45.0,
        },
    }
