"""Multi-edge node fleet for cloud–edge collaborative inspection.

One logical cloud reviewer is shared; each edge node has its own category,
geo site, and link into a shared physical ``NetworkEnvironment``.
Node count is configurable via ``collab.edge_fleet.num_nodes`` (default 3).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.network_env import NetworkEnvironment, default_env_config
from src.network_sim import NetworkSimulator, resolve_profile

# Cycled when auto-building nodes (distinct product lines).
DEFAULT_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]
# Legacy fallback only when network_env.enabled=false
DEFAULT_NETWORK_PROFILES = ["good", "fair", "weak", "outage"]


@dataclass
class EdgeNodeStats:
    n_infer: int = 0
    n_local: int = 0
    n_upload_want: int = 0
    n_upload_ok: int = 0
    n_upload_fail: int = 0
    n_outage_gate: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def record_path(self, *, path_type: str, upload_want: bool, network_profile: str) -> None:
        self.n_infer += 1
        if str(network_profile).lower() == "outage" and not upload_want:
            self.n_outage_gate += 1
        if upload_want:
            self.n_upload_want += 1
        if path_type == "CLOUD_REVIEW":
            self.n_upload_ok += 1
        elif path_type == "LOCAL_NET_FALLBACK":
            self.n_upload_fail += 1
        elif path_type == "LOCAL":
            self.n_local += 1


@dataclass
class EdgeNode:
    """One edge inspection site (logical node)."""

    id: str
    name: str
    category: str
    network: dict[str, Any] = field(default_factory=lambda: {"profile": "geo", "seed": 42})
    enabled: bool = True
    index: int = 0
    city: str | None = None
    access: str | None = None
    stats: EdgeNodeStats = field(default_factory=EdgeNodeStats)
    _sim: NetworkSimulator | None = field(default=None, repr=False, compare=False)
    _env: NetworkEnvironment | None = field(default=None, repr=False, compare=False)

    def bind_env(self, env: NetworkEnvironment) -> None:
        """Attach shared physical environment (preferred)."""
        self._env = env
        if self.id not in env.edges:
            raise KeyError(f"env has no site for {self.id}")
        site = env.edges[self.id]
        self.city = site.city
        self.access = site.access
        self.name = self.name if self.name and not self.name.startswith("Edge-") else site.name
        self._sim = NetworkSimulator.from_env(env, self.id, seed=42 + int(self.index))
        self.network = self.network_snapshot()

    def __post_init__(self) -> None:
        self.network = dict(self.network or {})
        self.network.setdefault("profile", "geo")
        self.network.setdefault("seed", 42 + int(self.index))
        # Legacy path until bind_env() is called
        if self._sim is None and self._env is None:
            if str(self.network.get("profile") or "geo") in {"good", "fair", "weak", "outage", "custom"}:
                self._sim = NetworkSimulator.from_config(self.network)

    @property
    def sim(self) -> NetworkSimulator:
        if self._sim is None:
            if self._env is not None:
                self._sim = NetworkSimulator.from_env(self._env, self.id, seed=42 + int(self.index))
            else:
                self._sim = NetworkSimulator.from_config(self.network)
        return self._sim

    def set_network(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """Update link controls. For geo env, profile=outage forces outage."""
        merged = dict(self.network)
        merged.update(cfg or {})
        merged.setdefault("seed", 42 + int(self.index))
        profile = str(merged.get("profile") or "geo").lower()

        if self._env is not None:
            if profile == "outage":
                self._env.set_force_outage(self.id, True)
            else:
                self._env.set_force_outage(self.id, False)
                # ignore static rtt overrides in geo mode; physics owns RTT
            self._sim = NetworkSimulator.from_env(self._env, self.id, seed=int(merged.get("seed") or 42))
            self.network = self.network_snapshot()
            return self.network

        if "profile" not in merged:
            merged["profile"] = "fair"
        self.network = merged
        self._sim = NetworkSimulator.from_config(self.network)
        return self.network_snapshot()

    def network_snapshot(self) -> dict[str, Any]:
        if self._env is not None:
            link = self._env.sample_link(self.id)
            d = link.to_profile_dict()
            d["seed"] = int(self.network.get("seed") or 42 + self.index)
            return d
        prof = resolve_profile(self.network)
        d = prof.to_dict()
        d["profile"] = prof.name
        d["seed"] = int(self.network.get("seed") or 42)
        return d

    def to_dict(self) -> dict[str, Any]:
        net = self.network_snapshot()
        site = None
        if self._env is not None and self.id in self._env.edges:
            site = self._env.edges[self.id].to_dict()
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "index": self.index,
            "enabled": self.enabled,
            "city": self.city or (site or {}).get("city"),
            "access": self.access or (site or {}).get("access"),
            "site": site,
            "network": net,
            "stats": self.stats.to_dict(),
        }


def _as_int(val: Any, default: int) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def resolve_num_nodes(fleet_cfg: dict | None, collab_cfg: dict | None = None) -> int:
    """Resolve node count: edge_fleet.num_nodes > collab.num_edge_nodes > 3."""
    fleet = dict(fleet_cfg or {})
    collab = dict(collab_cfg or {})
    n = fleet.get("num_nodes", collab.get("num_edge_nodes", 3))
    n = _as_int(n, 3)
    return max(1, min(n, 32))


def build_edge_node_specs(
    fleet_cfg: dict | None = None,
    collab_cfg: dict | None = None,
    *,
    data_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Build ``num_nodes`` node specs (explicit list padded/truncated as needed)."""
    fleet = dict(fleet_cfg or {})
    collab = dict(collab_cfg or {})
    n = resolve_num_nodes(fleet, collab)

    cats = list(fleet.get("default_categories") or DEFAULT_CATEGORIES)
    if data_root is not None:
        root = Path(data_root)
        if root.exists():
            present = sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "test").exists())
            if present:
                # keep configured order when possible
                preferred = [c for c in cats if c in present]
                rest = [c for c in present if c not in preferred]
                cats = preferred + rest if preferred else present
    if not cats:
        cats = list(DEFAULT_CATEGORIES)

    env_cfg = dict(collab.get("network_env") or fleet.get("network_env") or {})
    cities = list(
        fleet.get("default_cities")
        or env_cfg.get("default_edge_cities")
        or ["Suzhou", "Shenzhen", "Chengdu"]
    )
    access_list = list(
        fleet.get("default_access")
        or env_cfg.get("default_edge_access")
        or ["fiber_enterprise", "broadband", "5g"]
    )
    profiles = list(fleet.get("default_network_profiles") or DEFAULT_NETWORK_PROFILES)
    use_geo = bool(env_cfg.get("enabled", True))

    explicit = list(fleet.get("nodes") or [])
    specs: list[dict[str, Any]] = []
    for i in range(n):
        city = cities[i % len(cities)]
        access = access_list[i % len(access_list)]
        base: dict[str, Any] = {
            "id": f"edge-{i}",
            "name": f"Edge-{city}",
            "category": cats[i % len(cats)],
            "city": city,
            "access": access,
            "network": {
                "profile": "geo" if use_geo else profiles[i % len(profiles)],
                "seed": 42 + i,
            },
            "enabled": True,
            "index": i,
        }
        if i < len(explicit) and isinstance(explicit[i], dict):
            ov = dict(explicit[i])
            net_ov = dict(base["network"])
            if isinstance(ov.get("network"), dict):
                net_ov.update(ov["network"])
            elif ov.get("network_profile"):
                net_ov["profile"] = ov["network_profile"]
            base.update({k: v for k, v in ov.items() if k != "network"})
            base["network"] = net_ov
            base.setdefault("id", f"edge-{i}")
            base.setdefault("city", city)
            base.setdefault("access", access)
            base.setdefault("name", f"Edge-{base.get('city') or city}")
            base["index"] = i
        specs.append(base)
    return specs


class EdgeFleet:
    """Collection of edge nodes sharing one cloud + one physical network world."""

    def __init__(
        self,
        nodes: list[EdgeNode],
        *,
        num_nodes: int | None = None,
        env: NetworkEnvironment | None = None,
    ):
        if not nodes:
            raise ValueError("EdgeFleet requires at least one node")
        self.nodes: dict[str, EdgeNode] = {n.id: n for n in nodes}
        self.order: list[str] = [n.id for n in nodes]
        self.num_nodes = int(num_nodes if num_nodes is not None else len(nodes))
        self.active_id: str = self.order[0]
        self.env = env
        if env is not None:
            for n in nodes:
                n.bind_env(env)

    @classmethod
    def from_config(
        cls,
        cfg: dict | None = None,
        *,
        collab_cfg: dict | None = None,
        data_root: str | Path | None = None,
    ) -> "EdgeFleet":
        """Load from full yaml dict or a collab/edge_fleet fragment."""
        root_cfg = dict(cfg or {})
        collab = dict(collab_cfg or root_cfg.get("collab") or {})
        fleet_cfg = dict(root_cfg.get("edge_fleet") or collab.get("edge_fleet") or {})
        if "num_nodes" not in fleet_cfg and "num_edge_nodes" in collab:
            fleet_cfg["num_nodes"] = collab["num_edge_nodes"]
        if data_root is None:
            data_root = root_cfg.get("data_root")
        specs = build_edge_node_specs(fleet_cfg, collab, data_root=data_root)

        env_cfg = dict(collab.get("network_env") or {})
        use_geo = bool(env_cfg.get("enabled", True))
        env: NetworkEnvironment | None = None
        if use_geo:
            # merge defaults + user cfg; align edge ids/cities with fleet specs
            merged = default_env_config(num_edges=len(specs))
            merged.update({k: v for k, v in env_cfg.items() if k != "dynamics"})
            if isinstance(env_cfg.get("dynamics"), dict):
                dyn = dict(merged.get("dynamics") or {})
                dyn.update(env_cfg["dynamics"])
                merged["dynamics"] = dyn
            merged["num_edges"] = len(specs)
            # push per-node city/access into env edge list
            merged["edges"] = [
                {
                    "id": s["id"],
                    "name": s.get("name"),
                    "city": s.get("city"),
                    "access": s.get("access"),
                    **(dict(s.get("site") or {})),
                }
                for s in specs
            ]
            env = NetworkEnvironment.from_config(merged, edge_ids=[s["id"] for s in specs])

        nodes = [
            EdgeNode(
                id=str(s["id"]),
                name=str(s.get("name") or s["id"]),
                category=str(s.get("category") or "bottle"),
                network=dict(s.get("network") or {}),
                enabled=bool(s.get("enabled", True)),
                index=int(s.get("index") or i),
                city=s.get("city"),
                access=s.get("access"),
            )
            for i, s in enumerate(specs)
        ]
        return cls(nodes, num_nodes=len(nodes), env=env)

    def get(self, node_id: str | None = None) -> EdgeNode:
        nid = node_id or self.active_id
        if nid not in self.nodes:
            raise KeyError(f"unknown edge node: {nid}; known={self.order}")
        return self.nodes[nid]

    def set_active(self, node_id: str) -> EdgeNode:
        node = self.get(node_id)
        self.active_id = node.id
        return node

    def list_nodes(self) -> list[dict[str, Any]]:
        return [self.nodes[i].to_dict() for i in self.order]

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "num_nodes": self.num_nodes,
            "active_id": self.active_id,
            "network_mode": "physical_geo_temporal" if self.env is not None else "legacy_profile",
            "nodes": self.list_nodes(),
        }
        if self.env is not None:
            out["network_env"] = {
                "cloud": self.env.cloud.to_dict(),
                "route_stretch": self.env.route_stretch,
                "fiber_km_per_ms": self.env.fiber_km_per_ms,
                "time_scale": self.env.time_scale,
            }
        return out

    def clone_for_worker(self) -> "EdgeFleet":
        """Rebuild from node specs (fresh env/simulators)."""
        nodes = [
            EdgeNode(
                id=n.id,
                name=n.name,
                category=n.category,
                network=deepcopy(n.network),
                enabled=n.enabled,
                index=n.index,
                city=n.city,
                access=n.access,
            )
            for n in (self.nodes[i] for i in self.order)
        ]
        env = None
        if self.env is not None:
            # rebuild env with same sites
            cfg = {
                "seed": self.env.seed,
                "route_stretch": self.env.route_stretch,
                "fiber_km_per_ms": self.env.fiber_km_per_ms,
                "processing_rtt_ms": self.env.processing_rtt_ms,
                "timeout_ms": self.env.timeout_ms,
                "time_scale": self.env.time_scale,
                "cloud": self.env.cloud.to_dict(),
                "edges": [self.env.edges[i].to_dict() for i in self.order],
            }
            env = NetworkEnvironment.from_config(cfg, edge_ids=list(self.order))
        return EdgeFleet(nodes, num_nodes=self.num_nodes, env=env)
