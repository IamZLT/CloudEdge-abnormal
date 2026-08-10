"""Helpers to bind benches / agents to physical geo-temporal network simulation.

Legacy static profiles (fair/weak fixed RTT tables) are not used for routing CONTEXT.
Scenarios only select edge city + access tech; live RTT/BW/loss come from NetworkEnvironment.
"""
from __future__ import annotations

from typing import Any

from src.network_env import NetworkEnvironment
from src.network_sim import NetworkSimulator

# Named scenarios → geo site (city, access). Not static RTT presets.
GEO_SCENARIOS: dict[str, dict[str, str]] = {
    "good": {"city": "Suzhou", "access": "fiber_enterprise"},
    "fair": {"city": "Shenzhen", "access": "broadband"},
    "weak": {"city": "Urumqi", "access": "weak_backhaul"},
}


def resolve_scenario(name: str) -> dict[str, str]:
    key = str(name or "fair").strip().lower()
    if key in GEO_SCENARIOS:
        return dict(GEO_SCENARIOS[key])
    # allow raw "City:access" overrides
    if ":" in key:
        city, access = key.split(":", 1)
        return {"city": city.strip().title(), "access": access.strip()}
    return dict(GEO_SCENARIOS["fair"])


def make_geo_simulator(
    collab: dict[str, Any] | None,
    scenario: str,
    *,
    seed: int = 42,
    edge_id: str | None = None,
) -> NetworkSimulator:
    """Build a NetworkSimulator bound to NetworkEnvironment for ``scenario``."""
    sc = resolve_scenario(scenario)
    eid = edge_id or f"edge-{str(scenario).strip().lower() or 'fair'}"
    env_cfg = dict((collab or {}).get("network_env") or {})
    env_cfg["seed"] = int(seed)
    env_cfg["edges"] = [{"id": eid, "city": sc["city"], "access": sc["access"]}]
    env_cfg["num_edges"] = 1
    # keep dynamics / cloud from config; force physical mode
    env = NetworkEnvironment.from_config(env_cfg, edge_ids=[eid])
    return NetworkSimulator.from_env(env, eid, seed=seed)


def live_network_dict(sim: NetworkSimulator) -> dict[str, Any]:
    """Refresh geo link and return a network dict for RouteContext / CRR."""
    sim.refresh_profile_from_env()
    net = dict(sim.last_link or {})
    if not net:
        net = dict(sim.profile.to_dict())
        net["profile"] = sim.profile.name
        net["outage"] = sim.profile.name == "outage"
    else:
        net.setdefault("profile", "outage" if net.get("outage") else "geo")
        net.setdefault("outage", bool(net.get("outage")))
    return net
