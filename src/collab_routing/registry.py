"""Router registry: build / cache / list collaboration algorithms."""
from __future__ import annotations

from typing import Any

from src.collab_routing.base import CollabRouter
from src.collab_routing.baseline import BaselineMarginRouter
from src.collab_routing.cost_risk import CostRiskRouter

# Canonical names + aliases for collab.route_policy
_REGISTRY: dict[str, type[CollabRouter]] = {
    "baseline": BaselineMarginRouter,
    "margin": BaselineMarginRouter,
    "heuristic": BaselineMarginRouter,
    "cost_risk": CostRiskRouter,
    "crr": CostRiskRouter,
    "cost-risk": CostRiskRouter,
}

_DEFAULT_POLICY = "cost_risk"
_cached: CollabRouter | None = None
_cached_key: str | None = None


def list_policies() -> list[str]:
    """Unique canonical policy names (no aliases)."""
    seen: list[str] = []
    for name, cls in _REGISTRY.items():
        if cls.name not in seen and name == cls.name:
            seen.append(cls.name)
    # ensure order: baseline then cost_risk
    ordered = [n for n in ("baseline", "cost_risk") if n in seen]
    ordered.extend([n for n in seen if n not in ordered])
    return ordered


def normalize_policy(name: str | None) -> str:
    key = str(name or _DEFAULT_POLICY).strip().lower().replace(" ", "_")
    if key not in _REGISTRY:
        known = sorted(set(_REGISTRY) | set(list_policies()))
        raise KeyError(f"unknown route_policy={name!r}; choose one of {known}")
    return _REGISTRY[key].name


def build_router(
    policy: str | None = None,
    collab_cfg: dict[str, Any] | None = None,
) -> CollabRouter:
    """Construct a router. ``collab_cfg`` is the ``collab:`` yaml block (or fragment)."""
    cfg = dict(collab_cfg or {})
    name = normalize_policy(policy if policy is not None else cfg.get("route_policy"))
    cls = _REGISTRY[name]
    # pass full collab cfg so CRR can read cost_risk / cloud_admission
    return cls(cfg)


def configure_routing(collab_cfg: dict[str, Any] | None = None) -> CollabRouter:
    """Set process-wide default router from collab config (call at app/bench startup)."""
    global _cached, _cached_key
    cfg = dict(collab_cfg or {})
    policy = normalize_policy(cfg.get("route_policy"))
    key = f"{policy}:{sorted((cfg.get('cost_risk') or {}).items())!r}:{sorted((cfg.get('cloud_admission') or {}).items())!r}"
    if _cached is not None and _cached_key == key:
        return _cached
    _cached = build_router(policy, cfg)
    _cached_key = key
    return _cached


def get_router(
    collab_cfg: dict[str, Any] | None = None,
    *,
    policy: str | None = None,
) -> CollabRouter:
    """Return cached default router, or build from overrides for A/B comparison."""
    if policy is not None or collab_cfg is not None:
        cfg = dict(collab_cfg or {})
        if policy is not None:
            cfg["route_policy"] = policy
        return build_router(cfg.get("route_policy"), cfg)
    if _cached is None:
        return configure_routing({"route_policy": _DEFAULT_POLICY})
    return _cached


def reset_router_cache() -> None:
    global _cached, _cached_key
    _cached = None
    _cached_key = None
