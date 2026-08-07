"""Pluggable cloud–edge collaboration routing algorithms.

Usage:
    from src.collab_routing import configure_routing, get_router, rule_upload

    configure_routing(collab_yaml_block)   # once at startup
    verdict = get_router().decide(signal, cloud)
    # A/B:
    v_base = get_router(policy="baseline").decide(signal)
    v_crr = get_router(policy="cost_risk").decide(signal)
"""
from src.collab_routing.adapters import (
    cloud_state_from_fleet,
    push_routing_context,
    reset_routing_context,
    rule_decide,
    rule_upload,
    signal_from_route_context,
)
from src.collab_routing.base import (
    AdmitCandidate,
    AdmitResult,
    CloudState,
    CollabRouter,
    RouteSignal,
    RouteVerdict,
)
from src.collab_routing.baseline import BaselineMarginRouter
from src.collab_routing.cost_risk import CostRiskRouter
from src.collab_routing.registry import (
    build_router,
    configure_routing,
    get_router,
    list_policies,
    normalize_policy,
    reset_router_cache,
)

__all__ = [
    "AdmitCandidate",
    "AdmitResult",
    "BaselineMarginRouter",
    "CloudState",
    "CollabRouter",
    "CostRiskRouter",
    "RouteSignal",
    "RouteVerdict",
    "build_router",
    "cloud_state_from_fleet",
    "configure_routing",
    "get_router",
    "list_policies",
    "normalize_policy",
    "push_routing_context",
    "reset_router_cache",
    "reset_routing_context",
    "rule_decide",
    "rule_upload",
    "signal_from_route_context",
]
