"""Adapters between RouteAgent CONTEXT and collab_routing signals."""
from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any

from src.collab_routing.base import CloudState, RouteSignal, RouteVerdict
from src.collab_routing.registry import get_router

# So RouteAgent.rules_snap -> heuristic_upload can see fleet cloud load.
_cloud_ctx: ContextVar[CloudState | None] = ContextVar("collab_cloud_state", default=None)
_node_ctx: ContextVar[str | None] = ContextVar("collab_edge_node_id", default=None)
_recent_ctx: ContextVar[float] = ContextVar("collab_recent_cloud", default=0.0)


def push_routing_context(
    *,
    cloud: CloudState | None = None,
    edge_node_id: str | None = None,
    recent_cloud: float = 0.0,
) -> tuple[Token, Token, Token]:
    return (
        _cloud_ctx.set(cloud),
        _node_ctx.set(edge_node_id),
        _recent_ctx.set(float(recent_cloud)),
    )


def reset_routing_context(tokens: tuple[Token, Token, Token]) -> None:
    _cloud_ctx.reset(tokens[0])
    _node_ctx.reset(tokens[1])
    _recent_ctx.reset(tokens[2])


def signal_from_route_context(
    ctx: Any,
    *,
    edge_node_id: str | None = None,
    recent_cloud: float = 0.0,
) -> RouteSignal:
    """Build RouteSignal from ``route_agent.RouteContext`` (duck-typed)."""
    net = dict(getattr(ctx, "network", None) or {})
    profile = str(getattr(ctx, "network_profile", None) or net.get("profile") or "fair")
    return RouteSignal(
        category=str(getattr(ctx, "category", "") or ""),
        n_gallery=int(getattr(ctx, "n_gallery", 0) or 0),
        edge_score=float(getattr(ctx, "edge_score", 0.5)),
        edge_thr=float(getattr(ctx, "edge_thr", 0.5)),
        edge_decision=str(getattr(ctx, "edge_decision", "OK") or "OK"),
        network_profile=profile,
        network=net,
        hard_margin=float(getattr(ctx, "hard_margin", 0.05) or 0.05),
        edge_node_id=edge_node_id,
        recent_cloud=float(recent_cloud),
        conflict=float(getattr(ctx, "conflict", 0.0) or 0.0),
    )


def cloud_state_from_fleet(fleet: Any | None, collab_cfg: dict | None = None) -> CloudState:
    """Read CloudState from EdgeFleet if present; else empty/single-node defaults."""
    adm = dict((collab_cfg or {}).get("cloud_admission") or {})
    k = int(adm.get("max_inflight", 2))
    if fleet is None:
        return CloudState(inflight=0, queue=0, max_inflight=k)
    state = getattr(fleet, "cloud_state", None)
    if state is not None and hasattr(state, "inflight"):
        return CloudState(
            inflight=int(getattr(state, "inflight", 0)),
            queue=int(getattr(state, "queue", 0)),
            max_inflight=int(getattr(state, "max_inflight", k)),
        )
    # duck-typed attributes on fleet
    return CloudState(
        inflight=int(getattr(fleet, "cloud_inflight", 0) or 0),
        queue=int(getattr(fleet, "cloud_queue", 0) or 0),
        max_inflight=int(getattr(fleet, "cloud_max_inflight", k) or k),
    )


def rule_decide(
    ctx: Any,
    *,
    collab_cfg: dict | None = None,
    policy: str | None = None,
    cloud: CloudState | None = None,
    edge_node_id: str | None = None,
    recent_cloud: float | None = None,
) -> RouteVerdict:
    """Run configured collab router on a RouteContext."""
    router = get_router(collab_cfg, policy=policy)
    eid = edge_node_id if edge_node_id is not None else _node_ctx.get()
    recent = float(recent_cloud if recent_cloud is not None else _recent_ctx.get())
    signal = signal_from_route_context(ctx, edge_node_id=eid, recent_cloud=recent)
    cloud_state = cloud if cloud is not None else _cloud_ctx.get()
    return router.decide(signal, cloud_state)


def rule_upload(
    ctx: Any,
    *,
    collab_cfg: dict | None = None,
    policy: str | None = None,
    cloud: CloudState | None = None,
    edge_node_id: str | None = None,
) -> bool:
    """Boolean upload decision (drop-in replacement for heuristic_upload)."""
    return bool(
        rule_decide(
            ctx,
            collab_cfg=collab_cfg,
            policy=policy,
            cloud=cloud,
            edge_node_id=edge_node_id,
        ).upload
    )
