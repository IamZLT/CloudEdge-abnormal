#!/usr/bin/env python3
"""Smoke + A/B compare for pluggable collab routing (baseline vs cost_risk)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.collab_routing import (  # noqa: E402
    AdmitCandidate,
    CloudState,
    RouteSignal,
    build_router,
    list_policies,
)


def _signal(**kw) -> RouteSignal:
    base = dict(
        category="bottle",
        n_gallery=16,
        edge_score=0.65,
        edge_thr=0.487,
        edge_decision="NG",
        network_profile="geo",
        network={"rtt_ms": 40.0, "bandwidth_mbps": 80.0, "loss_prob": 0.01},
        hard_margin=0.05,
        edge_node_id="edge-0",
    )
    base.update(kw)
    return RouteSignal(**base)


def main() -> int:
    print("policies:", list_policies())
    cases = [
        ("confident_good_link", _signal()),
        (
            "near_thr_good_link",
            _signal(edge_score=0.50, edge_thr=0.487),
        ),
        (
            "near_thr_weak_link",
            _signal(
                edge_score=0.50,
                edge_thr=0.487,
                network={"rtt_ms": 250.0, "bandwidth_mbps": 2.0, "loss_prob": 0.25},
            ),
        ),
        ("cold_start", _signal(n_gallery=0)),
        ("outage", _signal(network_profile="outage", network={"outage": True})),
    ]
    cloud = CloudState(inflight=1, queue=0, max_inflight=2)
    rows = []
    for name, sig in cases:
        row = {"case": name}
        for policy in ("baseline", "cost_risk"):
            r = build_router(policy, {"route_policy": policy, "cost_risk": {}, "cloud_admission": {}})
            v = r.decide(sig, cloud)
            row[policy] = {"upload": v.upload, "U": round(v.utility, 4) if v.utility != float("-inf") else "-inf"}
        rows.append(row)
    print(json.dumps(rows, indent=2, ensure_ascii=False))

    # admission Top-K
    crr = build_router("cost_risk", {"cloud_admission": {"max_inflight": 1, "fairness_gamma": 0.1}})
    cands = []
    for i, (name, sig) in enumerate(cases[:3]):
        sig.edge_node_id = f"edge-{i}"
        sig.recent_cloud = float(i)
        v = crr.decide(sig, cloud)
        cands.append(AdmitCandidate(signal=sig, verdict=v, request_id=name))
    adm = crr.admit(cands, max_inflight=1, cloud=cloud)
    print(
        "admit_accept:",
        [c.request_id for c in adm.accepted],
        "reject:",
        [c.request_id for c in adm.rejected if c.verdict.upload],
    )
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
