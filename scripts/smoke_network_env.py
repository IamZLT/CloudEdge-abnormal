#!/usr/bin/env python3
"""Smoke-test physical geo + temporal network environment.

Shows per-edge distance / propagation RTT and how live RTT wanders over time.

Example:
  python scripts/smoke_network_env.py --config configs/default.yaml --seconds 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.edge_fleet import EdgeFleet  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--seconds", type=float, default=5.0, help="wall time to sample dynamics")
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--tag", default="net_env_smoke")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    fleet = EdgeFleet.from_config(cfg, data_root=ROOT / (cfg.get("data_root") or "datasets/mvtec"))
    assert fleet.env is not None, "network_env.enabled must be true"

    env = fleet.env
    print(f"[net_env] cloud={env.cloud.city} ({env.cloud.lat:.2f},{env.cloud.lon:.2f})")
    print(f"[net_env] stretch={env.route_stretch} fiber_km/ms={env.fiber_km_per_ms} time_scale={env.time_scale}")

    baseline = []
    for nid in fleet.order:
        link = env.sample_link(nid)
        node = fleet.get(nid)
        print(
            f"  {nid}: {link.city:10s} access={link.access:16s} "
            f"geo={link.distance_geo_km:7.1f}km fiber={link.distance_fiber_km:7.1f}km "
            f"prop_rtt={link.prop_rtt_ms:5.2f}ms live_rtt={link.rtt_ms:6.1f}ms "
            f"bw={link.bandwidth_mbps:6.1f}Mbps cat={node.category}"
        )
        baseline.append(link.to_dict())

    # prove temporal variation
    series = {nid: [] for nid in fleet.order}
    t_end = time.time() + float(args.seconds)
    while time.time() < t_end:
        for nid in fleet.order:
            link = env.sample_link(nid)
            series[nid].append(
                {
                    "t": link.t,
                    "rtt_ms": link.rtt_ms,
                    "bw": link.bandwidth_mbps,
                    "loss": link.loss_prob,
                    "cong": link.congestion,
                    "diurnal": link.diurnal,
                    "burst": link.burst,
                    "outage": link.outage,
                }
            )
        time.sleep(float(args.dt))

    print("\n[net_env] RTT range over window (should differ across cities & vary in time):")
    stats = {}
    for nid, pts in series.items():
        rtts = [p["rtt_ms"] for p in pts]
        stats[nid] = {
            "n": len(rtts),
            "rtt_min": min(rtts),
            "rtt_max": max(rtts),
            "rtt_mean": sum(rtts) / len(rtts),
            "city": env.edges[nid].city,
        }
        print(
            f"  {nid} ({stats[nid]['city']}): "
            f"rtt {stats[nid]['rtt_min']:.1f}–{stats[nid]['rtt_max']:.1f} "
            f"(mean {stats[nid]['rtt_mean']:.1f}) n={stats[nid]['n']}"
        )

    # distant edge should have higher prop RTT than near edge
    props = [(nid, env.sample_link(nid).prop_rtt_ms) for nid in fleet.order]
    props_sorted = sorted(props, key=lambda x: x[1])
    assert props_sorted[0][1] < props_sorted[-1][1], props_sorted
    print(f"[net_env] prop RTT ordered: {props_sorted} (near < far) OK")

    out_dir = ROOT / "outputs" / "network_env"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}.json"
    out_path.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "cloud": env.cloud.to_dict(),
                "baseline_links": baseline,
                "rtt_stats": stats,
                "series_tail": {k: v[-6:] for k, v in series.items()},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
