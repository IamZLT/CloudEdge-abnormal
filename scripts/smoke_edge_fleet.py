#!/usr/bin/env python3
"""Smoke-test multi-edge fleet from config (default num_nodes=3).

Example:
  python scripts/smoke_edge_fleet.py --config configs/default.yaml
  python scripts/smoke_edge_fleet.py --num-nodes 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.edge_fleet import EdgeFleet, resolve_num_nodes  # noqa: E402
from src.vlm.route_agent import heuristic_upload  # noqa: E402
from src.vlm.route_agent import RouteContext  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--num-nodes", type=int, default=None, help="override config num_nodes")
    ap.add_argument("--tag", default="fleet_smoke")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    if args.num_nodes is not None:
        collab = cfg.setdefault("collab", {})
        collab["num_edge_nodes"] = int(args.num_nodes)
        fleet_cfg = collab.setdefault("edge_fleet", {})
        fleet_cfg["num_nodes"] = int(args.num_nodes)

    data_root = Path(cfg.get("data_root") or ROOT / "datasets" / "mvtec")
    if not data_root.is_absolute():
        data_root = ROOT / data_root

    fleet = EdgeFleet.from_config(cfg, data_root=data_root)
    collab = cfg.get("collab") or {}
    expected = resolve_num_nodes(collab.get("edge_fleet"), collab)
    assert fleet.num_nodes == expected, (fleet.num_nodes, expected)

    rows = []
    for nid in fleet.order:
        node = fleet.get(nid)
        # synthetic near-threshold sample → wants upload unless outage
        ctx = RouteContext(
            image=data_root / node.category / "test" / "good" / "000.png",
            category=node.category,
            n_gallery=16,
            edge_score=0.52,
            edge_thr=0.50,
            edge_decision="NG",
            network_profile=str(node.network.get("profile") or "fair"),
            network=node.network_snapshot(),
            hard_margin=0.05,
        )
        want = heuristic_upload(ctx)
        path_type = "LOCAL"
        outcome = None
        if want:
            out = node.sim.try_upload(int(collab.get("upload_bytes_hard") or 80000))
            outcome = out.to_dict()
            path_type = "CLOUD_REVIEW" if out.ok else "LOCAL_NET_FALLBACK"
        else:
            path_type = "LOCAL"
        node.stats.record_path(
            path_type=path_type, upload_want=want, network_profile=ctx.network_profile
        )
        rows.append(
            {
                "id": node.id,
                "name": node.name,
                "category": node.category,
                "network_profile": ctx.network_profile,
                "upload_want": want,
                "path_type": path_type,
                "network_outcome": outcome,
                "stats": node.stats.to_dict(),
            }
        )

    summary = {
        "tag": args.tag,
        "num_nodes": fleet.num_nodes,
        "active_id": fleet.active_id,
        "nodes": fleet.list_nodes(),
        "smoke_rows": rows,
    }
    out_dir = ROOT / "outputs" / "edge_fleet"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}_n{fleet.num_nodes}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[fleet] num_nodes={fleet.num_nodes} (config default 3)")
    for r in rows:
        print(
            f"  {r['id']:8s} cat={r['category']:12s} net={r['network_profile']:7s} "
            f"want={r['upload_want']} -> {r['path_type']}"
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
