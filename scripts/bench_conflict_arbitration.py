#!/usr/bin/env python3
"""Multi-node conflict + arbitration bench (hand-written algorithm, no LLM routing).

Pipeline per sample:
  1) N edge nodes judge the SAME image under deterministic augmentations
     (src/collab_conflict.multi_node_consensus) -> conflict + majority vote.
  2) CRR router (hand-written cost-risk) decides upload from
     uncertainty + conflict + live geo network + cloud load.
  3) If upload: network sim try_upload.
     - ok + vlm arbiter  -> cloud VLM detects -> cloud arbitration.
     - ok + vote arbiter -> margin-weighted local vote (no LLM).
     - fail              -> fail-safe (conservative NG + divert).

Reports M6 (conflict ratio) + M7 (cloud arbitration success rate) + fail-safe /
business-keep rates and final vs consensus F1.

Example (fast, no cloud model):
  conda activate clip
  CUDA_VISIBLE_DEVICES=0 python scripts/bench_conflict_arbitration.py \
    --categories bottle --scenario good --limit 20 --arbiter vote

Full (real cloud VLM arbiter):
  CUDA_VISIBLE_DEVICES=0 python scripts/bench_conflict_arbitration.py \
    --categories bottle --scenario good --limit 20 --arbiter vlm
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics as st
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge.infer import DEFAULT_PATHS, _gallery_paths  # noqa: E402
from edge.methods.encoders import load_qwen35_vision_encoder  # noqa: E402
from edge.methods.gallery_ad import mvtec_test_split  # noqa: E402
from edge.methods.patch_gallery_ad import PatchGalleryAD  # noqa: E402
from src.collab_conflict import arbitrate, multi_node_consensus, resolve_augs  # noqa: E402
from src.collab_routing import CloudState, RouteSignal, build_router  # noqa: E402
from src.network_geo import live_network_dict, make_geo_simulator  # noqa: E402
from src.vlm import QwenVLClient  # noqa: E402

OUT_DIR = ROOT / "outputs" / "reports" / "conflict_arbitration"


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _subsample(items: list[tuple[Path, int]], limit: int, seed: int) -> list[tuple[Path, int]]:
    if limit <= 0 or len(items) <= limit:
        return items
    rng = np.random.default_rng(seed)
    ok = [it for it in items if it[1] == 0]
    ng = [it for it in items if it[1] == 1]
    n_ok = max(1, limit // 3)
    n_ng = limit - n_ok
    pick_ok = ok if len(ok) <= n_ok else [ok[i] for i in sorted(rng.choice(len(ok), n_ok, replace=False))]
    pick_ng = ng if len(ng) <= n_ng else [ng[i] for i in sorted(rng.choice(len(ng), n_ng, replace=False))]
    out = pick_ok + pick_ng
    rng.shuffle(out)
    return out[:limit]


def _free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_edges(
    *,
    cfg: dict[str, Any],
    categories: list[str],
    limit: int,
    seed: int,
    device: str,
    n_nodes: int,
    aug_names: list[str],
    warmup: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    edge_cfg = dict(cfg.get("edge") or {})
    data_root = Path(cfg.get("data_root") or ROOT / "datasets" / "mvtec")
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    image_size = int(cfg.get("image_size") or 224)
    max_gallery = int(edge_cfg.get("max_gallery") or 16)
    layers = edge_cfg.get("layers") or [6, 8, 10, 12]
    fusion_temp = float(edge_cfg.get("fusion_temp") or 0.5)
    model_path = edge_cfg.get("model_path") or DEFAULT_PATHS["qwen35"]
    augs = resolve_augs(aug_names or None, n_nodes)

    print(f"[edge] load Qwen3.5 vision @ {device} (nodes={n_nodes} augs={[a.name for a in augs]}) ...")
    t0 = time.perf_counter()
    _, encode_patches, meta = load_qwen35_vision_encoder(
        model_path,
        device=device,
        max_pixels=int(edge_cfg.get("max_pixels") or image_size * image_size),
        layers=layers,
    )
    load_s = time.perf_counter() - t0

    rows: list[dict[str, Any]] = []
    for cat in categories:
        ad = PatchGalleryAD(encode_patches, device=device, name="qwen35_0.8b_vision_mlpatch", fusion_temperature=fusion_temp)
        gallery = _gallery_paths(data_root, cat, max_gallery, seed)
        thr = edge_cfg.get("threshold")
        if thr is None:
            thr = ad.calibrate_threshold_loo(
                gallery,
                seed=seed,
                quantile=float(edge_cfg.get("thr_quantile") or 0.95),
            )
        else:
            thr = float(thr)
            ad.build_gallery(gallery, seed=seed)

        test_items = mvtec_test_split(data_root, cat)
        items = _subsample(test_items, limit, seed + zlib.crc32(cat.encode()) % 1000)
        print(f"[edge] {cat}: gallery={len(gallery)} thr={thr:.4f} n_test={len(items)}")

        def _score(img: Image.Image) -> float:
            return float(ad.score_image(img)[0])

        # warmup
        for i in range(min(warmup, len(items))):
            _ = multi_node_consensus(_score, Image.open(items[i][0]).convert("RGB"), thr, augs=augs)
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize()

        for path, y in items:
            img = Image.open(path).convert("RGB")
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            cons = multi_node_consensus(_score, img, thr, augs=augs)
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.synchronize()
            edge_ms = (time.perf_counter() - t1) * 1000.0
            edge_score = float(np.median(cons.node_scores))
            rows.append(
                {
                    "category": cat,
                    "path": str(path),
                    "label": int(y),
                    "edge_score": edge_score,
                    "edge_thr": float(thr),
                    "edge_decision": cons.majority_decision,
                    "edge_pred": 1 if cons.majority_decision == "NG" else 0,
                    "edge_ms": float(edge_ms),
                    "n_gallery": len(gallery),
                    "consensus": cons.to_dict(),
                }
            )

    del encode_patches, ad
    _free_cuda()
    print("[edge] unloaded")
    return rows, {"edge_load_s": load_s, "n_nodes": n_nodes, "augs": [a.name for a in augs]}


def run_route_arbitrate(
    *,
    edge_rows: list[dict[str, Any]],
    collab: dict[str, Any],
    cloud_cfg: dict[str, Any],
    prompt: str | None,
    scenario: str,
    device: str,
    seed: int,
    arbiter: str,
    up_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    router = build_router("cost_risk", {**collab, "route_policy": "cost_risk"})
    adm = dict(collab.get("cloud_admission") or {})
    max_inflight = int(adm.get("max_inflight", 2))
    min_cloud_conf = float(adm.get("min_cloud_conf", 0.6))
    hard_margin = float(collab.get("hard_margin") or collab.get("thr_margin") or 0.05)
    sim = make_geo_simulator(collab, scenario, seed=seed, edge_id=f"edge-{scenario}")

    cloud_categories = {str(c).strip() for c in (cloud_cfg.get("categories") or [])}

    cloud = None
    cloud_load_s = 0.0
    if arbiter == "vlm":
        cats = {er["category"] for er in edge_rows}
        need_cloud = (not cloud_categories) or any(c in cloud_categories for c in cats)
        if not need_cloud:
            print(
                f"[cloud] skipped: categories {sorted(cats)} not in LoRA domain "
                f"{sorted(cloud_categories)} → weighted vote"
            )
        else:
            print("[cloud] load Qwen3-VL arbiter ...")
            t1 = time.perf_counter()
            cloud = QwenVLClient(
                model_path=cloud_cfg["model_path"],
                adapter_path=cloud_cfg.get("adapter_path"),
                device=device,
                dtype=cloud_cfg.get("dtype", "bfloat16"),
                max_new_tokens=int(cloud_cfg.get("max_new_tokens", 160)),
                role="cloud",
                prompt=prompt,
            )
            cloud_load_s = time.perf_counter() - t1
            print(f"[cloud] loaded in {cloud_load_s:.1f}s")

    cloud_state = CloudState(inflight=0, queue=0, max_inflight=max_inflight)
    all_rows: list[dict[str, Any]] = []
    for i, er in enumerate(edge_rows):
        cons = er["consensus"]
        net = live_network_dict(sim)
        sig = RouteSignal(
            category=er["category"],
            n_gallery=int(er["n_gallery"]),
            edge_score=float(er["edge_score"]),
            edge_thr=float(er["edge_thr"]),
            edge_decision=str(er["edge_decision"]),
            network_profile=str(net.get("profile") or "geo"),
            network=net,
            hard_margin=hard_margin,
            edge_node_id=f"edge-{scenario}",
            conflict=float(cons["conflict_score"]),
        )
        verd = router.decide(sig, cloud_state)
        upload = bool(verd.upload)

        net_ms = 0.0
        net_ok = False
        if upload:
            out = sim.try_upload(up_bytes)
            net_ms = float(out.rtt_ms) + float(out.tx_ms)
            net_ok = bool(out.ok)

        cloud_decision = None
        cloud_ms = 0.0
        cloud_conf = None
        cloud_fallback = False
        cloud_used = False
        if upload and net_ok:
            use_vlm = (
                arbiter == "vlm"
                and cloud is not None
                and (not cloud_categories or er["category"] in cloud_categories)
            )
            if use_vlm:
                cloud_used = True
                t_c = time.perf_counter()
                vlm = cloud.infer(Path(er["path"]))
                cloud_ms = float(vlm.latency_ms or (time.perf_counter() - t_c) * 1000.0)
                cloud_conf = float(vlm.confidence)
                if vlm.parse_ok and vlm.decision in {"OK", "NG"} and cloud_conf >= min_cloud_conf:
                    cloud_decision = vlm.decision
                else:
                    # Low-confidence cloud review (or parse failure): don't let a
                    # possibly out-of-domain LoRA override a correct edge call.
                    cloud_fallback = True
                    cloud_decision = _weighted_vote(cons)
            else:
                # hand-written tiebreaker (no LLM, or out-of-domain category)
                cloud_decision = _weighted_vote(cons)

        arb = arbitrate(_consensus_from_dict(cons), cloud_decision=cloud_decision, upload_ok=(upload and net_ok))
        final_pred = 1 if arb.decision == "NG" else 0

        row = {
            "scenario": scenario,
            "category": er["category"],
            "path": er["path"],
            "label": int(er["label"]),
            "edge_score": float(er["edge_score"]),
            "conflict": bool(cons["conflict"]),
            "conflict_score": float(cons["conflict_score"]),
            "edge_pred": int(er["edge_pred"]),
            "edge_ms": float(er["edge_ms"]),
            "upload": upload,
            "net_ok": net_ok,
            "net_ms": float(net_ms),
            "cloud_ms": float(cloud_ms),
            "cloud_conf": cloud_conf,
            "cloud_fallback": cloud_fallback,
            "cloud_used": cloud_used,
            "path_type": arb.path,
            "provisional": arb.provisional,
            "final_pred": int(final_pred),
            "crr_reason": verd.reason,
            "arbiter": arbiter,
        }
        all_rows.append(row)
        print(
            f"[{scenario} {i}] {er['category']}/{Path(er['path']).name} "
            f"conflict={row['conflict']} up={upload} ok={net_ok} path={arb.path} "
            f"final={'NG' if final_pred else 'OK'} gt={'NG' if er['label'] else 'OK'}"
        )

    return all_rows, {"cloud_load_s": cloud_load_s, "arbiter": arbiter}


def _weighted_vote(cons: dict[str, Any]) -> str:
    """Margin-weighted local vote (hand-written, no LLM)."""
    thr = float(cons["thr"])
    w_ng = w_ok = 0.0
    for s, d in zip(cons["node_scores"], cons["node_decisions"]):
        w = abs(float(s) - thr) + 1e-6
        if d == "NG":
            w_ng += w
        else:
            w_ok += w
    return "NG" if w_ng >= w_ok else "OK"


def _consensus_from_dict(cons: dict[str, Any]):
    from src.collab_conflict import MultiNodeConsensus

    return MultiNodeConsensus(
        n_nodes=int(cons["n_nodes"]),
        node_scores=list(cons["node_scores"]),
        node_decisions=list(cons["node_decisions"]),
        thr=float(cons["thr"]),
        majority_decision=cons["majority_decision"],
        n_ng=int(cons["n_ng"]),
        conflict=bool(cons["conflict"]),
        conflict_score=float(cons["conflict_score"]),
        vote_entropy=float(cons["vote_entropy"]),
    )


def _agg(xs: list[float]) -> dict[str, float] | None:
    if not xs:
        return None
    return {
        "n": len(xs),
        "mean": float(st.mean(xs)),
        "p50": float(st.median(xs)),
        "p95": float(np.percentile(xs, 95)),
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


def _f1(yt: list[int], yp: list[int]) -> float:
    return float(f1_score(yt, yp, zero_division=0)) if yt else 0.0


def summarize(rows: list[dict[str, Any]], scenario: str) -> dict[str, Any]:
    rs = [r for r in rows if r["scenario"] == scenario]
    yt = [r["label"] for r in rs]
    y_edge = [r["edge_pred"] for r in rs]
    y_final = [r["final_pred"] for r in rs]
    conflicts = [r for r in rs if r["conflict"]]
    n_conf = len(conflicts)
    arb_ok = sum(1 for r in conflicts if r["upload"] and r["net_ok"])
    fail_safe = sum(1 for r in conflicts if not (r["upload"] and r["net_ok"]))
    auroc = float(roc_auc_score(yt, [r["edge_score"] for r in rs])) if len(set(yt)) > 1 else float("nan")

    return {
        "scenario": scenario,
        "n": len(rs),
        "conflict_ratio": n_conf / max(1, len(rs)),
        "arbitration_success_rate": arb_ok / max(1, n_conf),  # M7 解决成功率
        "fail_safe_divert_rate": fail_safe / max(1, n_conf),
        "resolution_rate": (arb_ok + fail_safe) / max(1, n_conf),
        "edge_consensus_f1": _f1(yt, y_edge),
        "final_f1": _f1(yt, y_final),
        "edge_auroc": auroc,
        "path_counts": {
            k: sum(1 for r in rs if r["path_type"] == k)
            for k in ("LOCAL", "CLOUD_ARBITRATED", "FAIL_SAFE_DIVERT")
        },
        "edge_ms": _agg([r["edge_ms"] for r in rs]),
        "cloud_ms_when_arb": _agg([r["cloud_ms"] for r in rs if r["path_type"] == "CLOUD_ARBITRATED"]),
        "net_ms_when_upload": _agg([r["net_ms"] for r in rs if r["upload"]]),
    }


def _md(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    s = summary
    return "\n".join(
        [
            "# Multi-node conflict + arbitration",
            "",
            f"Measured: {time.strftime('%Y-%m-%d %H:%M')}. scenario={s['scenario']} "
            f"nodes={meta.get('n_nodes')} arbiter={meta.get('arbiter')} augs={meta.get('augs')}.",
            "",
            "## Contest metrics",
            "",
            f"| metric | value | target |",
            f"|---|---:|---|",
            f"| M6 conflict ratio | {s['conflict_ratio']:.2%} | <=5% |",
            f"| M7 arbitration success rate | {s['arbitration_success_rate']:.2%} | >=90% |",
            f"| fail-safe divert rate (conflict, cloud unreachable) | {s['fail_safe_divert_rate']:.2%} | — |",
            f"| overall resolution rate | {s['resolution_rate']:.2%} | — |",
            "",
            "## Detection",
            "",
            f"| metric | value |",
            f"|---|---:|",
            f"| edge consensus F1 | {s['edge_consensus_f1']:.3f} |",
            f"| final F1 (after arbitration) | {s['final_f1']:.3f} |",
            f"| edge AUROC | {s['edge_auroc']:.3f} |",
            "",
            "## Path distribution",
            "",
            f"| path | n |",
            f"|---|---:|",
            f"| LOCAL | {s['path_counts']['LOCAL']} |",
            f"| CLOUD_ARBITRATED | {s['path_counts']['CLOUD_ARBITRATED']} |",
            f"| FAIL_SAFE_DIVERT | {s['path_counts']['FAIL_SAFE_DIVERT']} |",
            "",
            "Notes: conflict = multi-node augmented disagreement; arbitration = cloud "
            "tiebreaker (VLM detection) or margin-weighted vote when arbiter=vote; "
            "fail-safe = conservative NG + divert for human review.",
            "",
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="bottle")
    ap.add_argument("--scenario", default="good", help="geo scenario: good|fair|weak (or City:access)")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-nodes", type=int, default=None, help="override collab.conflict.n_nodes")
    ap.add_argument("--arbiter", default="vote", choices=["vlm", "vote"])
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--cloud-config", default=str(ROOT / "configs" / "hybrid_lora.yaml"))
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    collab = dict(cfg.get("collab") or {})
    conf_cfg = dict(collab.get("conflict") or {})
    n_nodes = int(args.n_nodes or conf_cfg.get("n_nodes") or 3)
    aug_names = list(conf_cfg.get("augs") or [])
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    up_bytes = int(collab.get("upload_bytes_hard") or 80000)

    edge_rows, edge_meta = score_edges(
        cfg=cfg,
        categories=cats,
        limit=args.limit,
        seed=args.seed,
        device=args.device,
        n_nodes=n_nodes,
        aug_names=aug_names,
    )
    if not edge_rows:
        print("no samples", file=sys.stderr)
        return 1

    cloud_yaml = _load_yaml(Path(args.cloud_config))
    rows, rc_meta = run_route_arbitrate(
        edge_rows=edge_rows,
        collab=collab,
        cloud_cfg=dict(cloud_yaml.get("cloud") or {}),
        prompt=cloud_yaml.get("prompt"),
        scenario=args.scenario,
        device=args.device,
        seed=args.seed,
        arbiter=args.arbiter,
        up_bytes=up_bytes,
    )

    meta = {**edge_meta, **rc_meta}
    summary = summarize(rows, args.scenario)

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "args": vars(args),
        "summary": summary,
        "rows": rows,
    }
    (args.out / "conflict.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md = _md(summary, meta)
    (args.out / "conflict.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"wrote {args.out / 'conflict.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
