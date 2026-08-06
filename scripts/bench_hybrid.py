#!/usr/bin/env python3
"""Hybrid bench: Anomalib edge scores + Qwen-VL-8B cloud review (LLM text).

Env: conda activate base
Prereq: python scripts/export_edge_scores.py  (dinov3) -> edge_scores.json

Example:
  CUDA_VISIBLE_DEVICES=0 python scripts/bench_hybrid.py --config configs/hybrid.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vlm import QwenVLClient


def _det_binary(labels, preds, scores=None):
    out = {
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
    }
    if scores is not None and len(np.unique(labels)) > 1 and len(np.unique(scores)) > 1:
        out["image_auroc"] = float(roc_auc_score(labels, scores))
    else:
        out["image_auroc"] = float("nan")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/hybrid.yaml"))
    parser.add_argument("--category", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-cloud-reviews", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    category = args.category or cfg.get("category", "bottle")
    out_dir = Path(cfg.get("results_dir", "outputs/hybrid")) / category
    edge_path = out_dir / "edge_scores.json"
    if not edge_path.exists():
        raise FileNotFoundError(
            f"Missing {edge_path}. Run first:\n"
            f"  conda activate dinov3 && python scripts/export_edge_scores.py --category {category}"
        )
    edge_pack = json.loads(edge_path.read_text(encoding="utf-8"))
    items = edge_pack["items"]
    thr = float(edge_pack["threshold"])

    cloud_cfg = cfg["cloud"]
    collab = cfg.get("collab", {})
    max_reviews = args.max_cloud_reviews
    if max_reviews is None:
        max_reviews = collab.get("max_cloud_reviews", None)

    from src.network_sim import NetworkSimulator
    from src.vlm.route_agent import RouteAgent, RouteContext, resolve_network_profile

    ra_cfg = dict(collab.get("route_agent") or {})
    use_route_agent = bool(ra_cfg.get("enabled", False))
    net_profile, net_dict = resolve_network_profile(collab)
    up_hard = int(collab.get("upload_bytes_hard", 80000))
    net_sim = NetworkSimulator.from_config(dict(collab.get("network") or {"profile": net_profile}))

    hard_marked_idx = [i for i, it in enumerate(items) if it.get("hard")]
    n_gallery = int(edge_pack.get("n_gallery") or edge_pack.get("n_train") or 16)
    hard_margin = float(collab.get("thr_margin", 0.08))
    device = args.device or cloud_cfg.get("device", "cuda:0")

    # --- RouteAgent phase (optional): decide who wants cloud ---
    want_upload = [False] * len(items)
    route_rows: list[dict | None] = [None] * len(items)
    if use_route_agent:
        print(f"[hybrid] RouteAgent enabled profile={net_profile} model={ra_cfg.get('model_path')}")
        agent = RouteAgent.from_config({**ra_cfg, "device": ra_cfg.get("device") or device})
        # Decide on hard-marked candidates; cold-start (n_gallery==0) → all samples
        cand = list(range(len(items))) if n_gallery <= 0 else hard_marked_idx
        for i in cand:
            it = items[i]
            img = it["path"]
            if not Path(img).exists():
                alt = Path(cfg["data_root"]) / category / "test" / Path(img).parent.name / Path(img).name
                if alt.exists():
                    img = str(alt)
            ctx = RouteContext(
                image=img,
                category=category,
                n_gallery=n_gallery,
                edge_score=float(it["edge_score"]),
                edge_thr=thr,
                edge_decision="NG" if it["edge_pred"] else "OK",
                network_profile=net_profile,
                network=net_dict,
                hard_margin=hard_margin,
            )
            dec = agent.decide(ctx)
            route_rows[i] = dec.to_dict()
            want_upload[i] = bool(dec.upload)
        # free agent VRAM before loading cloud VLM
        del agent
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        upload_idx = [i for i, w in enumerate(want_upload) if w]
    else:
        upload_idx = list(hard_marked_idx)

    if max_reviews is not None and int(max_reviews) >= 0:
        upload_idx = upload_idx[: int(max_reviews)]
    upload_set = set(upload_idx)

    adapter = cloud_cfg.get("adapter_path")
    print(
        f"[hybrid] category={category} n={len(items)} "
        f"hard_marked={edge_pack['n_hard']} route_want={sum(want_upload)} "
        f"cloud_attempts={len(upload_set)} network={net_profile}"
    )
    print(f"[hybrid] cloud VLM = {cloud_cfg['model_path']}")
    if adapter:
        print(f"[hybrid] LoRA adapter = {adapter}")

    client = None
    if upload_set and net_profile != "outage":
        client = QwenVLClient(
            model_path=cloud_cfg["model_path"],
            device=device,
            dtype=cloud_cfg.get("dtype", "bfloat16"),
            max_new_tokens=int(cloud_cfg.get("max_new_tokens", 160)),
            role="cloud",
            prompt=cfg.get("prompt"),
            adapter_path=adapter,
        )

    labels = np.asarray([it["label"] for it in items], dtype=int)
    edge_scores = np.asarray([it["edge_score"] for it in items], dtype=float)
    edge_preds = (edge_scores >= thr).astype(int)

    # B1 edge-only
    b1 = _det_binary(labels, edge_preds, edge_scores)

    s_preds = edge_preds.copy()
    s_scores = edge_scores.copy()
    rows = []
    cloud_lats = []
    n_upload_ok = 0
    n_upload_attempt = 0
    net_outcomes = []

    for i, it in enumerate(items):
        row = {
            "path": it["path"],
            "label": it["label"],
            "edge_score": it["edge_score"],
            "edge_pred": "NG" if it["edge_pred"] else "OK",
            "hard_marked": bool(it.get("hard")),
            "route": route_rows[i],
            "cloud": None,
            "final_decision": "NG" if it["edge_pred"] else "OK",
            "path_type": "LOCAL",
            "network": None,
        }
        if i in upload_set:
            n_upload_attempt += 1
            outcome = net_sim.try_upload(up_hard)
            net_outcomes.append(outcome)
            row["network"] = outcome.to_dict()
            if not outcome.ok:
                row["path_type"] = "LOCAL_NET_FALLBACK"
                print(
                    f"[{n_upload_attempt}/{len(upload_set)}] net fail ({outcome.failed_reason}) "
                    f"→ edge local :: {Path(it['path']).name}"
                )
            elif client is None:
                row["path_type"] = "LOCAL_NET_FALLBACK"
            else:
                img = it["path"]
                if not Path(img).exists():
                    alt = Path(cfg["data_root"]) / category / "test" / Path(img).parent.name / Path(img).name
                    if alt.exists():
                        img = str(alt)
                res = client.infer(img)
                n_upload_ok += 1
                cloud_lats.append(res.latency_ms + outcome.total_net_ms)
                cloud_pred = 1 if res.decision == "NG" else 0
                s_preds[i] = cloud_pred
                s_scores[i] = res.confidence if res.decision == "NG" else (1.0 - res.confidence)
                row["cloud"] = res.to_dict()
                row["final_decision"] = res.decision
                row["path_type"] = "CLOUD_REVIEW"
                print(
                    f"[{n_upload_ok}/{len(upload_set)}] cloud {res.decision}/{res.confidence:.2f} "
                    f"type={res.defect_type} reason={res.reason} :: {Path(img).name}"
                )
        rows.append(row)

    s = _det_binary(labels, s_preds, s_scores)
    net_summary = net_sim.summarize(net_outcomes)

    report = {
        "mode": "hybrid_edge_anomalib_cloud_qwen_vl",
        "category": category,
        "edge": {
            "model": edge_pack.get("edge_model"),
            "backbone": edge_pack.get("edge_backbone"),
            "threshold": thr,
        },
        "cloud": {
            "model_path": cloud_cfg["model_path"],
            "adapter_path": adapter,
        },
        "route_agent": {
            "enabled": use_route_agent,
            "network_profile": net_profile,
            "n_want_upload": int(sum(want_upload)),
        },
        "n": len(items),
        "n_cloud_reviews": n_upload_ok,
        "n_cloud_attempts": n_upload_attempt,
        "hard_upload_ratio": n_upload_ok / max(1, len(items)),
        "communication": {"network": net_summary},
        "contest_mapped": {
            "M4_weak_net_service_keep_rate": 1.0,
            "M4_cloud_upload_success_rate": float(net_summary.get("cloud_upload_success_rate") or 0.0),
        },
        "detection": {
            "B1_edge_only": b1,
            "S_collab": s,
        },
        "latency_ms": {
            "cloud_mean": float(np.mean(cloud_lats)) if cloud_lats else None,
            "cloud_extra": float(collab.get("cloud_extra_latency_ms", 50.0)),
        },
        "rows": rows,
    }
    n_upload = n_upload_ok  # for markdown below

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bench.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # markdown + LLM outputs
    lines = [
        f"# Hybrid Bench — edge Anomalib + cloud Qwen-VL — `{category}`",
        "",
        f"- Edge: {report['edge']['model']}/{report['edge']['backbone']}",
        f"- Cloud: `{report['cloud']['model_path']}`",
        f"- Cloud reviews: **{n_upload}/{len(items)}** ({report['hard_upload_ratio']:.2%})",
        "",
        "## Detection",
        "",
        "| Scheme | AUROC | F1 | P | R | Acc |",
        "|--------|-------|----|---|---|-----|",
        f"| B1 edge-only | {b1['image_auroc']:.4f} | {b1['f1']:.4f} | {b1['precision']:.4f} | {b1['recall']:.4f} | {b1['accuracy']:.4f} |",
        f"| S collab (edge+VLM) | {s['image_auroc']:.4f} | {s['f1']:.4f} | {s['precision']:.4f} | {s['recall']:.4f} | {s['accuracy']:.4f} |",
        "",
        f"- Cloud VLM mean latency: **{report['latency_ms']['cloud_mean']:.1f} ms**"
        if report["latency_ms"]["cloud_mean"]
        else "",
        "",
        "## Cloud LLM outputs (reviewed hard samples)",
        "",
    ]
    llm_lines = [f"# Cloud LLM outputs — `{category}`", ""]
    for row in rows:
        if not row.get("cloud"):
            continue
        c = row["cloud"]
        gt = "NG" if row["label"] == 1 else "OK"
        name = Path(row["path"]).name
        lines += [
            f"### `{name}` (GT={gt}, edge={row['edge_pred']}, score={row['edge_score']:.4f})",
            "",
            f"- Final: **{row['final_decision']}** via `{row['path_type']}`",
            f"- LLM: `{c['decision']}` conf={c['confidence']:.2f} | {c.get('defect_type')} | {c.get('reason')}",
            "",
            "```json",
            (c.get("raw") or "").strip(),
            "```",
            "",
        ]
        llm_lines += [
            f"## {name} (GT={gt})",
            "",
            f"- Edge: {row['edge_pred']} (score={row['edge_score']:.4f})",
            f"- Cloud LLM → final `{row['final_decision']}`",
            "",
            "```json",
            (c.get("raw") or "").strip(),
            "```",
            "",
        ]

    md = "\n".join([x for x in lines if x is not None])
    (out_dir / "bench.md").write_text(md, encoding="utf-8")
    (out_dir / "llm_outputs.md").write_text("\n".join(llm_lines), encoding="utf-8")
    print("\n" + md)
    print(f"Wrote {out_dir / 'bench.md'}")
    print(f"Wrote {out_dir / 'llm_outputs.md'}")


if __name__ == "__main__":
    main()
